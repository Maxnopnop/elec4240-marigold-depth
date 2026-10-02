"""Exploratory, training-only paired gradient probe; never updates model weights."""
from pathlib import Path
import sys
import json
import hashlib
import time
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from peft import LoraConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment import load_pipe, cache_latents, seed_all


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cosine(a, b):
    denom = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    assert denom > 1e-20
    return float(torch.dot(a, b) / denom)


def main():
    out = ROOT / 'research_v6'
    protocol = json.loads((out/'gradient_protocol.json').read_text())
    assert digest(__file__) == protocol['source_sha256']
    work = ROOT.parents[1]/'work/marigold-local'
    assets = work/'assets'
    manifest = json.loads((ROOT/'results/robustness_v3/split_manifest.json').read_text())
    rows = [r for r in manifest['draws']['draw1'] if r['id'] in protocol['training_ids']]
    rows.sort(key=lambda r: r['id'])
    assert [r['id'] for r in rows] == protocol['training_ids']
    assert not {r['scene'] for r in rows} & {r['scene'] for r in manifest['validation']}
    assert not (out/'gradient_results.json').exists(), 'Do not overwrite completed diagnostics'
    seed_all(17)
    pipe = load_pipe(json.loads((assets/'model_path.json').read_text())['path'])
    caches = {r: cache_latents(pipe, rows, assets, r) for r in [256,512]}
    unet = pipe.unet
    unet.add_adapter(LoraConfig(r=4,lora_alpha=4,init_lora_weights='gaussian',
                               target_modules=['to_q','to_k','to_v','to_out.0']))
    unet.enable_gradient_checkpointing()
    params = {n:p for n,p in unet.named_parameters() if p.requires_grad}
    for p in params.values(): p.data = p.data.float()
    assert sum(p.numel() for p in params.values()) == 829952
    pipe.vae.to('cpu')
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config, rescale_betas_zero_snr=True,
                                        timestep_spacing='trailing')
    unet.train()
    torch.cuda.empty_cache()
    records=[]
    started=time.perf_counter()

    def grad(res, idx, timestep, noise):
        rgb, depth, valid = [v.to('cuda') for v in caches[res][idx]]
        valid = valid.expand(-1,4,-1,-1)
        t=torch.tensor([timestep],device='cuda',dtype=torch.long)
        noisy=scheduler.add_noise(depth,noise,t)
        target=scheduler.get_velocity(depth,noise,t)
        unet.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred=unet(torch.cat([rgb,noisy],1),t,pipe.empty_text_embedding).sample.float()
            loss=F.mse_loss(pred[valid],target.float()[valid])
        loss.backward()
        g=torch.cat([p.grad.detach().float().flatten().cpu() for p in params.values()])
        assert torch.isfinite(g).all() and torch.isfinite(loss)
        return g, float(loss.detach())

    for state, spec in protocol['checkpoints'].items():
        path=work/spec['path']
        assert digest(path)==spec['sha256']
        saved=torch.load(path,map_location='cpu',weights_only=True)
        assert set(saved)==set(params)
        with torch.no_grad():
            for n,p in params.items(): p.copy_(saved[n].to(device='cuda',dtype=torch.float32))
        for idx,row in enumerate(rows):
            for t in protocol['timesteps']:
                gs={256:[],512:[]}; losses={256:[],512:[]}
                for replica in [0,1]:
                    gen=torch.Generator(device='cuda').manual_seed(604240+row['id']*10000+t*2+replica)
                    high=torch.randn(caches[512][idx][1].shape,device='cuda',generator=gen)
                    # 2x2 average times two preserves unit Gaussian variance.
                    low=F.avg_pool2d(high,2,2)*2
                    for res,noise in [(256,low),(512,high)]:
                        g,loss=grad(res,idx,t,noise)
                        gs[res].append(g); losses[res].append(loss)
                records.append({'state':state,'id':row['id'],'timestep':t,
                                'cross_resolution_cosines':[cosine(gs[256][i],gs[512][i]) for i in [0,1]],
                                'same_resolution_independent_noise_cosines':{str(r):cosine(gs[r][0],gs[r][1]) for r in gs},
                                'gradient_norms':{str(r):[float(torch.linalg.vector_norm(g)) for g in gs[r]] for r in gs},
                                'losses':losses})
            print('GRADIENT_PROBE',state,row['id'],flush=True)
        # The diagnostic must not mutate adapter weights.
        assert all(torch.equal(p.detach().cpu(),saved[n]) for n,p in params.items())
        assert digest(path)==spec['sha256']
    groups={}
    for state in protocol['checkpoints']:
        for t in protocol['timesteps']:
            values=[r for r in records if r['state']==state and r['timestep']==t]
            cross=np.array([v for r in values for v in r['cross_resolution_cosines']])
            groups[f'{state}_t{t}']={'cross_mean_cosine':float(cross.mean()),
                                    'cross_negative_fraction':float((cross<0).mean()),
                                    'within_resolution_means':{str(res):float(np.mean([r['same_resolution_independent_noise_cosines'][str(res)] for r in values])) for res in [256,512]}}
    result={'status':'passed','protocol_sha256':digest(out/'gradient_protocol.json'),
            'records':records,'summary':groups,'backward_passes':len(records)*4,
            'elapsed_seconds':time.perf_counter()-started,'parameters_unchanged':True,
            'limits':'Eight training images; two checkpoints from one draw/seed; selected timesteps; descriptive local gradients, not generalization evidence or causal identification.'}
    (out/'gradient_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'summary':groups,'backward_passes':len(records)*4,'parameters_unchanged':True},indent=2),flush=True)


if __name__=='__main__': main()
