"""No optimizer updates: verify integration, gradients, identity and GPU fit."""
import time
import numpy as np
import torch
import torch.nn.functional as F
from multitask_v7 import engine
from multitask_v7.common import read, write, sha, rays
from .core import geometry_loss, shuffled_weights
from .study import OUT, WORK, V7OUT, verify_protocol


def paired_loss(pipe,item,noise_seed,weights,coefficient=.1):
    rgb=item['rgb'].to('cuda');noise=torch.randn(rgb.shape,device='cuda',dtype=torch.float32,
        generator=torch.Generator(device='cuda').manual_seed(noise_seed))
    predictions={};parts={}
    for task in ['depth','normal']:
        target=item[task].to('cuda');mask=item[task+'_mask'].to('cuda').expand_as(target)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            velocity=pipe.unet(torch.cat([rgb,noise],dim=1),999,pipe.task_embeddings[task]).sample.float()
        parts[task]=F.mse_loss(velocity[mask],-target[mask])
        predictions[task]=engine.decode(pipe,engine.clean_latent(pipe,velocity,noise),task)
    ray=rays(*engine.SIZE,read(V7OUT/'camera.json')['intrinsics'],device='cuda')
    geo=geometry_loss(predictions['depth'],predictions['normal'],ray,weights.to('cuda'),item['geo_mask'].to('cuda'))
    return torch.stack(list(parts.values())).mean()+coefficient*geo,parts,geo


def main():
    verify_protocol(); torch.set_num_threads(2)
    analysis=read(OUT/'analysis.json')
    rows=read(V7OUT/'manifest.json')['training'][:3]
    pipe,params=engine.setup(17);pipe.unet.train();pipe.vae.eval()
    cache=engine.build_cache(pipe,rows);records=[]
    try:
        item=cache[0]; ones=torch.ones_like(item['geo_mask'],dtype=torch.float32)
        original,_,_=engine.paired_loss(pipe,item,['depth','normal'],73117,.1)
        replacement,_,_=paired_loss(pipe,item,73117,ones)
        torch.testing.assert_close(original.detach(),replacement.detach(),rtol=1e-5,atol=1e-6)
        identity_difference=float(abs(original.detach()-replacement.detach()))
        del original,replacement
        for row,item in zip(rows,cache):
            with np.load(WORK/'weights'/row['file']) as z:
                weights=torch.from_numpy(z['weights']);mask=z['mask']
            assert np.array_equal(mask,item['geo_mask'].numpy())
            shuffled=torch.from_numpy(shuffled_weights(weights.numpy(),mask,90000+row['id']))
            for mode,w,lam in [('uniform',torch.ones_like(weights),.1),('weaker_uniform',torch.ones_like(weights),.03),
                               ('weighted',weights,.1),('shuffled',shuffled,.1)]:
                for p in params.values():p.grad=None
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                loss,parts,geo=paired_loss(pipe,item,73117,w,lam)
                assert torch.isfinite(loss)
                loss.backward()
                grads=[p.grad for p in params.values() if p.grad is not None]
                assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
                norm=float(torch.sqrt(sum(g.float().square().sum() for g in grads)))
                assert norm>0
                torch.cuda.synchronize()
                records.append({'id':row['id'],'mode':mode,'loss':float(loss.detach()),'geometry':float(geo.detach()),
                    'task_losses':{k:float(v.detach()) for k,v in parts.items()},'gradient_norm':norm,
                    'forward_backward_seconds':time.perf_counter()-start,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20})
                print('GPU_INTEGRATION',row['id'],mode,'finite gradient',round(norm,4),flush=True)
                del loss,parts,geo,grads
        write(OUT/'gpu_check.json',{'scope':'three training images; twelve forward/backward checks; zero optimizer steps; no model accuracy result',
            'engineering_gate_passed':analysis['engineering_gate_passed'],'device':torch.cuda.get_device_name(),
            'uniform_vs_v7_loss_abs_difference':identity_difference,'records':records,
            'protocol_sha256':sha(OUT/'component_protocol.json')})
    finally:
        engine.release(pipe)


if __name__=='__main__':main()
