"""Bounded training-only collapse recovery experiment, separate from frozen baseline."""
import argparse
import ctypes
import gc
import json
import random
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from multitask_v7.engine import setup
from .prepare_training import ROOT, WORK, OUT, sha, write
from .train_baseline import SIZE, rows, latent_prediction, iou

DEST=OUT/'loss_pilot'
CHECKPOINT=WORK/'seed17/adapter_1000.pt'
ARMS=['latent_mse','latent_plus_balanced_pixels']
STEPS=128

def selected():
    train=[r for r in rows() if r['split']=='train']
    image_ids=list(dict.fromkeys(r['image_id'] for r in train))[:8]
    result=[];seen=set()
    for r in train:
        if r['image_id'] in image_ids and r['ann_id'] not in seen:
            result.append(r);seen.add(r['ann_id'])
    assert len(result)==16
    return result

def freeze():
    rr=selected()
    config={'scope':'post-hoc training-only recovery pilot; not generalization or novelty evidence',
        'initialization':'same seed17 step1000 adapter; fresh optimizer in both arms',
        'checkpoint_sha256':sha(CHECKPOINT),'manifest_sha256':sha(OUT/'data_manifest.json'),
        'cache_sha256':sha(WORK/'train_cache.pt'),
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [
            ROOT/'generative_ris_v11/loss_pilot.py',ROOT/'generative_ris_v11/train_baseline.py',
            ROOT/'multitask_v7/engine.py',ROOT/'experiment.py']},
        'selected_sent_ids':[r['sent_id'] for r in rr], 'images':8,'expressions':16,
        'arms':ARMS,'updates':STEPS,'seed':424024,'lr':1e-4,'accumulation':1,
        'rank':4,'resolution':list(SIZE),'optimizer':'AdamW weight_decay0.01, grad_clip1',
        'pixel_loss':'0.5*mean(softplus(-4*d) on foreground)+0.5*mean(softplus(4*d) on background)+soft Dice(sigmoid(4*d)); coefficient1 added to velocity MSE',
        'evaluation':[0,64,128],'threshold':0,'noise':'matched between arms; eval shared per image',
        'selection_rule':'first8 train image groups, first expression per distinct annotation; no dev/reserved used',
        'interpretation':'can these losses recover nonempty masks and fit different targets in the same training images? No held-out efficacy claim.',
        'stop':'finite128 updates per arm; stop on nonfinite/OOM; no automatic parameter search'}
    p=DEST/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==config,'pilot protocol changed'
    else:write(p,config)
    return config

def pixel_loss(decoded,target):
    fg=target>0.5; bg=~fg
    assert fg.any() and bg.any()
    logits=4*decoded
    balanced=.5*F.softplus(-logits)[fg].mean()+.5*F.softplus(logits)[bg].mean()
    probability=logits.sigmoid()
    dice=1-(2*(probability*target).sum()+1)/(probability.sum()+target.sum()+1)
    return balanced+dice

@torch.no_grad()
def evaluate(pipe,cache,rr,targets,arm,step):
    pipe.unet.eval();records=[];dest=DEST/arm/f'step{step}';dest.mkdir(parents=True,exist_ok=True)
    for r in rr:
        rgb=cache['rgb'][r['image_id']].cuda()
        noise=torch.randn(rgb.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(710000+r['image_id']))
        d=latent_prediction(pipe,rgb,cache['text'][r['sent_id']].cuda(),noise)
        pred=(d[0,0]>0).cpu().numpy();truth=targets[r['ann_id']][0,0].numpy()>0
        other=next(x for x in rr if x['image_id']==r['image_id'] and x['ann_id']!=r['ann_id'])
        own=iou(pred,truth);wrong=iou(pred,targets[other['ann_id']][0,0].numpy()>0)
        path=dest/f"{r['ann_id']}.png";Image.fromarray(pred.astype(np.uint8)*255).save(path)
        records.append({'ann_id':r['ann_id'],'iou':own,'distractor_iou':wrong,'empty':not pred.any().item(),
            'target_selected':own>wrong,'prediction':str(path.relative_to(DEST)),'sha256':sha(path)})
    result={'arm':arm,'step':step,'mean_iou':float(np.mean([r['iou'] for r in records])),
        'empty_rate':float(np.mean([r['empty'] for r in records])),
        'target_selection_rate':float(np.mean([r['target_selected'] for r in records])),'records':records}
    write(DEST/arm/f'eval_{step}.json',result);pipe.unet.train();return result

def run():
    protocol=freeze()
    for seed in [17,29,43]:
        assert json.loads((OUT/f'seed{seed}/complete.json').read_text())['status']=='complete','baseline must finish first'
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        rr=selected();cache=torch.load(WORK/'train_cache.pt',map_location='cpu',weights_only=True)
        targets={}
        for r in rr:
            assert sha(WORK/r['mask'])==r['mask_sha256']
            a=np.array(Image.open(WORK/r['mask']).resize((SIZE[1],SIZE[0]),Image.Resampling.NEAREST),copy=True)
            targets[r['ann_id']]=torch.from_numpy(a)[None,None].float()/255
        order=[];rng=random.Random(424024)
        while len(order)<STEPS:
            ids=list(range(len(rr)));rng.shuffle(ids);order.extend(ids)
        for arm in ARMS:
            assert not (DEST/arm/'complete.json').exists(),'refuse overwrite completed pilot'
            pipe,params=setup(424024,CHECKPOINT)
            optimizer=torch.optim.AdamW(params.values(),lr=1e-4,weight_decay=.01)
            evaluate(pipe,cache,rr,targets,arm,0);history=[]
            torch.cuda.reset_peak_memory_stats()
            for step in range(1,STEPS+1):
                r=rr[order[step-1]];rgb=cache['rgb'][r['image_id']].cuda();target=cache['mask'][r['ann_id']].cuda()
                emb=cache['text'][r['sent_id']].cuda();optimizer.zero_grad(set_to_none=True)
                noise=torch.randn(rgb.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(424024000+step))
                begin=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    v=pipe.unet(torch.cat([rgb,noise],1),999,emb).sample.float()
                    latent=F.mse_loss(v,-target);extra=torch.zeros_like(latent)
                    if arm==ARMS[1]:
                        z=pipe.scheduler.step(v,999,noise.float(),eta=0.).prev_sample
                        d=pipe.vae.decode(z.to(torch.bfloat16)/pipe.vae.config.scaling_factor).sample.float().mean(1,keepdim=True)
                        extra=pixel_loss(d,targets[r['ann_id']].cuda())
                    loss=latent+extra
                assert torch.isfinite(loss).item()
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True)
                optimizer.step();torch.cuda.synchronize()
                history.append({'step':step,'latent_loss':latent.item(),'pixel_loss':extra.item(),'gradient_norm':float(norm),'seconds':time.perf_counter()-begin})
                write(DEST/'status.json',{'stage':'running','arm':arm,'step':step,'total':STEPS})
                if step%32==0:
                    adapter={k:p.detach().cpu().clone() for k,p in params.items()}
                    torch.save(adapter,DEST/arm/f'adapter_{step}.pt')
                    torch.save({'step':step,'adapter':adapter,'optimizer':optimizer.state_dict(),'protocol_sha256':sha(DEST/'protocol.json')},DEST/arm/'resume.pt')
                    write(DEST/arm/'history.json',history)
                if step in [64,128]:final=evaluate(pipe,cache,rr,targets,arm,step)
            for record in final['records']:
                p=DEST/record['prediction'];assert sha(p)==record['sha256']
                assert abs(iou(np.array(Image.open(p))>0,targets[record['ann_id']][0,0].numpy()>0)-record['iou'])<1e-12
            peak=torch.cuda.max_memory_allocated()/2**20
            del pipe,params,optimizer;gc.collect();torch.cuda.empty_cache()
            pipe,params=setup(424024,DEST/arm/f'adapter_{STEPS}.pt')
            restored=evaluate(pipe,cache,rr,targets,arm,'restored')
            assert [r['sha256'] for r in restored['records']]==[r['sha256'] for r in final['records']]
            write(DEST/arm/'complete.json',{'status':'complete','peak_allocated_mib':peak,'all16_restored_predictions_exact':True,'mean_training_iou':final['mean_iou']})
            del pipe,params;gc.collect();torch.cuda.empty_cache()
        write(DEST/'status.json',{'stage':'complete','scope':'training-only pilot; no held-out evidence'})
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    try:
        if args.freeze:freeze()
        else:run()
    except Exception:
        write(DEST/'error.json',{'traceback':traceback.format_exc()});raise
