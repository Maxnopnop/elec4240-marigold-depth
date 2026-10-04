import gc
import time
import numpy as np
import torch
import torch.nn.functional as F
from multitask_v7 import engine
from training_v9.run import cache_for
from reliability_v8.core import geometry_loss
from .common import *


def losses(pipe,item,noise_seed):
    rgb=item['rgb'].to('cuda')
    noise=torch.randn(rgb.shape,device='cuda',dtype=torch.float32,generator=torch.Generator(device='cuda').manual_seed(noise_seed))
    parts={};pred={}
    for task in ['depth','normal']:
        target=item[task].to('cuda');mask=item[task+'_mask'].to('cuda').expand_as(target)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            v=pipe.unet(torch.cat([rgb,noise],1),999,pipe.task_embeddings[task]).sample.float()
        parts[task]=F.mse_loss(v[mask],-target[mask])
        pred[task]=engine.decode(pipe,engine.clean_latent(pipe,v,noise),task)
    ray=rays(*engine.SIZE,read(V7OUT/'camera.json')['intrinsics'],device='cuda')
    for name,w in [('uniform',torch.ones_like(item['weights'])),('weighted',item['weights'])]:
        parts[name+'_geometry']=geometry_loss(pred['depth'],pred['normal'],ray,w.to('cuda'),item['geo_mask'].to('cuda'))
    return parts


def main():
    p=verify();torch.set_num_threads(2)
    cache=cache_for(read(V7OUT/'manifest.json'));records=[];start=time.perf_counter()
    for state in p['gradient']['states']:
        seed=int(state.split('seed')[-1]);cp=None if state.startswith('initial') else V9WORK/'checkpoints'/state/'step1280/adapter.pt'
        if cp:
            assert sha(cp)==read(V9OUT/'runs'/state/'complete.json')['final_checkpoint_sha256']
        pipe,params=engine.setup(seed,cp);pipe.unet.train();pipe.vae.eval();parameters=tuple(params.values())
        for index in p['gradient']['training_indices']:
            item=cache[index]
            for noise in p['gradient']['noise_seeds']:
                parts=losses(pipe,item,noise);vectors={}
                for name,value in parts.items():
                    grads=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
                    vectors[name]=torch.cat([(g.detach().float().flatten().cpu() if g is not None else torch.zeros(param.numel()))
                                              for g,param in zip(grads,parameters)]).numpy()
                    assert np.isfinite(vectors[name]).all()
                    del grads
                total=.5*(parts['depth']+parts['normal'])+.1*parts['weighted_geometry']
                combined=torch.autograd.grad(total,parameters,allow_unused=True)
                actual=torch.cat([(g.detach().float().flatten().cpu() if g is not None else torch.zeros(param.numel()))
                                  for g,param in zip(combined,parameters)]).numpy()
                expected=.5*(vectors['depth']+vectors['normal'])+.1*vectors['weighted_geometry']
                discrepancy=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-12))
                assert discrepancy<.02,('mixed precision gradient sum discrepancy',state,discrepancy)
                records.append(dict(state=state,id=item['id'],noise_seed=noise,losses={k:float(v.detach()) for k,v in parts.items()},
                                    sum_relative_error=discrepancy,**vector_stats(vectors)))
                del parts,total,combined,actual,expected,vectors
        del pipe,params,parameters;gc.collect();torch.cuda.empty_cache()
        write(OUT/'gradient_progress.json',dict(completed_states=state,records=len(records)))
        print('GRADIENT_STATE_DONE',state,len(records),flush=True)
    assert len(records)==56
    write(OUT/'gradients.json',dict(records=records,seconds=time.perf_counter()-start,optimizer_updates=0,
                                   interpretation='Fixed training-only probes; gradient cosine diagnoses local interference, not its causal effect on validation.'))


if __name__=='__main__':main()
