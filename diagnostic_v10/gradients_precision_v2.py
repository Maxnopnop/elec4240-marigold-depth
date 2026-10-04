"""Recorded numerical amendment: preserve BF16 probes; inspect flagged probes in FP32."""
import gc
import time
import torch
import torch.nn.functional as F
import numpy as np
from multitask_v7 import engine
from multitask_v7.common import decode_depth,decode_normal
from training_v9.run import cache_for
from reliability_v8.core import geometry_loss
from .gradients import losses
from .common import *


def fp32_losses(pipe,item,noise_seed):
    rgb=item['rgb'].to('cuda');noise=torch.randn(rgb.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(noise_seed))
    parts={};pred={}
    for task in ['depth','normal']:
        target=item[task].to('cuda');mask=item[task+'_mask'].to('cuda').expand_as(target)
        v=pipe.unet(torch.cat([rgb,noise],1),999,pipe.task_embeddings[task].float()).sample.float()
        parts[task]=F.mse_loss(v[mask],-target[mask])
        latent=engine.clean_latent(pipe,v,noise)
        value=pipe.vae.decode(latent/pipe.vae.config.scaling_factor,return_dict=False)[0]
        pred[task]=decode_depth(value.mean(1,keepdim=True)) if task=='depth' else decode_normal(value)
    ray=rays(*engine.SIZE,read(V7OUT/'camera.json')['intrinsics'],device='cuda')
    for name,w in [('uniform',torch.ones_like(item['weights'])),('weighted',item['weights'])]:
        parts[name+'_geometry']=geometry_loss(pred['depth'],pred['normal'],ray,w.to('cuda'),item['geo_mask'].to('cuda'))
    return parts


def measure(pipe,params,item,noise,fp32=False):
    parts=(fp32_losses if fp32 else losses)(pipe,item,noise);vectors={};parameters=tuple(params.values())
    for name,value in parts.items():
        grads=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
        vectors[name]=torch.cat([(g.detach().float().flatten().cpu() if g is not None else torch.zeros(param.numel())) for g,param in zip(grads,parameters)]).numpy()
        assert np.isfinite(vectors[name]).all()
    total=.5*(parts['depth']+parts['normal'])+.1*parts['weighted_geometry']
    combined=torch.autograd.grad(total,parameters,allow_unused=True)
    actual=torch.cat([(g.detach().float().flatten().cpu() if g is not None else torch.zeros(param.numel())) for g,param in zip(combined,parameters)]).numpy()
    expected=.5*(vectors['depth']+vectors['normal'])+.1*vectors['weighted_geometry']
    discrepancy=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-12))
    assert np.isfinite(discrepancy)
    if fp32:assert discrepancy<.001,('FP32 gradient linearity failure',discrepancy)
    return dict(losses={k:float(v.detach()) for k,v in parts.items()},sum_relative_error=discrepancy,**vector_stats(vectors))


def verify_amendment():
    p=verify();a=read(OUT/'numerical_amendment.json')
    assert a['base_protocol_sha256']==sha(OUT/'protocol.json')
    for name,value in a['source_sha256'].items():assert sha(ROOT/name)==value,name
    return p


def main():
    p=verify_amendment();torch.set_num_threads(2);cache=cache_for(read(V7OUT/'manifest.json'));records=[];start=time.perf_counter()
    for state in p['gradient']['states']:
        seed=int(state.split('seed')[-1]);cp=None if state.startswith('initial') else V9WORK/'checkpoints'/state/'step1280/adapter.pt'
        if cp:assert sha(cp)==read(V9OUT/'runs'/state/'complete.json')['final_checkpoint_sha256']
        pipe,params=engine.setup(seed,cp);pipe.unet.train();pipe.vae.eval();state_records=[]
        for index in p['gradient']['training_indices']:
            for noise in p['gradient']['noise_seeds']:
                result=measure(pipe,params,cache[index],noise)
                state_records.append(dict(state=state,id=cache[index]['id'],training_index=index,noise_seed=noise,
                                          precision_flag=result['sum_relative_error']>=.02,**result))
        del pipe,params;gc.collect();torch.cuda.empty_cache()
        flagged=[r for r in state_records if r['precision_flag']]
        if flagged:
            pipe,params=engine.setup(seed,cp);pipe.unet.float();pipe.vae.float();pipe.unet.train();pipe.vae.eval()
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            for row in flagged:
                row['fp32_reference']=measure(pipe,params,cache[row['training_index']],row['noise_seed'],fp32=True)
            del pipe,params;gc.collect();torch.cuda.empty_cache()
        records.extend(state_records)
        write(OUT/'gradient_progress.json',dict(completed_states=state,records=len(records),precision_flags=sum(r['precision_flag'] for r in records)))
        print('GRADIENT_STATE_DONE',state,len(records),'FP32 references',len(flagged),flush=True)
    assert len(records)==56
    write(OUT/'gradients.json',dict(records=records,seconds=time.perf_counter()-start,optimizer_updates=0,
                                   numerical_amendment_sha256=sha(OUT/'numerical_amendment.json'),
                                   interpretation='BF16 primary probes retained, including precision flags. Flagged probes additionally checked in FP32 with1e-3 linearity tolerance. FP32 is a numerical sensitivity check, not the training precision. Near-zero cosines must not be overinterpreted.'))


if __name__=='__main__':main()
