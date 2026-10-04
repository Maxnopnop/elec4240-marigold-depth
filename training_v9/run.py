"""Sequential GPU training with atomic resume state and exact restore checks."""
import argparse,ctypes,gc,os,random,time
from pathlib import Path
import numpy as np
import torch
from multitask_v7 import engine
from multitask_v7.common import ROOT,ASSETS,WORK as V7WORK,OUT as V7OUT,read,write,sha,sample
from reliability_v8.study import WORK as COMPONENT_WORK,OUT as COMPONENT_OUT
from reliability_v8.gpu_check import paired_loss as weighted_loss
from reliability_v8.core import shuffled_weights
from .protocol import OUT,WORK,verify


def atomic_torch(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.partial');torch.save(obj,temp);os.replace(temp,path)


def random_state():
    return {'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all(),'python':random.getstate()}


def restore_random(state):
    torch.set_rng_state(state['torch_cpu'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in state['torch_cuda']]);random.setstate(state['python'])


def cache_for(manifest):
    path=V7WORK/'train_cache.pt';meta=read(V7WORK/'train_cache.json')
    assert meta['sha256']==sha(path)
    assert meta['fingerprint']=={'manifest_sha256':sha(V7OUT/'manifest.json'),'engine_sha256':sha(ROOT/'multitask_v7/engine.py'),'common_sha256':sha(ROOT/'multitask_v7/common.py')}
    cache=torch.load(path,map_location='cpu',weights_only=True)
    audit={r['id']:r for r in read(COMPONENT_OUT/'training_audit.json')['records']}
    assert [r['id'] for r in cache]==[r['id'] for r in manifest['training']]
    for item,row in zip(cache,manifest['training']):
        weightpath=COMPONENT_WORK/'weights'/row['file'];assert sha(weightpath)==audit[row['id']]['weight_sha256']
        with np.load(weightpath) as z:
            weight=z['weights'];mask=z['mask']
        assert np.array_equal(mask,item['geo_mask'].numpy())
        item['weights']=torch.from_numpy(weight.copy())
        item['shuffled']=torch.from_numpy(shuffled_weights(weight,mask,90000+row['id']))
    return cache


def save_state(path,params,optimizer,rng,order,history,seconds,peak,protocol_hash):
    atomic_torch(path,{'adapter':{n:p.detach().float().cpu() for n,p in params.items()},'optimizer':optimizer.state_dict(),
        'sampler':rng.bit_generator.state,'order':order,'history':history,'optimizer_seconds':seconds,'peak_allocated_mib':peak,
        'random':random_state(),'protocol_sha256':protocol_hash})
    write(path.with_suffix('.json'),{'sha256':sha(path),'step':len(history),'protocol_sha256':protocol_hash})


def one_run(stage,mode,seed,p,manifest,cache):
    name=f'{mode}_seed{seed}';dest=OUT/'runs'/name;local=WORK/'checkpoints'/name
    dest.mkdir(parents=True,exist_ok=True);local.mkdir(parents=True,exist_ok=True)
    ph=sha(OUT/f'protocol_{stage}.json')
    if (dest/'complete.json').exists():
        done=read(dest/'complete.json');assert done['protocol_sha256']==ph
        assert done['training_sha256']==sha(dest/'training.json')
        for step,v in done['metrics_sha256'].items():
            assert sha(dest/f'validation_{step}.json')==v
        assert sha(local/'step1280/adapter.pt')==done['final_checkpoint_sha256']
        print('V9_SKIP_COMPLETE',name,flush=True);return
    print('V9_START',name,flush=True)
    pipe,params=engine.setup(seed);optimizer=torch.optim.AdamW(params.values(),lr=1e-4,weight_decay=.01)
    tasks=[mode] if mode in ['depth','normal'] else ['depth','normal']
    coefficient=p['geometry_coefficients'][mode];rng=np.random.default_rng(seed);order=[];history=[];seconds=0.;peak=0.
    resume=local/'resume.pt'
    if resume.exists():
        meta=read(resume.with_suffix('.json'));assert sha(resume)==meta['sha256']
        state=torch.load(resume,map_location='cpu',weights_only=True);assert state['protocol_sha256']==ph
        with torch.no_grad():
            for n,param in params.items():param.copy_(state['adapter'][n].to('cuda'))
        optimizer.load_state_dict(state['optimizer']);rng.bit_generator.state=state['sampler'];order=state['order'];history=state['history']
        seconds=state['optimizer_seconds'];peak=state['peak_allocated_mib'];restore_random(state['random']);del state
        print('V9_RESUME',name,len(history),flush=True)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()

    def checkpoint(step):
        cp=local/f'step{step}'/'adapter.pt'
        atomic_torch(cp,{n:param.detach().float().cpu() for n,param in params.items()})
        keep=random_state()
        records=engine.evaluate(pipe,manifest['validation'],tasks,WORK/'predictions'/name/f'step{step}')
        write(dest/f'validation_{step}.json',records)
        write(dest/f'checkpoint_{step}.json',{'adapter_sha256':sha(cp),'metrics_sha256':sha(dest/f'validation_{step}.json'),'protocol_sha256':ph})
        restore_random(keep);pipe.unet.train();pipe.vae.eval()

    # If interruption occurred after a state save but before checkpoint evaluation.
    if len(history) in p['evaluation_checkpoints'] and not (dest/f'checkpoint_{len(history)}.json').exists():checkpoint(len(history))
    pipe.unet.train();pipe.vae.eval()
    for step in range(len(history),p['steps']):
        if not order:order=rng.permutation(len(cache)).tolist()
        idx=int(order.pop());item=cache[idx]
        optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
        if mode in ['weighted','shuffled']:
            loss,parts,geo=weighted_loss(pipe,item,seed+1000*step,item['weights' if mode=='weighted' else 'shuffled'],coefficient)
        else:loss,parts,geo=engine.paired_loss(pipe,item,tasks,seed+1000*step,coefficient)
        assert torch.isfinite(loss),('nonfinite loss',name,step)
        loss.backward();gradient=torch.nn.utils.clip_grad_norm_(list(params.values()),1.)
        assert torch.isfinite(gradient),('nonfinite gradient',name,step)
        optimizer.step();torch.cuda.synchronize();seconds+=time.perf_counter()-start
        peak=max(peak,torch.cuda.max_memory_allocated()/2**20)
        history.append({'step':step+1,'id':item['id'],'loss':float(loss.detach()),'task_losses':{k:float(v.detach()) for k,v in parts.items()},'geometry_loss':float(geo.detach()),'gradient_norm':float(gradient)})
        del loss,parts,geo
        if (step+1)%80==0:
            write(OUT/'status.json',{'stage':stage,'state':'training','run':name,'step':step+1,'total':p['steps'],'optimizer_seconds':seconds,'peak_allocated_mib':peak})
            print('V9_TRAIN',name,step+1,round(history[-1]['loss'],6),'seconds',round(seconds,1),flush=True)
        if (step+1)%160==0:save_state(resume,params,optimizer,rng,order,history,seconds,peak,ph)
        if step+1 in p['evaluation_checkpoints']:checkpoint(step+1)
    stats={'mode':mode,'seed':seed,'steps':len(history),'tasks':tasks,'task_examples':{t:len(history) for t in tasks},'train_images':len(cache),
           'trainable_parameters':sum(param.numel() for param in params.values()),'optimizer_seconds':seconds,'peak_allocated_mib':peak,
           'geometry_weight':coefficient,'history':history,'protocol_sha256':ph,
           'cost_scope':'synchronized optimizer iterations including geometry decode/backprop; excludes caching, evaluation and checkpoint I/O'}
    write(dest/'training.json',stats)
    first=manifest['validation'][0];image=sample(ASSETS/'subset'/first['file'])['image']
    del pipe,params,optimizer;gc.collect();torch.cuda.empty_cache()
    restored,restored_params=engine.setup(seed,local/'step1280/adapter.pt')
    predicted,_=engine.predict(restored,image,tasks,73000+first['id'])
    for task,value in predicted.items():
        saved=np.load(WORK/'predictions'/name/'step1280'/f"{first['id']:04d}_{task}.npy")
        assert np.array_equal(value,saved),('restore mismatch',name,task)
    write(dest/'complete.json',{'protocol_sha256':ph,'training_sha256':sha(dest/'training.json'),
        'final_checkpoint_sha256':sha(local/'step1280/adapter.pt'),'exact_restored_tasks':tasks,
        'metrics_sha256':{str(step):sha(dest/f'validation_{step}.json') for step in p['evaluation_checkpoints']}})
    del restored,restored_params;gc.collect();torch.cuda.empty_cache()
    print('V9_COMPLETE',name,flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['b','c'],required=True);a.add_argument('--mode');args=a.parse_args()
    torch.set_num_threads(2);p=verify(args.stage);manifest=read(V7OUT/'manifest.json')
    if args.mode:assert args.mode in p['modes']
    cache=cache_for(manifest)
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for mode in p['modes']:
            if args.mode and mode!=args.mode:continue
            for seed in p['seeds']:one_run(args.stage,mode,seed,p,manifest,cache)
        write(OUT/'status.json',{'stage':args.stage,'state':'finished','selected_mode':args.mode})
    except BaseException as ex:
        write(OUT/'failure.json',{'stage':args.stage,'error_type':type(ex).__name__,'message':str(ex)})
        raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
