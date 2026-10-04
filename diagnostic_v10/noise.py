import gc
import hashlib
import time
import numpy as np
import torch
import torch.nn.functional as F
from multitask_v7 import engine
from multitask_v7.common import encode_depth,sample,depth_metrics,normal_metrics
from reliability_v8.core import native_weights,shuffled_weights,angles
from reliability_v8.gpu_check import paired_loss
from training_v9.run import atomic_torch,random_state,restore_random
from .common import *


def prepare():
    p=verify();torch.set_num_threads(2)
    if (OUT/'noise_data.json').exists():
        meta=read(OUT/'noise_data.json')
        for name,digest in meta['files'].items():assert sha(WORK/name)==digest,name
        return meta
    camera=read(V7OUT/'camera.json')['intrinsics'];ray=rays(480,640,camera)
    pipe,params=engine.setup(17);caches={};files={};quality=[];eval_rows=[]
    for split,seeds in [('train',p['noise']['train_scene_seeds']),('eval',p['noise']['evaluation_scene_seeds'])]:
        conditions=CONDITIONS if split=='train' else ['clean']
        for condition in conditions:
            cache=[]
            for seed in seeds:
                for kind in SURFACES:
                    case=synthetic_case(kind,seed,condition,ray);name=f'{split}/{condition}/{kind}_{seed}.npz'
                    path=WORK/name;path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**case);files[name]=sha(path)
                    key=f'{kind}_{seed}'
                    if split=='eval':eval_rows.append(dict(id=key,file=name));continue
                    with torch.no_grad():
                        rgb=engine.encode(pipe,engine.resized_rgb(case['image'])).float().cpu()
                        d=F.interpolate(torch.from_numpy(case['observed']).to('cuda')[None,None],size=engine.SIZE,mode='bilinear',align_corners=False,antialias=True)
                        n=F.normalize(F.interpolate(torch.from_numpy(case['target_normal']).to('cuda')[None],size=engine.SIZE,mode='area'),dim=1,eps=1e-6)
                        mask=F.interpolate(torch.from_numpy(case['mask']).to('cuda').float()[None,None],size=engine.SIZE,mode='area')
                        weights,support=native_weights(case['weight'],case['mask']);oracle,oracle_support=native_weights(case['oracle'],case['mask'])
                        assert np.array_equal(support.numpy(),oracle_support.numpy())
                        item=dict(id=key,rgb=rgb,depth=engine.encode(pipe,encode_depth(d).repeat(1,3,1,1)).float().cpu(),
                                  normal=engine.encode(pipe,n).float().cpu(),depth_mask=(F.avg_pool2d(mask,8,8)>.99).cpu(),
                                  normal_mask=(F.avg_pool2d(mask,8,8)>.5).cpu(),geo_mask=support,
                                  weighted=weights,oracle=oracle,uniform=torch.ones_like(weights),
                                  shuffled=torch.from_numpy(shuffled_weights(weights.numpy(),support.numpy(),90000+seed+SURFACES.index(kind))))
                        assert item['depth_mask'].sum()>10 and item['normal_mask'].sum()>10
                        cache.append(item)
                    error=angles(case['target_normal'],case['normal']);valid=case['mask']
                    quality.append(dict(condition=condition,id=key,normal_target_error=float(error[valid].mean()),
                                        weighted_target_error=float(np.average(error[valid],weights=case['weight'][valid])),
                                        oracle_target_error=float(np.average(error[valid],weights=case['oracle'][valid]))))
            if split=='train':
                name=f'cache_{condition}.pt';atomic_torch(WORK/name,cache);files[name]=sha(WORK/name)
    del pipe,params;gc.collect();torch.cuda.empty_cache()
    meta=dict(files=files,evaluation=eval_rows,label_quality=quality,protocol_sha256=sha(OUT/'protocol.json'))
    write(OUT/'noise_data.json',meta);return meta


def evaluate(pipe,rows,dest):
    records=[];dest.mkdir(parents=True,exist_ok=True)
    for index,row in enumerate(rows):
        case=sample(WORK/row['file']);pred,_=engine.predict(pipe,case['image'],['depth','normal'],93000+index)
        record=dict(id=row['id'],tasks={})
        for task,value in pred.items():
            path=dest/f"{row['id']}_{task}.npy";np.save(path,value)
            metrics=depth_metrics(value[0],case['depth'],case['mask']) if task=='depth' else normal_metrics(value,case['normal'],case['mask'])
            record['tasks'][task]={**metrics,'prediction_sha256':sha(path)}
        records.append(record)
    return records


def main():
    p=verify();torch.set_num_threads(2);meta=prepare();ph=sha(OUT/'protocol.json')
    for condition in CONDITIONS:
        cache=torch.load(WORK/f'cache_{condition}.pt',map_location='cpu',weights_only=True)
        for rule in RULES:
            for seed in SEEDS:
                name=f'{condition}_{rule}_seed{seed}';dest=OUT/'noise_runs'/name;local=WORK/'noise_runs'/name
                dest.mkdir(parents=True,exist_ok=True);local.mkdir(parents=True,exist_ok=True)
                if (dest/'complete.json').exists():
                    done=read(dest/'complete.json')
                    for key,file in [('training_sha256','training.json'),('evaluation_sha256','evaluation.json')]:assert done[key]==sha(dest/file)
                    assert done['checkpoint_sha256']==sha(local/'adapter.pt')
                    continue
                pipe,params=engine.setup(seed);optimizer=torch.optim.AdamW(params.values(),lr=1e-4,weight_decay=.01)
                initialization_sha=hashlib.sha256(b''.join(v.detach().float().cpu().numpy().tobytes() for v in params.values())).hexdigest()
                initial=OUT/f'initial_seed{seed}.json'
                if not initial.exists():
                    write(initial,dict(initialization_sha256=initialization_sha,records=evaluate(pipe,meta['evaluation'],WORK/f'initial_seed{seed}')))
                assert read(initial)['initialization_sha256']==initialization_sha
                rng=np.random.default_rng(seed);order=np.concatenate([rng.permutation(16)[::-1] for _ in range(4)]).tolist()
                history=[];seconds=0.;resume=local/'resume.pt'
                if resume.exists():
                    mark=read(local/'resume.json');assert mark['sha256']==sha(resume)
                    state=torch.load(resume,map_location='cpu',weights_only=True);assert state['protocol_sha256']==ph
                    with torch.no_grad():
                        for key,param in params.items():param.copy_(state['adapter'][key].to('cuda'))
                    optimizer.load_state_dict(state['optimizer']);restore_random(state['random']);history=state['history'];seconds=state['seconds'];del state
                pipe.unet.train();pipe.vae.eval();torch.cuda.reset_peak_memory_stats()
                for step in range(len(history),64):
                    item=cache[order[step]];optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
                    loss,parts,geo=paired_loss(pipe,item,seed+1000*step,item[rule],.1)
                    assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.)
                    assert torch.isfinite(norm);optimizer.step();torch.cuda.synchronize();seconds+=time.perf_counter()-start
                    history.append(dict(step=step+1,id=item['id'],loss=float(loss.detach()),geometry=float(geo.detach()),
                                        task_losses={k:float(v.detach()) for k,v in parts.items()},gradient_norm=float(norm)))
                    del loss,parts,geo
                    if (step+1)%16==0:
                        atomic_torch(resume,dict(adapter={n:v.detach().float().cpu() for n,v in params.items()},optimizer=optimizer.state_dict(),
                                                random=random_state(),history=history,seconds=seconds,protocol_sha256=ph))
                        write(local/'resume.json',dict(sha256=sha(resume),step=step+1))
                        write(OUT/'noise_status.json',dict(run=name,step=step+1,total=64))
                atomic_torch(local/'adapter.pt',{n:v.detach().float().cpu() for n,v in params.items()})
                stats=dict(condition=condition,rule=rule,seed=seed,steps=64,initialization_sha256=initialization_sha,history=history,
                           optimizer_seconds=seconds,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,protocol_sha256=ph)
                write(dest/'training.json',stats)
                records=evaluate(pipe,meta['evaluation'],local/'predictions');write(dest/'evaluation.json',records)
                del pipe,params,optimizer;gc.collect();torch.cuda.empty_cache()
                pipe,params=engine.setup(seed,local/'adapter.pt');first=sample(WORK/meta['evaluation'][0]['file'])
                restored,_=engine.predict(pipe,first['image'],['depth','normal'],93000)
                for task,value in restored.items():
                    assert np.array_equal(value,np.load(local/'predictions'/f"{meta['evaluation'][0]['id']}_{task}.npy"))
                write(dest/'complete.json',dict(training_sha256=sha(dest/'training.json'),evaluation_sha256=sha(dest/'evaluation.json'),
                                               checkpoint_sha256=sha(local/'adapter.pt'),exact_restored_tasks=['depth','normal'],protocol_sha256=ph))
                del pipe,params;gc.collect();torch.cuda.empty_cache()
                print('NOISE_RUN_COMPLETE',name,seconds,flush=True)
    write(OUT/'noise_status.json',dict(state='complete',runs=24,optimizer_updates=1536))


if __name__=='__main__':main()
