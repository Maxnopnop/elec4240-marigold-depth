"""Bounded training-only V13 engineering pilot; never evaluates held-out images."""
import sys,json,time,random,hashlib,gc,ctypes
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'outputs/marigold-depth'
sys.path.insert(0,str(REPO))
import numpy as np
import torch
from generative_ris_v12.train import update,masks
from multitask_v7.engine import setup
from generative_ris_v11.prepare_training import sha,write
from collections import defaultdict
OUT=Path(__file__).resolve().parent
OLD=ROOT/'work/marigold-local/generative_ris_v11'
V12=REPO/'results/generative_ris_v12'
ARMS=['pixel','pair_always','pair_ready']

def main():
    rr=json.loads((V12/'data_manifest.json').read_text())['records']
    grouped=defaultdict(lambda:defaultdict(list))
    for r in rr:
        if r['split']=='train':grouped[r['image_id']][r['ann_id']].append(r)
    ids=sorted(grouped);random.Random(424113).shuffle(ids);ids=ids[:64]
    rows=[r for r in rr if r['split']=='train' and r['image_id'] in ids]
    pairs=[list(grouped[i].values()) for i in ids]
    protocol={'scope':'engineering and training-only, not effectiveness validation','arms':ARMS,'seed':17,'updates_per_arm':128,'training_images':ids,'original_manifest_sha256':sha(V12/'data_manifest.json'),'initial_adapter_sha256':sha(OLD/'seed17/adapter_1000.pt'),'cache_sha256':sha(ROOT/'work/marigold-local/generative_ris_v12/cache.pt'),'source_sha256':{str(p):sha(p) for p in [Path(__file__),REPO/'generative_ris_v12/train.py',REPO/'multitask_v7/engine.py',REPO/'generative_ris_v11/train_baseline.py',REPO/'experiment.py']},'wall_cap_seconds':7200,'no_heldout_access':True}
    pp=OUT/'pilot_protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:write(pp,protocol)
    start=time.monotonic();results=[]
    for arm in ARMS:
        receipt=OUT/f'pilot_{arm}_complete.json'
        if receipt.exists():
            result=json.loads(receipt.read_text());assert result['protocol_sha256']==sha(pp);results.append(result);continue
        pipe,params=setup(17,OLD/'seed17/adapter_1000.pt')
        cache=torch.load(ROOT/'work/marigold-local/generative_ris_v12/cache.pt',map_location='cpu',weights_only=True)
        assert cache['manifest_sha256']==protocol['original_manifest_sha256']
        targets=masks(rows);opt=torch.optim.AdamW(params.values(),lr=1e-4,weight_decay=.01);history=[]
        checkpoint=OUT/f'pilot_{arm}_resume.pt'
        if checkpoint.exists():
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True);assert saved['protocol_sha256']==sha(pp)
            with torch.no_grad():
                for k,p in params.items():p.copy_(saved['adapter'][k].cuda())
            opt.load_state_dict(saved['optimizer']);history=saved['history'];torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
        pipe.unet.train();torch.cuda.reset_peak_memory_stats()
        for step in range(len(history)+1,129):
            assert time.monotonic()-start<7200,'Pilot wall cap reached'
            groups=pairs[(step-1)%len(pairs)]
            pair=[g[(step+g[0]['ann_id'])%len(g)] for g in groups]
            torch.cuda.synchronize();t=time.perf_counter();row=update(pipe,params,opt,cache,targets,pair,arm,17,step);torch.cuda.synchronize()
            row.update(step=step,seconds=time.perf_counter()-t);history.append(row)
            if step%16==0:
                write(OUT/'status.json',{'stage':'training_only_pilot','arm':arm,'step':step,'total_per_arm':128});print(arm,step,flush=True)
            if step%64==0:
                tmp=checkpoint.with_suffix('.tmp');torch.save({'adapter':{k:p.detach().cpu().clone() for k,p in params.items()},'optimizer':opt.state_dict(),'history':history,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'protocol_sha256':sha(pp)},tmp);tmp.replace(checkpoint)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert all(torch.equal(p.detach().cpu(),saved['adapter'][k]) for k,p in params.items())
        result={'arm':arm,'updates':128,'protocol_sha256':sha(pp),'checkpoint_sha256':sha(checkpoint),'checkpoint_parameters_exact':True,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'median_warm_step_seconds':float(np.median([r['seconds'] for r in history[16:]])),'optimizer_seconds':sum(r['seconds'] for r in history),'all_losses_finite':all(np.isfinite(r['loss']) for r in history),'heldout_evaluation':False}
        write(OUT/f'pilot_{arm}_history.json',history);write(receipt,result);results.append(result)
        del pipe,params,opt,cache,targets,saved;gc.collect();torch.cuda.empty_cache()
    estimate=sum(r['median_warm_step_seconds']*2*3*8192 for r in results)/3600
    write(OUT/'pilot_completion.json',{'status':'complete','results':results,'estimated_18run_8192step_optimizer_hours':estimate,'estimate_excludes_data_cache_evaluation_and_io':True,'formal_training_started':False})
    write(OUT/'status.json',{'stage':'pilot_complete','formal_training_started':False});print('PILOT COMPLETE',flush=True)

if __name__=='__main__':
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:main()
    except Exception as e:write(OUT/'error.json',{'type':type(e).__name__,'message':str(e)});raise
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
