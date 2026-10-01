"""Controlled repeated inference and one-draw fixed-time validation diagnostic."""
import gc,time,subprocess,argparse
from pathlib import Path
import numpy as np
import torch
from experiment import load_pipe,load_sample,cache_latents,infer_marigold,evaluate_depth
from run_expanded import read,write,sha,restore_adapter,utc
from run_final_control import ROOT,ASSETS,WORK,OUT
from train_budget_v5 import train_budget

def telemetry():
    try:return subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    except Exception as e:return type(e).__name__

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    bench=OUT/'cost';protocol={'source_sha256':{f:sha(f) for f in ['benchmark_final_v5.py','train_budget_v5.py','FINAL_EXTENSION_PROTOCOL.md']},'rounds':5,'images':8,'warmup_calls':3,'seed':4285,'budget_seconds':120.,'budget_seeds':[17,29,43],'draw':'draw1'}
    if a.freeze:write(bench/'protocol.json',protocol);print('BENCHMARK_FROZEN',flush=True);return
    assert read(bench/'protocol.json')==protocol
    assert read(OUT/'verification.json')['status']=='passed'
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    manifest=read('results/robustness_v3/split_manifest.json');val=manifest['validation'];rng=np.random.default_rng(4285)
    configs=[(mode,res) for mode in ['base','high512','mixed','low256'] for res in [256,512]]
    orders=[rng.permutation(len(configs)).tolist() for _ in range(5)];write(bench/'inference_order.json',orders)
    for round_idx,order in enumerate(orders):
      for idx in order:
        mode,res=configs[idx];label=f'{mode}_r{res}';dest=bench/'inference'/f'round{round_idx+1}_{label}.json'
        if dest.exists():continue
        pipe=load_pipe(read(ASSETS/'model_path.json')['path'])
        prior=WORK if mode=='low256' else ROOT/'robustness_v3'
        rid=f'draw1_{mode}_seed17'
        if mode!='base':restore_adapter(pipe,'lora',prior/'checkpoints'/rid/'adapter.pt')
        reference_label=('base' if mode=='base' else rid)+f'_r{res}'
        im,_=load_sample(ASSETS,val[0])
        for _ in range(3):infer_marigold(pipe,im,1,res,17+val[0]['id'])
        before=telemetry();records=[]
        for row in val[:8]:
            image,_=load_sample(ASSETS,row);pred,seconds,memory=infer_marigold(pipe,image,1,res,17+row['id'])
            ref=prior/'predictions/validation'/reference_label/f"{row['id']:04d}.npy"
            assert np.array_equal(pred,np.load(ref)),str(ref)
            records.append({'id':row['id'],'seconds':seconds,'peak_mib':memory,'reference_sha256':sha(ref),'exact_prediction_match':True})
        write(dest,{'round':round_idx+1,'condition':label,'before_telemetry':before,'after_telemetry':telemetry(),'records':records})
        del pipe;gc.collect();torch.cuda.empty_cache();print('TIMING_DONE',round_idx+1,label,flush=True)
    order=[(s,m) for s in [17,29,43] for m in ['low256','high512','mixed']];rng.shuffle(order);write(bench/'budget_order.json',order)
    for seed,mode in order:
        label=f'{mode}_seed{seed}';dest=bench/'budget'/label;cp=WORK/'budget_checkpoints'/label/'adapter.pt'
        if (dest/'complete.json').exists():
            mark=read(dest/'complete.json');assert sha(dest/'evaluation.json')==mark['evaluation_sha256'];assert sha(cp)==read(dest/'training.json')['checkpoint_sha256'];continue
        pipe=load_pipe(read(ASSETS/'model_path.json')['path'])
        if (dest/'training.json').exists():
            stats=read(dest/'training.json');assert sha(cp)==stats['checkpoint_sha256'];restore_adapter(pipe,'lora',cp)
        else:
            start=time.perf_counter();resolutions=[256] if mode=='low256' else [512] if mode=='high512' else [256,512]
            caches={r:cache_latents(pipe,manifest['draws']['draw1'],ASSETS,r) for r in resolutions};cache_seconds=time.perf_counter()-start
            before=telemetry();stats=train_budget(pipe,caches,mode,seed,cp.parent,120.);del caches
            assert stats['training_seconds']>=120 and stats['training_seconds']<130
            stats.update({'cache_seconds':cache_seconds,'before_telemetry':before,'after_telemetry':telemetry(),'checkpoint_sha256':sha(cp)})
            write(dest/'training.json',stats)
        records=[];hashes={}
        for res in [256,512]:
          for row in val:
            image,gt=load_sample(ASSETS,row);pred,sec,mem=infer_marigold(pipe,image,1,res,17+row['id'])
            raw=WORK/'budget_predictions'/label/f'r{res}'/f"{row['id']:04d}.npy";raw.parent.mkdir(parents=True,exist_ok=True);np.save(raw,pred)
            hashes[str(raw.relative_to(WORK))]=sha(raw)
            records.append({'id':row['id'],'resolution':res,'inference_seconds':sec,'peak_mib':mem,**evaluate_depth(pred,gt)[0]})
        write(dest/'evaluation.json',records)
        for row in records:
            source=next(x for x in val if x['id']==row['id']);_,gt=load_sample(ASSETS,source)
            raw=WORK/'budget_predictions'/label/f"r{row['resolution']}"/f"{row['id']:04d}.npy"
            actual=evaluate_depth(np.load(raw),gt)[0];assert all(np.isclose(row[k],v,atol=1e-10,rtol=0) for k,v in actual.items())
        write(dest/'complete.json',{'evaluation_sha256':sha(dest/'evaluation.json'),'predictions_sha256':hashes,'metrics_recomputed':64})
        del pipe;gc.collect();torch.cuda.empty_cache();print('BUDGET_DONE',label,stats['steps'],flush=True)
    write(bench/'verification.json',{'status':'passed','exact_timing_prediction_matches':320,'budget_runs':9,'budget_metrics_recomputed':576,'finished_utc':utc()})
    print('COST_AUDIT_PASSED',flush=True)

if __name__=='__main__':main()
