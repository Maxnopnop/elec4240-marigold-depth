"""Fixed low256 control; explicitly exploratory on already observed cohorts."""
import argparse,gc
from pathlib import Path
import numpy as np
import torch
from experiment import load_pipe,load_sample,cache_latents,infer_marigold,evaluate_depth
from train_scaleup import train
from run_expanded import read,write,sha,restore_adapter,utc
from run_robustness import fit_calibration,fixed_metrics
from run_prospective import score

ROOT=Path(r'E:\Codex\2026-09-27\yo\work\marigold-local')
OUT=Path('results/final_extension_v5');WORK=ROOT/'final_extension_v5';ASSETS=ROOT/'assets'
SEEDS=[17,29,43,59,71]
SOURCES=['run_final_control.py','FINAL_EXTENSION_PROTOCOL.md','experiment.py','train_scaleup.py','run_expanded.py','run_robustness.py','run_prospective.py','prospective_metrics.py']

def sample(stage,row):
    if stage=='external':
        p=ASSETS/'external_sun3d_v4'/row['file']
        assert sha(p)==row['sha256']
        with np.load(p) as z:return z['image'],z['depth']
    p=ASSETS/'subset'/row['file'];assert sha(p)==row['sha256']
    return load_sample(ASSETS,row)

def scored(pred,gt,stage,cal):
    if stage=='external':return score(pred,gt,cal)
    m=evaluate_depth(pred,gt)[0]
    if cal:m.update({'calibrated_'+k:v for k,v in fixed_metrics(pred,gt,cal['scale'],cal['shift']).items()})
    return m

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--audit-only',action='store_true');a=ap.parse_args()
    old=read('results/robustness_v3/split_manifest.json');external=read('results/prospective_v4/manifest.json')['samples']
    stages={'validation':old['validation'],'nyu31':old['fresh31'],'external':external}
    fp={'study':'exploratory_final_extension_v5','sources':{f:sha(f) for f in SOURCES},
        'nyu_manifest':sha('results/robustness_v3/split_manifest.json'),'external_manifest':sha('results/prospective_v4/manifest.json'),
        'historical_commit':'d14701c251ac1df58c9bfb9025287c476b127d3a','seeds':SEEDS,'steps':320}
    if a.freeze:
        assert not (OUT/'runs').exists();write(OUT/'protocol.json',fp);print('CONTROL_FROZEN',flush=True);return
    assert read(OUT/'protocol.json')==fp;protocol_sha=sha(OUT/'protocol.json')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    calibrations=read(OUT/'calibration.json') if (OUT/'calibration.json').exists() else {}
    total=0;artifacts={};restores=read(OUT/'restoration.json') if (OUT/'restoration.json').exists() else {}
    for draw in ['draw1','draw2','draw3']:
      for seed in SEEDS:
        rid=f'{draw}_low256_seed{seed}';rd=OUT/'runs'/rid;cp=WORK/'checkpoints'/rid/'adapter.pt';pipe=None
        if not a.audit_only:
            pipe=load_pipe(read(ASSETS/'model_path.json')['path'])
            if (rd/'training.json').exists():
                tr=read(rd/'training.json');assert sha(cp)==tr['checkpoint_sha256'];restore_adapter(pipe,'lora',cp)
            else:
                import time
                started=time.perf_counter();c={256:cache_latents(pipe,old['draws'][draw],ASSETS,256)};elapsed=time.perf_counter()-started
                tr=train(pipe,c,'low256',seed,cp.parent,320);del c
                tr.update({'draw':draw,'training_ids':[r['id'] for r in old['draws'][draw]],'checkpoint_sha256':sha(cp),'latent_cache_seconds':elapsed,'protocol_sha256':protocol_sha})
                write(rd/'training.json',tr)
            print('CONTROL_TRAIN_READY',rid,flush=True)
        else:assert sha(cp)==read(rd/'training.json')['checkpoint_sha256']
        for stage,rows in stages.items():
          for resolution in [256,512]:
            label=rid+f'_r{resolution}';dest=OUT/'evaluations'/stage/label;raw=WORK/'predictions'/stage/label
            if not (dest/'complete.json').exists():
                assert not a.audit_only
                raw.mkdir(parents=True,exist_ok=True);records=[];hashes={}
                write(OUT/'status.json',{'state':'running','stage':stage,'condition':label,'updated_utc':utc()})
                image,_=sample(stage,rows[0]);infer_marigold(pipe,image,1,resolution,(427200 if stage=='external' else 17)+rows[0]['id'])
                for i,row in enumerate(rows):
                    image,gt=sample(stage,row);pred,seconds,memory=infer_marigold(pipe,image,1,resolution,(427200 if stage=='external' else 17)+row['id'])
                    name=f"{row['id']:04d}.npy";np.save(raw/name,pred);hashes[name]=sha(raw/name)
                    records.append({'id':row['id'],'condition':label,'draw':draw,'seed':seed,'inference_seconds':seconds,'peak_allocated_mib':memory,**scored(pred,gt,stage,calibrations.get(label))})
                    if (i+1)%50==0:print('CONTROL_PROGRESS',stage,label,i+1,len(rows),flush=True)
                write(dest/'per_image.json',records);write(dest/'complete.json',{'protocol_sha256':protocol_sha,'metrics_sha256':sha(dest/'per_image.json'),'prediction_sha256':hashes})
                print('CONTROL_EVAL_COMPLETE',stage,label,len(rows),flush=True)
            mark=read(dest/'complete.json');assert mark['protocol_sha256']==protocol_sha and sha(dest/'per_image.json')==mark['metrics_sha256']
            recs=read(dest/'per_image.json');assert [r['id'] for r in recs]==[r['id'] for r in rows]
            for row,rec in zip(rows,recs):
                name=f"{row['id']:04d}.npy";assert sha(raw/name)==mark['prediction_sha256'][name]
                pred=np.load(raw/name);_,gt=sample(stage,row)
                actual=scored(pred,gt,stage,None if stage=='validation' else calibrations[label])
                assert all(np.isclose(v,rec[k],atol=1e-10,rtol=0) for k,v in actual.items());total+=1
            if stage=='validation':
                cal=fit_calibration([(np.load(raw/f"{r['id']:04d}.npy"),sample(stage,r)[1]) for r in rows])
                if label in calibrations:assert calibrations[label]==cal
                calibrations[label]=cal;write(OUT/'calibration.json',calibrations)
            for name in ['per_image.json','complete.json']:artifacts[str((dest/name).relative_to(OUT))]=sha(dest/name)
        if pipe is not None:del pipe;gc.collect();torch.cuda.empty_cache()
        if rid not in restores and not a.audit_only:
            pipe=load_pipe(read(ASSETS/'model_path.json')['path']);restore_adapter(pipe,'lora',cp)
            row=stages['validation'][0];image,_=sample('validation',row)
            for resolution in [256,512]:
                pred,_,_=infer_marigold(pipe,image,1,resolution,17+row['id'])
                assert np.array_equal(pred,np.load(WORK/'predictions/validation'/f'{rid}_r{resolution}'/f"{row['id']:04d}.npy"))
            restores[rid]={'resolutions':[256,512],'max_abs_difference':0};write(OUT/'restoration.json',restores)
            del pipe;gc.collect();torch.cuda.empty_cache()
    assert total==7890 and len(restores)==15
    write(OUT/'verification.json',{'status':'passed','metrics_recomputed':total,'checkpoint_restorations':30,'artifacts':artifacts,'protocol_sha256':protocol_sha})
    write(OUT/'status.json',{'state':'complete','predictions':total,'updated_utc':utc()});print('CONTROL_AUDIT_PASSED',total,flush=True)

if __name__=='__main__':main()
