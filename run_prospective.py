"""Frozen external evaluation, using every v3 checkpoint without new selection."""
import argparse, gc, time, platform
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from experiment import load_pipe, infer_marigold, load_sample
from run_expanded import read,write,sha,utc,restore_adapter
from run_robustness import matrix
from prospective_metrics import metrics,aligned_metrics

SOURCE_FILES=['run_prospective.py','prospective_metrics.py','prospective_statistics.py','experiment.py','run_expanded.py',
              'run_robustness.py','robustness_stats.py','crossed_bootstrap_sensitivity.py','PROSPECTIVE_PROTOCOL.md']

def args():
    p=argparse.ArgumentParser()
    p.add_argument('--assets',required=True,type=Path)
    p.add_argument('--prior-work',required=True,type=Path)
    p.add_argument('--data',required=True,type=Path)
    p.add_argument('--work',required=True,type=Path)
    p.add_argument('--results',type=Path,default=Path('results/prospective_v4'))
    p.add_argument('--freeze',action='store_true')
    p.add_argument('--audit-only',action='store_true')
    return p.parse_args()

def configurations():
    return [c for c in matrix() if c['mode']!='expert']+[{'run_id':'expert_default','mode':'expert','draw':None,'seed':None}]

def sample(a,row):
    path=a.data/row['file']
    assert sha(path)==row['sha256']
    with np.load(path) as f:return f['image'],f['depth']

def score(pred,gt,cal,expert=False):
    r=aligned_metrics(pred,gt)
    r.update({'calibrated_'+k:v for k,v in metrics(pred,gt,cal['scale'],cal['shift']).items()})
    if expert:r.update({'native_'+k:v for k,v in metrics(pred,gt).items()})
    return r

def main():
    a=args();out=a.results
    manifest=read(out/'manifest.json');rows=manifest['samples']
    assert len({r['space'] for r in rows})==len(rows)
    prior=Path('results/robustness_v3')
    checkpoint_hashes={c['run_id']:sha(a.prior_work/'checkpoints'/c['run_id']/'adapter.pt') for c in configurations() if c['mode'] in ['mixed','high512']}
    for rid,h in checkpoint_hashes.items():assert h==read(prior/'runs'/rid/'training.json')['checkpoint_sha256']
    fingerprint={'study':'prospective_v4','manifest_sha256':sha(out/'manifest.json'),
                 'decision_sha256':sha(out/'sample_size_decision.json'),
                 'power_results_sha256':sha(out/'power/design_and_results.json'),
                 'source_sha256':{n:sha(n) for n in SOURCE_FILES},'checkpoint_sha256':checkpoint_hashes,
                 'calibration_sha256':sha(prior/'calibration.json'),
                 'expert_calibration_sha256':sha('results/expert_default_diagnostic_v1/calibration.json'),
                 'conditions':63,'scenes':len(rows),'inference_seed_offset':427200,
                 'primary_family':'six aligned AbsRel contrasts; both nested and crossed Holm<.05 required',
                 'stopping_rule':'complete every condition on the frozen cohort; no significance-based stopping'}
    if a.freeze:
        assert not (a.work/'predictions').exists() and not (out/'evaluations').exists()
        for row in rows:sample(a,row)
        if (out/'protocol.json').exists():assert read(out/'protocol.json')['fingerprint']==fingerprint
        else:write(out/'protocol.json',{'frozen_utc':utc(),'fingerprint':fingerprint})
        print('EXTERNAL_PROTOCOL_FROZEN',len(rows),'scenes',63*len(rows),'predictions',flush=True)
        return
    protocol=read(out/'protocol.json');assert protocol['fingerprint']==fingerprint
    frozen_sha=sha(out/'protocol.json')
    cal=read(prior/'calibration.json')['conditions']
    expert_cal=read('results/expert_default_diagnostic_v1/calibration.json')
    for row in rows:sample(a,row)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    restores=read(out/'restoration_checks.json') if (out/'restoration_checks.json').exists() else {}
    validation=read(prior/'split_manifest.json')['validation'][0]
    warm_image,_=load_sample(a.assets,validation)
    total=0;artifacts={}
    for cfg in configurations():
        labels=[cfg['run_id']] if cfg['mode']=='expert' else [cfg['run_id']+f'_r{r}' for r in [256,512]]
        def complete(label):
            dest=out/'evaluations'/label
            if not (dest/'complete.json').exists():return False
            mark=read(dest/'complete.json')
            assert mark['protocol_sha256']==frozen_sha and mark['metrics_sha256']==sha(dest/'per_image.json')
            assert len(mark['prediction_sha256'])==len(rows)
            for name,h in mark['prediction_sha256'].items():assert sha(a.work/'predictions'/label/name)==h
            return True
        needed=[label for label in labels if not complete(label)]
        if needed and a.audit_only:raise RuntimeError('Incomplete evaluation')
        if needed:
            if cfg['mode']=='expert':
                proc=AutoImageProcessor.from_pretrained(str(a.assets/'expert'),local_files_only=True,use_fast=False)
                net=AutoModelForDepthEstimation.from_pretrained(str(a.assets/'expert'),local_files_only=True).to('cuda').eval()
                @torch.inference_mode()
                def infer(image,res,seed):
                    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                    inputs=proc(images=image,return_tensors='pt').to('cuda')
                    pred=net(**inputs).predicted_depth
                    pred=F.interpolate(pred[:,None],size=image.shape[:2],mode='bicubic',align_corners=False)[0,0].cpu().numpy()
                    torch.cuda.synchronize()
                    return pred,time.perf_counter()-start,torch.cuda.max_memory_allocated()/2**20
            else:
                pipe=load_pipe(read(a.assets/'model_path.json')['path'])
                if cfg['mode']!='base':restore_adapter(pipe,'lora',a.prior_work/'checkpoints'/cfg['run_id']/'adapter.pt')
                def infer(image,res,seed):return infer_marigold(pipe,image,1,res,seed)
            for label in needed:
                resolution=int(label.rsplit('_r',1)[1]) if cfg['mode']!='expert' else 518
                warm,_,_=infer(warm_image,resolution,17+validation['id'])
                reference=(a.prior_work.parent/'expert_default_diagnostic_v1/predictions/validation'/f"{validation['id']:04d}.npy" if cfg['mode']=='expert' else a.prior_work/'predictions/validation'/label/f"{validation['id']:04d}.npy")
                assert np.array_equal(warm,np.load(reference)),('Restoration mismatch',label)
                restores[label]={'validation_id':validation['id'],'max_difference':0.,'reference_sha256':sha(reference)}
                write(out/'restoration_checks.json',restores)
                dest=out/'evaluations'/label;raw=a.work/'predictions'/label;raw.mkdir(parents=True,exist_ok=True)
                records=[];hashes={};coeff=expert_cal if cfg['mode']=='expert' else cal[label]
                write(out/'status.json',{'state':'running','label':label,'updated_utc':utc(),'completed_images':0})
                for index,row in enumerate(rows):
                    rgb,gt=sample(a,row)
                    pred,seconds,memory=infer(rgb,resolution,427200+row['id'])
                    filename=f"{row['id']:04d}.npy";np.save(raw/filename,pred);hashes[filename]=sha(raw/filename)
                    records.append({'id':row['id'],'space':row['space'],'condition':label,'draw':cfg['draw'],'seed':cfg['seed'],
                                    'inference_seconds':seconds,'peak_allocated_mib':memory,**score(pred,gt,coeff,cfg['mode']=='expert')})
                    if (index+1)%25==0:print('EXTERNAL_PROGRESS',label,index+1,len(rows),flush=True)
                write(dest/'per_image.json',records)
                write(dest/'complete.json',{'protocol_sha256':frozen_sha,'metrics_sha256':sha(dest/'per_image.json'),'prediction_sha256':hashes,'completed_utc':utc()})
                print('EXTERNAL_CONDITION_COMPLETE',label,len(rows),flush=True)
            if cfg['mode']=='expert':del net,proc
            else:del pipe
            gc.collect();torch.cuda.empty_cache()
        for label in labels:
            assert complete(label)
            dest=out/'evaluations'/label;records=read(dest/'per_image.json')
            assert [r['id'] for r in records]==[r['id'] for r in rows]
            coeff=expert_cal if cfg['mode']=='expert' else cal[label]
            for row,rec in zip(rows,records):
                _,gt=sample(a,row);pred=np.load(a.work/'predictions'/label/f"{row['id']:04d}.npy")
                actual=score(pred,gt,coeff,cfg['mode']=='expert')
                assert all(np.isclose(rec[k],v,rtol=0,atol=1e-10) for k,v in actual.items())
                total+=1
            for name in ['per_image.json','complete.json']:artifacts[str((dest/name).relative_to(out))]=sha(dest/name)
    assert total==63*len(rows) and len(restores)==63
    write(out/'verification.json',{'status':'passed','metrics_recomputed':total,'scenes':len(rows),'conditions':63,
        'protocol_sha256':frozen_sha,'data_hashes_verified':len(rows),'checkpoint_hashes_verified':30,
        'restoration_checks':len(restores),'audited_artifact_sha256':artifacts})
    write(out/'status.json',{'state':'complete','predictions':total,'updated_utc':utc()})
    print('EXTERNAL_AUDIT_PASSED',total,flush=True)

if __name__=='__main__':main()
