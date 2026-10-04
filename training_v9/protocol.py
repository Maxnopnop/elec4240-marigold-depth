import argparse
import datetime
import numpy as np
from multitask_v7.common import ROOT,read,write,sha
from reliability_v8.study import verify_protocol as verify_component, OUT as COMPONENT, WORK as COMPONENT_WORK

OUT=ROOT/'results/reliability_training_v9'
WORK=COMPONENT_WORK.parent/'reliability_training_v9'
SEEDS=[17,29,43]
JOINT=['joint','uniform','weaker_uniform','weighted','shuffled']


def verify(stage):
    verify_component()
    p=read(OUT/f'protocol_{stage}.json')
    for name,value in p['source_sha256'].items():assert sha(ROOT/name)==value,name
    for name,value in p['inputs'].items():assert sha(ROOT/name)==value,name
    return p


def freeze(stage):
    path=OUT/f'protocol_{stage}.json'
    if path.exists():return verify(stage)
    component=verify_component()
    assert read(COMPONENT/'analysis.json')['engineering_gate_passed']
    if stage=='c':
        assert gate()['passed'],'Baseline gate failed; do not silently extend training'
    sources={**component['source_sha256'],**component['v7_source_sha256'],'experiment.py':sha(ROOT/'experiment.py')}
    sources.update({str(f.relative_to(ROOT)).replace('\\','/'):sha(f) for f in sorted((ROOT/'training_v9').glob('*.py'))})
    inputs={f'results/metric_multitask_v7/{name}.json':sha(ROOT/f'results/metric_multitask_v7/{name}.json') for name in ['manifest','camera']}
    inputs.update({f'results/reliability_v8/{name}.json':sha(COMPONENT/f'{name}.json') for name in ['component_protocol','analysis','training_audit','gpu_check']})
    if stage=='c':
        inputs['results/reliability_training_v9/protocol_b.json']=sha(OUT/'protocol_b.json')
        inputs['results/reliability_training_v9/baseline_gate.json']=sha(OUT/'baseline_gate.json')
    p={'version':f'reliability_training_v9_stage_{stage}','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'modes':['depth','normal'] if stage=='b' else JOINT,'seeds':SEEDS,'steps':1280,
       'evaluation_checkpoints':[320,640,1280] if stage=='b' else [1280],
       'training_images':128,'validation_images':32,'validation_status':'previously observed development scenes; exploratory only',
       'size':[192,256],'rank':4,'trainable_parameters':829952,'optimizer':{'type':'AdamW','lr':1e-4,'weight_decay':.01,'clip':1.},
       'loss':'unchanged V7 single-step zero-SNR latent supervision; per-task mean, optional decoded geometry',
       'geometry_coefficients':{'depth':0.,'normal':0.,'joint':0.,'uniform':.1,'weaker_uniform':.03,'weighted':.1,'shuffled':.1},
       'weighting':'frozen V8 fullres sigma1/2/4, temperature10deg; native valid-normalized area resampling; normalized detached weights',
       'shuffle':'permute native weights within each image original support; seed90000+image_id, same map in all model seeds',
       'order':'per-seed numpy default_rng permutations, pop last; same order across all modes;10 complete passes',
       'noise':'seed+1000*zero_based_update; shared across tasks each update',
       'inference':'one-step fixed seed73000+image_id; fixed metric codec; no GT alignment',
       'selection':'fixed final1280checkpoint; report all seeds and all scheduled learning-curve points',
       'baseline_gate':'all runs finite and exact final checkpoint restoration; task-wise mean final error <=1.05*mean320 error; feasibility only, not convergence',
       'resume':'atomic adapter/optimizer/local sampler/global RNG state every160 updates; failure/interrupt retained; no skip without hash validation',
       'statistics':{'family':[[a,b,t,m] for b in ['uniform','weaker_uniform','shuffled'] for a,t,m in [('weighted','depth','abs_rel'),('weighted','normal','mean_deg')]],
                     'bootstrap':'paired crossed seed/scene resampling,20000 replicates,seed934240; centered two-sided tests and Holm6',
                     'other_comparisons':'descriptive only; no non-inferiority claim or fresh-confirmation claim'},
       'diagnostics':'descriptive per-task metrics, geometry agreement, high-depth-gradient/other-valid pixel errors, time/memory; all methods identical masks',
       'stop':'nonfinite/OOM/integrity failure; diagnose before any new protocol. No accuracy-based early stopping within a frozen run.',
       'source_sha256':sources,'inputs':inputs}
    write(path,p)
    if stage=='b':
        historical={str(f.relative_to(ROOT)).replace('\\','/'):sha(f) for directory in [ROOT/'results',ROOT/'reports'] for f in directory.rglob('*') if f.is_file() and OUT not in f.parents}
        write(WORK/'historical_hashes.json',historical)
    print('FROZEN',stage,sha(path),flush=True)
    return p


def gate():
    p=verify('b');out={};summaries={}
    for mode in ['depth','normal']:
        metric='abs_rel' if mode=='depth' else 'mean_deg'
        vals={step:[] for step in p['evaluation_checkpoints']}
        for seed in SEEDS:
            d=OUT/'runs'/f'{mode}_seed{seed}'
            complete=read(d/'complete.json');stats=read(d/'training.json')
            assert complete['exact_restored_tasks']==[mode] and len(stats['history'])==1280
            assert complete['training_sha256']==sha(d/'training.json')
            for step in vals:
                rows=read(d/f'validation_{step}.json')
                assert sha(d/f'validation_{step}.json')==complete['metrics_sha256'][str(step)]
                vals[step].append(float(np.mean([r['tasks'][mode][metric] for r in rows])))
        summaries[mode]={str(step):{'seed_means':v,'mean':float(np.mean(v)),'seed_sd':float(np.std(v,ddof=1))} for step,v in vals.items()}
        out[mode]=bool(np.mean(vals[1280])<=1.05*np.mean(vals[320]))
    result={'passed':all(out.values()),'checks':out,'learning_curves':summaries,
            'interpretation':'Feasibility screen on observed development scenes; does not establish convergence or superiority.',
            'protocol_sha256':sha(OUT/'protocol_b.json')}
    path=OUT/'baseline_gate.json'
    if path.exists():assert read(path)==result
    else:write(path,result)
    print('BASELINE_GATE',result,flush=True)
    return result


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze_b','gate','freeze_c']);x=a.parse_args()
    gate() if x.action=='gate' else freeze(x.action[-1])
