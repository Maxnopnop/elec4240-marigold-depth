"""Frozen cloud scale execution. No paid resources or local GPU execution."""
from scale_runtime import *
import platform
import sys
import traceback
import psutil
from importlib.metadata import version

SOURCES=['scale_runtime.py','scale_train.py','scale_evaluate.py','scale_report.py','scale_preflight.py','scale_run.py','scale_queue.py','frozen_functions.py','prepare_scale_cache.py']


def verify_inputs():
    rr=rows();seen={}
    for r in rr:
        for key,digest in [('image',r['image_sha256']),('mask',r['cloud_mask_png_sha256'])]:
            if r[key] not in seen:seen[r[key]]=sha(r[key])
            assert seen[r[key]]==digest,r[key]
    assert read(ROOT/'migration_audit.json')['status']=='passed'
    return rr


def freeze():
    from scale_report import contrasts
    preflight=read(DRIVE/'scale_preflight/passed.json')
    assert preflight['status']=='passed'
    source_hashes={n:sha(ROOT/n) for n in SOURCES}
    assert source_hashes==preflight['sources'],'Code changed after production preflight'
    cache=read(DRIVE/'scale_cache/cache_audit.json')
    assert cache['status']=='complete' and cache['manifest_sha256']==EXPECTED_MANIFEST
    assert cache['source_sha256']==source_hashes['prepare_scale_cache.py']
    plan=read(ROOT/'scale_plan.json')
    assert plan['arms']==ARMS and plan['seeds']==SEEDS and plan['training_image_counts']==SIZES
    assert plan['steps_per_run']==STEPS and plan['checkpoints']==[2048,4096]
    environment={'gpu':torch.cuda.get_device_name(),'python':platform.python_version(),
                 'packages':{k:version(k) for k in ['torch','diffusers','transformers','peft','accelerate']}}
    assert environment['gpu']==cache['environment']['gpu'] and environment['packages']==cache['environment']['packages']
    # The registered 20% optimizer margin and 8h overhead floor cannot shrink.
    step_seconds=max(2.323987743124983,max(np.mean(r['warm_step_seconds']) for r in preflight['results'].values()))
    train_hours=float(step_seconds)*1.2*12*4096/3600
    inference_seconds=max(r['inference_seconds_per_expression'] for r in preflight['results'].values())
    eval_hours=inference_seconds*3*320*24*2/3600
    cache_hours=sum(s['seconds'] for s in cache['sessions'])/3600
    engineering_hours=cache_hours+preflight['seconds']/3600+1 # conservative prior migration/probe/persistence allowance
    overhead_hours=max(8.,engineering_hours+eval_hours+1.)
    budget={'training_hours_with_20percent_margin':train_hours,'evaluation_estimate_hours_with_2x_margin':eval_hours,
            'cache_hours':cache_hours,'engineering_hours_charged':engineering_hours,'overhead_hours':overhead_hours,
            'estimated_total_hours':train_hours+overhead_hours,'cap_hours':48,'passed':train_hours+overhead_hours<=48}
    write(WORK/'budget_review.json',budget)
    assert budget['passed'],'Measured budget exceeds48h: do not launch or alter matrix silently'
    protocol={'version':'v13-cloud-scale','arms':ARMS,'seeds':SEEDS,'sizes':SIZES,'steps':STEPS,'checkpoints':[2048,4096],
        'source_sha256':source_hashes,'plan_sha256':sha(ROOT/'scale_plan.json'),'manifest_sha256':EXPECTED_MANIFEST,
        'cache_audit_sha256':sha(DRIVE/'scale_cache/cache_audit.json'),'preflight_sha256':sha(DRIVE/'scale_preflight/passed.json'),
        'initial_sha256':{str(s):sha(ROOT/f'initial_seed{s}.pt') for s in SEEDS},'environment':environment,'budget':budget,
        'optimizer':'fresh AdamW lr1e-4 wd0.01 gradient clip1; frozen V12 update: latent+pixel+0.25 gate*pair',
        'schedule':'sorted IDs Python Random424113+epoch, common arms/seeds; sentences (step+ann_id)%count',
        'precision':'BF16 model, FP32 trainable parameters; deterministic algorithms/CUBLAS4096:8/cuDNN',
        'evaluation':'Only after all12 runs4096 verified. Fresh320,2expressions/image,matched noise700000+imageID,empty prompt; native bilinear resize then >0',
        'statistics':{'contrasts':contrasts(),'family_size':26,'metrics':['mean_iou','both_targets_selected_rate'],
                      'bootstrap':5000,'sign_flips':10000,'rng':424113,'unit':'image; average two seeds first; pointwise intervals; Holm p-values'},
        'limitations':['two seeds only','same-category COCO subset, not full benchmark or cross-domain validation',
                      'equal steps not equal epochs; no convergence claim','no cross-hardware bitwise claim','negative results retained'],
        'hard_budget_seconds':48*3600,'no_shutdown':True}
    path=WORK/'protocol.json'
    if path.exists():assert read(path)==json.loads(json.dumps(protocol)),'Frozen protocol differs'
    else:write(path,protocol)
    for name in SOURCES:copy_verified(ROOT/name,WORK/'frozen_sources'/name)
    return protocol


def verify(p):
    for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h,n
    assert sha(ROOT/'scale_plan.json')==p['plan_sha256']
    assert sha(ROOT/'cloud_data_manifest.json')==p['manifest_sha256']
    assert sha(DRIVE/'scale_cache/cache_audit.json')==p['cache_audit_sha256']
    for seed,h in p['initial_sha256'].items():assert sha(ROOT/f'initial_seed{seed}.pt')==h


def main():
    assert sys.platform=='linux' and ROOT.is_dir() and DRIVE.is_dir(),'Cloud-only authorized Drive project'
    deterministic();WORK.mkdir(parents=True,exist_ok=True)
    lock=ROOT/'scale_formal_lock.json'
    if lock.exists():
        prior=read(lock)
        try:
            process=psutil.Process(prior['pid'])
            assert process.create_time()!=prior['create_time'] or process.status()=='zombie','Formal process already active'
        except psutil.NoSuchProcess:pass
    write(lock,{'pid':os.getpid(),'create_time':psutil.Process().create_time()})
    p=freeze();verify(p)
    ph=sha(WORK/'protocol.json')
    budget=Budget(WORK/'execution.json',p['budget']['engineering_hours_charged']*3600)
    budget.reserve(1200,'input_and_cache_audit')
    rr=verify_inputs();cache=load_cache()
    budget.settle()
    if not (WORK/'TRAINING_STARTED.json').exists():
        write(WORK/'TRAINING_STARTED.json',{'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocol_sha256':ph})
    from scale_train import train
    for size in SIZES:
        for seed in SEEDS:
            for arm in ARMS:
                verify(p);train(arm,size,seed,rr,cache,budget)
    del cache;gc.collect();torch.cuda.empty_cache()
    verify_training_barrier(ph)
    write(WORK/'training_complete.json',{'runs':12,'protocol_sha256':ph})
    from scale_evaluate import evaluate
    for size in SIZES:
        for seed in SEEDS:
            for arm in ARMS:
                for step in [2048,4096]:
                    verify(p);evaluate(arm,size,seed,step,rr,budget)
    from scale_report import main as report
    budget.reserve(3600,'audit_and_report')
    audit=report(lambda stage,**detail:write(WORK/'status.json',{'stage':stage,**detail}))
    verify(p);budget.settle()
    write(WORK/'completion.json',{'status':'complete','audit':audit,'protocol_sha256':ph,
          'charged_hours':budget.state['charged_seconds']/3600,'shutdown_requested':False})
    print('CLOUD SCALE MATRIX COMPLETE',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        write(WORK/'error.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
        raise
