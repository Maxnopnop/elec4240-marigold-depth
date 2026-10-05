"""Finite V13 pipeline. Never requests shutdown; errors preserve state and receipts."""
from .common import *
import argparse
import ctypes
import gc
import platform
import shutil
import time
import psutil
from .train import build_cache, load_cache, run_training
from .evaluate import evaluate
from .report import main as report, contrasts
from multitask_v7.engine import setup


def verify_inputs():
    rr=rows();seen={}
    for r in rr:
        for key in ['image','mask']:
            path=r[key];expected=r[key+'_sha256']
            if path not in seen:seen[path]=sha(path)
            assert seen[path]==expected,path
    assert sha(WORK/'selection.json')=='cd51f861e1ab56d65222c2ffdd9c00c966b08b82237d81a1136348a63458dd83'
    write(WORK/'formal_input_recheck.json',{'files':len(seen),'status':'verified','manifest_sha256':sha(WORK/'data_manifest.json')})
    return rr


def protocol():
    benchmark=read(WORK/'budget_review.json')
    assert benchmark['benchmark_sha256']==sha(WORK/'deterministic_benchmark.json')
    assert benchmark['cache_audit_sha256']==sha(WORK/'cache_audit.json')
    assert benchmark['budget_gate_pass'] and benchmark['estimated_total_hours']<=48,'Measured budget exceeds48h; no formal launch'
    assert read(WORK/'resume_test.json')['status']=='passed'
    for path,digest in read(WORK/'resume_test.json')['source_sha256'].items():assert sha(path)==digest
    for path,digest in read(WORK/'pilot_protocol.json')['source_sha256'].items():assert sha(path)==digest
    sources=list((REPO/'generative_ris_v13').glob('*.py'))
    sources += [REPO/p for p in ['generative_ris_v12/train.py','generative_ris_v12/data.py','generative_ris_v11/train_baseline.py','generative_ris_v11/prepare_training.py','multitask_v7/engine.py','multitask_v7/common.py','experiment.py']]
    p={'version':13,'design':read(WORK/'scale_design.json'),'arms':ARMS,'sizes':SIZES,'seeds':SEEDS,'steps':STEPS,
       'checkpoints':[2048,8192],'initial_sha256':{str(s):sha(OLD/f'seed{s}/adapter_1000.pt') for s in SEEDS},
       'input_sha256':{name:sha(WORK/name) for name in ['selection.json','data_manifest.json','input_audit.json','resume_test.json','deterministic_benchmark.json','budget_review.json','cache_audit.json']},
       'source_sha256':{str(path.relative_to(REPO)):sha(path) for path in sources},
       'optimizer':'AdamW lr1e-4 wd0.01 clip1.0, identical fresh state; frozenV12 loss implementations',
       'schedule':'sorted image IDs shuffled with Python Random(424113+epoch), shared across seeds/arms, deterministic original sentence rotation(step+ann_id)%count',
       'deterministic':'CUBLAS_WORKSPACE_CONFIG=:4096:8; deterministic_algorithms=True; cudnn benchmarkFalse deterministicTrue',
       'evaluation':'all18trained before any fresh320 prediction; both fixed checkpoints; same noise700000+image_id; zero threshold; native bilinearresize; ties fail targetselection',
       'statistics':{'contrasts':contrasts(),'family_size':26,'metrics':['mean_iou','both_targets_selected_rate'],'bootstrap':5000,'sign_flips':10000,'rng':424113,'unit':'image, average3seeds first','intervals':'pointwise95%; Holm adjusts p-values only'},
       'budget':benchmark,'execution_cap_seconds':48*3600,'no_shutdown':True,
       'environment':{'python':platform.python_version(),'torch':torch.__version__,'gpu':torch.cuda.get_device_name()}}
    path=OUT/'protocol.json'
    if path.exists():assert read(path)==json.loads(json.dumps(p)),'Frozen protocol mismatch'
    else:write(path,p)
    return p


def verify(p):
    for path,digest in p['source_sha256'].items():assert sha(REPO/path)==digest,path
    for name,digest in p['input_sha256'].items():assert sha(WORK/name)==digest,name
    for seed,digest in p['initial_sha256'].items():assert sha(OLD/f'seed{seed}/adapter_1000.pt')==digest


def main():
    deterministic();OUT.mkdir(parents=True,exist_ok=True)
    lock=WORK/'formal.run.lock'
    identity={'pid':os.getpid(),'create_time':psutil.Process().create_time(),'command':psutil.Process().cmdline()}
    # A stale lock needs external identity verification before removal, never auto-kill/restart.
    with lock.open('x',encoding='utf-8') as f:json.dump(identity,f)
    start=time.monotonic();prior=read(WORK/'execution.json').get('elapsed_seconds',0) if (WORK/'execution.json').exists() else 0
    def tick(stage,**detail):
        elapsed=prior+time.monotonic()-start
        write(WORK/'execution.json',{'elapsed_seconds':elapsed,'pid':os.getpid()})
        state={'stage':stage,'elapsed_hours':elapsed/3600,'formal_training_started':(WORK/'TRAINING_STARTED.json').exists(),**detail}
        write(WORK/'status.json',state);write(OUT/'status.json',state)
        assert elapsed<48*3600,'48h finite execution cap reached; checkpoints preserved'
        assert not (WORK/'PAUSE').exists(),'User pause marker present'
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        assert shutil.disk_usage(WORK).free>15*2**30,'Need at least15GiB free for caches/predictions/checkpoints'
        tick('verifying_inputs');rr=verify_inputs();p=protocol();verify(p)
        # Protocol/source frozen before any formal cache or optimization.
        pipe,params=setup(17,OLD/'seed17/adapter_1000.pt')
        cache=build_cache(pipe,rr,tick)
        del pipe,params;gc.collect();torch.cuda.empty_cache()
        write(WORK/'TRAINING_STARTED.json',{'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocol_sha256':sha(OUT/'protocol.json')})
        for size in SIZES:
            for seed in SEEDS:
                for arm in ARMS:
                    verify(p);run_training(arm,size,seed,rr,cache,tick)
        del cache;gc.collect();torch.cuda.empty_cache()
        # Strong completion barrier: verify every checkpoint before any new evaluation.
        for size in SIZES:
            for seed in SEEDS:
                for arm in ARMS:
                    d=read(OUT/'runs'/label(arm,size,seed)/'trained.json');assert d['updates']==8192
                    assert d['protocol_sha256']==sha(OUT/'protocol.json')
                    for step,digest in d['checkpoints'].items():assert sha(WORK/'runs'/label(arm,size,seed)/f'adapter_{step}.pt')==digest
        write(OUT/'training_complete.json',{'runs':18,'protocol_sha256':sha(OUT/'protocol.json')})
        for size in SIZES:
            for seed in SEEDS:
                for arm in ARMS:
                    for step in [2048,8192]:
                        verify(p);evaluate(arm,size,seed,step,rr,tick)
        audit=report(tick);verify(p);tick('complete')
        write(OUT/'completion.json',{'status':'complete','audit':audit,'protocol_sha256':sha(OUT/'protocol.json'),'shutdown_requested':False})
        print('V13 COMPLETE',flush=True)
    except Exception as e:
        write(WORK/'formal_error.json',{'type':type(e).__name__,'message':str(e),'elapsed_seconds':prior+time.monotonic()-start})
        raise
    finally:
        write(WORK/'execution.json',{'elapsed_seconds':prior+time.monotonic()-start,'pid':os.getpid(),'process_exited':True})
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink()


if __name__=='__main__':main()
