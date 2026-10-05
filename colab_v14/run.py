"""One owned finite Colab process: prepare, verify, freeze, train, evaluate."""
from runtime import *
import sys,subprocess,psutil,traceback,platform
from importlib.metadata import version
SOURCES=['runtime.py','train.py','evaluate.py','report.py','preflight.py','run.py','model_setup.py','prepare.py','prepare_migration.py','frozen_functions.py','io_utils.py']

def freeze(prior):
    pre=read(DRIVE/'preflight/passed.json')
    sources={n:sha(ROOT/n) for n in SOURCES}
    assert all(sources[n]==h for n,h in pre['sources'].items())
    results=pre['results']
    assert len({r['initialization_audit']['lora_sha256'] for r in results.values()})==1
    assert len({r['initialization_audit']['architecture_shapes_sha256'] for r in results.values()})==1
    rate=max(float(np.mean(r['warm_step_seconds'])) for r in results.values())
    inference=max(r['inference_seconds_per_expression'] for r in results.values())
    engineering=read(DRIVE/'engineering.json')['charged_seconds']+pre['seconds']
    overhead=max(8*3600,engineering+inference*3*320*36*2+3600)
    estimated=prior+rate*1.2*18*2048+overhead
    budget={'prior_v13_seconds':prior,'engineering_seconds':engineering,'seconds_per_update':rate,
        'train_seconds_with_20percent_margin':rate*1.2*18*2048,'overhead_seconds':overhead,
        'total_estimated_hours_including_prior':estimated/3600,'cap_hours':50,'passed':estimated<=180000}
    write(DRIVE/'budget_review.json',budget)
    assert budget['passed'],'Measured total exceeds cumulative 50h; no matrix change or launch'
    from report import contrasts
    p={'version':'v14-lora-initialization','arms':ARMS,'seeds':SEEDS,'sizes':SIZES,'steps':2048,'checkpoints':[1024,2048],
       'sources':sources,'data':read(DRIVE/'data_receipt.json'),'models':read(DRIVE/'model_receipts.json'),
       'cache_audit_sha256':sha(OLD_DRIVE/'scale_cache/cache_audit.json'),
       'environment':{'gpu':torch.cuda.get_device_name(),'python':platform.python_version(),'packages':{n:version(n) for n in ['torch','diffusers','transformers','peft','accelerate']}},
       'budget':budget,'hard_budget_seconds':180000,
       'initialization':'A random BF16 UNet; B pinned community SD2 768 v-prediction weights, conv_in repeat/2; C Marigold depth. Common C architecture, VAE, text encoder, tokenizer and scheduler. New identical LoRA per seed.',
       'optimizer':'AdamW lr1e-4 wd0.01 clip1, LoRA rank4 alpha4, 829952 FP32 trainable params, BF16 frozen weights, checkpointing enabled',
       'loss':'same latent MSE + balanced pixel BCE/Dice, two expressions per image, no pair loss',
       'schedule':'Random424113+epoch on sorted nested IDs, sentences(step+ann_id)%count; matched seeds/noise/order',
       'evaluation':'all18 runs complete before any holdout inference;320 images2expressions, common noise700000+imageID;native IoU and both-target selection;1024/2048 milestones',
       'statistics':{'contrasts':contrasts(),'family_size':14,'metrics':['mean_iou','both_targets_selected_rate'],'bootstrap':5000,'sign_flips':10000,'unit':'image averaged over3seeds; seed-wise effects also reported; intervals conditional on tested seeds'},
       'limitations':['not fully from scratch: shared pretrained encoders','frozen random backbone with LoRA is not a fully optimized random model','community B mirror lineage not independently authenticated against unavailable original bytes','C versus B also differs in training recipe; not pure objective causality','no conventional segmentation baseline in this phase','small selected COCO subset; not benchmark or cross-domain proof','equal updates not equal epochs or exact compute; no convergence claim','downstream compute excludes historic pretraining costs','three seeds; retain negative results']}
    write(WORK/'protocol.json',p)
    for n in SOURCES:copy_verified(ROOT/n,WORK/'frozen_sources'/n)
    return p

def main():
    assert sys.platform=='linux' and (OLD_DRIVE/'PAUSE').exists()
    assert torch.cuda.is_available() and torch.cuda.get_device_name()=='Tesla T4','Free T4 required'
    DRIVE.mkdir(exist_ok=True)
    lock=ROOT/'process.json'
    if lock.exists():
        a=read(lock)
        try:
            p=psutil.Process(a['pid']);assert p.create_time()!=a['create_time'] or p.status()=='zombie','Existing V14 process active'
        except psutil.NoSuchProcess:pass
    write(lock,{'pid':os.getpid(),'create_time':psutil.Process().create_time()})
    # Take the authoritative old ledger; never reset already charged compute.
    entries=sorted((OLD_DRIVE/'scale_experiment/execution_journal').glob('entry_*.json'))
    prior=read(entries[-1])['charged_seconds'] if entries else read(OLD_DRIVE/'scale_experiment/execution.json')['charged_seconds']
    if not (WORK/'protocol.json').exists():
        eng=Budget(DRIVE/'engineering.json',300,cap=7200) # current environment setup allowance
        if not (DRIVE/'data_receipt.json').exists():
            eng.reserve(6000,'model_and_original_data_restore')
            subprocess.run([sys.executable,str(ROOT/'prepare.py')],check=True,timeout=5900)
            eng.settle()
        write(DRIVE/'status.json',{'stage':'production_preflight','heldout_inference':False})
        subprocess.run([sys.executable,str(ROOT/'preflight.py')],check=True,timeout=6500)
        p=freeze(prior)
    else:p=read(WORK/'protocol.json')
    for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
    assert sha(ROOT/'cloud_data_manifest.json')==p['data']['manifest_sha256']
    assert sha(ROOT/'scale_plan.json')==p['data']['plan_sha256']
    assert read(DRIVE/'model_receipts.json')==p['models']
    assert sha(OLD_DRIVE/'scale_cache/cache_audit.json')==p['cache_audit_sha256']
    assert p['environment']['packages']=={n:version(n) for n in p['environment']['packages']},'Environment changed'
    paths=read(ROOT/'model_paths.json')
    for arm,m in p['models'].items():
        for n,h in m['sha256'].items():assert sha(Path(paths[arm])/n)==h,'Model bytes changed'
    deterministic()
    budget=Budget(WORK/'execution.json',prior+p['budget']['engineering_seconds'],cap=180000)
    budget.reserve(900,'cache_and_input_audit')
    rr=rows();seen={}
    for r in rr:
        for key,h in [('image',r['image_sha256']),('mask',r['cloud_mask_png_sha256'])]:
            if r[key] not in seen:seen[r[key]]=sha(r[key])
            assert seen[r[key]]==h
    cache=load_cache();budget.settle()
    if not (WORK/'TRAINING_STARTED.json').exists():
        write(WORK/'TRAINING_STARTED.json',{'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocol_sha256':sha(WORK/'protocol.json')})
    from train import train
    for size in SIZES:
        for seed in SEEDS:
            for arm in ARMS:train(arm,size,seed,rr,cache,budget)
    del cache;gc.collect();torch.cuda.empty_cache()
    verify_training_barrier(sha(WORK/'protocol.json'))
    write(WORK/'TRAINING_COMPLETE.json',{'runs':18,'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    from evaluate import evaluate
    for size in SIZES:
        for seed in SEEDS:
            for arm in ARMS:
                for step in [1024,2048]:evaluate(arm,size,seed,step,rr,budget)
    from report import main as report
    budget.reserve(1800,'all_prediction_audit_and_report')
    result=report(lambda stage,**kw:write(WORK/'status.json',dict(stage=stage,**kw)))
    budget.settle()
    write(WORK/'COMPLETE.json',dict(result,status='complete',protocol_sha256=sha(WORK/'protocol.json'),charged_hours=budget.state['charged_seconds']/3600))
    write(WORK/'status.json',{'stage':'complete'})

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(DRIVE/'error.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
        write(DRIVE/'status.json',{'stage':'stopped','error':str(e)})
        raise
