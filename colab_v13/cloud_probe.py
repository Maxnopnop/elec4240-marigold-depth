"""Bounded T4 training compatibility, throughput and exact-resume experiment."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['USE_TF']='0'
import gc, hashlib, json, time, traceback, platform
from pathlib import Path
from collections import defaultdict
import torch
from huggingface_hub import snapshot_download
from importlib.metadata import version
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,data):(OUT/name).write_text(json.dumps(data,indent=2))
def main():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    for name,h in manifest['files'].items():assert sha(ROOT/name)==h,name
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    assert torch.cuda.is_available()
    write('environment.json',{'gpu':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),'native_bf16':torch.cuda.is_bf16_supported(including_emulation=False),
        'python':platform.python_version(),'packages':{k:version(k) for k in ['torch','diffusers','transformers','peft','accelerate']},'input_manifest_sha256':sha(ROOT/'MANIFEST.json')})
    model=snapshot_download('prs-eth/marigold-depth-v1-1',revision='9571e7123e258cf052b4e54241f17971c290e9a8',allow_patterns=['*.json','*/*.json','*/*.txt','*/*fp16.safetensors'])
    (ROOT/'model_path.json').write_text(json.dumps({'path':model}))
    from frozen_functions import setup, update
    data=torch.load(ROOT/'training_only_inputs.pt',map_location='cpu',weights_only=True)
    groups=defaultdict(lambda:defaultdict(list))
    for r in data['rows']:groups[r['image_id']][r['ann_id']].append(r)
    pairs=[[v[0] for v in groups[i].values()] for i in sorted(groups)];assert len(pairs)==64
    start=time.monotonic();results=[]
    for arm in ['pixel','pair_always','pair_ready']:
        pipe,params=setup(17,ROOT/'initial.pt');pipe.unet.train()
        opt=torch.optim.AdamW(params.values(),lr=1e-4,weight_decay=.01);history=[]
        torch.cuda.reset_peak_memory_stats()
        for step in range(1,65):
            assert time.monotonic()-start<7200,'2h engineering cap'
            torch.cuda.synchronize();t=time.perf_counter()
            row=update(pipe,params,opt,data['cache'],data['targets'],pairs[step-1],arm,17,step)
            torch.cuda.synchronize();row.update(step=step,seconds=time.perf_counter()-t);history.append(row)
            if step==32:
                torch.save({'adapter':{k:p.detach().cpu().clone() for k,p in params.items()},'optimizer':opt.state_dict(),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},OUT/f'{arm}_resume32.pt')
            if step==33:
                expected={k:p.detach().cpu().clone() for k,p in params.items()};expected_row={k:v for k,v in row.items() if k not in ['seconds','step']}
            if step%16==0:
                write(f'{arm}_history.json',history);print(arm,step,row['loss'],flush=True)
        peak=torch.cuda.max_memory_allocated()/2**20
        torch.save({k:p.detach().cpu().clone() for k,p in params.items()},OUT/f'{arm}_adapter64.pt')
        del pipe,params,opt;gc.collect();torch.cuda.empty_cache()
        pipe,params=setup(17,ROOT/'initial.pt');pipe.unet.train();opt=torch.optim.AdamW(params.values(),lr=1e-4,weight_decay=.01)
        saved=torch.load(OUT/f'{arm}_resume32.pt',map_location='cpu',weights_only=True)
        with torch.no_grad():
            for k,p in params.items():p.copy_(saved['adapter'][k].cuda())
        opt.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
        row=update(pipe,params,opt,data['cache'],data['targets'],pairs[32],arm,17,33)
        maximum=max((p.detach().cpu()-expected[k]).abs().max().item() for k,p in params.items())
        exact=maximum==0 and row==expected_row
        result={'arm':arm,'updates':64,'warm_mean_seconds':sum(r['seconds'] for r in history[16:])/48,'peak_mib':peak,'resume_exact':exact,'resume_max_parameter_difference':maximum,'same_dtype_as_local':True}
        results.append(result);write('results.json',results)
        assert exact,('Restored update differs',result)
        del pipe,params,opt,saved,expected;gc.collect();torch.cuda.empty_cache()
    write('completion.json',{'status':'engineering_complete','results':results,'heldout_evaluated':False,'formal_training_started':False,
        'note':'Does not establish cross-hardware bitwise equivalence or segmentation generalization.'})
    print('CLOUD ENGINEERING COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        write('error.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
