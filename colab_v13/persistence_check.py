"""Copy an existing checkpoint to approved durable storage and replay in fresh processes."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['USE_TF']='0'
import argparse,hashlib,json,shutil,subprocess,sys,time
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def replay(origin,checkpoint,arm,output):
    import torch
    from collections import defaultdict
    sys.path.insert(0,str(origin))
    from frozen_functions import setup,update
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    data=torch.load(origin/'training_only_inputs.pt',map_location='cpu',weights_only=True)
    groups=defaultdict(lambda:defaultdict(list))
    for r in data['rows']:groups[r['image_id']][r['ann_id']].append(r)
    pairs=[[v[0] for v in groups[i].values()] for i in sorted(groups)]
    pipe,params=setup(17,origin/'initial.pt');pipe.unet.train()
    opt=torch.optim.AdamW(params.values(),lr=1e-4,weight_decay=.01)
    d=torch.load(checkpoint,map_location='cpu',weights_only=True)
    with torch.no_grad():
        for k,p in params.items():p.copy_(d['adapter'][k].cuda())
    opt.load_state_dict(d['optimizer']);torch.set_rng_state(d['rng']);torch.cuda.set_rng_state(d['cuda_rng'])
    row=update(pipe,params,opt,data['cache'],data['targets'],pairs[32],arm,17,33)
    h=hashlib.sha256()
    for k,p in sorted(params.items()):h.update(k.encode());h.update(p.detach().cpu().numpy().tobytes())
    result={'row':row,'next_adapter_tensor_sha256':h.hexdigest(),'checkpoint_sha256':sha(checkpoint)}
    output.write_text(json.dumps(result,indent=2))

def main(origin,dest):
    dest.mkdir(parents=True,exist_ok=True)
    report=[]
    for arm in ['pixel','pair_always','pair_ready']:
        source=origin/'results'/f'{arm}_resume32.pt';target=dest/source.name
        if target.exists():assert sha(target)==sha(source),'Existing durable checkpoint differs'
        else:
            tmp=target.with_suffix('.tmp');shutil.copyfile(source,tmp);assert sha(tmp)==sha(source);tmp.replace(target)
        # Each invocation constructs a new model, optimizer and CUDA RNG state.
        for tag,path in [('original',source),('durable',target)]:
            out=origin/'results'/f'persistence_{arm}_{tag}.json'
            subprocess.run([sys.executable,__file__,'--origin',str(origin),'--replay',str(path),'--arm',arm,'--output',str(out)],check=True,timeout=240)
        a=json.loads((origin/'results'/f'persistence_{arm}_original.json').read_text())
        b=json.loads((origin/'results'/f'persistence_{arm}_durable.json').read_text())
        assert a==b,(arm,'Durable replay differs')
        historical=json.loads((origin/'results'/f'{arm}_history.json').read_text())[32]
        assert a['row']=={k:v for k,v in historical.items() if k not in ['step','seconds']}
        report.append({'arm':arm,'byte_copy_verified':True,'fresh_process_next_update_exact':True,**a})
    receipt={'status':'passed','directory':str(dest),'results':report,'source_sha256':sha(__file__),
        'scope':'Existing training-only step32 restored from Drive in a new process, matches original step33. Does not prove storage survival after account revocation or cross-hardware equality.'}
    (dest/'PERSISTENCE_VERIFIED.json').write_text(json.dumps(receipt,indent=2))
    (origin/'results/PERSISTENCE_VERIFIED.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--origin',type=Path,default=Path('/content/v13_cloud_engineering'));p.add_argument('--dest',type=Path)
    p.add_argument('--replay',type=Path);p.add_argument('--arm');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.replay:replay(a.origin,a.replay,a.arm,a.output)
    else:
        assert a.dest is not None,'Explicit approved durable directory required'
        main(a.origin,a.dest)
