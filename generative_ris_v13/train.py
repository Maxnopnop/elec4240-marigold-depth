"""Fixed 18-run continuation study; no validation-based choice of updates."""
from .common import *
from .state import schedule, save, restore
import gc
import time
import random
from collections import defaultdict
import numpy as np
from PIL import Image
from multitask_v7.engine import setup, encode
from generative_ris_v11.train_baseline import SIZE, text_prompt
from generative_ris_v12.train import update


def pairs_by_image(rr):
    result = defaultdict(lambda: defaultdict(list))
    for r in rr:
        if r['split'] == 'train': result[r['image_id']][r['ann_id']].append(r)
    assert len(result) == 8192 and all(len(v) == 2 for v in result.values())
    return {i: [sorted(v[a], key=lambda r:r['sent_id']) for a in sorted(v)] for i,v in result.items()}


@torch.no_grad()
def build_cache(pipe, rr, tick):
    """256-image resumable shards, only training rows; old cache is not rewritten."""
    groups = pairs_by_image(rr); ids = sorted(groups)
    mh = sha(WORK/'data_manifest.json')
    oldpath = ROOT/'work/marigold-local/generative_ris_v12/cache.pt'
    assert sha(oldpath) == read(WORK/'pilot_protocol.json')['cache_sha256']
    old = torch.load(oldpath, map_location='cpu', weights_only=True)
    oldrows = {r['sent_id']:r for r in read(REPO/'results/generative_ris_v12/data_manifest.json')['records'] if r['split']=='train'}
    dest = WORK/'cache'; dest.mkdir(exist_ok=True)
    receipt_path = WORK/'cache_audit.json'
    receipt = read(receipt_path) if receipt_path.exists() else {'manifest_sha256':mh,'source_sha256':sha(__file__),'shards':{}}
    assert receipt['manifest_sha256'] == mh
    assert receipt['source_sha256'] == sha(__file__)
    merged = {'rgb':{},'mask':{},'text':{}}
    for start in range(0,len(ids),256):
        tick('caching', images=start, total=len(ids))
        name = f'shard_{start:05d}.pt'; path = dest/name
        if name in receipt['shards']:
            assert sha(path)==receipt['shards'][name]
            shard = torch.load(path,map_location='cpu',weights_only=True)
            assert shard['manifest_sha256']==mh
        else:
            shard = {'manifest_sha256':mh,'rgb':{},'mask':{},'text':{}}
            for iid in ids[start:start+256]:
                for group in groups[iid]:
                    for r in group:
                        aid,sid=r['ann_id'],r['sent_id']
                        if sid in oldrows:
                            source=oldrows[sid]
                            assert all(source[k]==r[k] for k in ['text','image_sha256','mask_sha256'])
                            shard['rgb'][iid]=old['rgb'][iid];shard['mask'][aid]=old['mask'][aid];shard['text'][sid]=old['text'][sid]
                        if iid not in shard['rgb']:
                            a=np.array(Image.open(r['image']).convert('RGB').resize((SIZE[1],SIZE[0]),Image.Resampling.BILINEAR),copy=True)
                            shard['rgb'][iid]=encode(pipe,torch.from_numpy(a).permute(2,0,1)[None].float().cuda()/127.5-1).float().cpu()
                        if aid not in shard['mask']:
                            a=np.array(Image.open(r['mask']).resize((SIZE[1],SIZE[0]),Image.Resampling.NEAREST),copy=True)
                            shard['mask'][aid]=encode(pipe,(torch.from_numpy(a)[None,None].float().cuda()/127.5-1).repeat(1,3,1,1)).float().cpu()
                        assert len(pipe.tokenizer(text_prompt(r['text']),truncation=False).input_ids)<=77, ('text exceeds77',sid)
                        if sid not in shard['text']:
                            tokens=pipe.tokenizer(text_prompt(r['text']),padding='max_length',max_length=77,return_tensors='pt').input_ids.cuda()
                            shard['text'][sid]=pipe.text_encoder(tokens)[0].cpu()
            tmp=path.with_suffix('.tmp');torch.save(shard,tmp);tmp.replace(path)
            receipt['shards'][name]=sha(path);write(receipt_path,receipt)
        for key in merged: merged[key].update(shard[key])
        del shard
    assert len(merged['rgb'])==8192 and len(merged['mask'])==16384
    receipt['status']='complete';receipt['counts']={k:len(v) for k,v in merged.items()};write(receipt_path,receipt)
    del old
    return merged


def load_cache():
    receipt=read(WORK/'cache_audit.json');assert receipt['status']=='complete'
    assert receipt['manifest_sha256']==sha(WORK/'data_manifest.json')
    cache={'rgb':{},'mask':{},'text':{}}
    for name,digest in receipt['shards'].items():
        path=WORK/'cache'/name;assert sha(path)==digest
        shard=torch.load(path,map_location='cpu',weights_only=True)
        for key in cache:cache[key].update(shard[key])
    return cache


def run_training(arm,size,seed,rr,cache,tick):
    name=label(arm,size,seed); dest=WORK/'runs'/name; receipt=OUT/'runs'/name/'trained.json'
    ph=sha(OUT/'protocol.json')
    if receipt.exists():
        done=read(receipt);assert done['protocol_sha256']==ph
        for step,digest in done['checkpoints'].items():assert sha(dest/f'adapter_{step}.pt')==digest
        return
    groups=pairs_by_image(rr);selected=read(WORK/'selection.json')['small_train_images' if size==2048 else 'large_train_images']
    order=schedule(selected);assert len(set(order))==size
    pipe,params=setup(seed,OLD/f'seed{seed}/adapter_1000.pt');pipe.unet.train()
    opt=torch.optim.AdamW(params.values(),lr=1e-4,weight_decay=.01)
    step0=0;history=[]
    if (dest/'resume.pt').exists():step0,history=restore(dest/'resume.pt',params,opt,ph)
    assert [r['image_id'] for r in history]==order[:step0]
    torch.cuda.reset_peak_memory_stats()
    for step in range(step0+1,STEPS+1):
        if step==step0+1 or step%16==0:tick('training',run=name,step=step,total=STEPS)
        pair=[g[(step+g[0]['ann_id'])%len(g)] for g in groups[order[step-1]]]
        # Only two native labels are resized at a time, avoiding a multi-GB mask cache.
        targets={}
        for r in pair:
            a=np.array(Image.open(r['mask']).resize((SIZE[1],SIZE[0]),Image.Resampling.NEAREST),copy=True)
            targets[r['ann_id']]=torch.from_numpy(a)[None,None].float()/255
        torch.cuda.synchronize();begin=time.perf_counter()
        row=update(pipe,params,opt,cache,targets,pair,arm,seed,step)
        torch.cuda.synchronize();row.update(step=step,seconds=time.perf_counter()-begin,image_id=pair[0]['image_id'],sent_ids=[r['sent_id'] for r in pair]);history.append(row)
        if step in [2048,8192]:
            dest.mkdir(parents=True,exist_ok=True)
            path=dest/f'adapter_{step}.pt';tmp=path.with_suffix('.tmp')
            torch.save({k:p.detach().cpu().clone() for k,p in params.items()},tmp);tmp.replace(path)
        if step%128==0:
            # Preserve milestone first: a restart must never skip a missing analysis checkpoint.
            save(dest/'resume.pt',params,opt,step,history,ph)
            write(OUT/'runs'/name/'history.json',history)
            print(name,step,row['loss'],flush=True)
    assert len(set(r['image_id'] for r in history))==size
    assert len(set(r['image_id'] for r in history[:2048]))==2048
    write(receipt,{'status':'trained','updates':STEPS,'unique_images':size,'unique_at_2048':2048,
          'optimizer_seconds':sum(r['seconds'] for r in history),'peak_mib':torch.cuda.max_memory_allocated()/2**20,
          'protocol_sha256':ph,'checkpoints':{str(step):sha(dest/f'adapter_{step}.pt') for step in [2048,8192]}})
    del pipe,params,opt;gc.collect();torch.cuda.empty_cache()
