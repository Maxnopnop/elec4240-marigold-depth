"""Five fixed objectives, identical real-image pairs and random noise; no test selection."""
import argparse
import gc
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from multitask_v7.engine import setup, encode
from generative_ris_v11.train_baseline import SIZE, text_prompt, latent_prediction, iou
from .data import ROOT, OLDWORK, OLDOUT, WORK, OUT, sha, write

ARMS=['latent','pixel','pair_always','pair_ramp','pair_ready']
SEEDS=[17,29,43]
STEPS=2048

def records():return json.loads((OUT/'data_manifest.json').read_text())['records']

def pixel_loss(d,y):
    fg=y>.5;bg=~fg;logits=4*d
    # A target lost by fixed resolution still remains in the dataset and evaluation.
    parts=[]
    if fg.any():parts.append(F.softplus(-logits)[fg].mean())
    if bg.any():parts.append(F.softplus(logits)[bg].mean())
    p=logits.sigmoid();dice=(2*(p*y).sum()+1)/(p.sum()+y.sum()+1)
    return torch.stack(parts).mean()+1-dice

def pair_term(ds,ys):
    a=(ys[0]>.5)&(ys[1]<.5);b=(ys[1]>.5)&(ys[0]<.5)
    delta=4*(ds[0]-ds[1]);parts=[]
    if a.any():parts.append(F.softplus(1-delta)[a].mean())
    if b.any():parts.append(F.softplus(1+delta)[b].mean())
    return torch.stack(parts).mean() if parts else delta.sum()*0

def readiness(ds,ys):
    scores=[]
    for d,y in zip(ds,ys):
        p=(4*d).sigmoid()
        scores.append((2*(p*y).sum()+1)/(p.sum()+y.sum()+1))
    # Detached per-pair decoder-fit score: training labels only, no inference gate.
    return ((torch.stack(scores).min().detach()-.1)/.5).clamp(0,1)

def freeze():
    paths=[ROOT/'generative_ris_v12'/name for name in ['data.py','train.py','report.py','run.py']]
    paths += [ROOT/'multitask_v7/engine.py',ROOT/'experiment.py',ROOT/'generative_ris_v11/train_baseline.py']
    protocol={'version':12,'arms':ARMS,'seeds':SEEDS,'updates':STEPS,'training_images':2048,
        'starting_point':'each matching V11 seed adapter1000, optimizer reset equally in all arms; continuation study, not from scratch',
        'initial_sha256':{str(s):sha(OLDWORK/f'seed{s}/adapter_1000.pt') for s in SEEDS},
        'base_model_revision':'9571e7123e258cf052b4e54241f17971c290e9a8','lora_rank':4,'size':list(SIZE),
        'optimizer':'AdamW lr1e-4 weight_decay0.01 clip1.0;2expressions/sameimage/update',
        'pair_schedule':'same shuffle seed424026 across all arms/seeds; choose original sentence by stable update+annotation rotation',
        'noise':'same noise per paired description, seed-dependent but matched across arms',
        'objectives':{'latent':'mean2 velocity MSE','pixel':'latent+mean2 balanced decoded softplus(4*d)+Dice',
            'pair_always':'pixel+0.25*pair_term','pair_ramp':'pixel+0.25*min(step/512,1)*pair_term',
            'pair_ready':'pixel+0.25*clamp((min(detached softDiceA,softDiceB)-0.1)/0.5,0,1)*pair_term'},
        'pair_term':'mean region-balanced softplus(1-signed(4*dA-4*dB)) on exclusive original target pixels; overlap excluded from this auxiliary only; never alter labels',
        'dev_updates':[0,512,1024,2048],'selection':'always last2048; no early stopping or bestcheckpoint selection',
        'heldout':'reserved256 + externalRefCOCO128 + externalRefCOCOplus128, allmethods aftertraining; no retuning',
        'threshold':0,'primary_metrics':['native-resolution meanIoU','both_targets_selected_rate'],
        'secondary':['IoU>0.5','empty_rate','target_selection_rate','matched-minus-swapped IoU','empty-prompt IoU','GPUmemory','trainingtime','inference time'],
        'contrasts':[['pixel','latent'],['pair_always','pixel'],['pair_ramp','pair_always'],['pair_ready','pair_always'],['pair_ready','pair_ramp']],
        'statistics':'reserved image-cluster paired differences, first average3seeds;5000bootstrap95%CI;10000two-sided sign-flip MonteCarlo p plus-one;Holm across5contrasts x2primarymetrics. Conditional on fixed3seeds; seed spread separate. Exploratory, not universal confirmation.',
        'bound':'15runs x2048updates; no additionaladaptivearms; stop/fail onintegrity,nonfinite,OOM;wallclockcap36h excludingpreparation',
        'data_manifest_sha256':sha(OUT/'data_manifest.json'),
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths}}
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol,'Frozen V12 protocol changed'
    else:write(p,protocol)
    return protocol

def verify(protocol):
    assert sha(OUT/'data_manifest.json')==protocol['data_manifest_sha256']
    for p,h in protocol['source_sha256'].items():assert sha(ROOT/p)==h,p
    for seed,h in protocol['initial_sha256'].items():assert sha(OLDWORK/f'seed{seed}/adapter_1000.pt')==h

@torch.no_grad()
def cache_training(pipe,rr):
    path=WORK/'cache.pt'
    if path.exists():
        d=torch.load(path,map_location='cpu',weights_only=True)
        assert d['manifest_sha256']==sha(OUT/'data_manifest.json');return d
    d={'manifest_sha256':sha(OUT/'data_manifest.json'),'rgb':{},'mask':{},'text':{}}
    old=torch.load(OLDWORK/'train_cache.pt',map_location='cpu',weights_only=True)
    oldrows={r['sent_id']:r for r in json.loads((OLDOUT/'data_manifest.json').read_text())['records'] if r['split']=='train'}
    max_tokens=0
    for n,r in enumerate([r for r in rr if r['split']=='train']):
        iid,aid,sid=r['image_id'],r['ann_id'],r['sent_id']
        if sid in oldrows:
            assert r['text']==oldrows[sid]['text'] and r['mask_sha256']==oldrows[sid]['mask_sha256']
            d['rgb'][iid]=old['rgb'][iid];d['mask'][aid]=old['mask'][aid];d['text'][sid]=old['text'][sid]
        if iid not in d['rgb']:
            a=np.array(Image.open(r['image']).convert('RGB').resize((SIZE[1],SIZE[0]),Image.Resampling.BILINEAR),copy=True)
            d['rgb'][iid]=encode(pipe,torch.from_numpy(a).permute(2,0,1)[None].float().cuda()/127.5-1).float().cpu()
        if aid not in d['mask']:
            a=np.array(Image.open(r['mask']).resize((SIZE[1],SIZE[0]),Image.Resampling.NEAREST),copy=True)
            d['mask'][aid]=encode(pipe,(torch.from_numpy(a)[None,None].float().cuda()/127.5-1).repeat(1,3,1,1)).float().cpu()
        ids=pipe.tokenizer(text_prompt(r['text']),truncation=False).input_ids;max_tokens=max(max_tokens,len(ids))
        assert len(ids)<=77,('truncation forbidden',sid,len(ids))
        if sid not in d['text']:
            tokens=pipe.tokenizer(text_prompt(r['text']),padding='max_length',max_length=77,return_tensors='pt').input_ids.cuda()
            d['text'][sid]=pipe.text_encoder(tokens)[0].cpu()
        if (n+1)%256==0:write(OUT/'status.json',{'stage':'caching','expressions':n+1})
    tmp=path.with_suffix('.tmp');torch.save(d,tmp);tmp.replace(path)
    write(OUT/'cache_audit.json',{'sha256':sha(path),'images':len(d['rgb']),'masks':len(d['mask']),'texts':len(d['text']),'max_tokens':max_tokens})
    return d

def make_pairs(rr):
    pairs=defaultdict(lambda:defaultdict(list))
    for r in rr:
        if r['split']=='train':pairs[r['image_id']][r['ann_id']].append(r)
    assert len(pairs)==2048 and all(len(v)==2 for v in pairs.values())
    return [list(v.values()) for _,v in sorted(pairs.items())]

def masks(rr):
    result={}
    for r in rr:
        if r['split']=='train' and r['ann_id'] not in result:
            a=np.array(Image.open(r['mask']).resize((SIZE[1],SIZE[0]),Image.Resampling.NEAREST),copy=True)
            result[r['ann_id']]=torch.from_numpy(a)[None,None].float()/255
    return result

def update(pipe,params,optimizer,cache,targets,pair,arm,seed,step):
    optimizer.zero_grad(set_to_none=True);losses=[];ds=[];ys=[]
    rgb=cache['rgb'][pair[0]['image_id']].cuda()
    noise=torch.randn(rgb.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed*10000000+step))
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for r in pair:
            v=pipe.unet(torch.cat([rgb,noise],1),999,cache['text'][r['sent_id']].cuda()).sample.float()
            losses.append(F.mse_loss(v,-cache['mask'][r['ann_id']].cuda()))
            if arm!='latent':
                z=pipe.scheduler.step(v,999,noise.float(),eta=0.).prev_sample
                ds.append(pipe.vae.decode(z.to(torch.bfloat16)/pipe.vae.config.scaling_factor).sample.float().mean(1,keepdim=True))
                ys.append(targets[r['ann_id']].cuda())
        latent=torch.stack(losses).mean();pixel=latent*0;contrast=latent*0;gate=latent.detach()*0
        if ds:pixel=torch.stack([pixel_loss(d,y) for d,y in zip(ds,ys)]).mean()
        if arm.startswith('pair_'):
            contrast=pair_term(ds,ys)
            gate=readiness(ds,ys) if arm=='pair_ready' else torch.tensor(min(step/512,1) if arm=='pair_ramp' else 1.,device='cuda')
        loss=latent+pixel+.25*gate*contrast
    assert torch.isfinite(loss).item()
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True);optimizer.step()
    return {'loss':loss.item(),'latent':latent.item(),'pixel':pixel.item(),'pair':contrast.item(),'gate':gate.item(),'gradient_norm':float(norm)}

@torch.inference_mode()
def evaluate(pipe,rr,arm,seed,step,split):
    dest=OUT/'runs'/f'{arm}_s{seed}'/f'{split}_{step}';dest.mkdir(parents=True,exist_ok=True)
    pipe.unet.eval();start=time.perf_counter();items=[];byimage=defaultdict(list)
    for r in rr:
        if r['split']==split:byimage[r['image_id']].append(r)
    empty_ids=pipe.tokenizer('',padding='max_length',max_length=77,return_tensors='pt').input_ids.cuda()
    empty=pipe.text_encoder(empty_ids)[0]
    for iid,pair in sorted(byimage.items()):
        im=Image.open(pair[0]['image']).convert('RGB')
        a=np.array(im.resize((SIZE[1],SIZE[0]),Image.Resampling.BILINEAR),copy=True)
        rgb=encode(pipe,torch.from_numpy(a).permute(2,0,1)[None].float().cuda()/127.5-1).float()
        noise=torch.randn(rgb.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(700000+iid))
        truth=[np.array(Image.open(r['mask']))>0 for r in pair]
        blank=latent_prediction(pipe,rgb,empty,noise)
        blank=(F.interpolate(blank,size=(im.height,im.width),mode='bilinear',align_corners=False)[0,0]>0).cpu().numpy()
        bp=dest/f'{iid}_empty.png';Image.fromarray(blank.astype(np.uint8)*255).save(bp)
        for j,r in enumerate(pair):
            assert len(pipe.tokenizer(text_prompt(r['text']),truncation=False).input_ids)<=77
            ids=pipe.tokenizer(text_prompt(r['text']),padding='max_length',max_length=77,return_tensors='pt').input_ids.cuda()
            d=latent_prediction(pipe,rgb,pipe.text_encoder(ids)[0],noise)
            pred=(F.interpolate(d,size=(im.height,im.width),mode='bilinear',align_corners=False)[0,0]>0).cpu().numpy()
            own=iou(pred,truth[j]);other=iou(pred,truth[1-j]) if len(pair)==2 else None
            p=dest/f"{r['ann_id']}.png";Image.fromarray(pred.astype(np.uint8)*255).save(p)
            items.append({'image_id':iid,'ann_id':r['ann_id'],'iou':own,'distractor_iou':other,'target_selected':own>other if other is not None else None,
                'empty':not bool(pred.any()),'empty_prompt_iou':iou(blank,truth[j]),'prediction':str(p.relative_to(OUT)),
                'sha256':sha(p),'empty_prediction':str(bp.relative_to(OUT)),'empty_sha256':sha(bp),'truth':r['mask'],'truth_sha256':r['mask_sha256']})
    image_pairs=defaultdict(list)
    for r in items:
        if r['target_selected'] is not None:image_pairs[r['image_id']].append(r['target_selected'])
    selected=[r['target_selected'] for r in items if r['target_selected'] is not None]
    result={'arm':arm,'seed':seed,'step':step,'split':split,'images':len(byimage),'expressions':len(items),
        'mean_iou':float(np.mean([r['iou'] for r in items])),'precision_iou_above_05':float(np.mean([r['iou']>.5 for r in items])),
        'empty_rate':float(np.mean([r['empty'] for r in items])),'empty_prompt_mean_iou':float(np.mean([r['empty_prompt_iou'] for r in items])),
        'target_selection_rate':float(np.mean(selected)) if selected else None,
        'both_targets_selected_rate':float(np.mean([all(v) for v in image_pairs.values()])) if image_pairs else None,
        'matched_minus_swapped_iou':float(np.mean([r['iou']-r['distractor_iou'] for r in items if r['distractor_iou'] is not None])) if selected else None,
        'seconds':time.perf_counter()-start,'records':items}
    write(dest/'metrics.json',result);pipe.unet.train();return result

def save(pipe,params,optimizer,arm,seed,step,history):
    dest=WORK/f'{arm}_s{seed}';dest.mkdir(parents=True,exist_ok=True)
    adapter={k:p.detach().cpu().clone() for k,p in params.items()}
    tmp=dest/'resume.tmp';torch.save({'adapter':adapter,'optimizer':optimizer.state_dict(),'step':step,'history':history,
        'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'protocol_sha256':sha(OUT/'protocol.json')},tmp);tmp.replace(dest/'resume.pt')
    torch.save(adapter,dest/f'adapter_{step}.pt')
    write(OUT/'runs'/f'{arm}_s{seed}'/'history.json',history)

def run_training(arm,seed,protocol,deadline):
    verify(protocol);rr=records();pipe,params=setup(seed,OLDWORK/f'seed{seed}/adapter_1000.pt')
    cache=cache_training(pipe,rr);targets=masks(rr);pairs=make_pairs(rr)
    optimizer=torch.optim.AdamW(params.values(),lr=1e-4,weight_decay=.01);history=[];start_step=0
    resume=WORK/f'{arm}_s{seed}/resume.pt'
    if resume.exists():
        d=torch.load(resume,map_location='cpu',weights_only=True);assert d['protocol_sha256']==sha(OUT/'protocol.json')
        with torch.no_grad():
            for k,p in params.items():p.copy_(d['adapter'][k].cuda())
        optimizer.load_state_dict(d['optimizer']);history=d['history'];start_step=d['step']
        torch.set_rng_state(d['rng']);torch.cuda.set_rng_state(d['cuda_rng']);del d
    if start_step==0:evaluate(pipe,rr,arm,seed,0,'dev')
    order=list(range(len(pairs)));random.Random(424026).shuffle(order)
    pipe.unet.train();torch.cuda.reset_peak_memory_stats()
    for step in range(start_step+1,STEPS+1):
        assert time.time()<deadline,'36-hour batch limit reached; save checkpoints but no completion claim'
        pair=[group[(step+group[0]['ann_id'])%len(group)] for group in pairs[order[step-1]]]
        torch.cuda.synchronize();begin=time.perf_counter()
        row=update(pipe,params,optimizer,cache,targets,pair,arm,seed,step)
        torch.cuda.synchronize();row.update(step=step,seconds=time.perf_counter()-begin,image_id=pair[0]['image_id'],sent_ids=[r['sent_id'] for r in pair]);history.append(row)
        if step==1 or step%16==0:write(OUT/'status.json',{'stage':'training','arm':arm,'seed':seed,'step':step,'total':STEPS,'loss':row['loss']})
        if step%128==0:
            save(pipe,params,optimizer,arm,seed,step,history)
            print(arm,seed,step,row['loss'],flush=True)
        if step in [512,1024,2048]:evaluate(pipe,rr,arm,seed,step,'dev')
    peak=torch.cuda.max_memory_allocated()/2**20
    write(OUT/'runs'/f'{arm}_s{seed}'/'trained.json',{'status':'trained','updates':STEPS,'peak_mib':peak,'optimizer_seconds':sum(r['seconds'] for r in history),
        'adapter_sha256':sha(WORK/f'{arm}_s{seed}/adapter_{STEPS}.pt'),'protocol_sha256':sha(OUT/'protocol.json')})
    del pipe,params,optimizer,cache,targets;gc.collect();torch.cuda.empty_cache()

def final_evaluation(arm,seed,protocol):
    verify(protocol);rr=records();pipe,params=setup(seed,WORK/f'{arm}_s{seed}/adapter_{STEPS}.pt')
    # All holdout evaluation happens after the full training matrix; no model selection.
    for split in ['reserved','refcoco','refcocoplus']:
        p=OUT/'runs'/f'{arm}_s{seed}'/f'{split}_{STEPS}/metrics.json'
        if not p.exists():evaluate(pipe,rr,arm,seed,STEPS,split)
    reference=json.loads((OUT/'runs'/f'{arm}_s{seed}'/f'dev_{STEPS}/metrics.json').read_text())
    first_ids={r['image_id'] for r in reference['records'][:2]}
    check=evaluate(pipe,[r for r in rr if r['split']=='dev' and r['image_id'] in first_ids],arm,seed,'restore','dev')
    expected={r['ann_id']:r['sha256'] for r in reference['records'] if r['image_id'] in first_ids}
    assert all(expected[r['ann_id']]==r['sha256'] for r in check['records'])
    write(OUT/'runs'/f'{arm}_s{seed}'/'complete.json',{'status':'complete','restored_first_pair_exact':True,'protocol_sha256':sha(OUT/'protocol.json')})
    del pipe,params;gc.collect();torch.cuda.empty_cache()

def smoke():
    rr=records();pipe,params=setup(17,OLDWORK/'seed17/adapter_1000.pt');cache=cache_training(pipe,rr)
    targets=masks(rr);pair=[g[0] for g in make_pairs(rr)[0]];initial={k:p.detach().clone() for k,p in params.items()};measure=[]
    for arm in ARMS:
        with torch.no_grad():
            for k,p in params.items():p.copy_(initial[k])
        optimizer=torch.optim.AdamW(params.values(),lr=1e-4,weight_decay=.01)
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter();row=update(pipe,params,optimizer,cache,targets,pair,arm,17,1)
        torch.cuda.synchronize();row.update(arm=arm,seconds=time.perf_counter()-start,peak_mib=torch.cuda.max_memory_allocated()/2**20)
        assert any(not torch.equal(initial[k],p) for k,p in params.items());measure.append(row);del optimizer
    write(OUT/'smoke.json',{'status':'passed','training_only':True,'not_formal_updates':True,'measurements':measure})
    print(json.dumps(measure),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if args.smoke:smoke()
    else:freeze()
