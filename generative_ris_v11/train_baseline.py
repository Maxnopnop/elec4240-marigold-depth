"""Direct generative referring-mask adaptation; resumable finite baseline experiment."""
import argparse
import gc
import hashlib
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from multitask_v7.engine import setup, encode
from .prepare_training import ROOT, WORK, OUT, sha, write

SIZE=(192,256)
STEPS=3000
ACCUM=2
LR=1e-4
SEEDS=[17,29,43]

def text_prompt(text):return 'A binary segmentation mask of '+text+', white foreground and black background.'

def freeze():
    manifest=OUT/'data_manifest.json'
    assert manifest.exists()
    files=[Path(__file__),ROOT/'generative_ris_v11/prepare_training.py',ROOT/'multitask_v7/engine.py',ROOT/'experiment.py']
    protocol={'status':'frozen_before_training','model':'prs-eth/marigold-depth-v1-1',
        'revision':'9571e7123e258cf052b4e54241f17971c290e9a8','seeds':SEEDS,'updates':STEPS,'accumulation':ACCUM,
        'lr':LR,'optimizer':'AdamW weight_decay=0.01','gradient_clip':1.,'rank':4,'size':list(SIZE),
        'objective':'one-step t999 zero-SNR velocity MSE to VAE-encoded binary mask, foreground+1 background-1, three identical channels',
        'text':'individual referring sentence; tokenizer truncation forbidden','data_order':'identical shuffled repeated row order across seeds, seed424012',
        'evaluation_updates':[0,500,1000,1500,2000,2500,3000],'final_selection':'last update3000, no validation-based selection',
        'dev':'64 image-disjoint same-category pairs, 128 expressions; descriptive evidence',
        'reserved':'64 further validation images excluded from this baseline training/evaluation',
        'metrics':['native-resolution per-expression IoU','threshold IoU>.5','target-vs-paired-distractor IoU selection','both targets correctly selected','matched-minus-swapped expression IoU','empty-prompt IoU'],
        'no_claims':['full benchmark','statistical confirmation','policy efficacy','novel architecture'],
        'data_manifest_sha256':sha(manifest),'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files}}
    p=OUT/'training_protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol,'Frozen protocol differs'
    else:write(p,protocol)
    return protocol

def verify(protocol):
    assert sha(OUT/'data_manifest.json')==protocol['data_manifest_sha256']
    for f,h in protocol['source_sha256'].items():assert sha(ROOT/f)==h,f

def rows():return json.loads((OUT/'data_manifest.json').read_text())['records']

@torch.no_grad()
def build_cache(pipe,rr):
    cache_path=WORK/'train_cache.pt'
    fingerprint=sha(OUT/'data_manifest.json')
    if cache_path.exists():
        d=torch.load(cache_path,map_location='cpu',weights_only=True)
        assert d['manifest_sha256']==fingerprint
        return d
    images={};masks={};texts={};token_lengths=[]
    train=[r for r in rr if r['split']=='train']
    for i,r in enumerate(train):
        if r['image_id'] not in images:
            p=WORK/r['image'];assert sha(p)==r['image_sha256']
            a=np.array(Image.open(p).convert('RGB').resize((SIZE[1],SIZE[0]),Image.Resampling.BILINEAR),copy=True)
            x=torch.from_numpy(a).permute(2,0,1)[None].float().to('cuda')/127.5-1
            images[r['image_id']]=encode(pipe,x).float().cpu()
        if r['ann_id'] not in masks:
            p=WORK/r['mask'];assert sha(p)==r['mask_sha256']
            a=np.array(Image.open(p).resize((SIZE[1],SIZE[0]),Image.Resampling.NEAREST),copy=True)
            x=torch.from_numpy(a)[None,None].float().to('cuda')/127.5-1
            masks[r['ann_id']]=encode(pipe,x.repeat(1,3,1,1)).float().cpu()
        prompt=text_prompt(r['text'])
        ids=pipe.tokenizer(prompt,padding=False,truncation=False).input_ids
        assert len(ids)<=77,('text exceeds context',r['sent_id'],len(ids))
        token_lengths.append(len(ids))
        tokens=pipe.tokenizer(prompt,padding='max_length',max_length=77,return_tensors='pt').input_ids.to('cuda')
        texts[r['sent_id']]=pipe.text_encoder(tokens)[0].detach().cpu()
        if (i+1)%128==0:write(OUT/'status.json',{'stage':'caching_real_training_data','done':i+1,'total':len(train)})
    d={'manifest_sha256':fingerprint,'rgb':images,'mask':masks,'text':texts}
    torch.save(d,cache_path)
    write(OUT/'cache_audit.json',{'sha256':sha(cache_path),'images':len(images),'masks':len(masks),'expressions':len(texts),'maximum_tokens':max(token_lengths),'reserved_used':False})
    return d

def latent_prediction(pipe,rgb,embedding,noise):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        v=pipe.unet(torch.cat([rgb,noise],1),999,embedding).sample.float()
        z=pipe.scheduler.step(v,999,noise.float(),eta=0.).prev_sample
        decoded=pipe.vae.decode(z.to(torch.bfloat16)/pipe.vae.config.scaling_factor).sample.float().mean(1,keepdim=True)
    return decoded

def iou(pred,truth):
    union=np.logical_or(pred,truth).sum()
    return float(np.logical_and(pred,truth).sum()/union) if union else 1.

@torch.inference_mode()
def evaluate(pipe,rr,seed,step):
    pipe.unet.eval();start=time.perf_counter();records=[];byimage={}
    for r in rr:
        if r['split']=='dev':byimage.setdefault(r['image_id'],[]).append(r)
    output_dir=WORK/f'seed{seed}/predictions/step{step}';output_dir.mkdir(parents=True,exist_ok=True)
    empty=pipe.tokenizer('',padding='max_length',max_length=77,return_tensors='pt').input_ids.to('cuda')
    empty=pipe.text_encoder(empty)[0]
    for image_id,pair in byimage.items():
        assert len(pair)==2 and pair[0]['ann_id']!=pair[1]['ann_id'] and pair[0]['category_id']==pair[1]['category_id']
        image=Image.open(WORK/pair[0]['image']).convert('RGB')
        a=np.array(image.resize((SIZE[1],SIZE[0]),Image.Resampling.BILINEAR),copy=True)
        rgb=encode(pipe,torch.from_numpy(a).permute(2,0,1)[None].float().to('cuda')/127.5-1).float()
        noise=torch.randn(rgb.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(700000+image_id))
        masks=[np.array(Image.open(WORK/r['mask']))>0 for r in pair]
        blank=latent_prediction(pipe,rgb,empty,noise)
        blank=(F.interpolate(blank,size=(image.height,image.width),mode='bilinear',align_corners=False)[0,0]>0).cpu().numpy()
        bp=output_dir/f'{image_id}_empty.png';Image.fromarray(blank.astype(np.uint8)*255).save(bp)
        for j,r in enumerate(pair):
            prompt=text_prompt(r['text'])
            assert len(pipe.tokenizer(prompt,truncation=False).input_ids)<=77
            ids=pipe.tokenizer(prompt,padding='max_length',max_length=77,return_tensors='pt').input_ids.to('cuda')
            emb=pipe.text_encoder(ids)[0]
            decoded=latent_prediction(pipe,rgb,emb,noise)
            pred=(F.interpolate(decoded,size=(image.height,image.width),mode='bilinear',align_corners=False)[0,0]>0).cpu().numpy()
            p=output_dir/f"{r['ann_id']}.png";Image.fromarray(pred.astype(np.uint8)*255).save(p)
            own,other=iou(pred,masks[j]),iou(pred,masks[1-j])
            records.append({'image_id':image_id,'ann_id':r['ann_id'],'text':r['text'],'iou':own,'paired_distractor_iou':other,
                'target_selected':own>other,'empty_output':not bool(pred.any()),'empty_prompt_iou':iou(blank,masks[j]),
                'prediction':str(p.relative_to(WORK)),'sha256':sha(p),'empty_prediction':str(bp.relative_to(WORK)),'empty_sha256':sha(bp)})
    pairs={}
    for r in records:pairs.setdefault(r['image_id'],[]).append(r['target_selected'])
    result={'seed':seed,'step':step,'mean_iou':float(np.mean([r['iou'] for r in records])),
        'precision_iou_above_05':float(np.mean([r['iou']>.5 for r in records])),
        'target_selection_rate':float(np.mean([r['target_selected'] for r in records])),
        'both_targets_selected_rate':float(np.mean([all(v) for v in pairs.values()])),
        'matched_minus_swapped_iou':float(np.mean([r['iou']-r['paired_distractor_iou'] for r in records])),
        'empty_prompt_mean_iou':float(np.mean([r['empty_prompt_iou'] for r in records])),
        'seconds':time.perf_counter()-start,'records':records}
    write(OUT/f'seed{seed}/eval_{step}.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
    pipe.unet.train();return result

def save_state(pipe,params,optimizer,seed,step,history):
    dest=WORK/f'seed{seed}';dest.mkdir(exist_ok=True,parents=True)
    state={'step':step,'seed':seed,'adapter':{k:p.detach().cpu().clone() for k,p in params.items()},
        'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),
        'history':history,'protocol_sha256':sha(OUT/'training_protocol.json')}
    tmp=dest/'resume.tmp';torch.save(state,tmp);tmp.replace(dest/'resume.pt')
    torch.save(state['adapter'],dest/f'adapter_{step}.pt')
    write(OUT/f'seed{seed}/checkpoint.json',{'step':step,'resume_sha256':sha(dest/'resume.pt'),'adapter_sha256':sha(dest/f'adapter_{step}.pt')})

def run_seed(seed,protocol):
    verify(protocol);rr=rows();train=[r for r in rr if r['split']=='train']
    pipe,params=setup(seed);cache=build_cache(pipe,rr)
    optimizer=torch.optim.AdamW(params.values(),lr=LR,weight_decay=.01)
    history=[];start_step=0;resume=WORK/f'seed{seed}/resume.pt'
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=True)
        assert state['protocol_sha256']==sha(OUT/'training_protocol.json') and state['seed']==seed
        with torch.no_grad():
            for k,p in params.items():p.copy_(state['adapter'][k].to('cuda'))
        optimizer.load_state_dict(state['optimizer']);history=state['history'];start_step=state['step']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
    if start_step==0:evaluate(pipe,rr,seed,0)
    rng=random.Random(424012);order=[]
    while len(order)<STEPS*ACCUM:
        ids=list(range(len(train)));rng.shuffle(ids);order.extend(ids)
    torch.cuda.reset_peak_memory_stats();elapsed=0.;pipe.unet.train()
    for step in range(start_step+1,STEPS+1):
        torch.cuda.synchronize();begin=time.perf_counter();optimizer.zero_grad(set_to_none=True);total=0.
        for a in range(ACCUM):
            r=train[order[(step-1)*ACCUM+a]]
            rgb=cache['rgb'][r['image_id']].to('cuda');target=cache['mask'][r['ann_id']].to('cuda')
            emb=cache['text'][r['sent_id']].to('cuda')
            gen=torch.Generator(device='cuda').manual_seed(seed*1000000+step*ACCUM+a)
            noise=torch.randn(rgb.shape,device='cuda',generator=gen)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                v=pipe.unet(torch.cat([rgb,noise],1),999,emb).sample.float()
                loss=F.mse_loss(v,-target)
            assert torch.isfinite(loss).item(),'Nonfinite loss'
            (loss/ACCUM).backward();total+=loss.item()/ACCUM
        norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True)
        optimizer.step();torch.cuda.synchronize();seconds=time.perf_counter()-begin;elapsed+=seconds
        history.append({'step':step,'loss':total,'gradient_norm_before_clip':float(norm),'seconds':seconds})
        if step==1 or step%10==0:
            write(OUT/'status.json',{'stage':'training','seed':seed,'step':step,'total_steps':STEPS,'loss':total,
                'optimizer_seconds_this_process':elapsed,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20})
        if step%100==0:
            save_state(pipe,params,optimizer,seed,step,history)
            write(OUT/f'seed{seed}/history.json',history)
            print(f'seed {seed} step {step}/{STEPS} loss {total:.6f}',flush=True)
        if step%500==0:evaluate(pipe,rr,seed,step)
    if not (OUT/f'seed{seed}/eval_{STEPS}.json').exists():evaluate(pipe,rr,seed,STEPS)
    # Independent saved-mask metrics and restore predictions are required before completion.
    final=json.loads((OUT/f'seed{seed}/eval_{STEPS}.json').read_text())
    maskmap={r['ann_id']:r for r in rr}
    for result in final['records']:
        p=WORK/result['prediction'];assert sha(p)==result['sha256']
        truth=np.array(Image.open(WORK/maskmap[result['ann_id']]['mask']))>0
        assert abs(iou(np.array(Image.open(p))>0,truth)-result['iou'])<1e-12
    sample=final['records'][0];reference=(WORK/sample['prediction']).read_bytes()
    del optimizer,params,pipe,cache;gc.collect();torch.cuda.empty_cache()
    pipe,params=setup(seed,WORK/f'seed{seed}/adapter_{STEPS}.pt')
    first=[r for r in rr if r['split']=='dev' and r['image_id']==sample['image_id']]
    # Reuse eval on a single image pair and retain separate audit filename.
    restored=evaluate_restore(pipe,first,seed)
    assert restored==reference,'Checkpoint restoration changed prediction'
    write(OUT/f'seed{seed}/complete.json',{'status':'complete','updates':STEPS,'final_predictions_audited':len(final['records']),
        'restored_first_prediction_exact':True,'final_mean_iou':final['mean_iou'],'protocol_sha256':sha(OUT/'training_protocol.json')})
    del pipe,params;gc.collect();torch.cuda.empty_cache()

@torch.inference_mode()
def evaluate_restore(pipe,pair,seed):
    r=pair[0];image=Image.open(WORK/r['image']).convert('RGB')
    a=np.array(image.resize((SIZE[1],SIZE[0]),Image.Resampling.BILINEAR),copy=True)
    rgb=encode(pipe,torch.from_numpy(a).permute(2,0,1)[None].float().to('cuda')/127.5-1).float()
    noise=torch.randn(rgb.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(700000+r['image_id']))
    ids=pipe.tokenizer(text_prompt(r['text']),padding='max_length',max_length=77,return_tensors='pt').input_ids.to('cuda')
    pipe.unet.eval();d=latent_prediction(pipe,rgb,pipe.text_encoder(ids)[0],noise)
    p=(F.interpolate(d,size=(image.height,image.width),mode='bilinear',align_corners=False)[0,0]>0).cpu().numpy()
    import io
    b=io.BytesIO();Image.fromarray(p.astype(np.uint8)*255).save(b,format='PNG');return b.getvalue()

def main():
    args=argparse.ArgumentParser();args.add_argument('--freeze-only',action='store_true');args=args.parse_args()
    protocol=freeze()
    if args.freeze_only:print('Frozen',sha(OUT/'training_protocol.json'));return
    lock=WORK/'training.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    import os
    handle=lock.open('x');handle.write(str(os.getpid()));handle.close()
    try:
        for seed in SEEDS:
            if not (OUT/f'seed{seed}/complete.json').exists():run_seed(seed,protocol)
        write(OUT/'status.json',{'stage':'complete','seeds':SEEDS,'updates_per_seed':STEPS,'reserved_evaluated':False,'policy_evaluated':False})
    except Exception:
        write(OUT/'status.json',{'stage':'failed','error':traceback.format_exc()});raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
