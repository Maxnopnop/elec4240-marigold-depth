"""Sequential, resumable expanded evaluation; training and inference seeds are separate."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig
from experiment import (load_pipe,load_sample,cache_latents,train_adapter,
                        infer_marigold,evaluate_depth,seed_all,EXPERT_ID,EXPERT_REVISION)
from download_assets import MODEL_ID,MODEL_REVISION

SEEDS=[17,29,43]
PROTOCOL={'version':'expanded_v1','resolution':256,'training_steps':160,
          'training_seeds':SEEDS,'inference_seed_offset':17,'denoising_steps':[1,4],
          'lr':1e-4,'rank':4,'batch_size':1,'train_sizes':[32,64],
          'model_id':MODEL_ID,'model_revision':MODEL_REVISION,
          'expert_id':EXPERT_ID,'expert_revision':EXPERT_REVISION}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8');temp.replace(path)
def utc():return datetime.now(timezone.utc).isoformat()
def matrix():
    return ([{'run_id':'base','mode':'base','train_images':0,'train_seed':None}]+
            [{'run_id':f'{mode}{n}_seed{seed}','mode':mode,'train_images':n,'train_seed':seed}
             for mode,n in [('lora',32),('lora',64),('head',64)] for seed in SEEDS]+
            [{'run_id':'expert','mode':'expert','train_images':0,'train_seed':None}])

def restore_adapter(pipe,mode,path):
    if mode=='lora':
        pipe.unet.add_adapter(LoraConfig(r=4,lora_alpha=4,init_lora_weights='gaussian',target_modules=['to_q','to_k','to_v','to_out.0']))
    expected={n for n,p in pipe.unet.named_parameters() if p.requires_grad} if mode=='lora' else {'conv_out.weight','conv_out.bias'}
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if set(saved)!=expected:raise ValueError('Checkpoint parameter mismatch')
    with torch.no_grad():
        named=dict(pipe.unet.named_parameters())
        for n,v in saved.items():named[n].copy_(v.to(device='cuda',dtype=named[n].dtype))
    pipe.unet.requires_grad_(False);pipe.unet.eval()

def evaluate(cfg,label,rows,assets,infer,result_dir,pred_dir):
    pred_dir.mkdir(parents=True,exist_ok=True);records=[]
    image,_=load_sample(assets,rows[0]);infer(image,rows[0]['id'])
    for index,row in enumerate(rows,1):
        image,gt=load_sample(assets,row);pred,seconds,memory=infer(image,row['id'])
        metrics,_,_=evaluate_depth(pred,gt)
        records.append({'condition':label,'run_id':cfg['run_id'],'train_seed':cfg['train_seed'],
            'train_images':cfg['train_images'],'id':row['id'],'scene':row['scene'],'split':row['split'],
            'cohort':row.get('cohort','val16'),**metrics,'inference_seconds':seconds,'peak_allocated_mib':memory})
        np.save(pred_dir/f"{row['id']:04d}.npy",pred)
        if index%16==0 or index==len(rows):
            write(result_dir/f'{label}_per_image.json',records)
            print('EVAL',label,f'{index}/{len(rows)}',flush=True)
    return records

def execute(cfg,manifest,assets,work,out,fingerprint):
    rid=cfg['run_id'];rd=out/'runs'/rid;rd.mkdir(parents=True,exist_ok=True)
    config={**PROTOCOL,**cfg,**fingerprint}
    if (rd/'config.json').exists():
        if read(rd/'config.json')!=config:raise ValueError(f'Configuration changed: {rid}')
    else:write(rd/'config.json',config)
    rows=[r for r in manifest['samples'] if r['split']!='train']
    steps=[None] if cfg['mode']=='expert' else [1,4]
    labels=[rid if s is None else f'{rid}_s{s}' for s in steps]
    def evaluation_complete(label):
        path=rd/f'{label}_per_image.json';marker=rd/f'{label}_complete.json'
        if not path.exists() or not marker.exists():return False
        mark=read(marker);records=read(path)
        return (mark['metrics_sha256']==sha(path) and mark['config_sha256']==sha(rd/'config.json')
                and len(records)==len(rows) and {r['id'] for r in records}=={r['id'] for r in rows}
                and all((work/'predictions'/label/f"{r['id']:04d}.npy").exists() for r in rows))
    if all(evaluation_complete(label) for label in labels):
        print('SKIP_COMPLETE',rid,flush=True);return
    write(out/'status.json',{'state':'running','run_id':rid,'updated_utc':utc()})
    print('RUN_START',rid,flush=True)
    seed_all(cfg['train_seed'] or 17)
    if cfg['mode']=='expert':
        from transformers import AutoImageProcessor,AutoModelForDepthEstimation
        proc=AutoImageProcessor.from_pretrained(str(assets/'expert'),local_files_only=True,use_fast=False)
        net=AutoModelForDepthEstimation.from_pretrained(str(assets/'expert'),local_files_only=True).to('cuda').eval()
        @torch.inference_mode()
        def infer(image,idx):
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
            inputs=proc(images=image,return_tensors='pt',size={'height':256,'width':256}).to('cuda')
            pred=net(**inputs).predicted_depth
            pred=F.interpolate(pred[:,None],size=image.shape[:2],mode='bicubic',align_corners=False)[0,0].cpu().numpy()
            torch.cuda.synchronize()
            return pred,time.perf_counter()-started,torch.cuda.max_memory_allocated()/2**20
    else:
        pipe=load_pipe(read(assets/'model_path.json')['path'])
        if cfg['mode']!='base':
            checkpoint_dir=work/'checkpoints'/rid;checkpoint=checkpoint_dir/'adapter.pt'
            training_path=rd/'training.json'
            if training_path.exists() and checkpoint.exists():
                stats=read(training_path)
                if stats['checkpoint_sha256']!=sha(checkpoint):raise ValueError('Checkpoint SHA mismatch')
                if stats['config_sha256']!=sha(rd/'config.json'):raise ValueError('Training configuration mismatch')
                restore_adapter(pipe,cfg['mode'],checkpoint)
                print('RESTORED',rid,flush=True)
            else:
                tr=[r for r in manifest['samples'] if r['split']=='train'][:cfg['train_images']]
                start=time.perf_counter();cache=cache_latents(pipe,tr,assets,256);cache_seconds=time.perf_counter()-start
                stats=train_adapter(pipe,cache,cfg['mode'],160,cfg['train_seed'],checkpoint_dir)
                stats.update({'latent_cache_seconds':cache_seconds,'checkpoint_sha256':sha(checkpoint),
                              'train_ids':[r['id'] for r in tr],'config_sha256':sha(rd/'config.json')})
                write(training_path,stats);del cache
            print('TRAIN_READY',rid,f"{stats['training_seconds']:.1f}s",flush=True)
    for denoise,label in zip(steps,labels):
        if evaluation_complete(label):continue
        if denoise is not None:
            infer=lambda image,idx:infer_marigold(pipe,image,denoise,256,17+idx)
        records=evaluate(cfg,label,rows,assets,infer,rd,work/'predictions'/label)
        write(rd/f'{label}_complete.json',{'condition':label,'images':len(records),
              'metrics_sha256':sha(rd/f'{label}_per_image.json'),'config_sha256':sha(rd/'config.json'),
              'finished_utc':utc(),'inference_seed_offset':17,'denoising_steps':denoise})
    print('RUN_COMPLETE',rid,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--work',required=True)
    p.add_argument('--results',default='results/expanded_v1');p.add_argument('--only',nargs='*');a=p.parse_args()
    assets=Path(a.assets);work=Path(a.work);out=Path(a.results)
    manifest=read(out/'split_manifest.json');assert manifest['counts']=={'train':64,'val':16,'test':120}
    protocol_path=Path(__file__).with_name('EXPANDED_PROTOCOL.md')
    fingerprint={'manifest_sha256':sha(out/'split_manifest.json'),'protocol_sha256':sha(protocol_path),
                 'source_sha256':{name:sha(Path(__file__).with_name(name)) for name in ['run_expanded.py','experiment.py']}}
    fixed={**PROTOCOL,**fingerprint,'matrix':matrix()}
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==fixed,'Protocol changed'
    else:write(out/'protocol.json',fixed)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    write(out/'environment.json',{'gpu':torch.cuda.get_device_name(),'total_vram_mib':torch.cuda.get_device_properties(0).total_memory/2**20,
          'python':platform.python_version(),'cuda':torch.version.cuda,
          'packages':{n:importlib.metadata.version(n) for n in ['torch','torchvision','diffusers','transformers','peft','accelerate','numpy','Pillow','h5py','scipy']}})
    chosen=matrix()
    if a.only:
        assert set(a.only)<={c['run_id'] for c in chosen};chosen=[c for c in chosen if c['run_id'] in a.only]
    for cfg in chosen:
        try:execute(cfg,manifest,assets,work,out,fingerprint)
        except Exception as exc:
            write(out/'status.json',{'state':'failed','run_id':cfg['run_id'],'updated_utc':utc(),'error':str(exc)})
            raise
        gc.collect();torch.cuda.empty_cache()
    write(out/'status.json',{'state':'requested_runs_finished','runs':[c['run_id'] for c in chosen],'updated_utc':utc()})
    print('EXPANDED_RUNS_COMPLETE',flush=True)

if __name__=='__main__':main()
