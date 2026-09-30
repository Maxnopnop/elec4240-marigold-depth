"""Small, scene-disjoint NYUv2 pilot on an 8 GB GPU.

Marigold latent denoising follows prs-eth/Marigold's Apache-2.0 trainer.
This compact implementation freezes VAE/text encoder and studies UNet adapters.
See REFERENCES.md for attribution and differences from the full training recipe.
"""
import argparse
import csv
import gc
import importlib.metadata
import json
import platform
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from diffusers import MarigoldDepthPipeline, DDPMScheduler
from diffusers.pipelines.marigold.marigold_image_processing import MarigoldImageProcessor
from peft import LoraConfig

EXPERT_ID='depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf'
EXPERT_REVISION='8078d68a9c75a972131914f6afd0c1723be0da7f'

class FloatResizeProcessor(MarigoldImageProcessor):
    """Keep antialiased resizing in FP32; the Windows CPU kernel lacks BF16."""
    def preprocess(self,image,processing_resolution,resample_method_input,device,dtype):
        image,padding,original=super().preprocess(image,processing_resolution,resample_method_input,device,torch.float32)
        return image.to(dtype=dtype),padding,original

    @staticmethod
    def resize_antialias(image,size,mode,is_aa=None):
        return MarigoldImageProcessor.resize_antialias(image.float(),size,mode,is_aa).to(image.dtype)

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2),encoding='utf-8')

def mask_for(depth):
    mask=np.isfinite(depth)&(depth>1e-3)&(depth<10)
    crop=np.zeros(depth.shape,dtype=bool);crop[45:471,41:601]=True
    return mask&crop

def evaluate_depth(pred,gt):
    """GT-based per-image least-squares affine alignment, then clip to depth range.

    This assesses relative depth structure, NOT deployable metric depth accuracy.
    The same alignment, crop, range, resolution and metrics apply to all methods.
    """
    pred=np.asarray(pred,dtype=np.float64);gt=np.asarray(gt,dtype=np.float64)
    if pred.shape!=gt.shape or not np.isfinite(pred).all():raise ValueError('Invalid prediction')
    mask=mask_for(gt);x=pred[mask];y=gt[mask]
    if mask.sum()<100:raise ValueError('Too few valid pixels')
    vx=np.var(x)
    scale=float(np.mean((x-x.mean())*(y-y.mean()))/vx) if vx>1e-12 else 0.0
    shift=float(y.mean()-scale*x.mean())
    aligned=np.clip(pred*scale+shift,1e-3,10)
    z=aligned[mask];ratio=np.maximum(z/y,y/z)
    metrics={'abs_rel':float(np.mean(np.abs(z-y)/y)),
             'rmse_m':float(np.sqrt(np.mean((z-y)**2))),
             'delta1':float(np.mean(ratio<1.25)),
             'scale':scale,'shift':shift,'valid_pixels':int(mask.sum())}
    return metrics,aligned,mask

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def load_pipe(path):
    pipe=MarigoldDepthPipeline.from_pretrained(path,variant='fp16',torch_dtype=torch.bfloat16,local_files_only=True)
    pipe.image_processor=FloatResizeProcessor.from_config(pipe.image_processor.config)
    pipe.set_progress_bar_config(disable=True);pipe.to('cuda')
    with torch.no_grad():
        ids=pipe.tokenizer('',padding='do_not_pad',return_tensors='pt').input_ids.to('cuda')
        pipe.empty_text_embedding=pipe.text_encoder(ids)[0].detach()
    pipe.vae.requires_grad_(False);pipe.unet.requires_grad_(False);pipe.text_encoder.requires_grad_(False)
    torch.cuda.empty_cache()
    return pipe

def load_sample(root,row):
    with np.load(Path(root)/'subset'/row['file']) as f:return f['image'],f['depth']

@torch.inference_mode()
def infer_marigold(pipe,image,steps,resolution,seed):
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    out=pipe(Image.fromarray(image),num_inference_steps=steps,ensemble_size=1,
             processing_resolution=resolution,match_input_resolution=True,
             generator=torch.Generator(device='cuda').manual_seed(seed))
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    return out.prediction[0,:,:,0].copy(),elapsed,torch.cuda.max_memory_allocated()/2**20

@torch.no_grad()
def cache_latents(pipe,rows,root,resolution):
    cache=[]
    for row in rows:
        rgb,dep=load_sample(root,row);size=(int(resolution*3/4)//8*8,resolution)
        image=torch.tensor(rgb.copy(),device='cuda',dtype=torch.float32).permute(2,0,1)[None]/127.5-1
        depth=torch.tensor(dep.copy(),device='cuda',dtype=torch.float32)[None,None]
        mask=torch.tensor(mask_for(dep),device='cuda')[None,None]
        lo,hi=torch.quantile(depth[mask],torch.tensor([.02,.98],device='cuda'))
        depth=((depth-lo)/(hi-lo).clamp_min(1e-6)*2-1).clamp(-1,1)
        image=F.interpolate(image,size=size,mode='bilinear',align_corners=False)
        depth=F.interpolate(depth,size=size,mode='bilinear',align_corners=False)
        valid=F.interpolate(mask.float(),size=size,mode='nearest')>.5
        latent_mask=~F.max_pool2d((~valid).float(),8,8).bool()
        def encode(x):return pipe.vae.encode(x.to(torch.bfloat16)).latent_dist.mode()*pipe.vae.config.scaling_factor
        cache.append((encode(image).float().cpu(),encode(depth.repeat(1,3,1,1)).float().cpu(),latent_mask.cpu()))
    return cache

def train_adapter(pipe,cache,mode,steps,seed,outdir):
    seed_all(seed);unet=pipe.unet;unet.requires_grad_(False)
    if mode=='lora':
        unet.add_adapter(LoraConfig(r=4,lora_alpha=4,init_lora_weights='gaussian',target_modules=['to_q','to_k','to_v','to_out.0']))
        lr=1e-4;unet.enable_gradient_checkpointing()
    elif mode=='head':
        unet.conv_out.requires_grad_(True);lr=1e-4
    else:raise ValueError(mode)
    params=[p for p in unet.parameters() if p.requires_grad]
    for p in params:p.data=p.data.float()
    count=sum(p.numel() for p in params);total=sum(p.numel() for p in unet.parameters())
    optimizer=torch.optim.AdamW(params,lr=lr,weight_decay=.01)
    scheduler=DDPMScheduler.from_config(pipe.scheduler.config,rescale_betas_zero_snr=True,timestep_spacing='trailing')
    unet.train();pipe.vae.to('cpu');torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    gen=torch.Generator(device='cuda').manual_seed(seed)
    rng=np.random.default_rng(seed);order=[];history=[]
    torch.cuda.synchronize();start=time.perf_counter()
    for step in range(steps):
        if not order:order=list(rng.permutation(len(cache)))
        rgb,target_latent,valid=cache[order.pop()]
        rgb=rgb.to('cuda');target_latent=target_latent.to('cuda');valid=valid.to('cuda').expand(-1,4,-1,-1)
        t=torch.randint(0,scheduler.config.num_train_timesteps,(1,),device='cuda',generator=gen)
        noise=torch.randn(target_latent.shape,device='cuda',generator=gen)
        noisy=scheduler.add_noise(target_latent,noise,t)
        target=scheduler.get_velocity(target_latent,noise,t) if scheduler.config.prediction_type=='v_prediction' else noise
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            predicted=unet(torch.cat([rgb,noisy],dim=1),t,pipe.empty_text_embedding).sample
            loss=F.mse_loss(predicted.float()[valid],target.float()[valid])
        if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(params,1.0);optimizer.step()
        history.append({'step':step+1,'loss':float(loss.detach()),'grad_norm':float(grad)})
        if (step+1)%10==0:print(f'{mode} n={len(cache)} step={step+1}/{steps} loss={loss.item():.5f}',flush=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    stats={'mode':mode,'train_images':len(cache),'steps':steps,'seed':seed,'lr':lr,'rank':4 if mode=='lora' else None,
           'trainable_parameters':count,'unet_parameters':total,'trainable_percent':100*count/total,
           'training_seconds':elapsed,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
           'history':history,'latent_cache_time_excluded':True,'precision':'bfloat16 autocast, fp32 trainable params'}
    outdir.mkdir(parents=True,exist_ok=True)
    # Checkpoints remain in private local work storage, never GitHub.
    torch.save({n:p.detach().float().cpu() for n,p in unet.named_parameters() if p.requires_grad},outdir/'adapter.pt')
    dump(outdir/'training.json',stats)
    del optimizer;unet.eval();unet.to(dtype=torch.bfloat16);pipe.vae.to('cuda');torch.cuda.empty_cache()
    return stats

def save_eval(method,rows,root,infer,out,predroot):
    metrics=[];predroot.mkdir(parents=True,exist_ok=True)
    # Warm-up is excluded from latency statistics.
    image,_=load_sample(root,rows[0]);infer(image,rows[0]['id'])
    for row in rows:
        image,gt=load_sample(root,row);pred,seconds,mem=infer(image,row['id'])
        m,_,_=evaluate_depth(pred,gt)
        record={'method':method,'id':row['id'],'scene':row['scene'],'split':row['split'],**m,
                'inference_seconds':seconds,'peak_allocated_mib':mem};metrics.append(record)
        np.save(predroot/f"{row['id']:04d}.npy",pred)
        print(method,row['split'],row['id'],f"AbsRel={m['abs_rel']:.4f}",f'{seconds:.2f}s',flush=True)
        dump(out/ f'{method}_per_image.json',metrics)
    return metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--results',default='results')
    p.add_argument('--work',required=True);p.add_argument('--method',required=True,choices=['base','lora8','lora32','head32','expert'])
    p.add_argument('--steps',type=int,default=80);p.add_argument('--resolution',type=int,default=256)
    p.add_argument('--denoise-steps',type=int,default=4);p.add_argument('--seed',type=int,default=17)
    a=p.parse_args();out=Path(a.results);work=Path(a.work);assets=Path(a.assets)
    manifest=json.loads((out/'split_manifest.json').read_text());rows=manifest['samples']
    evalrows=[r for r in rows if r['split'] in ['val','test']]
    seed_all(a.seed);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    info={'gpu':torch.cuda.get_device_name(),'total_vram_mib':torch.cuda.get_device_properties(0).total_memory/2**20,
          'python':platform.python_version(),'cuda':torch.version.cuda,
          'packages':{n:importlib.metadata.version(n) for n in ['torch','torchvision','diffusers','transformers','peft','accelerate','numpy','Pillow','h5py','scipy']}}
    dump(out/'environment.json',info)
    label=a.method
    if a.method=='expert':
        from transformers import AutoImageProcessor,AutoModelForDepthEstimation
        proc=AutoImageProcessor.from_pretrained(str(assets/'expert'),local_files_only=True)
        net=AutoModelForDepthEstimation.from_pretrained(str(assets/'expert'),local_files_only=True).to('cuda').eval()
        @torch.inference_mode()
        def infer(image,seed):
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            inputs=proc(images=image,return_tensors='pt',size={'height':a.resolution,'width':a.resolution}).to('cuda')
            pred=net(**inputs).predicted_depth
            pred=F.interpolate(pred[:,None],size=image.shape[:2],mode='bicubic',align_corners=False)[0,0].cpu().numpy()
            torch.cuda.synchronize();return pred,time.perf_counter()-start,torch.cuda.max_memory_allocated()/2**20
        example,_=load_sample(assets,evalrows[0])
        actual_size=list(proc(images=example,return_tensors='pt',size={'height':a.resolution,'width':a.resolution})['pixel_values'].shape[-2:])
        dump(out/'expert_config.json',{'model_id':EXPERT_ID,'revision':EXPERT_REVISION,'requested_resolution':a.resolution,'actual_input_height_width':actual_size,
             'note':'NYUv2-trained metric-depth specialist; different prior supervision; same affine-aligned evaluation.'})
    else:
        model_path=json.loads((assets/'model_path.json').read_text())['path'];pipe=load_pipe(model_path)
        if a.method!='base':
            n=8 if a.method=='lora8' else 32
            trainrows=[r for r in rows if r['split']=='train'][:n]
            started=time.perf_counter();cache=cache_latents(pipe,trainrows,assets,a.resolution)
            cache_seconds=time.perf_counter()-started
            stats=train_adapter(pipe,cache,'head' if a.method=='head32' else 'lora',a.steps,a.seed,work/label)
            stats['latent_cache_seconds']=cache_seconds;dump(out/f'{label}_training.json',stats)
        label=f'{a.method}_s{a.denoise_steps}'
        infer=lambda image,idx:infer_marigold(pipe,image,a.denoise_steps,a.resolution,a.seed+idx)
        dump(out/f'{label}_config.json',vars(a))
    save_eval(label,evalrows,assets,infer,out,work/'predictions'/label)
    print('EXPERIMENT_COMPLETE',label,flush=True)
if __name__=='__main__':main()
