"""Task-conditioned one-step LoRA. Metric targets never use per-image alignment."""
import gc
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig
from experiment import load_pipe,seed_all
from .common import (ASSETS,WORK,OUT,PROMPTS,read,write,sha,sample,encode_depth,
                     decode_depth,decode_normal,rays,depth_to_normals,depth_metrics,normal_metrics)

RESOLUTION=256
SIZE=(192,256)


def setup(seed,checkpoint=None):
    seed_all(seed)
    pipe=load_pipe(read(ASSETS/'model_path.json')['path'])
    embeddings={}
    with torch.no_grad():
        for task,prompt in PROMPTS.items():
            ids=pipe.tokenizer(prompt,padding='max_length',max_length=77,truncation=True,
                               return_tensors='pt').input_ids.to('cuda')
            embeddings[task]=pipe.text_encoder(ids)[0].detach()
    assert not torch.equal(embeddings['depth'],embeddings['normal'])
    pipe.task_embeddings=embeddings
    pipe.unet.add_adapter(LoraConfig(r=4,lora_alpha=4,init_lora_weights='gaussian',
                                    target_modules=['to_q','to_k','to_v','to_out.0']))
    params={n:p for n,p in pipe.unet.named_parameters() if p.requires_grad}
    for p in params.values():p.data=p.data.float()
    if checkpoint:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert set(saved)==set(params)
        with torch.no_grad():
            for n,p in params.items():p.copy_(saved[n].to('cuda'))
    pipe.unet.enable_gradient_checkpointing()
    pipe.vae.enable_gradient_checkpointing()
    pipe.scheduler.set_timesteps(1,device='cuda')
    assert pipe.scheduler.timesteps.tolist()==[999]
    assert float(pipe.scheduler.alphas_cumprod[999])==0.
    assert pipe.scheduler.config.prediction_type=='v_prediction'
    assert sum(p.numel() for p in params.values())==829952
    return pipe,params


def resized_rgb(image):
    x=torch.from_numpy(image.copy()).to(device='cuda',dtype=torch.float32).permute(2,0,1)[None]/127.5-1
    return F.interpolate(x,size=SIZE,mode='bilinear',align_corners=False,antialias=True)


def encode(pipe,tensor):
    return pipe.vae.encode(tensor.to(torch.bfloat16)).latent_dist.mode()*pipe.vae.config.scaling_factor


@torch.no_grad()
def build_cache(pipe,rows):
    cache=[]
    for row in rows:
        source=sample(ASSETS/'subset'/row['file']);derived=sample(WORK/'derived'/row['file'])
        assert sha(ASSETS/'subset'/row['file'])==row['source_sha256']
        assert sha(WORK/'derived'/row['file'])==row['derived_sha256']
        rgb=encode(pipe,resized_rgb(source['image'])).float().cpu()
        depth=torch.from_numpy(source['depth']).to('cuda')[None,None]
        # Physical depth is resized before applying the fixed nonlinear codec.
        depth=F.interpolate(depth,size=SIZE,mode='bilinear',align_corners=False,antialias=True)
        normal=torch.from_numpy(derived['normal']).to('cuda')[None]
        normal=F.normalize(F.interpolate(normal,size=SIZE,mode='area'),dim=1,eps=1e-6)
        dmask=F.interpolate(torch.from_numpy(derived['depth_valid']).to('cuda').float()[None,None],size=SIZE,mode='area')
        nmask=F.interpolate(torch.from_numpy(derived['normal_valid']).to('cuda').float()[None,None],size=SIZE,mode='area')
        latent_dmask=F.avg_pool2d(dmask,8,8)>.99
        latent_nmask=F.avg_pool2d(nmask,8,8)>.5
        geo_mask=nmask>.9
        assert latent_dmask.sum()>10 and latent_nmask.sum()>10 and geo_mask.sum()>100,(row['id'],int(latent_nmask.sum()))
        cache.append({'id':row['id'],'rgb':rgb,
                      'depth':encode(pipe,encode_depth(depth).repeat(1,3,1,1)).float().cpu(),
                      'normal':encode(pipe,normal).float().cpu(),
                      'depth_mask':latent_dmask.cpu(),'normal_mask':latent_nmask.cpu(),
                      'geo_mask':geo_mask.cpu()})
    return cache


def clean_latent(pipe,velocity,noise):
    # DDIM is differentiable; exactly the same step is used in training geometry and inference.
    return pipe.scheduler.step(velocity.float(),999,noise.float(),eta=0.).prev_sample


def decode(pipe,latent,task):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        value=pipe.vae.decode(latent.to(torch.bfloat16)/pipe.vae.config.scaling_factor,return_dict=False)[0].float()
    return decode_depth(value.mean(1,keepdim=True)) if task=='depth' else decode_normal(value)


def paired_loss(pipe,item,tasks,seed,geometry_weight=0.):
    rgb=item['rgb'].to('cuda')
    gen=torch.Generator(device='cuda').manual_seed(seed)
    noise=torch.randn(rgb.shape,device='cuda',dtype=torch.float32,generator=gen)
    predictions={}; losses={}
    for task in tasks:
        target=item[task].to('cuda')
        mask=item[task+'_mask'].to('cuda').expand_as(target)
        # At zero SNR t=999, noisy target is N(0,I) and v target is -z_0.
        with torch.autocast('cuda',dtype=torch.bfloat16):
            velocity=pipe.unet(torch.cat([rgb,noise],dim=1),999,pipe.task_embeddings[task]).sample.float()
        losses[task]=F.mse_loss(velocity[mask],-target[mask])
        if geometry_weight:predictions[task]=decode(pipe,clean_latent(pipe,velocity,noise),task)
    supervised=torch.stack(list(losses.values())).mean()
    geo=supervised.new_zeros(())
    if geometry_weight:
        ray=rays(*SIZE,read(OUT/'camera.json')['intrinsics'],device='cuda')
        nd=depth_to_normals(predictions['depth'],ray)
        mask=item['geo_mask'].to('cuda')[:,0]
        geo=(1-(nd*predictions['normal']).sum(1).clamp(-1,1))[mask].mean()
    total=supervised+geometry_weight*geo
    return total,losses,geo


def train(pipe,params,cache,mode,seed,steps,dest):
    tasks=['depth'] if mode=='depth' else ['normal'] if mode=='normal' else ['depth','normal']
    geometry_weight=.1 if mode=='joint_geometry' else 0.
    optimizer=torch.optim.AdamW(params.values(),lr=1e-4,weight_decay=.01)
    pipe.unet.train();pipe.vae.eval()
    rng=np.random.default_rng(seed);order=[];history=[]
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    for step in range(steps):
        if not order:order=list(rng.permutation(len(cache)))
        idx=int(order.pop());optimizer.zero_grad(set_to_none=True)
        loss,parts,geo=paired_loss(pipe,cache[idx],tasks,seed+1000*step,geometry_weight)
        assert torch.isfinite(loss)
        loss.backward()
        gradient=torch.nn.utils.clip_grad_norm_(list(params.values()),1.)
        assert torch.isfinite(gradient)
        optimizer.step()
        history.append({'step':step+1,'id':cache[idx]['id'],'loss':float(loss.detach()),
                        'task_losses':{k:float(v.detach()) for k,v in parts.items()},
                        'geometry_loss':float(geo.detach()),'gradient_norm':float(gradient)})
        if (step+1)%40==0 or step+1==steps:print('V7_TRAIN',mode,seed,step+1,round(float(loss.detach()),5),flush=True)
    torch.cuda.synchronize()
    stats={'mode':mode,'seed':seed,'steps':steps,'tasks':tasks,'task_examples':{t:steps for t in tasks},
           'train_images':len(cache),'training_seconds':time.perf_counter()-start,
           'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
           'trainable_parameters':sum(p.numel() for p in params.values()),
           'geometry_weight':geometry_weight,'history':history,
           'objective':'single-step zero-SNR masked latent velocity MSE, mean across present tasks; optional decoded two-task geometric loss',
           'cost_scope':'optimizer loop includes geometry decoding, excludes initial target caching/loading'}
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    torch.save({n:p.detach().float().cpu() for n,p in params.items()},dest/'adapter.pt')
    stats['checkpoint_sha256']=sha(dest/'adapter.pt')
    del optimizer;pipe.unet.eval();torch.cuda.empty_cache()
    return stats


@torch.inference_mode()
def predict(pipe,image,tasks,seed):
    pipe.unet.eval()
    torch.cuda.synchronize();start=time.perf_counter()
    rgb=encode(pipe,resized_rgb(image)).float()
    gen=torch.Generator(device='cuda').manual_seed(seed)
    noise=torch.randn(rgb.shape,device='cuda',dtype=torch.float32,generator=gen)
    outputs={};native={}
    for task in tasks:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            v=pipe.unet(torch.cat([rgb,noise],1),999,pipe.task_embeddings[task]).sample.float()
        value=decode(pipe,clean_latent(pipe,v,noise),task)
        native[task]=value
        value=F.interpolate(value,size=image.shape[:2],mode='bilinear',align_corners=False)
        if task=='normal':value=F.normalize(value,dim=1,eps=1e-6)
        outputs[task]=value[0].float().cpu().numpy()
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    return outputs,seconds


def evaluate(pipe,rows,tasks,predroot):
    records=[];predroot=Path(predroot);predroot.mkdir(parents=True,exist_ok=True)
    for i,row in enumerate(rows):
        data=sample(ASSETS/'subset'/row['file']);gt=sample(WORK/'derived'/row['file'])
        predictions,seconds=predict(pipe,data['image'],tasks,73000+row['id'])
        record={'id':row['id'],'scene':row['scene'],'seconds_all_requested_tasks':seconds,'tasks':{}}
        for task,pred in predictions.items():
            dest=predroot/f"{row['id']:04d}_{task}.npy";np.save(dest,pred)
            metrics=depth_metrics(pred[0],data['depth'],gt['depth_valid']) if task=='depth' else normal_metrics(pred,gt['normal'],gt['normal_valid'])
            if task=='depth':metrics['raw_depth_abs_rel']=depth_metrics(pred[0],gt['raw_depth'],gt['raw_valid'])['abs_rel']
            record['tasks'][task]={**metrics,'prediction_sha256':sha(dest)}
        records.append(record)
        if (i+1)%8==0:print('V7_EVAL',predroot.name,i+1,len(rows),flush=True)
    return records


def release(pipe):
    del pipe;gc.collect();torch.cuda.empty_cache()
