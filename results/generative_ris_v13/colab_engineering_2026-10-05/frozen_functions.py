import json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import MarigoldDepthPipeline
from diffusers.pipelines.marigold.marigold_image_processing import MarigoldImageProcessor
from peft import LoraConfig
ASSETS=Path(__file__).parent
read=lambda p:json.loads(Path(p).read_text())
PROMPTS={"depth":"A metric depth map of the indoor scene, encoded as logarithmic depth.","normal":"A camera-facing surface normal map of the indoor scene, encoded in RGB."}


class FloatResizeProcessor(MarigoldImageProcessor):
    """Keep antialiased resizing in FP32; the Windows CPU kernel lacks BF16."""
    def preprocess(self,image,processing_resolution,resample_method_input,device,dtype):
        image,padding,original=super().preprocess(image,processing_resolution,resample_method_input,device,torch.float32)
        return image.to(dtype=dtype),padding,original

    @staticmethod
    def resize_antialias(image,size,mode,is_aa=None):
        return MarigoldImageProcessor.resize_antialias(image.float(),size,mode,is_aa).to(image.dtype)

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