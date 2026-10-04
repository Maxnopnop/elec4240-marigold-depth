# Engineering probe only: zero optimizer updates; not RIS accuracy.
import subprocess, sys, os, json, time, hashlib
from pathlib import Path
os.environ['USE_TF']='0'
subprocess.run([sys.executable,'-m','pip','install','-q','diffusers==0.35.2','transformers==4.57.1','peft==0.17.1','accelerate==1.11.0'],check=True)
import torch
from diffusers import MarigoldDepthPipeline
from peft import LoraConfig
from importlib.metadata import version
assert torch.cuda.is_available()
torch.manual_seed(17)
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
revision='9571e7123e258cf052b4e54241f17971c290e9a8'
start=time.perf_counter()
pipe=MarigoldDepthPipeline.from_pretrained('prs-eth/marigold-depth-v1-1',revision=revision,variant='fp16',torch_dtype=dtype).to('cuda')
pipe.vae.requires_grad_(False); pipe.unet.requires_grad_(False); pipe.text_encoder.requires_grad_(False)
pipe.unet.add_adapter(LoraConfig(r=4,lora_alpha=4,init_lora_weights='gaussian',target_modules=['to_q','to_k','to_v','to_out.0']))
params=[p for p in pipe.unet.parameters() if p.requires_grad]
for p in params: p.data=p.data.float()
pipe.unet.enable_gradient_checkpointing()
pipe.scheduler.set_timesteps(1,device='cuda')
assert pipe.scheduler.timesteps.tolist()==[999] and float(pipe.scheduler.alphas_cumprod[999])==0.
assert pipe.scheduler.config.prediction_type=='v_prediction'
torch.cuda.reset_peak_memory_stats()
rgb=torch.full((1,3,192,256),-.7,device='cuda')
rgb[:,:,48:144,24:104]=torch.tensor([1.,-1.,-1.],device='cuda')[None,:,None,None]
rgb[:,:,48:144,152:232]=torch.tensor([-1.,-1.,1.],device='cuda')[None,:,None,None]
mask=torch.full_like(rgb,-1.); mask[:,:,48:144,24:104]=1.
prompts=['A binary segmentation mask of the red rectangle, white foreground and black background.','A binary segmentation mask of the blue rectangle, white foreground and black background.']
pipe.unet.eval()
with torch.no_grad():
    ids=pipe.tokenizer(prompts,padding='max_length',max_length=77,truncation=True,return_tensors='pt').input_ids.to('cuda')
    embeddings=pipe.text_encoder(ids)[0]
    zrgb=pipe.vae.encode(rgb.to(dtype)).latent_dist.mode()*pipe.vae.config.scaling_factor
    target=pipe.vae.encode(mask.to(dtype)).latent_dist.mode()*pipe.vae.config.scaling_factor
    recon=pipe.vae.decode(target/pipe.vae.config.scaling_factor).sample.float().mean(1)>0
    truth=mask.mean(1)>0
    noise=torch.randn(zrgb.shape,device='cuda',dtype=dtype,generator=torch.Generator(device='cuda').manual_seed(91001))
    inp=torch.cat([zrgb,noise],1)
    a=pipe.unet(inp,999,embeddings[:1]).sample.float()
    b=pipe.unet(inp,999,embeddings[1:]).sample.float()
pipe.unet.train()
with torch.autocast('cuda',dtype=dtype,enabled=(dtype!=torch.float32)):
    loss=torch.nn.functional.mse_loss(pipe.unet(inp,999,embeddings[:1]).sample.float(),-target.float())
loss.backward()
grads=[p.grad for p in params if p.grad is not None]
finite=all(torch.isfinite(g).all().item() for g in grads)
norm=sum(g.float().square().sum().item() for g in grads)**.5
assert finite and norm>0
torch.cuda.synchronize()
report={'status':'engineering_probe_complete','gpu':torch.cuda.get_device_name(),'dtype':str(dtype),'model_revision':revision,'trainable_parameters':sum(p.numel() for p in params),'prompt_output_mae':(a-b).abs().mean().item(),'mask_vae_roundtrip_iou':((recon&truth).sum()/(recon|truth).sum()).item(),'roundtrip_is_not_prediction':True,'loss':loss.item(),'gradient_norm':norm,'finite_gradients':finite,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'seconds_including_download':time.perf_counter()-start,'optimizer_updates':0,'ris_accuracy_validated':False,'selector_validated':False,'packages':{n:version(n) for n in ['torch','diffusers','transformers','peft','accelerate']}}
Path('/content/generative_ris_v11').mkdir(exist_ok=True)
Path('/content/generative_ris_v11/cloud_preflight.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
del pipe,params,grads,a,b,loss
import gc
gc.collect(); torch.cuda.empty_cache()
