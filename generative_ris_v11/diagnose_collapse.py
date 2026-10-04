"""Post-hoc diagnosis, not a new confirmatory experiment or training change."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw
from multitask_v7.engine import setup,encode
from .prepare_training import WORK,OUT,sha,write
from .train_baseline import SIZE,text_prompt,iou

@torch.inference_mode()
def main():
    dest=OUT/'collapse_diagnostic';dest.mkdir(exist_ok=True)
    allrows=json.loads((OUT/'data_manifest.json').read_text())['records']
    rr=sum(([r for r in allrows if r['split']==s][:8] for s in ['train','dev']),[])
    cp=WORK/'seed17/adapter_1000.pt'
    write(dest/'selection.json',{'checkpoint':str(cp),'checkpoint_sha256':sha(cp),'rows':[{'split':r['split'],'sent_id':r['sent_id']} for r in rr],
        'scope':'post-hoc first8train+8dev expressions, not independent test; no parameter updates'})
    pipe,params=setup(17,cp);pipe.unet.eval()
    black=encode(pipe,torch.full((1,3,*SIZE),-1.,device='cuda')).float()
    cache=torch.load(WORK/'train_cache.pt',map_location='cpu',weights_only=True)
    records=[];panels=[]
    for r in rr:
        im=Image.open(WORK/r['image']).convert('RGB').resize((SIZE[1],SIZE[0]),Image.Resampling.BILINEAR)
        gt=np.array(Image.open(WORK/r['mask']).resize((SIZE[1],SIZE[0]),Image.Resampling.NEAREST))>0
        rgb=encode(pipe,torch.from_numpy(np.array(im,copy=True)).permute(2,0,1)[None].float().to('cuda')/127.5-1).float()
        target=encode(pipe,torch.from_numpy(gt.astype(np.float32))[None,None].repeat(1,3,1,1).to('cuda')*2-1).float()
        ids=pipe.tokenizer(text_prompt(r['text']),padding='max_length',max_length=77,return_tensors='pt').input_ids.to('cuda')
        emb=pipe.text_encoder(ids)[0];noise=torch.randn(rgb.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(700000+r['image_id']))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            v=pipe.unet(torch.cat([rgb,noise],1),999,emb).sample.float()
            z=pipe.scheduler.step(v,999,noise,eta=0.).prev_sample
            pred=pipe.vae.decode(z.to(torch.bfloat16)/pipe.vae.config.scaling_factor).sample.float().mean(1)[0]
            ideal=pipe.vae.decode(target.to(torch.bfloat16)/pipe.vae.config.scaling_factor).sample.float().mean(1)[0]
        a=pred.cpu().numpy();m=a>0
        record={'split':r['split'],'sent_id':r['sent_id'],'ann_id':r['ann_id'],'foreground_fraction':float(gt.mean()),
            'prediction_foreground_fraction':float(m.mean()),'prediction_iou':iou(m,gt),'vae_roundtrip_iou':iou(ideal.cpu().numpy()>0,gt),
            'prediction_min':float(a.min()),'prediction_max':float(a.max()),'prediction_mean':float(a.mean()),
            'foreground_score_mean':float(a[gt].mean()),'background_score_mean':float(a[~gt].mean()),
            'predicted_latent_mse':F.mse_loss(z,target).item(),'all_black_latent_mse':F.mse_loss(black,target).item(),
            'scheduler_vs_negative_velocity_maxabs':(z+v).abs().max().item()}
        if r['split']=='train':
            record['cache_target_maxabs']=(cache['mask'][r['ann_id']].to('cuda')-target).abs().max().item()
            record['cache_rgb_maxabs']=(cache['rgb'][r['image_id']].to('cuda')-rgb).abs().max().item()
            record['cache_text_maxabs']=(cache['text'][r['sent_id']].to('cuda')-emb).abs().max().item()
        records.append(record)
        if len(panels)<3 or (r['split']=='dev' and len(panels)<6):
            panels.append((im,gt,m,((a+1)/2*255).clip(0,255).astype(np.uint8),r))
    canvas=Image.new('RGB',(1024,len(panels)*226),'white');d=ImageDraw.Draw(canvas)
    for i,(im,gt,m,soft,r) in enumerate(panels):
        for j,img in enumerate([im,Image.fromarray(gt.astype(np.uint8)*255),Image.fromarray(m.astype(np.uint8)*255),Image.fromarray(soft)]):canvas.paste(img,(j*256,i*226))
        d.text((3,i*226+195),r['split']+' | '+r['text'][:115],fill='black')
    canvas.save(dest/'diagnostic.png')
    saved=[]
    for step in [0,500,1000]:
        p=OUT/f'seed17/eval_{step}.json'
        if not p.exists():continue
        er=json.loads(p.read_text())['records'];empty=0
        for row in er:
            f=WORK/row['prediction'];assert sha(f)==row['sha256'];empty+=not np.array(Image.open(f)).any()
        saved.append({'step':step,'images':len(er),'empty_masks':empty})
    result={'scope':'post-hoc diagnostic, no training changes','checkpoint_step':1000,'records':records,'saved_prediction_audit':saved}
    write(dest/'results.json',result)
    print(json.dumps({'saved_prediction_audit':saved,'means':{s:{k:float(np.mean([r[k] for r in records if r['split']==s])) for k in ['foreground_fraction','prediction_foreground_fraction','prediction_iou','vae_roundtrip_iou','predicted_latent_mse','all_black_latent_mse']} for s in ['train','dev']}}),flush=True)

if __name__=='__main__':main()
