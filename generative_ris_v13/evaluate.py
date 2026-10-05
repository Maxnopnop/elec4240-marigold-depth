"""Native-resolution masks on preregistered fresh320; invoked after all training."""
from .common import *
import gc
import time
from collections import defaultdict
import numpy as np
from PIL import Image
import torch.nn.functional as F
from multitask_v7.engine import setup, encode
from generative_ris_v11.train_baseline import SIZE, text_prompt, latent_prediction, iou


@torch.inference_mode()
def evaluate(arm,size,seed,step,rr,tick):
    name=label(arm,size,seed);dest=OUT/'runs'/name/f'fresh_holdout_{step}'
    checkpoint=WORK/'runs'/name/f'adapter_{step}.pt';digest=sha(checkpoint)
    if (dest/'metrics.json').exists():
        saved=read(dest/'metrics.json');assert saved['checkpoint_sha256']==digest
        assert saved['protocol_sha256']==sha(OUT/'protocol.json');return
    dest.mkdir(parents=True,exist_ok=True)
    pipe,params=setup(seed,checkpoint);pipe.unet.eval()
    grouped=defaultdict(list)
    for r in rr:
        if r['split']=='fresh_holdout':grouped[r['image_id']].append(r)
    assert len(grouped)==320 and all(len(v)==2 for v in grouped.values())
    empty_ids=pipe.tokenizer('',padding='max_length',max_length=77,return_tensors='pt').input_ids.cuda()
    empty=pipe.text_encoder(empty_ids)[0];items=[];begin=time.perf_counter()
    for n,(iid,pair) in enumerate(sorted(grouped.items())):
        if n%16==0:tick('evaluating',run=name,step=step,images=n,total=320)
        pair=sorted(pair,key=lambda r:r['ann_id'])
        im=Image.open(pair[0]['image']).convert('RGB')
        a=np.array(im.resize((SIZE[1],SIZE[0]),Image.Resampling.BILINEAR),copy=True)
        rgb=encode(pipe,torch.from_numpy(a).permute(2,0,1)[None].float().cuda()/127.5-1).float()
        noise=torch.randn(rgb.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(700000+iid))
        truths=[np.array(Image.open(r['mask']))>0 for r in pair]
        def native(embedding):
            d=latent_prediction(pipe,rgb,embedding,noise)
            return (F.interpolate(d,size=(im.height,im.width),mode='bilinear',align_corners=False)[0,0]>0).cpu().numpy()
        blank=native(empty);bp=dest/f'{iid}_empty.png';Image.fromarray(blank.astype(np.uint8)*255).save(bp)
        for j,r in enumerate(pair):
            assert len(pipe.tokenizer(text_prompt(r['text']),truncation=False).input_ids)<=77
            tokens=pipe.tokenizer(text_prompt(r['text']),padding='max_length',max_length=77,return_tensors='pt').input_ids.cuda()
            pred=native(pipe.text_encoder(tokens)[0]);own=iou(pred,truths[j]);other=iou(pred,truths[1-j])
            p=dest/f"{r['ann_id']}.png";Image.fromarray(pred.astype(np.uint8)*255).save(p)
            items.append({'image_id':iid,'ann_id':r['ann_id'],'iou':own,'distractor_iou':other,'target_selected':own>other,
                'empty':not bool(pred.any()),'empty_prompt_iou':iou(blank,truths[j]),'prediction':str(p.relative_to(OUT)),
                'sha256':sha(p),'empty_prediction':str(bp.relative_to(OUT)),'empty_sha256':sha(bp),
                'truth':r['mask'],'truth_sha256':r['mask_sha256']})
    groups=defaultdict(list)
    for r in items:groups[r['image_id']].append(r)
    write(dest/'metrics.json',{'arm':arm,'size':size,'seed':seed,'step':step,'images':320,'expressions':640,
        'protocol_sha256':sha(OUT/'protocol.json'),'checkpoint_sha256':digest,'seconds':time.perf_counter()-begin,
        'mean_iou':float(np.mean([r['iou'] for r in items])),
        'both_targets_selected_rate':float(np.mean([all(r['target_selected'] for r in pair) for pair in groups.values()])),
        'empty_rate':float(np.mean([r['empty'] for r in items])),
        'empty_prompt_mean_iou':float(np.mean([r['empty_prompt_iou'] for r in items])),
        'records':items})
    del pipe,params;gc.collect();torch.cuda.empty_cache()
