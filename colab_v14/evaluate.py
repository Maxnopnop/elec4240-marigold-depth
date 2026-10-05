"""Native predictions after the verified full-matrix barrier, resume per image."""
from runtime import *


@torch.inference_mode()
def evaluate(arm,size,seed,step,rr,budget):
    ph=sha(WORK/'protocol.json')
    verify_training_barrier(ph)
    name=label(arm,size,seed)
    dest=WORK/'runs'/name/f'fresh_holdout_{step}'
    checkpoint=WORK/'runs'/name/f'adapter_{step}.pt'
    digest=sha(checkpoint)
    if (dest/'metrics.json').exists():
        d=read(dest/'metrics.json')
        assert d['protocol_sha256']==ph and d['checkpoint_sha256']==digest
        return
    dest.mkdir(parents=True,exist_ok=True)
    budget.reserve(600,name+':evaluation_setup')
    pipe,params=setup(seed,checkpoint,arm=arm)
    pipe.unet.eval()
    grouped=defaultdict(list)
    for r in rr:
        if r['split']=='fresh_holdout':grouped[r['image_id']].append(r)
    assert len(grouped)==320 and all(len(v)==2 for v in grouped.values())
    tokens=pipe.tokenizer('',padding='max_length',max_length=77,return_tensors='pt').input_ids.cuda()
    empty=pipe.text_encoder(tokens)[0]
    budget.settle()
    items=[];total_seconds=0
    for index,(iid,pair) in enumerate(sorted(grouped.items())):
        saved=dest/f'{iid}.json'
        if saved.exists():
            d=read(saved)
            assert d['protocol_sha256']==ph and d['checkpoint_sha256']==digest
            for row in d['records']:
                assert sha(WORK/row['prediction'])==row['sha256']
                assert sha(WORK/row['empty_prediction'])==row['empty_sha256']
            items.extend(d['records']);total_seconds+=d['seconds']
            continue
        budget.reserve(180,name+':evaluation_image')
        begin=time.monotonic()
        pair=sorted(pair,key=lambda r:r['ann_id'])
        im=Image.open(pair[0]['image']).convert('RGB')
        a=np.array(im.resize((256,192),Image.Resampling.BILINEAR),copy=True)
        rgb=encode(pipe,torch.from_numpy(a).permute(2,0,1)[None].float().cuda()/127.5-1).float()
        noise=torch.randn(rgb.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(700000+iid))
        truths=[np.array(Image.open(r['mask']))>0 for r in pair]
        def native(emb):
            d=latent_prediction(pipe,rgb,emb,noise)
            return (F.interpolate(d,size=(im.height,im.width),mode='bilinear',align_corners=False)[0,0]>0).cpu().numpy()
        blank=native(empty)
        bp=dest/f'{iid}_empty.png'
        Image.fromarray(blank.astype(np.uint8)*255).save(bp)
        image_records=[]
        for j,r in enumerate(pair):
            prompt=text_prompt(r['text'])
            assert len(pipe.tokenizer(prompt,truncation=False).input_ids)<=77
            tokens=pipe.tokenizer(prompt,padding='max_length',max_length=77,return_tensors='pt').input_ids.cuda()
            pred=native(pipe.text_encoder(tokens)[0])
            own,other=iou(pred,truths[j]),iou(pred,truths[1-j])
            p=dest/f"{r['ann_id']}.png"
            Image.fromarray(pred.astype(np.uint8)*255).save(p)
            image_records.append({'image_id':iid,'ann_id':r['ann_id'],'iou':own,'distractor_iou':other,
                'target_selected':own>other,'empty':not bool(pred.any()),'empty_prompt_iou':iou(blank,truths[j]),
                'prediction':str(p.relative_to(WORK)),'sha256':sha(p),'empty_prediction':str(bp.relative_to(WORK)),
                'empty_sha256':sha(bp),'truth':r['mask'],'truth_sha256':r['cloud_mask_png_sha256']})
        elapsed=time.monotonic()-begin
        write(saved,{'protocol_sha256':ph,'checkpoint_sha256':digest,'records':image_records,'seconds':elapsed})
        items.extend(image_records);total_seconds+=elapsed
        budget.settle()
        if index%16==0:write(WORK/'status.json',{'stage':'evaluating','run':name,'step':step,'images':index+1,'total':320})
    gg=defaultdict(list)
    for r in items:gg[r['image_id']].append(r)
    write(dest/'metrics.json',{'arm':arm,'size':size,'seed':seed,'step':step,'images':320,'expressions':640,
        'protocol_sha256':ph,'checkpoint_sha256':digest,'seconds':total_seconds,
        'mean_iou':float(np.mean([r['iou'] for r in items])),
        'both_targets_selected_rate':float(np.mean([all(r['target_selected'] for r in pair) for pair in gg.values()])),
        'empty_rate':float(np.mean([r['empty'] for r in items])),
        'empty_prompt_mean_iou':float(np.mean([r['empty_prompt_iou'] for r in items])),'records':items})
    del pipe,params
    gc.collect();torch.cuda.empty_cache()
