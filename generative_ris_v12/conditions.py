"""Exploratory condition reliability/budget/harm studies; box diagnostics, not mask GT."""
import gc
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw
from multitask_v7.engine import setup,encode
from generative_ris_v11.train_baseline import SIZE,text_prompt,latent_prediction
from .data import ROOT,OUT,WORK,OLDWORK,sha,write
from .gqa_data import DEST,DATA,COLORS

SEEDS=[17,29,43]
VARIANTS=['clean','lowres48','occluded50']
POLICIES=['original','hard_all','safe_all','color_first_1','color_first_2','relation_first_1','relation_first_2','random_1','random_2','adaptive_1','adaptive_2']

def rows():return json.loads((DEST/'data_manifest.json').read_text())['records']

def freeze():
    sources=[ROOT/'generative_ris_v12/conditions.py',ROOT/'generative_ris_v12/gqa_data.py']
    p={'scope':'3 exploratory mechanisms; frozen generative candidates + small color verifier; GQA box/condition diagnostic, NOT RIS mask benchmark',
        'images':{'train':512,'calibration':64,'test':128},'seeds':SEEDS,'colors':COLORS,
        'generator':'fixedV11seed17step3000, selected before V12outcomes; Marigold/VAE/CLIP allfrozen',
        'generator_sha256':sha(OLDWORK/'seed17/adapter_3000.pt'),
        'verifier':'VAEcrop latent adaptivepool4x4(64dims)+RGBmean/std(6dims), standardized trainingfeatures;MLP70-64-11,CPUAdamW1e-3,50epochs,batch128,square-root-inverse-frequencyCE',
        'relation':'geometry on predicted candidate mask bounding boxes and oracle reference-object boxes; naivefirstreference vs all-reference agreement/abstention; no learned relation detector',
        'training_variants':['clean','lowres96','occluded25'],'evaluation_variants':VARIANTS,
        'corruptions':'separate derived imagecopies: resizefullimage thenupsample; blackrectangle covering leftfraction25/50% originaltargetbox; sourcefiles/scenegraphsunchanged',
        'labels':'original sole color attributes and original left/right edges; graph-derived queries; opposite relation is contradiction; no colour guessed from text',
        'threshold':'smallest in[.5,.6,.7,.8,.9,.95,.99,.999] with Wilson95%upper falseveto<=.10 oncalibration original target+distractor colors,3trainingperturbations; otherwise1(abstain). No test tuning.',
        'candidates':'3 direct generative masks:fullquery/noise0,fullquery/noise1,nounonly/noise0; samepool allpolicies; no GTcandidate added',
        'verifier_feature_training':'oracle object boxes, separately diagnosed; application uses generatedmaskboundingboxes',
        'policies':POLICIES,'adaptive':'untrained heuristic choose geometry center spread/(referencecount*fixedrelativecostprior0.01) vs RGBmean spread; only cheap preview statistics; not a learned policy or novelty claim',
        'decision':'veto thenretain lowestoriginalcandidateindex; ifallvetoedretain original;budgetmethods stop atunique survivor; safe_all executes both; hard_all threshold.5 naivefirstreference',
        'uncertainty':'support ifp>=threshold,refute ifp<=1-threshold,otherwiseunknown; unknown never vetoes',
        'metrics':['originalcolor accuracy','false veto oftruecolor','contradiction coverage','boxIoU','boxIoU>=.5','correction count','harm count','candidateboxrecall','conditioncalls','decision latency onfirst16cleantestimages'],
        'limitations':['oracle reference boxes','scenegraph annotationnoise','derivedqueries notoriginalGQAquestions','three candidate masks may miss target','no segmaskGT','adaptive heuristic not trained','stress results separatefromoriginalbenchmarks'],
        'manifest_sha256':sha(DEST/'data_manifest.json'),'sources':{str(p.relative_to(ROOT)):sha(p) for p in sources}}
    f=DEST/'protocol.json'
    if f.exists():assert json.loads(f.read_text())==p
    else:write(f,p)
    write(OUT/'extra_plan.json',{'module':'generative_ris_v12.conditions','source_sha256':p['sources'],'protocol_sha256':sha(f)})
    return p

def box(o):return [o['x'],o['y'],o['x']+o['w'],o['y']+o['h']]

def box_iou(a,b):
    if a is None or b is None:return 0.
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-intersection
    return intersection/union if union else 0.

def image_variant(r,variant):
    im=Image.open(r['image']).convert('RGB')
    if variant.startswith('lowres'):
        n=int(variant[6:]);im=im.resize((n,n),Image.Resampling.BILINEAR).resize(im.size,Image.Resampling.BILINEAR)
    elif variant.startswith('occluded'):
        fraction=int(variant[8:])/100;x0,y0,x1,y1=box(r['objects'][r['target']])
        im=im.copy();ImageDraw.Draw(im).rectangle([x0,y0,x0+(x1-x0)*fraction,y1],fill=(0,0,0))
    return im

@torch.no_grad()
def feature(pipe,im,b):
    if b is None:return torch.zeros(70)
    x0,y0,x1,y1=b;x0=max(0,min(im.width-1,x0));y0=max(0,min(im.height-1,y0));x1=max(x0+1,min(im.width,x1));y1=max(y0+1,min(im.height,y1))
    crop=im.crop((x0,y0,x1,y1)).resize((96,96),Image.Resampling.BILINEAR)
    a=np.array(crop,copy=True);x=torch.from_numpy(a).permute(2,0,1)[None].float().cuda()/127.5-1
    z=encode(pipe,x).float();latent=F.adaptive_avg_pool2d(z,4).flatten().cpu()
    stats=torch.cat([x.mean((0,2,3)),x.std((0,2,3))]).cpu()
    return torch.cat([latent,stats])

def feature_cache(pipe,rr):
    path=DATA/'condition_features.pt'
    if path.exists():
        d=torch.load(path,map_location='cpu',weights_only=True);assert d['protocol_sha256']==sha(DEST/'protocol.json');return d
    x=[];y=[];meta=[]
    for n,r in enumerate(rr):
        variants=['clean','lowres96','occluded25'] if r['split']!='test' else VARIANTS
        for variant in variants:
            im=image_variant(r,variant)
            for oid in [r['target'],r['distractor']]:
                obj=r['objects'][oid];color=next(c for c in COLORS if c in obj['attributes'])
                x.append(feature(pipe,im,box(obj)));y.append(COLORS.index(color))
                meta.append({'split':r['split'],'image_id':r['image_id'],'object_id':oid,'variant':variant})
        if (n+1)%32==0:write(DEST/'status.json',{'stage':'frozen_vae_features','images':n+1,'total':len(rr)})
    d={'x':torch.stack(x),'y':torch.tensor(y),'meta':meta,'protocol_sha256':sha(DEST/'protocol.json')};torch.save(d,path)
    write(DEST/'feature_audit.json',{'sha256':sha(path),'examples':len(meta)});return d

def head():return torch.nn.Sequential(torch.nn.Linear(70,64),torch.nn.ReLU(),torch.nn.Linear(64,len(COLORS)))

def wilson_upper(k,n):
    if n==0:return 1.
    z=1.96;p=k/n
    return (p+z*z/(2*n)+z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)

def train_heads(d):
    torch.set_num_threads(4);train=torch.tensor([i for i,m in enumerate(d['meta']) if m['split']=='train']);cal=torch.tensor([i for i,m in enumerate(d['meta']) if m['split']=='calibration'])
    mean=d['x'][train].mean(0);std=d['x'][train].std(0).clamp_min(.01);x=(d['x']-mean)/std
    counts=torch.bincount(d['y'][train],minlength=len(COLORS)).float();weight=(counts.sum()/counts.clamp_min(1)).sqrt();weight/=weight.mean()
    result=[]
    for seed in SEEDS:
        torch.manual_seed(seed);net=head();optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3);history=[]
        for epoch in range(50):
            order=train[torch.randperm(len(train))];total=0.
            for batch in order.split(128):
                optimizer.zero_grad();loss=F.cross_entropy(net(x[batch]),d['y'][batch],weight=weight);assert torch.isfinite(loss);loss.backward();optimizer.step();total+=loss.item()
            history.append(total)
        net.eval()
        with torch.no_grad():p=net(x[cal]).softmax(-1);true=p[torch.arange(len(cal)),d['y'][cal]]
        table=[];threshold=1.
        for t in [.5,.6,.7,.8,.9,.95,.99,.999]:
            k=int((true<=1-t).sum());upper=wilson_upper(k,len(true));table.append({'threshold':t,'false_veto':k,'n':len(true),'wilson_upper':upper})
            if upper<=.1 and threshold==1.:threshold=t
        checkpoint={'state':net.state_dict(),'mean':mean,'std':std,'threshold':threshold,'protocol_sha256':sha(DEST/'protocol.json')}
        torch.save(checkpoint,DATA/f'color_head_{seed}.pt')
        restored=head();restored.load_state_dict(torch.load(DATA/f'color_head_{seed}.pt',weights_only=True)['state']);restored.eval()
        with torch.no_grad():assert torch.equal(net(x[cal]),restored(x[cal]))
        write(DEST/f'head_{seed}.json',{'threshold':threshold,'calibration':table,'loss_history':history,'checkpoint_sha256':sha(DATA/f'color_head_{seed}.pt'),'restored_logits_exact':True})
        result.append((seed,net,mean,std,threshold))
    return result

@torch.no_grad()
def generate(pipe,r,im):
    a=np.array(im.resize((SIZE[1],SIZE[0]),Image.Resampling.BILINEAR),copy=True)
    rgb=encode(pipe,torch.from_numpy(a).permute(2,0,1)[None].float().cuda()/127.5-1).float();candidates=[]
    noun=r['objects'][r['target']]['name'];start=time.perf_counter()
    for i,(text,offset) in enumerate([(r['query'],0),(r['query'],1),('the '+noun,0)]):
        prompt=text_prompt(text);assert len(pipe.tokenizer(prompt,truncation=False).input_ids)<=77
        ids=pipe.tokenizer(prompt,padding='max_length',max_length=77,return_tensors='pt').input_ids.cuda()
        noise=torch.randn(rgb.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(424030+int(r['image_id'])*2+offset))
        d=latent_prediction(pipe,rgb,pipe.text_encoder(ids)[0],noise)
        mask=(F.interpolate(d,size=(im.height,im.width),mode='bilinear',align_corners=False)[0,0]>0).cpu().numpy()
        yy,xx=np.where(mask);b=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if len(xx) else None
        candidates.append({'box':b,'mask':mask})
    torch.cuda.synchronize();return candidates,time.perf_counter()-start

def relation_prob(r,b,safe=True):
    if b is None:return .5
    name=r['objects'][r['reference']]['name'];refs=[o for _,o in sorted(r['objects'].items()) if o['name']==name]
    if not safe:refs=refs[:1]
    center=(b[0]+b[2])/2;sign=-1 if r['relation']=='to the left of' else 1
    p=[1/(1+math.exp(-float(np.clip(sign*(center-(o['x']+o['w']/2))/r['width']*20,-40,40)))) for o in refs]
    if min(p)<.5<max(p):return .5
    return min(p) if min(p)>=.5 else max(p)

def choose(policy,probs,threshold,boxes,r,preview):
    if policy=='original':return 0,0,[]
    hard=policy=='hard_all';budget=2 if policy.endswith('all') else int(policy[-1]);alive=list(range(3));used=[]
    if policy.startswith('color'):order=[0,1]
    elif policy.startswith('relation'):order=[1,0]
    elif policy.startswith('random'):
        order=[0,1];random.Random(424031+int(r['image_id'])).shuffle(order)
    elif policy.startswith('adaptive'):
        refs=sum(o['name']==r['objects'][r['reference']]['name'] for o in r['objects'].values())
        centers=[(b[0]+b[2])/2/r['width'] for b in boxes if b is not None]
        geometry=(max(centers)-min(centers))/max(1,refs) if centers else 0
        color=float(np.std(preview,axis=0).mean()) if preview else 0
        # Fixed relative cost prior; preflight decision-time benchmark reports actual costs.
        order=[1,0] if geometry/.01>=color else [0,1]
    else:order=[0,1]
    for condition in order[:budget]:
        used.append(condition);p=probs(condition,not hard)
        cutoff=.5 if hard else 1-threshold if condition==0 else .1
        alive=[i for i in alive if p[i]>cutoff]
        if len(alive)<=1 and not policy.endswith('all'):break
    return (alive[0] if alive else 0),len(used),used

def main():
    protocol=freeze();rr=rows()
    manifest=json.loads((DEST/'data_manifest.json').read_text())
    assert sha(DATA/'raw/sceneGraphs.zip')==manifest['scenegraphs_sha256']
    assert sha(DATA/'raw/image_data.json.zip')==manifest['metadata_sha256']
    assert sha(DEST/'selection.json')==manifest['selection_sha256']
    for r in rr:assert sha(r['image'])==r['image_sha256']
    if (DEST/'complete.json').exists():
        receipt=json.loads((DEST/'complete.json').read_text())
        for p,h in receipt['output_sha256'].items():assert sha(p)==h
        return
    pipe,params=setup(17,OLDWORK/'seed17/adapter_3000.pt');pipe.unet.eval()
    for p in pipe.unet.parameters():p.requires_grad_(False)
    d=feature_cache(pipe,rr);heads=train_heads(d);reliability=[]
    for seed,net,mean,std,threshold in heads:
        with torch.no_grad():p=net((d['x']-mean)/std).softmax(-1)
        for variant in VARIANTS:
            ids=[i for i,m in enumerate(d['meta']) if m['split']=='test' and m['variant']==variant]
            labels=d['y'][ids];true=p[ids,labels];false=p[ids,(labels+1)%len(COLORS)]
            reliability.append({'seed':seed,'variant':variant,'n':len(ids),'color_accuracy':float((p[ids].argmax(-1)==labels).float().mean()),
                'hard_false_veto':float((true<=.5).float().mean()),'safe_false_veto':float((true<=1-threshold).float().mean()),
                'hard_contradiction_coverage':float((false<=.5).float().mean()),'safe_contradiction_coverage':float((false<=1-threshold).float().mean()),'threshold':threshold})
    write(DEST/'condition_reliability.json',reliability)
    results=[];timings=[];test=[r for r in rr if r['split']=='test']
    for n,r in enumerate(test):
        for variant in VARIANTS:
            im=image_variant(r,variant);candidates,generation_seconds=generate(pipe,r,im);bboxes=[c['box'] for c in candidates]
            target=box(r['objects'][r['target']]);ious=[box_iou(b,target) for b in bboxes];feats=torch.stack([feature(pipe,im,b) for b in bboxes])
            dest=DATA/'predictions'/r['image_id']/variant;dest.mkdir(parents=True,exist_ok=True);im.save(dest/'derived_input.png')
            for i,c in enumerate(candidates):Image.fromarray(c['mask'].astype(np.uint8)*255).save(dest/f'candidate{i}.png')
            preview=[np.array(im.crop(b).resize((8,8))).mean((0,1))/255 if b else np.zeros(3) for b in bboxes]
            refs=sum(o['name']==r['objects'][r['reference']]['name'] for o in r['objects'].values())
            for seed,net,mean,std,threshold in heads:
                with torch.no_grad():colors=net((feats-mean)/std).softmax(-1)[:,COLORS.index(r['color'])].numpy()
                def probs(condition,safe):return colors if condition==0 else np.array([relation_prob(r,b,safe) for b in bboxes])
                for policy in POLICIES:
                    chosen,calls,used=choose(policy,probs,threshold,bboxes,r,preview)
                    before=ious[0]>=.5;after=ious[chosen]>=.5
                    results.append({'image_id':r['image_id'],'variant':variant,'seed':seed,'policy':policy,'selected':chosen,
                        'box_iou':ious[chosen],'correct':after,'baseline_correct':before,'corrected':not before and after,'harmed':before and not after,
                        'candidate_recall':max(ious)>=.5,'condition_calls':calls,'conditions_used':used,'generator_calls':3,'reference_candidates':refs,
                        'generation_seconds_shared':generation_seconds,'candidate_boxes':bboxes,'target_box':target})
                    # Real decision-phase latency on fixed first16 cleanimages, seed17 only.
                    if n<16 and variant=='clean' and seed==17:
                        def actual(condition,safe):
                            if condition==1:return np.array([relation_prob(r,b,safe) for b in bboxes])
                            ff=torch.stack([feature(pipe,im,b) for b in bboxes])
                            with torch.no_grad():return net((ff-mean)/std).softmax(-1)[:,COLORS.index(r['color'])].numpy()
                        torch.cuda.synchronize();start=time.perf_counter();cc,nn,_=choose(policy,actual,threshold,bboxes,r,preview);torch.cuda.synchronize()
                        assert cc==chosen and nn==calls
                        timings.append({'image_id':r['image_id'],'policy':policy,'decision_seconds':time.perf_counter()-start,'calls':calls,
                            'caveat':'shared3callgenerativefrontend excluded; live verifier feature computation included; preview generation excluded'})
        write(DEST/'status.json',{'stage':'condition_policy_evaluation','images':n+1,'total':len(test)})
    write(DEST/'per_case.json',results);write(DEST/'latency.json',timings)
    summaries=[]
    for variant in VARIANTS:
        for policy in POLICIES:
            selected=[r for r in results if r['variant']==variant and r['policy']==policy]
            row={'variant':variant,'policy':policy,'observations':len(selected),'independent_images':128}
            for key in ['box_iou','correct','corrected','harmed','candidate_recall','condition_calls']:
                row[key]=float(np.mean([r[key] for r in selected]))
            correct=sum(r['baseline_correct'] for r in selected)
            row['harm_given_original_correct']=sum(r['harmed'] for r in selected)/correct if correct else None
            row['repair_given_original_wrong']=sum(r['corrected'] for r in selected)/(len(selected)-correct) if len(selected)>correct else None
            row['mean_corrected_count_over_seeds']=sum(r['corrected'] for r in selected)/len(SEEDS)
            row['mean_harmed_count_over_seeds']=sum(r['harmed'] for r in selected)/len(SEEDS)
            summaries.append(row)
    strata=[]
    for ambiguous in [False,True]:
        for variant in VARIANTS:
            for policy in POLICIES:
                selected=[r for r in results if r['variant']==variant and r['policy']==policy and (r['reference_candidates']>1)==ambiguous]
                strata.append({'ambiguous_reference':ambiguous,'variant':variant,'policy':policy,'observations':len(selected),
                    'correct':float(np.mean([r['correct'] for r in selected])) if selected else None,
                    'harmed':float(np.mean([r['harmed'] for r in selected])) if selected else None})
    write(DEST/'summary.json',{'rows':summaries,'condition_reliability':reliability,
        'reference_ambiguity_strata':strata,
        'interpretation':'Exploratory box-grounding and known-condition diagnostics; repeated3heads are not384independentimages. No segmentation IoU claim, no learned selector or universal innovation claim.'})
    # Independent metric reconstruction from saved candidate masks, not cached boxes.
    verified=0
    for r in results:
        p=DATA/'predictions'/r['image_id']/r['variant']/f"candidate{r['selected']}.png"
        yy,xx=np.where(np.array(Image.open(p))>0);b=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if len(xx) else None
        value=box_iou(b,r['target_box']);assert abs(value-r['box_iou'])<1e-12;assert (value>=.5)==r['correct'];verified+=1
    for r in rr:assert sha(r['image'])==r['image_sha256']
    for p,h in protocol['sources'].items():assert sha(ROOT/p)==h
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for policy in ['original','hard_all','safe_all','adaptive_1','adaptive_2']:
        r=[x for x in summaries if x['policy']==policy]
        axs[0].plot(VARIANTS,[x['correct'] for x in r],marker='o',label=policy)
        axs[1].plot(VARIANTS,[x['harmed'] for x in r],marker='o',label=policy)
    axs[0].set_ylabel('Box grounding accuracy');axs[1].set_ylabel('Previously correct cases harmed / all cases')
    axs[0].legend();fig.tight_layout();fig.savefig(DEST/'correction_harm.png',dpi=150);plt.close(fig)
    table=''.join(f"<tr><td>{r['variant']}</td><td>{r['policy']}</td><td>{r['correct']:.3f}</td><td>{r['corrected']:.3f}</td><td>{r['harmed']:.3f}</td><td>{r['condition_calls']:.2f}</td></tr>" for r in summaries)
    (DEST/'RESULTS.html').write_text('''<!doctype html><meta charset="utf-8"><title>Condition checks: exploratory diagnostics</title><style>body{max-width:1000px;margin:30px auto;font:16px/1.6 system-ui}td,th{padding:5px;border:1px solid #ccc}img{max-width:100%}</style><h1>条件可靠性、检查预算、纠错与误伤</h1><p>原始 GQA 图像及图结构保持不变。使用原始颜色属性和左右关系构造单独诊断问题；不是官方 GQA 问答成绩，也没有像素掩码真值，以下只报告对象框定位。主干为冻结的 Marigold，三个候选均来自生成式掩码。</p><p>512训练图像、64校准图像、128测试图像；颜色小模块3种子，空间检查使用真实参照物候选框，因此仍有 oracle 输入。自适应顺序为固定启发式，尚非训练出来的新策略。遮挡和降分辨率副本与原图分开保存，不修改原始标签。</p><p>正确率阈值为box IoU≥0.5。纠正/误伤均以全部样本为分母；完整逐样本结果可另行计算条件误伤率。实际延迟只测固定16张清晰图像的检查阶段，包含重新提取颜色特征，排除共享生成前端和预览，不声称整体系统已加速。</p><table><tr><th>Variant</th><th>Policy</th><th>Accuracy</th><th>Correction</th><th>Harm</th><th>Calls</th></tr>'''+table+'''</table><img src="correction_harm.png"><p><a href="summary.json">完整汇总</a> · <a href="condition_reliability.json">条件可靠性</a> · <a href="latency.json">实际检查阶段延迟</a> · <a href="protocol.json">固定方案</a></p>''',encoding='utf-8')
    hashes={str(p.resolve()):sha(p) for folder in [DEST,DATA/'predictions'] for p in folder.rglob('*') if p.is_file() and p.name not in ['complete.json','status.json']}
    for seed in SEEDS:hashes[str((DATA/f'color_head_{seed}.pt').resolve())]=sha(DATA/f'color_head_{seed}.pt')
    write(DEST/'complete.json',{'status':'complete','policies':len(POLICIES),'test_images':128,'variants':3,'verifier_seeds':3,'per_case_metrics_verified':verified,
        'output_sha256':hashes,
        'original_images_unchanged':True,'report':str(DEST/'RESULTS.html'),'claims_limited_to_exploratory_condition_and_box_diagnostics':True})
    del pipe,params;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    if args.freeze:freeze()
    else:main()
