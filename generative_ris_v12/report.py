"""Recompute saved predictions, fixed image-cluster contrasts, figures and final report."""
import html
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .data import OUT, WORK, sha, write
from .train import ARMS,SEEDS,STEPS,iou,records

def read_run(arm,seed,split):return json.loads((OUT/'runs'/f'{arm}_s{seed}'/f'{split}_{STEPS}/metrics.json').read_text())

def verify_all():
    count=0
    for arm in ARMS:
        for seed in SEEDS:
            d=OUT/'runs'/f'{arm}_s{seed}'
            assert json.loads((d/'complete.json').read_text())['restored_first_pair_exact']
            trained=json.loads((d/'trained.json').read_text())
            assert sha(WORK/f'{arm}_s{seed}/adapter_{STEPS}.pt')==trained['adapter_sha256']
            for split in ['dev','reserved','refcoco','refcocoplus']:
                result=read_run(arm,seed,split)
                byimage=defaultdict(list)
                for r in result['records']:
                    p=OUT/r['prediction'];bp=OUT/r['empty_prediction']
                    assert sha(p)==r['sha256'] and sha(bp)==r['empty_sha256'] and sha(r['truth'])==r['truth_sha256']
                    pred=np.array(Image.open(p))>0;truth=np.array(Image.open(r['truth']))>0
                    assert abs(iou(pred,truth)-r['iou'])<1e-12
                    assert abs(iou(np.array(Image.open(bp))>0,truth)-r['empty_prompt_iou'])<1e-12
                    assert (not bool(pred.any()))==r['empty'];byimage[r['image_id']].append(r);count+=1
                selected=[];both=[]
                for pair in byimage.values():
                    if len(pair)==2:
                        flags=[]
                        for j,r in enumerate(pair):
                            other=iou(np.array(Image.open(OUT/r['prediction']))>0,np.array(Image.open(pair[1-j]['truth']))>0)
                            assert abs(other-r['distractor_iou'])<1e-12
                            flag=r['iou']>other;assert flag==r['target_selected'];flags.append(flag);selected.append(flag)
                        both.append(all(flags))
                assert abs(np.mean([r['iou'] for r in result['records']])-result['mean_iou'])<1e-12
                if both:assert abs(np.mean(both)-result['both_targets_selected_rate'])<1e-12
    return {'final_predictions_recomputed':count,'runs':15,'all_restored_pairs_verified':True}

def cluster(result,metric):
    groups=defaultdict(list)
    for r in result['records']:groups[r['image_id']].append(r)
    if metric=='mean_iou':return {i:np.mean([r['iou'] for r in rr]) for i,rr in groups.items()}
    return {i:float(all(r['target_selected'] for r in rr)) for i,rr in groups.items() if len(rr)==2}

def statistics(protocol):
    rng=np.random.default_rng(424027);tests=[]
    for a,b in protocol['contrasts']:
        for metric in ['mean_iou','both_targets_selected_rate']:
            differences=[];ids=None
            for seed in SEEDS:
                aa=cluster(read_run(a,seed,'reserved'),metric);bb=cluster(read_run(b,seed,'reserved'),metric)
                if ids is None:ids=sorted(aa)
                assert sorted(aa)==ids==sorted(bb)
                differences.append([aa[i]-bb[i] for i in ids])
            dif=np.mean(differences,axis=0);obs=float(dif.mean())
            boot=np.mean(dif[rng.integers(0,len(dif),size=(5000,len(dif)))],axis=1)
            perm=np.mean(dif[None,:]*rng.choice([-1,1],size=(10000,len(dif))),axis=1)
            tests.append({'a':a,'b':b,'metric':metric,'difference':obs,'ci95':np.quantile(boot,[.025,.975]).tolist(),
                'p':float((1+np.sum(np.abs(perm)>=abs(obs)))/10001),'images':len(dif),'per_seed_differences':np.mean(differences,axis=1).tolist()})
    order=sorted(range(len(tests)),key=lambda i:tests[i]['p']);previous=0
    for rank,i in enumerate(order):
        previous=max(previous,min(1,tests[i]['p']*(len(tests)-rank)));tests[i]['holm_p']=previous
    return tests

def main():
    audit=verify_all();protocol=json.loads((OUT/'protocol.json').read_text());tests=statistics(protocol)
    summaries=[]
    for split in ['dev','reserved','refcoco','refcocoplus']:
        for arm in ARMS:
            rr=[read_run(arm,s,split) for s in SEEDS]
            row={'split':split,'arm':arm,'per_seed_iou':[r['mean_iou'] for r in rr]}
            for metric in ['mean_iou','both_targets_selected_rate','empty_rate','target_selection_rate']:
                row[metric]=float(np.mean([r[metric] for r in rr]))
            summaries.append(row)
    costs=[]
    for arm in ARMS:
        rr=[json.loads((OUT/'runs'/f'{arm}_s{s}'/'trained.json').read_text()) for s in SEEDS]
        costs.append({'arm':arm,'mean_optimizer_seconds':float(np.mean([r['optimizer_seconds'] for r in rr])),
            'max_peak_mib':max(r['peak_mib'] for r in rr),
            'mean_reserved_inference_seconds':float(np.mean([read_run(arm,s,'reserved')['seconds'] for s in SEEDS]))})
    write(OUT/'analysis.json',{'verification':audit,'summaries':summaries,'contrasts':tests,'costs':costs,
        'caution':'CI/sign-flip condition on these3seeds and selected images. Same COCO visual domain. No test-driven model selection. No universal novelty/superiority claim.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4))
    for arm in ARMS:
        curve=[]
        for step in [0,512,1024,2048]:
            curve.append(np.mean([json.loads((OUT/'runs'/f'{arm}_s{s}'/f'dev_{step}/metrics.json').read_text())['mean_iou'] for s in SEEDS]))
        ax.plot([0,512,1024,2048],curve,marker='o',label=arm)
    ax.set(xlabel='Additional optimizer updates',ylabel='Development mean IoU');ax.legend();fig.tight_layout();fig.savefig(OUT/'learning_curves.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,metric in zip(axs,['mean_iou','both_targets_selected_rate']):
        vals=[next(r for r in summaries if r['split']=='reserved' and r['arm']==a)[metric] for a in ARMS]
        ax.bar(ARMS,vals);ax.set_ylabel(metric);ax.tick_params(axis='x',rotation=25)
    fig.tight_layout();fig.savefig(OUT/'reserved_results.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for c in costs:
        y=next(r for r in summaries if r['split']=='reserved' and r['arm']==c['arm'])['mean_iou']
        ax.scatter(c['mean_optimizer_seconds']/60,y);ax.annotate(c['arm'],(c['mean_optimizer_seconds']/60,y))
    ax.set(xlabel='Mean training optimizer minutes',ylabel='Reserved mean IoU');fig.tight_layout();fig.savefig(OUT/'cost_accuracy.png',dpi=150);plt.close(fig)
    # Fixed first three heldout images, never selected for favourable predictions.
    rr=[r for r in records() if r['split']=='reserved'];ids=sorted({r['image_id'] for r in rr})[:3]
    canvas=Image.new('RGB',(7*192,6*160),'white');draw=ImageDraw.Draw(canvas)
    for row,r in enumerate([r for i in ids for r in rr if r['image_id']==i]):
        images=[Image.open(r['image']).convert('RGB'),Image.open(r['mask']).convert('RGB')]
        for arm in ARMS:
            pred=next(x for x in read_run(arm,17,'reserved')['records'] if x['ann_id']==r['ann_id'])
            images.append(Image.open(OUT/pred['prediction']).convert('RGB'))
        for col,im in enumerate(images):canvas.paste(im.resize((192,136)),(col*192,row*160))
        draw.text((3,row*160+136),r['text'][:130],fill='black')
    canvas.save(OUT/'fixed_examples.png')
    table=''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [r['split'],r['arm'],f"{r['mean_iou']:.4f}",f"{r['both_targets_selected_rate']:.4f}",f"{r['empty_rate']:.4f}"])+'</tr>' for r in summaries)
    contrast=''.join(f"<tr><td>{r['a']} - {r['b']}</td><td>{r['metric']}</td><td>{r['difference']:.4f}</td><td>{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}</td><td>{r['holm_p']:.4f}</td></tr>" for r in tests)
    body=f'''<!doctype html><meta charset="utf-8"><title>V12 generative RIS ablations</title><style>body{{max-width:1100px;margin:30px auto;font:16px/1.65 system-ui}}table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:6px}}img{{max-width:100%}}</style>
    <h1>生成式指代表达分割：固定的大规模消融实验</h1><p>15 runs × 2048 updates, 2048 training images. All methods continue matching V11 step1000 checkpoints. Original images, text and labels preserved. Final checkpoints only.</p>
    <p>本轮探索解码器拟合程度控制实例对比强度是否有效。实例对比、BCE/Dice及不确定性门控已有文献；组合不自动构成创新。所有正面与负面结果均保留。</p>
    <table><tr><th>Split</th><th>Method</th><th>Mean IoU</th><th>Both targets selected</th><th>Empty rate</th></tr>{table}</table>
    <h2>冻结比较与不确定性</h2><p>256个保留图像聚类，先对3个种子求平均；95%bootstrap区间和10项Holm校正。种子太少，不能将图像区间解释为完整训练随机性区间。外部数据集仍共享COCO图像来源，本轮采样图像相互隔离，不能声称跨视觉域泛化。</p>
    <table><tr><th>Contrast</th><th>Metric</th><th>Difference</th><th>95%CI</th><th>Holm p</th></tr>{contrast}</table>
    <img src="learning_curves.png"><img src="reserved_results.png"><img src="cost_accuracy.png">
    <h2>固定示例</h2><p>列：输入、原始标签、latent、pixel、pair_always、pair_ramp、pair_ready；种子17，按图像ID取前三张，未按效果挑选。</p><img src="fixed_examples.png">
    <p><a href="analysis.json">完整统计和审计</a> · <a href="protocol.json">冻结方案</a> · <a href="input_audit.json">输入完整性</a> · <a href="literature.json">文献与创新边界</a></p>'''
    (OUT/'RESULTS.html').write_text(body,encoding='utf-8')
    for name in ['learning_curves.png','reserved_results.png','cost_accuracy.png','fixed_examples.png']:
        with Image.open(OUT/name) as im:im.verify()
    return audit

if __name__=='__main__':main()
