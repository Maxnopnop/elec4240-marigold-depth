"""Audit every saved prediction and report the entire fixed 26-test family."""
from scale_runtime import *
from collections import defaultdict
import html
import numpy as np
from PIL import Image



def contrasts():
    result=[]
    for size in SIZES:
        for control in ['pixel','pair_always']:
            result.append((('pair_ready',size,4096),(control,size,4096)))
    for arm in ARMS:result.append(((arm,4096,4096),(arm,2048,4096)))
    for arm in ARMS:
        for size in SIZES:result.append(((arm,size,4096),(arm,size,2048)))
    assert len(result)==13
    return result


def holm(tests):
    previous=0
    for rank,index in enumerate(sorted(range(len(tests)),key=lambda i:tests[i]['p'])):
        previous=max(previous,min(1,tests[index]['p']*(len(tests)-rank)))
        tests[index]['holm_p']=previous


def main(tick):
    values={};summaries=[];count=0
    ids=sorted(read(ROOT/'scale_plan.json')['fresh_holdout_images'])
    for arm in ARMS:
        for size in SIZES:
            for step in [2048,4096]:
                allmetrics=[]
                for seed in SEEDS:
                    name=label(arm,size,seed);tick('auditing_predictions',run=name,step=step)
                    d=read(OUT/'runs'/name/f'fresh_holdout_{step}/metrics.json')
                    assert d['protocol_sha256']==sha(OUT/'protocol.json')
                    assert d['checkpoint_sha256']==sha(WORK/'runs'/name/f'adapter_{step}.pt')
                    groups=defaultdict(list)
                    for r in d['records']:groups[r['image_id']].append(r)
                    assert sorted(groups)==ids
                    rows_=[]
                    for iid in ids:
                        pair=groups[iid];assert len(pair)==2
                        truth=[np.array(Image.open(r['truth']))>0 for r in pair]
                        for j,r in enumerate(pair):
                            assert sha(r['truth'])==r['truth_sha256']
                            assert sha(OUT/r['prediction'])==r['sha256']
                            assert sha(OUT/r['empty_prediction'])==r['empty_sha256']
                            pred=np.array(Image.open(OUT/r['prediction']))>0
                            assert iou(pred,truth[j])==r['iou']
                            other=iou(pred,truth[1-j]);assert other==r['distractor_iou']
                            assert (r['iou']>other)==r['target_selected']
                            assert (not bool(pred.any()))==r['empty']
                            assert iou(np.array(Image.open(OUT/r['empty_prediction']))>0,truth[j])==r['empty_prompt_iou']
                            count+=1
                        rows_.append([np.mean([r['iou'] for r in pair]),float(all(r['target_selected'] for r in pair))])
                    v=np.array(rows_);values[(arm,size,step,seed)]=v
                    assert abs(v[:,0].mean()-d['mean_iou'])<1e-12
                    assert abs(v[:,1].mean()-d['both_targets_selected_rate'])<1e-12
                    allmetrics.append(v.mean(0).tolist())
                summaries.append({'arm':arm,'size':size,'step':step,'per_seed_metrics':allmetrics,
                                  'mean_metrics':np.mean(allmetrics,axis=0).tolist()})
    rng=np.random.default_rng(424113);tests=[]
    for a,b in contrasts():
        delta=np.array([values[(*a,seed)]-values[(*b,seed)] for seed in SEEDS])
        for j,metric in enumerate(['mean_iou','both_targets_selected_rate']):
            dif=delta[:,:,j].mean(0);obs=float(dif.mean())
            boot=dif[rng.integers(0,len(ids),size=(5000,len(ids)))].mean(1)
            null=(dif[None,:]*rng.choice([-1,1],size=(10000,len(ids)))).mean(1)
            tests.append({'a':a,'b':b,'metric':metric,'difference':obs,
                'ci95':np.quantile(boot,[.025,.975]).tolist(),
                'p':float((1+(np.abs(null)>=abs(obs)).sum())/10001),
                'per_seed_differences':delta[:,:,j].mean(1).tolist()})
    assert len(tests)==26;holm(tests)
    costs=[dict(run=label(a,n,s),**read(OUT/'runs'/label(a,n,s)/'trained.json')) for a in ARMS for n in SIZES for s in SEEDS]
    caution='Exploratory fixed same-category-pair COCO subset, not full official benchmark or cross-domain proof. Image-cluster intervals condition on two seeds and are not multiplicity-adjusted; Holm p-values cover all26 primary tests. Equal steps do not imply equal epochs or convergence. Preserve all negative results.'
    descriptive=[]
    for arm in ['pair_always','pair_ready']:
        effects={}
        for size in SIZES:
            for step in [2048,4096]:
                delta=np.mean([values[(arm,size,step,seed)]-values[('pixel',size,step,seed)] for seed in SEEDS],axis=0)
                effects[(size,step)]=delta[:,0]
                descriptive.append({'arm':arm,'size':size,'step':step,'comparison':'versus pixel, averaged over two seeds per image',
                    'image_mean_iou_improved_fraction':float(np.mean(delta[:,0]>0)),
                    'image_mean_iou_harmed_fraction':float(np.mean(delta[:,0]<0)),
                    'both_selection_improved_fraction':float(np.mean(delta[:,1]>0)),
                    'both_selection_harmed_fraction':float(np.mean(delta[:,1]<0)),
                    'mean_iou_gain':float(delta[:,0].mean())})
        descriptive.append({'arm':arm,'data_by_steps_interaction_in_gain_vs_pixel':float(((effects[(4096,4096)]-effects[(2048,4096)])-(effects[(4096,2048)]-effects[(2048,2048)])).mean()),
            'scope':'descriptive interaction only; no added hypothesis test or causal claim'})
    write(OUT/'analysis.json',{'predictions_recomputed':count,'summaries':summaries,'contrasts':tests,'costs':costs,'descriptive_correction_harm_and_interaction':descriptive,'caution':caution})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for j,ax in enumerate(axs):
        for arm in ARMS:
            for size in SIZES:
                subset=[r for r in summaries if r['arm']==arm and r['size']==size]
                ax.plot([r['step'] for r in subset],[r['mean_metrics'][j] for r in subset],marker='o',label=f'{arm} n={size}')
        ax.set(xlabel='Additional updates',ylabel=['Mean IoU','Both targets selected'][j]);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'scaling.png',dpi=160);plt.close(fig)
    table=''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [t['a'],t['b'],t['metric'],t['difference'],t['ci95'],t['holm_p']])+'</tr>' for t in tests)
    (OUT/'RESULTS.html').write_text('<!doctype html><meta charset="utf-8"><title>V13 scaling study</title><h1>Generative referring segmentation: data and update scaling</h1><p>'+caution+'</p><img width="1000" src="scaling.png"><table border="1"><tr><th>A</th><th>B</th><th>Metric</th><th>Delta</th><th>95% CI</th><th>Holm p</th></tr>'+table+'</table><p><a href="analysis.json">All results and costs</a></p>',encoding='utf-8')
    return {'predictions_recomputed':count,'contrasts':26}
