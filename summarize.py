"""Aggregate completed runs and render fixed, non-cherry-picked comparisons."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import evaluate_depth, load_sample

METHODS = ['base_s4','lora8_s4','lora32_s4','head32_s4','expert','base_s1']
NAMES = {'base_s4':'Marigold, 4 steps','lora8_s4':'LoRA, 8 images',
         'lora32_s4':'LoRA, 32 images','head32_s4':'Output head, 32 images',
         'expert':'Depth Anything V2','base_s1':'Marigold, 1 step'}
COLORS = ['#315c9b','#9bb9db','#dd8733','#4a957b','#7750a1','#62717b']

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True)
    p.add_argument('--work',required=True);p.add_argument('--results',default='results')
    a=p.parse_args();out=Path(a.results);figdir=out/'figures';figdir.mkdir(exist_ok=True)
    manifest=read(out/'split_manifest.json');samples=manifest['samples']
    rows=[];raw={}
    for method in METHODS:
        records=read(out/f'{method}_per_image.json');raw[method]=records
        expected={r['id'] for r in samples if r['split']!='train'}
        assert {r['id'] for r in records}==expected and len(records)==len(expected),method
        for split in ['val','test']:
            rr=[r for r in records if r['split']==split]
            row={'method':method,'split':split,'images':len(rr)}
            for key in ['abs_rel','rmse_m','delta1','inference_seconds']:
                values=np.array([r[key] for r in rr]);assert np.isfinite(values).all()
                row[key]=float(values.mean())
            row['peak_allocated_mib']=max(r['peak_allocated_mib'] for r in rr)
            rows.append(row)
    write_csv(out/'summary.csv',rows)
    (out/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    # Scene-level paired bootstrap: one frame in each of the 24 test scenes.
    base={r['id']:r for r in raw['base_s4'] if r['split']=='test'}
    intervals=[]
    for method in METHODS[1:]:
        rr=[r for r in raw[method] if r['split']=='test']
        diff=np.array([r['abs_rel']-base[r['id']]['abs_rel'] for r in rr])
        rng=np.random.default_rng(4240)
        boot=diff[rng.integers(0,len(diff),size=(10000,len(diff)))].mean(1)
        lo,hi=np.quantile(boot,[.025,.975])
        intervals.append({'method':method,'minus':'base_s4','abs_rel_difference':float(diff.mean()),
                          'paired_bootstrap_95_low':float(lo),'paired_bootstrap_95_high':float(hi),
                          'resamples':10000,'seed':4240})
    write_csv(out/'paired_absrel_differences.csv',intervals)
    test=[r for r in rows if r['split']=='test']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    for ax,key,title in zip(axs,['abs_rel','delta1','inference_seconds'],
                             ['Aligned AbsRel (lower is better)','Aligned delta1 (higher is better)','Mean inference time (seconds)']):
        vals=[r[key] for r in test]
        ax.barh(np.arange(len(test)),vals,color=COLORS)
        ax.set_yticks(np.arange(len(test)),[NAMES[r['method']] for r in test] if ax is axs[0] else [])
        ax.invert_yaxis();ax.set_title(title,fontsize=11);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        ax.set_xlim(0,max(vals)*1.23)
        for i,v in enumerate(vals):ax.text(v+max(vals)*.02,i,f'{v:.3f}',va='center',fontsize=9)
    fig.suptitle('NYUv2 local pilot | 24 held-out scenes | GT affine alignment',fontsize=15)
    fig.savefig(figdir/'quality_and_latency.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
    training=[]
    for ax,method,color in zip(axs,['lora8','lora32','head32'],COLORS[1:4]):
        stats=read(out/f'{method}_training.json');hist=stats['history']
        ax.plot([h['step'] for h in hist],[h['loss'] for h in hist],color=color,alpha=.35,label='Step loss')
        losses=np.array([h['loss'] for h in hist]);ax.plot(np.arange(5,len(losses)+1),np.convolve(losses,np.ones(5)/5,'valid'),color=color,label='5-step mean')
        ax.set_title(NAMES[f'{method}_s4']);ax.set_xlabel('Optimizer step');ax.set_ylabel('Masked latent MSE');ax.grid(alpha=.15)
        training.append({k:v for k,v in stats.items() if k!='history'})
    axs[0].legend(fontsize=8);fig.suptitle('Training loss varies with sampled noise and timestep',fontsize=13)
    fig.savefig(figdir/'training_losses.png',dpi=170);plt.close(fig)
    write_csv(out/'training_cost.csv',training)
    # Show the first three test frames in the fixed manifest order, irrespective of score.
    chosen=[r for r in samples if r['split']=='test'][:3]
    show=['base_s4','lora32_s4','expert']
    fig,axs=plt.subplots(3,8,figsize=(19,7.6))
    fig.subplots_adjust(left=.055,right=.995,top=.91,bottom=.17,wspace=.07,hspace=.27)
    titles=['Input RGB','Ground truth','Marigold 4-step','Absolute error','LoRA 32','Absolute error','Depth Anything V2','Absolute error']
    for rowidx,row in enumerate(chosen):
        rgb,gt=load_sample(a.assets,row);axs[rowidx,0].imshow(rgb);axs[rowidx,1].imshow(gt,cmap='viridis',vmin=0,vmax=10)
        axs[rowidx,0].set_ylabel(f"Frame {row['id']}\n{row['scene']}",fontsize=8)
        for i,method in enumerate(show):
            pred=np.load(Path(a.work)/'predictions'/method/f"{row['id']:04d}.npy")
            m,aligned,mask=evaluate_depth(pred,gt)
            ax=axs[rowidx,2+i*2];im=ax.imshow(aligned,cmap='viridis',vmin=0,vmax=10)
            ax.set_xlabel(f"AbsRel {m['abs_rel']:.3f}",fontsize=9)
            err=np.ma.array(np.abs(aligned-gt),mask=~mask)
            er=axs[rowidx,3+i*2].imshow(err,cmap='magma',vmin=0,vmax=2)
        for ax in axs[rowidx]:ax.set_xticks([]);ax.set_yticks([])
    for ax,title in zip(axs[0],titles):ax.set_title(title,fontsize=10)
    fig.colorbar(im,cax=fig.add_axes([.23,.07,.30,.025]),label='GT-aligned depth (m)',orientation='horizontal')
    fig.colorbar(er,cax=fig.add_axes([.63,.07,.30,.025]),label='Absolute error (m; clipped at 2)',orientation='horizontal')
    fig.suptitle('Fixed first three test frames | Predictions aligned using ground truth | Crop applied to errors',fontsize=14)
    fig.savefig(figdir/'qualitative.png',dpi=150);plt.close(fig)
    # A compact machine-generated Markdown table is included in the report.
    lines=['# Recorded pilot results','',
           'Means over 24 held-out test frames, one per scene. All scores use per-image ground-truth affine alignment. Lower AbsRel/RMSE and higher delta1 are better. These are relative-depth scores, not uncalibrated metric-depth accuracy.','',
           '| Method | AbsRel | RMSE (m) | delta1 | Seconds/image | Peak allocated MiB |',
           '|---|---:|---:|---:|---:|---:|']
    for r in test:lines.append(f"| {NAMES[r['method']]} | {r['abs_rel']:.4f} | {r['rmse_m']:.4f} | {r['delta1']:.4f} | {r['inference_seconds']:.3f} | {r['peak_allocated_mib']:.0f} |")
    lines+=['','## Adaptation cost','', '| Method | Images | Trainable parameters | Training seconds | Latent-cache seconds | Peak allocated MiB |','|---|---:|---:|---:|---:|---:|']
    for s in training:lines.append(f"| {s['mode']} | {s['train_images']} | {s['trainable_parameters']:,} | {s['training_seconds']:.1f} | {s['latent_cache_seconds']:.1f} | {s['peak_allocated_mib']:.0f} |")
    lines+=['','## Paired uncertainty','',
            'Paired scene bootstrap of test AbsRel differences versus pretrained Marigold (4 steps), 10,000 resamples, seed 4240. Negative differences favor the compared method. This interval only reflects the selected scenes, not training-seed or dataset uncertainty; comparisons are exploratory and not corrected for multiplicity.','',
            '| Method | AbsRel difference | 95% bootstrap interval |','|---|---:|---|']
    for s in intervals:lines.append(f"| {NAMES[s['method']]} | {s['abs_rel_difference']:+.4f} | [{s['paired_bootstrap_95_low']:+.4f}, {s['paired_bootstrap_95_high']:+.4f}] |")
    lines+=['','## Figures','','![Quality and latency](figures/quality_and_latency.png)','','![Training losses](figures/training_losses.png)','','![Fixed test frames](figures/qualitative.png)','','Validation results and individual-image measurements are available in the CSV and JSON files in this directory. No checkpoint was selected using test results.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(test,indent=2));print('SUMMARY_COMPLETE')

if __name__=='__main__':main()
