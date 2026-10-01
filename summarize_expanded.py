"""Report all predefined expanded conditions, separating scene and seed variation."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import load_sample,evaluate_depth
from run_expanded import read,write,matrix

ORDER=['base_s1','base_s4','lora32_s1','lora32_s4','lora64_s1','lora64_s4','head64_s1','head64_s4','expert']
NAMES={'base_s1':'Pretrained, 1 step','base_s4':'Pretrained, 4 steps',
       'lora32_s1':'LoRA 32, 1 step','lora32_s4':'LoRA 32, 4 steps',
       'lora64_s1':'LoRA 64, 1 step','lora64_s4':'LoRA 64, 4 steps',
       'head64_s1':'Head 64, 1 step','head64_s4':'Head 64, 4 steps','expert':'Depth Anything V2'}
METRICS=['abs_rel','rmse_m','delta1','inference_seconds','peak_allocated_mib']

def csv_out(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--work',required=True)
    p.add_argument('--results',default='results/expanded_v1');a=p.parse_args()
    out=Path(a.results);work=Path(a.work);figdir=out/'figures';figdir.mkdir(exist_ok=True)
    manifest=read(out/'split_manifest.json');allrecords=[];runs=[];group_records=defaultdict(list);costs=[]
    for cfg in matrix():
        rid=cfg['run_id'];rd=out/'runs'/rid
        if cfg['mode'] not in ['base','expert']:
            stats=read(rd/'training.json')
            costs.append({'run_id':rid,'mode':cfg['mode'],'train_images':cfg['train_images'],'seed':cfg['train_seed'],
                **{k:stats[k] for k in ['steps','trainable_parameters','training_seconds','latent_cache_seconds','peak_allocated_mib']}})
        for steps in ([None] if cfg['mode']=='expert' else [1,4]):
            label=rid if steps is None else f'{rid}_s{steps}'
            group='expert' if steps is None else (f'base_s{steps}' if cfg['mode']=='base' else f"{cfg['mode']}{cfg['train_images']}_s{steps}")
            records=read(rd/f'{label}_per_image.json');assert len(records)==136
            group_records[group].append(records);allrecords.extend(records)
            for cohort in ['val16','pilot24','fresh96','test120']:
                selected=[r for r in records if (r['split']=='test' if cohort=='test120' else r['cohort']==cohort)]
                row={'condition':label,'group':group,'train_seed':cfg['train_seed'],'cohort':cohort,'images':len(selected)}
                for key in METRICS:
                    vals=[r[key] for r in selected]
                    row[key]=float(max(vals) if key=='peak_allocated_mib' else np.mean(vals))
                runs.append(row)
    assert len(allrecords)==2856
    csv_out(out/'per_run_summary.csv',runs);csv_out(out/'training_cost.csv',costs)
    groups=[]
    for cohort in ['fresh96','test120','pilot24','val16']:
        for group in ORDER:
            selected=[r for r in runs if r['group']==group and r['cohort']==cohort]
            row={'group':group,'cohort':cohort,'images_per_run':selected[0]['images'],'runs':len(selected)}
            for key in METRICS:
                vals=[r[key] for r in selected]
                row[key+'_mean']=float(np.mean(vals))
                row[key+'_seed_std']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
            groups.append(row)
    csv_out(out/'group_summary.csv',groups);write(out/'group_summary.json',groups)
    # Average seed-level metrics per scene before resampling scenes; do not pool seeds as independent scenes.
    scene_means={}
    for group,batches in group_records.items():
        accum=defaultdict(list)
        for batch in batches:
            for row in batch:
                if row['cohort']=='fresh96':accum[row['id']].append(row['abs_rel'])
        scene_means[group]={i:float(np.mean(vals)) for i,vals in accum.items()}
    contrasts=[]
    pairs=[(g,f"base_s{g[-1]}") for g in ORDER if g.startswith(('lora','head'))]
    pairs += [('base_s1','base_s4'),('lora64_s1','lora64_s4'),('lora64_s1','lora32_s1'),('lora64_s4','lora32_s4'),('expert','base_s1')]
    ids=sorted(scene_means['base_s1'])
    boot_indices=np.random.default_rng(4240).integers(0,len(ids),size=(10000,len(ids)))
    for compared,reference in pairs:
        diff=np.array([scene_means[compared][i]-scene_means[reference][i] for i in ids])
        lo,hi=np.quantile(diff[boot_indices].mean(1),[.025,.975])
        contrasts.append({'compared':compared,'reference':reference,'cohort':'fresh96',
            'mean_absrel_difference':float(diff.mean()),'paired_scene_bootstrap_low':float(lo),
            'paired_scene_bootstrap_high':float(hi),'resamples':10000,'seed':4240})
    csv_out(out/'paired_comparisons.csv',contrasts)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    primary={r['group']:r for r in groups if r['cohort']=='fresh96'}
    colors=['#638ac0','#315c9b','#f2b374','#d28232','#7ab5a5','#377e68','#c998a0','#965462','#755a9b']
    fig,axs=plt.subplots(1,3,figsize=(14,6),layout='constrained')
    for ax,key,title in zip(axs,['abs_rel','delta1','inference_seconds'],
                              ['Aligned AbsRel (lower is better)','Aligned delta1 (higher is better)','Mean seconds per image']):
        vals=[primary[g][key+'_mean'] for g in ORDER]
        errs=[primary[g][key+'_seed_std'] or 0 for g in ORDER]
        ax.barh(range(9),vals,xerr=errs,color=colors,capsize=2,error_kw={'elinewidth':1})
        ax.set_yticks(range(9),[NAMES[g] for g in ORDER] if ax is axs[0] else [])
        ax.invert_yaxis();ax.set_title(title,fontsize=11);ax.set_xlim(0,max(np.array(vals)+errs)*1.25)
        ax.set_axisbelow(True);ax.grid(axis='x',alpha=.15)
        for i,v in enumerate(vals):ax.text(v+errs[i]+max(vals)*.02,i,f'{v:.3f}',va='center',fontsize=9)
    fig.suptitle('Expanded evaluation | 96 new test scenes | Error bars: SD across 3 training seeds',fontsize=14)
    fig.savefig(figdir/'fresh96_quality_cost.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,family in zip(axs,['lora32','lora64','head64']):
        for seed,color in zip([17,29,43],['#315c9b','#d28232','#377e68']):
            history=read(out/'runs'/f'{family}_seed{seed}'/'training.json')['history']
            loss=np.array([r['loss'] for r in history])
            ax.plot(np.arange(10,161),np.convolve(loss,np.ones(10)/10,'valid'),color=color,label=f'Seed {seed}')
        ax.set_title(family);ax.set_xlabel('Optimizer step');ax.set_ylabel('10-step mean latent MSE');ax.grid(alpha=.15)
    axs[0].legend(fontsize=8);fig.suptitle('All nine training runs | Random timesteps and noise affect loss')
    fig.savefig(figdir/'training_seeds.png',dpi=170);plt.close(fig)
    selected=[r for r in manifest['samples'] if r.get('cohort')=='fresh96'][:3]
    show=['base_s1','base_s4','lora64_seed17_s1','lora64_seed17_s4','expert']
    fig,axs=plt.subplots(3,8,figsize=(19,7.6));fig.subplots_adjust(left=.055,right=.995,top=.91,bottom=.17,wspace=.07,hspace=.27)
    for ri,row in enumerate(selected):
        rgb,gt=load_sample(a.assets,row);axs[ri,0].imshow(rgb);axs[ri,1].imshow(gt,cmap='viridis',vmin=0,vmax=10)
        axs[ri,0].set_ylabel(f"Frame {row['id']}\n{row['scene']}",fontsize=8)
        for label,col in zip(show,[2,3,4,5,7]):
            pred=np.load(work/'predictions'/label/f"{row['id']:04d}.npy")
            metrics,aligned,mask=evaluate_depth(pred,gt)
            im=axs[ri,col].imshow(aligned,cmap='viridis',vmin=0,vmax=10)
            axs[ri,col].set_xlabel(f"AbsRel {metrics['abs_rel']:.3f}",fontsize=9)
            if col==5:err=axs[ri,6].imshow(np.ma.array(np.abs(aligned-gt),mask=~mask),cmap='magma',vmin=0,vmax=2)
        for ax in axs[ri]:ax.set_xticks([]);ax.set_yticks([])
    for ax,title in zip(axs[0],['Input RGB','Ground truth','Pretrained 1 step','Pretrained 4 steps','LoRA 64, 1 step','LoRA 64, 4 steps','LoRA 4-step error','Depth Anything V2']):ax.set_title(title,fontsize=9)
    fig.colorbar(im,cax=fig.add_axes([.23,.07,.30,.025]),label='GT-aligned depth (m)',orientation='horizontal')
    fig.colorbar(err,cax=fig.add_axes([.63,.07,.30,.025]),label='Absolute error (m; clipped at 2)',orientation='horizontal')
    fig.suptitle('First three fresh test scenes | Fixed training seed 17 | Predictions aligned using ground truth',fontsize=14)
    fig.savefig(figdir/'fresh96_examples.png',dpi=150);plt.close(fig)
    hardest=sorted(scene_means['lora64_s4'],key=scene_means['lora64_s4'].get,reverse=True)[:10]
    samples={r['id']:r for r in manifest['samples']}
    csv_out(out/'lora64_s4_hardest_scenes.csv',[{'id':i,'scene':samples[i]['scene'],
        'lora64_s4_mean_absrel':scene_means['lora64_s4'][i],'base_s4_absrel':scene_means['base_s4'][i]} for i in hardest])
    lines=['# Expanded evaluation v1 results','',
        'Primary evaluation: **96 new test scenes**, excluding the 24 scenes observed in the pilot. Each scene contributes one frame. Training uses the nested original32 or expanded64 pools, validation uses 16 distinct scenes, and the combined test set has 120 scenes. All nine adaptation runs use 160 updates and seeds 17, 29 and 43. Inference noise is fixed independently of training seed.','',
        'All depth metrics use the same per-image ground-truth affine alignment and valid-pixel protocol. They measure relative depth structure, not uncalibrated metric depth. Adapted entries are mean +/- sample SD across three training seeds; baseline entries have one fixed inference realization. These SDs are not scene confidence intervals.','',
        '## Primary results on fresh96','',
        '| Method | Runs | AbsRel | RMSE (m) | delta1 | Seconds/image |','|---|---:|---:|---:|---:|---:|']
    def metric(row,key):
        mean=row[key+'_mean'];std=row[key+'_seed_std']
        return f'{mean:.4f}' if std is None else f'{mean:.4f} +/- {std:.4f}'
    for group in ORDER:
        row=primary[group];lines.append(f"| {NAMES[group]} | {row['runs']} | {metric(row,'abs_rel')} | {metric(row,'rmse_m')} | {metric(row,'delta1')} | {metric(row,'inference_seconds')} |")
    lines+=['','## Matched and paired comparisons','',
        'The difference is compared minus reference AbsRel on fresh96; negative favors the compared method. Scene metrics are averaged across training seeds before a paired bootstrap (10,000 resamples, seed 4240). Intervals reflect scene variation only, with no multiplicity correction. The training-seed SD above must be considered separately.','',
        '| Compared | Reference | AbsRel difference | 95% scene bootstrap interval |','|---|---|---:|---|']
    for r in contrasts:lines.append(f"| {NAMES[r['compared']]} | {NAMES[r['reference']]} | {r['mean_absrel_difference']:+.4f} | [{r['paired_scene_bootstrap_low']:+.4f}, {r['paired_scene_bootstrap_high']:+.4f}] |")
    lines+=['','## Training cost','',
        'Training and latent preparation are timed separately; model loading and downloads are excluded. Peak memory is PyTorch allocated memory, not whole-device VRAM.','',
        '| Condition | Seed | Trainable parameters | Training seconds | Cache seconds | Peak MiB |','|---|---:|---:|---:|---:|---:|']
    for c in costs:lines.append(f"| {c['mode']}{c['train_images']} | {c['seed']} | {c['trainable_parameters']:,} | {c['training_seconds']:.1f} | {c['latent_cache_seconds']:.1f} | {c['peak_allocated_mib']:.0f} |")
    lines+=['','## Secondary cohort checks','',
        '| Method | Combined120 AbsRel | Original pilot24 AbsRel | Validation16 AbsRel |','|---|---:|---:|---:|']
    for group in ORDER:
        lookup={r['cohort']:r for r in groups if r['group']==group}
        lines.append(f"| {NAMES[group]} | {metric(lookup['test120'],'abs_rel')} | {metric(lookup['pilot24'],'abs_rel')} | {metric(lookup['val16'],'abs_rel')} |")
    lines+=['','The specialist has prior NYUv2 supervision, so validation16 is not clean held-out validation for it. It also uses different image processing (252x336 versus Marigold 192x256) and FP32 inference. It is a reference rather than a controlled architecture ablation.','',
        '## Figures','','![Fresh test quality and cost](figures/fresh96_quality_cost.png)','','![Training seed curves](figures/training_seeds.png)','','![Fixed fresh examples](figures/fresh96_examples.png)','',
        '## Scope and limitations','',
        'This fixed matrix extends the pilot; no winner was selected and no hyperparameters were changed using these new test scores. Compared with the pilot, the optimizer budget doubles from 80 to 160, so differences across phases cannot be attributed solely to more data. Within this phase, training sizes share equal update budgets, not equal epochs. Only three training seeds, one data selection, one inference-noise realization, one dataset and low input resolution were studied. Timing is a single sequential pass on a working laptop, not a controlled performance benchmark. Later development must use validation data; these reported test scenes are now observed research evaluation data.','',
        'The companion CSV files preserve every run, cohort, comparison and training cost. `lora64_s4_hardest_scenes.csv` is a transparent post-hoc diagnostic list, not a basis for choosing training parameters. See `verification.json` for the completed provenance and prediction checks.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('FRESH96_SUMMARY')
    for group in ORDER:print(group,metric(primary[group],'abs_rel'),flush=True)
    print('EXPANDED_SUMMARY_COMPLETE')

if __name__=='__main__':main()
