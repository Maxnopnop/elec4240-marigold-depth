"""Report all frozen external results after the complete technical audit."""
import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_expanded import read,write,sha
from prospective_statistics import analyze
from audit_bootstrap_intervals import matched_intervals
from prospective_metrics import aligned_metrics,valid_mask

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path('results/prospective_v4'))
    p.add_argument('--data',type=Path,default=Path(r'E:\Codex\2026-09-27\yo\work\marigold-local\assets\external_sun3d_v4'))
    p.add_argument('--work',type=Path,default=Path(r'E:\Codex\2026-09-27\yo\work\marigold-local\prospective_v4'))
    a=p.parse_args();out=a.results
    audit=read(out/'verification.json');assert audit['status']=='passed' and audit['metrics_recomputed']==12600
    manifest=read(out/'manifest.json');rows=manifest['samples'];n=len(rows);assert n==200
    for name,checksum in audit['audited_artifact_sha256'].items():assert sha(out/name)==checksum
    groupnames=['base_r256','high512_r256','mixed_r256','base_r512','high512_r512','mixed_r512','expert_default']
    metricnames=['abs_rel','rmse_m','delta1','calibrated_abs_rel','calibrated_rmse_m','calibrated_delta1']
    arrays={g:{m:np.full((3,5,n) if g.startswith(('high512','mixed')) else (1,1,n),np.nan) for m in metricnames} for g in groupnames}
    runs=[];allrecs=[]
    for path in sorted((out/'evaluations').glob('*/per_image.json')):
        rec=read(path);label=path.parent.name
        assert [r['id'] for r in rec]==[r['id'] for r in rows]
        if label.startswith('draw'):
            pieces=label.split('_');d=int(pieces[0][4:])-1;s=[17,29,43,59,71].index(int(pieces[2][4:]));g=pieces[1]+'_'+pieces[3]
        else:d=s=0;g=label
        for m in metricnames:arrays[g][m][d,s]=[r[m] for r in rec]
        runs.append({'condition':label,'group':g,'draw':d+1 if label.startswith('draw') else None,
                     'seed':[17,29,43,59,71][s] if label.startswith('draw') else None,
                     **{m:float(np.mean([r[m] for r in rec])) for m in metricnames},
                     'inference_seconds_mean':float(np.mean([r['inference_seconds'] for r in rec]))})
        allrecs.extend(rec)
    assert all(np.isfinite(arr).all() for v in arrays.values() for arr in v.values())
    scores=[]
    for g in groupnames:
        scores.append({'group':g,**{m:float(arrays[g][m].mean()) for m in metricnames},
                       'draw_absrel_means':arrays[g]['abs_rel'].mean(axis=(1,2)).tolist()})
    comparisons=analyze({g:arrays[g]['abs_rel'] for g in groupnames if g!='expert_default'},[r['location_proxy'] for r in rows])
    intervals=[]
    for r in comparisons:
        delta=arrays[r['compared']]['abs_rel']-arrays[r['reference']]['abs_rel']
        intervals.append({'compared':r['compared'],'reference':r['reference'],**{kind:matched_intervals(delta,kind,r[kind]['p_bootstrap']) for kind in ['hierarchical','crossed']}})
    write(out/'summary.json',scores);write(out/'paired_comparisons.json',comparisons);write(out/'matched_intervals.json',intervals)
    with (out/'per_run_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(runs[0]));w.writeheader();w.writerows(runs)
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    selected=['base_r512','high512_r512','mixed_r512','expert_default']
    for ax,metric,title in zip(axes,['abs_rel','calibrated_abs_rel'],['Per-image GT alignment','Fixed NYUv2 validation calibration']):
        vals=[float(arrays[g][metric].mean()) for g in selected]
        ax.bar(range(4),vals,color=['#64748b','#059669','#d97706','#7c3aed'])
        ax.set(xticks=range(4),xticklabels=['Original512','Train512','Mixed512','Expert default'],title=title,ylabel='AbsRel (lower is better)',ylim=(0,max(vals)*1.2))
        for j,v in enumerate(vals):ax.annotate(f'{v:.4f}',(j,v),xytext=(0,4),textcoords='offset points',ha='center')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Frozen external SUN3D-source cohort | 200 scene groups | one frame per group')
    fig.savefig(figdir/'external_scores.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5.7),layout='constrained');labels=[]
    for i,(r,ci) in enumerate(zip(comparisons,intervals)):
        h=ci['hierarchical']
        ax.plot([h['bonferroni_ci_low'],h['bonferroni_ci_high']],[i,i],lw=3,color='#94a3b8')
        ax.plot([h['ci_low'],h['ci_high']],[i,i],lw=5,color='#2563eb')
        ax.scatter(h['difference'],i,color='#0f172a',zorder=3)
        labels.append(f"{r['compared']} - {r['reference']}\nHolm nested={r['hierarchical']['p_holm']:.4f}, crossed={r['crossed']['p_holm']:.4f}")
    ax.axvline(0,ls='--',color='black',lw=1);ax.set(yticks=range(6),yticklabels=labels,xlabel='Aligned AbsRel difference; negative favors first method')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.suptitle('Six frozen comparisons | blue: matched 95%; gray: Bonferroni | no optional stopping')
    fig.savefig(figdir/'external_comparisons.png',dpi=150);plt.close(fig)
    # Fixed before reading outcomes: first three manifest rows, draw 1 / seed 17.
    cal=read('results/robustness_v3/calibration.json')['conditions']['draw1_mixed_seed17_r512']
    fig,axes=plt.subplots(3,7,figsize=(17,8),layout='constrained')
    titles=['RGB','Measured GT','Original: GT aligned','Mixed: GT aligned',
            'Mixed: fixed calibration','Expert: native metric','Mixed: metric abs. error']
    for i,row in enumerate(rows[:3]):
        with np.load(a.data/row['file']) as data:rgb=data['image'];gt=data['depth']
        mask=valid_mask(gt)
        base,mixed,expert=[np.load(a.work/'predictions'/label/f"{row['id']:04d}.npy")
                           for label in ['base_r512','draw1_mixed_seed17_r512','expert_default']]
        aligned=[]
        for pred in [base,mixed]:
            fit=aligned_metrics(pred,gt);aligned.append(np.clip(pred*fit['scale']+fit['shift'],.1,10))
        metric=np.clip(mixed*cal['scale']+cal['shift'],.1,10)
        fields=[rgb,gt,*aligned,metric,np.clip(expert,.1,10),abs(metric-gt)]
        for j,field in enumerate(fields):
            ax=axes[i,j]
            if j==0:ax.imshow(field)
            else:
                shown=ax.imshow(np.where(mask,field,np.nan),cmap='inferno' if j==6 else 'magma_r',
                                vmin=0 if j==6 else .1,vmax=2 if j==6 else 7)
                if j==6:error_image=shown
                else:depth_image=shown
            if i==0:ax.set_title(titles[j],fontsize=9)
            ax.set_xticks([]);ax.set_yticks([])
            if j==0:ax.set_ylabel(row['space'],fontsize=9)
    fig.colorbar(depth_image,ax=axes[:,1:6],shrink=.6,label='Depth (m; color clipped at 7)')
    fig.colorbar(error_image,ax=axes[:,6],shrink=.6,label='Absolute error (m; clipped at 2)')
    fig.suptitle('Fixed examples: first 3 frozen scene IDs | mixed adapter: draw 1, seed 17\n'
                 'All depth panels show measured-GT pixels only; GT-aligned panels use target depth')
    fig.savefig(figdir/'external_fixed_examples.png',dpi=150);plt.close(fig)
    target=next(r for r in comparisons if r['compared']=='mixed_r512' and r['reference']=='base_r512')
    lookup={r['group']:r for r in scores}
    low=next(r for r in comparisons if r['compared']=='mixed_r256' and r['reference']=='base_r256')
    low_high=next(r for r in comparisons if r['compared']=='mixed_r256' and r['reference']=='high512_r256')
    reduction=100*(1-lookup['mixed_r512']['abs_rel']/lookup['base_r512']['abs_rel'])
    native=read(out/'evaluations/expert_default/per_image.json')
    native_summary={m:float(np.mean([r[m] for r in native])) for m in ['native_abs_rel','native_rmse_m','native_delta1']}
    write(out/'native_expert_summary.json',native_summary)
    lines=['# Prospective external validation after sample-size planning','',
           f"All {audit['metrics_recomputed']:,} predictions across 63 conditions and 200 distinct SUN3D-source scene groups passed the audit. "
           'The cohort, all 30 existing adapters, preprocessing, six comparisons, and stopping rule were frozen before external inference. '
           'No additional training, model selection, external calibration, or significance-based stopping occurred.','',
           '## Primary finding','',
           f"At 512 inference, original Marigold has aligned AbsRel {lookup['base_r512']['abs_rel']:.5f}, "
           f"and mixed-resolution adaptation has {lookup['mixed_r512']['abs_rel']:.5f} ({reduction:+.1f}% relative error reduction). "
           f"The predefined nested/crossed Holm-adjusted p-values are {target['hierarchical']['p_holm']:.5f} / {target['crossed']['p_holm']:.5f}. "
           f"Directional improvement satisfies both predefined tests: **{target['primary_improvement_supported']}**. "
           f"The broader-location proxy sensitivity has Holm p={target['location']['p_holm']:.5f}; "
           f"it supports the same directional claim: **{target['location_sensitivity_supports_improvement']}**.", '',
           'A favorable or unfavorable result on this fixed external cohort cannot establish universal model superiority. '
           'Nonsignificance is not equivalence or proof of zero effect. Any disagreement with the location sensitivity limits the robustness claim.','',
           'The earlier fresh31 NYUv2 mixed512-versus-original512 result remains unconfirmed '
           '(nested/crossed Holm p=0.1088/0.1094). This external experiment answers a separate generalization question; '
           'it neither changes that historical test nor pools its observations into a larger NYUv2 test.','',
           '## What the experiment supports','',
           f"At 256 inference, mixed training reduces aligned AbsRel from {lookup['base_r256']['abs_rel']:.5f} "
           f"to {lookup['mixed_r256']['abs_rel']:.5f} "
           f"({100*(1-lookup['mixed_r256']['abs_rel']/lookup['base_r256']['abs_rel']):.1f}% lower). "
           f"Both predefined Holm tests support this contrast (p={low['hierarchical']['p_holm']:.5f} / "
           f"{low['crossed']['p_holm']:.5f}), as does the location sensitivity (p={low['location']['p_holm']:.5f}). "
           f"Mixed training also improves on high-resolution-only training at 256 inference "
           f"(nested/crossed/location Holm p={low_high['hierarchical']['p_holm']:.5f} / "
           f"{low_high['crossed']['p_holm']:.5f} / {low_high['location']['p_holm']:.5f}). "
           'These comparisons were members of the original six-comparison family, not added after seeing the results.','',
           'The defensible interpretation is a benefit at low inference resolution on this external cohort. '
           'No tested 512-inference contrast passes the predefined improvement rule. '
           'This pattern is consistent with resolution-dependent transfer, but no formal interaction test was specified; '
           'a significant low-resolution result and a nonsignificant high-resolution result alone do not establish an interaction.','',
           f"Fixed-calibration mixed512 AbsRel is {lookup['mixed_r512']['calibrated_abs_rel']:.5f}, "
           f"compared with {lookup['expert_default']['calibrated_abs_rel']:.5f} for the specialist. "
           'Thus good GT-aligned structure does not establish accurate absolute distances. '
           'The specialist comparison is descriptive and is outside the six formal contrasts.','',
           '## Planning versus realized evidence','',
           'The [power study](power/RESULTS.md) used only earlier NYUv2 results. For a hypothesized 0.009 gain and fresh31-like variation, '
           'the known-pilot-distribution approximation estimated 84.3% power at 200 groups with 3 draws x 5 seeds. '
           'Inflating error SD by 1.5 lowered this estimate to 38.5%. Thus the budget was sufficient under a specific planning assumption, '
           'not a guarantee across domain shifts. These estimates are not observed post-hoc power and do not determine the final significance claims.', '',
           f"The realized mixed512 mean gain is {-target['hierarchical']['difference']:.5f}, "
           'well below the 0.009 planning assumption. This explains why the planning estimate cannot be cited as evidence '
           'that this external experiment should necessarily reject the null. The observed gain is itself uncertain; '
           'it is not a new sample-size target or a reason to extend this completed test until significant.','',
           '## Data and scope','',
           'Official SUNRGBD contains NYUv2, so only its SUN3D-source branch was eligible. The metadata census had 3,090 frames in 207 '
           'conservative source-space groups. SHA256-ranked groups/frames, raw-depth validity checks and duplicate checks were fixed before '
           'prediction. One frame per group prevents counting repeated video frames as independent scenes. '
           'This is a custom external cohort, not the full SUNRGBD benchmark or additional unseen NYUv2 data.',
           f"The 200 selected groups map to {len(manifest['broader_location_counts'])} broader location proxies. "
           'These are not verified building IDs; source groups can share physical context. Unknown image-pretraining overlap cannot be excluded. '
           'Per-group identities, source members, checksums, exclusions and availability are published in the manifests.', '',
           'Evaluation uses raw measured depth pixels with 0.1 < GT < 10 m across the full image. The NYUv2 crop is not transferred. '
           'Per-image GT affine alignment evaluates relative structure. The separate metric-depth diagnostic reuses the prior NYUv2-validation '
           'calibration without fitting any external target. Changing dataset, sensor-validity mask and crop precludes directly pooling old and new benchmark scores.','',
           '## Scores','',
           '| Method | Aligned AbsRel | Aligned RMSE | Aligned delta1 | Calibrated AbsRel | Calibrated RMSE (m) | Calibrated delta1 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in scores:lines.append('| '+r['group']+' | '+' | '.join(f'{r[m]:.5f}' for m in metricnames)+' |')
    lines+=['',f"Specialist native metric output: AbsRel={native_summary['native_abs_rel']:.5f}, RMSE={native_summary['native_rmse_m']:.5f}m, delta1={native_summary['native_delta1']:.5f}. "
            'This Hypersim-fine-tuned model uses a different architecture, prior supervision and larger default input; it is a descriptive reference.','',
            '![External scores](figures/external_scores.png)','','## Complete six-comparison family','',
            '| Comparison | Mean difference | Nested percentile 95% | Nested Holm p | Crossed Holm p | Location-proxy Holm p | Primary directional claim |',
            '|---|---:|---|---:|---:|---:|---|']
    for r in comparisons:
        h=r['hierarchical'];lines.append(f"| {r['compared']} - {r['reference']} | {h['difference']:+.5f} | [{h['ci_low']:+.5f},{h['ci_high']:+.5f}] | {h['p_holm']:.5f} | {r['crossed']['p_holm']:.5f} | {r['location']['p_holm']:.5f} | {r['primary_improvement_supported']} |")
    lines+=['','Percentile intervals and centered-bootstrap tests are different constructions; an interval excluding zero cannot override '
            'the predefined Holm decision. The plot shows symmetric centered-error intervals matched to the test, with all 12 underlying '
            'nested/crossed raw p-values exactly replayed. Bootstrap inference remains approximate with only three training draws.','',
            '![External comparisons](figures/external_comparisons.png)','','## Fixed qualitative examples','',
            'The first three frozen manifest entries and the draw-1/seed-17 mixed adapter were fixed before examining outcomes. '
            'These examples are not selected by prediction error. The metric error panel uses the old NYUv2 calibration; '
            'GT-aligned panels use each image\'s measured target depth and cannot demonstrate absolute-distance accuracy.','',
            '![Fixed external examples](figures/external_fixed_examples.png)','','## Cost and verification','',
            f"- No new training. Summed measured inference time: {sum(r['inference_seconds'] for r in allrecs)/60:.2f} minutes; loading, restoration checks, data acquisition and audit excluded.",
            f"- Peak allocated inference VRAM: {max(r['peak_allocated_mib'] for r in allrecs):.1f} MiB. Laptop timing is descriptive, not a controlled hardware comparison.",
            '- All 200 data hashes, 30 checkpoint hashes, 63 old-validation restoration checks, 12,600 raw predictions and recomputed scores verified.',
            '- All old experiments remain unchanged. Raw data, weights and prediction arrays remain local; code, provenance, metrics and figures are published.',
            '- The fixed 200-group result is final for this protocol, regardless of significance. Future modifications require a new independent validation plan.','',
            'Sources and frozen decisions: [protocol](../../PROSPECTIVE_PROTOCOL.md), [official SUNRGBD](https://rgbd.cs.princeton.edu/), '
            '[official SUN3D](https://sun3d.cs.princeton.edu/), [verification](verification.json).','']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('EXTERNAL_REPORT_READY',target,flush=True)

if __name__=='__main__':main()
