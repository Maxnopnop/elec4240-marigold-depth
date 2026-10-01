"""Publish complete exploratory extension, cost and diagnostic figures."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_expanded import read,write
from run_final_control import OUT

def main():
    assert read(OUT/'verification.json')['status']=='passed'
    assert read(OUT/'cost/verification.json')['status']=='passed'
    costs=[]
    for label in [f'{m}_r{r}' for m in ['base','high512','mixed','low256'] for r in [256,512]]:
        rounds=[read(OUT/'cost/inference'/f'round{i}_{label}.json') for i in range(1,6)]
        seconds=np.array([[r['seconds'] for r in row['records']] for row in rounds])
        costs.append({'condition':label,'median_ms':float(np.median(seconds)*1000),'p10_ms':float(np.quantile(seconds,.1)*1000),'p90_ms':float(np.quantile(seconds,.9)*1000),'round_median_ms':(np.median(seconds,axis=1)*1000).tolist(),'peak_mib':max(r['peak_mib'] for row in rounds for r in row['records'])})
    budget=[]
    for mode in ['low256','high512','mixed']:
        trs=[read(OUT/'cost/budget'/f'{mode}_seed{s}'/'training.json') for s in [17,29,43]]
        for res in [256,512]:
            vals=[np.mean([r['abs_rel'] for r in read(OUT/'cost/budget'/f'{mode}_seed{s}'/'evaluation.json') if r['resolution']==res]) for s in [17,29,43]]
            budget.append({'mode':mode,'resolution':res,'abs_rel_mean':float(np.mean(vals)),'abs_rel_sd':float(np.std(vals,ddof=1)),'seed_means':[float(x) for x in vals],'updates':[t['steps'] for t in trs],'seconds':[t['training_seconds'] for t in trs],'warmup_seconds':[t['warmup_seconds'] for t in trs]})
    write(OUT/'cost/summary.json',{'inference':costs,'budget':budget})
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    ext={r['group']:r for r in read(OUT/'external_summary.json')};fig,axes=plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
    for ax,res in zip(axes,[256,512]):
        names=['base','low256','high512','mixed'];vals=[ext[f'{m}_r{res}']['abs_rel'] for m in names]
        ax.bar(names,vals,color=['#64748b','#059669','#2563eb','#d97706']);ax.set(ylabel='Aligned AbsRel',title=f'External200 | inference {res}',ylim=(0,max(vals)*1.22))
        for i,v in enumerate(vals):ax.text(i,v+max(vals)*.02,f'{v:.4f}',ha='center',fontsize=9)
    fig.savefig(figdir/'low_control.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
    for mode in ['base','low256','high512','mixed']:
        rows=[r for r in costs if r['condition'].startswith(mode+'_')];rows.sort(key=lambda r:r['condition'])
        axes[0].plot([256,512],[r['median_ms'] for r in rows],'-o',label=mode)
    axes[0].set(xticks=[256,512],ylabel='Median end-to-end inference (ms)',xlabel='Processing long side');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    for res,style in [(256,'-o'),(512,'--s')]:
        rows=[r for r in budget if r['resolution']==res]
        axes[1].errorbar([r['mode'] for r in rows],[r['abs_rel_mean'] for r in rows],yerr=[r['abs_rel_sd'] for r in rows],fmt=style,capsize=3,label=f'inference {res}')
    axes[1].set(ylabel='Validation AbsRel (mean +/- seed SD)',title='120-second training budget | draw1');axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.savefig(figdir/'controlled_cost.png',dpi=180);plt.close(fig)
    lines=['# Final-report exploratory completion study','',
           '**All test cohorts were already observed. This extension is exploratory and does not change the original frozen six-comparison conclusions.**','',
           '## Matched low-resolution control','',
           'Fifteen new low256 adapters use exactly the v3 three training draws, five seeds and 320-update recipe. All 7,890 predictions on validation32, NYUv2-31 and external200 were recomputed from saved raw arrays; all 15 checkpoints were restored at both inference resolutions.','',
           '| Group | External aligned AbsRel | NYUv2-31 aligned AbsRel |','|---|---:|---:|']
    nyu={r['group']:r for r in read(OUT/'nyu31_summary.json')}
    for g,row in ext.items():lines.append(f"| {g} | {row['abs_rel']:.5f} | {nyu[g]['abs_rel']:.5f} |")
    lines+=['','![Low-resolution control](figures/low_control.png)','','## Separate eight-comparison exploratory family','',
            'The following Holm correction includes the six old contrast definitions plus mixed-versus-low256 at both resolutions. It is separate from the unchanged published six-test family.','',
            '| Contrast | Difference | Nested Holm8 | Crossed Holm8 | Location Holm8 |','|---|---:|---:|---:|---:|']
    for row in read(OUT/'external_comparisons.json'):
        lines.append(f"| {row['compared']} - {row['reference']} | {row['hierarchical']['difference']:+.5f} | {row['hierarchical']['p_holm8']:.5f} | {row['crossed']['p_holm8']:.5f} | {row['location']['p_holm8']:.5f} |")
    it=read(OUT/'external_interaction.json')['crossed']
    lines+=['',f"Separate post-hoc interaction diagnostic, (mixed256-base256)-(mixed512-base512): {it['difference']:+.5f}; crossed percentile95% [{it['ci_low']:+.5f}, {it['ci_high']:+.5f}], unadjusted p={it['p_bootstrap']:.5f}. This is an exploratory single contrast, not prospective confirmation or a causal mechanism test.",'','## Depth-range and boundary diagnostics','',
            'Each bin is averaged equally over contributing scenes and runs (at least100 measured pixels per scene). Boundaries come from GT neighbor jumps >0.1m and >5%, dilated two pixels. Different bins need not have the same scene counts. No additional significance claims are made.','',
            '![Stratified failure analysis](figures/failure_strata.png)','','![Fixed first three examples](figures/fixed_examples.png)','','![Explicitly error-selected failures](figures/selected_failures.png)','','## Controlled cost measurement','',
            'Five randomized condition rounds, first eight validation images, three warmups per condition. All320 measured predictions exactly reproduce saved references. Times cover preprocessing through resized CPU output, with CUDA synchronization; loading excluded. Telemetry is retained, but one laptop and one checkpoint per method do not establish general hardware efficiency.','',
            '| Condition | Median ms | p10 ms | p90 ms | Peak MiB |','|---|---:|---:|---:|---:|']
    for r in costs:lines.append(f"| {r['condition']} | {r['median_ms']:.2f} | {r['p10_ms']:.2f} | {r['p90_ms']:.2f} | {r['peak_mib']:.1f} |")
    lines+=['','The fixed-time diagnostic trains low/high/mixed for120seconds per seed on draw1, with three seeds. Warmups do not update weights; cache/loading/warmup excluded. Stop at the first completed update reaching the budget. Only validation32 is evaluated; this is not an unseen generalization test.','',
            '| Method | Inference | Validation AbsRel mean | Seed SD | Completed updates |','|---|---:|---:|---:|---|']
    for r in budget:lines.append(f"| {r['mode']} | {r['resolution']} | {r['abs_rel_mean']:.5f} | {r['abs_rel_sd']:.5f} | {r['updates']} |")
    lines+=['','![Controlled cost](figures/controlled_cost.png)','','All prior data, weights and prediction arrays remain local. This report supplements the unchanged [prospective results](../prospective_v4/RESULTS.md).','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8');print('FINAL_EXTENSION_REPORT_READY',flush=True)

if __name__=='__main__':main()
