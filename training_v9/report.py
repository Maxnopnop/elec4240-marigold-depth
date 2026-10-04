"""Readable report generated only from completed, audited runs."""
import argparse,html
import numpy as np
from multitask_v7.common import read,write,sha
from .protocol import OUT,SEEDS


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['b','c'],required=True);args=parser.parse_args()
    a=read(OUT/f'analysis_{args.stage}.json');gate=read(OUT/'baseline_gate.json')
    rows=[]
    for mode,tasks in a['summary'].items():
        d=tasks.get('depth',{}).get('abs_rel');n=tasks.get('normal',{}).get('mean_deg')
        def display(v):return f"{v['mean']:.5f} ± {v['seed_sd']:.5f}" if v else 'Not applicable'
        rows.append(f"<tr><td>{mode}</td><td>{display(d)}</td><td>{display(n)}</td><td>{a['cost'][mode]['optimizer_seconds_total']/60:.2f}</td><td>{a['cost'][mode]['peak_allocated_mib']:.1f}</td></tr>")
    comparisons=[]
    for r in a['comparisons']:
        low,high=r['unadjusted_symmetric_95_interval']
        interpretation=('weighted better' if r['difference']<0 else 'weighted worse') if r['passes_holm_0_05'] else 'difference not established'
        comparisons.append(f"<tr><td>weighted vs {r['reference']}</td><td>{r['task']}</td><td>{r['difference']:+.6f}</td><td>[{low:+.6f}, {high:+.6f}]</td><td>{r['holm6_p']:.6f}</td><td>{interpretation}</td></tr>")
    curves=[]
    for task,steps in gate['learning_curves'].items():
        for step,v in steps.items():curves.append(f"<tr><td>{task}</td><td>{step}</td><td>{v['mean']:.5f}</td><td>{', '.join(f'{x:.5f}' for x in v['seed_means'])}</td></tr>")
    diag=read(OUT/f'diagnostics_{args.stage}.json');dr=[]
    for mode in a['summary']:
        collection=[r for seed in SEEDS for r in diag[f'{mode}_seed{seed}']]
        for task in a['summary'][mode]:
            high=np.mean([r[task]['high_gradient'] for r in collection]);low=np.mean([r[task]['other_valid'] for r in collection])
            dr.append(f'<tr><td>{mode}</td><td>{task}</td><td>{high:.5f}</td><td>{low:.5f}</td></tr>')
    agreement=[]
    for mode in a['summary']:
        vals=[r['geometry_disagreement_deg'] for seed in SEEDS for r in diag[f'{mode}_seed{seed}'] if 'geometry_disagreement_deg' in r]
        if vals:agreement.append(f'<tr><td>{mode}</td><td>{np.mean(vals):.4f}</td></tr>')
    stage_label='Single-task learning curves' if args.stage=='b' else 'Matched reliability-weighted model ablation'
    introduction=('Six single-task runs are complete. The development feasibility gate '+('passed' if gate['passed'] else 'failed')+'. This screen compares final and step-320 means; it does not establish convergence.' if args.stage=='b' else 'All seven conditions and three seeds are complete. Read the task-wise errors and corrected comparisons together; no single consistency score establishes better geometry.')
    v=a['verification']
    report=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>V9: {stage_label}</title><style>
body{{font:16px/1.6 system-ui,sans-serif;background:#f3f7fa;color:#233747;margin:0}}main{{max-width:1120px;margin:25px auto;background:white;padding:32px 42px;border-radius:12px}}h1,h2{{color:#16324f;line-height:1.2}}h2{{margin-top:32px}}a{{color:#087f8c}}.box{{background:#edf5f7;padding:18px;border-left:4px solid #087f8c}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #d5e2e7}}th{{background:#16324f;color:white}}img{{width:100%;height:auto}}small{{color:#566975}}@media(max-width:700px){{main{{padding:15px;margin:0}}table{{font-size:11px}}}}
</style><main><small>ELEC4240 | V9 | Observed development scenes; exploratory evidence</small>
<h1>{stage_label}</h1><div class="box">{introduction}</div>
<p>Marigold Depth v1.1, rank-4 shared attention LoRA, 829,952 trainable parameters, 192×256 processing, fixed metric log-depth codec, 128 training scenes, 32 previously observed development scenes, seeds 17/29/43, 1,280 updates per run. Normal targets are derived from depth and are not independent surface-normal ground truth. No per-image ground-truth depth alignment is used.</p>
<h2>Final checkpoint results</h2><p>Accuracy values are the mean ± sample standard deviation of the three seed means. Lower AbsRel and normal angular error are better. Times sum three optimizer loops; loading, checkpoint I/O and validation are excluded. Joint models receive one image exposure per task per update.</p>
<table><tr><th>Condition</th><th>Depth AbsRel</th><th>Normal mean degrees</th><th>Optimizer minutes</th><th>Peak allocated MiB</th></tr>{''.join(rows)}</table>
{('<img src="figures/final_metrics.png" alt="Final depth and normal errors across matched conditions">' if args.stage=='c' else '')}
<h2>Single-task learning curves</h2><table><tr><th>Task</th><th>Update</th><th>Mean</th><th>Seeds 17 / 29 / 43</th></tr>{''.join(curves)}</table>
<img src="figures/learning_curves.png" alt="Fixed checkpoint learning curves for all baseline seeds">
<p>Baseline gate: final mean error no more than 5% worse than step-320 mean for each task, with finite runs and exact final-checkpoint inference restoration. Passing is not a stationarity or convergence test. All comparisons use the fixed final checkpoint.</p>
<h2>Six predeclared exploratory contrasts</h2>
{('<table><tr><th>Comparison</th><th>Task</th><th>Difference</th><th>Unadjusted 95% interval</th><th>Holm6 p</th><th>Interpretation</th></tr>'+''.join(comparisons)+'</table>') if comparisons else '<p>Joint-model comparisons have not been run in this stage.</p>'}
<p>Differences are weighted minus reference, so negative favors weighting. Intervals are unadjusted symmetric bootstrap intervals. Centered two-sided tests use 20,000 paired crossed seed/scene resamples and one Holm family of six contrasts. There are only three seeds and one training draw. A non-significant normal difference does not prove no harm; no non-inferiority margin or claim is introduced after results.</p>
<h2>Descriptive failure analysis</h2><p>High-gradient regions are the top 10% of ground-truth depth-gradient magnitude within each task's fixed valid mask. The remaining valid pixels form the comparison group. These regions are identical across methods and add no hypothesis tests. Depth values below are AbsRel; normal values are angular error in degrees.</p>
<table><tr><th>Condition</th><th>Task</th><th>High gradient</th><th>Other valid</th></tr>{''.join(dr)}</table>
{('<h2>Prediction-to-prediction agreement</h2><table><tr><th>Condition</th><th>Mean disagreement degrees</th></tr>'+''.join(agreement)+'</table><p>Computed at saved full resolution, unlike the native-resolution training loss. Better agreement is not proof of better accuracy.</p>') if agreement else ''}
<h2>Verification and reproducibility</h2><ul><li>{v['independently_recomputed_predictions']} saved task predictions independently recomputed and hash-checked.</li><li>{v['checkpoints']} saved adapter checkpoints verified; {v['exact_restored_tasks']} final checkpoint-task predictions reproduced exactly in freshly loaded models.</li><li>{v['history_steps']} training updates checked, with identical image order across corresponding seeds and matched task exposure.</li><li>All 160 source/derived data pairs checked; {v['historical_files_unchanged']} historical report/result files unchanged.</li></ul>
<p>The serialization/RNG resume test passes on its CUDA test model. Diffusion-model training itself is not guaranteed bitwise deterministic: the first new single-task run matched V7 input order and first forward loss but differed slightly in the first backward gradient. The exact cause has not been isolated. Old V7 numbers are historical context; current single-task controls were trained afresh under the same runner. This is distinct from exact saved-checkpoint inference restoration.</p>
<h2>Scope and next evidence</h2><ul>{''.join('<li>'+html.escape(x)+'</li>' for x in a['limitations'])}</ul>
<p>The V8 synthetic component gain describes selection of supervision, not a model gain. A useful model result requires actual task error improvements under matched controls. Any independent confirmation must freeze the selected method, comparisons and sample-size plan before obtaining unobserved scene groups.</p>
<p><a href="../../training_v9/README.md">Execution details</a> | <a href="../../reliability_v8/EXPERIMENT_PLAN.md">Experiment design</a> | <a href="analysis_{args.stage}.json">Audited analysis</a> | <a href="protocol_{args.stage}.json">Frozen protocol</a> | <a href="../reliability_v8/RESULTS.html">Earlier component study</a></p></main></html>'''
    destination=OUT/('RESULTS_b.html' if args.stage=='b' else 'RESULTS.html')
    destination.write_text(report,encoding='utf-8')
    write(OUT/f'report_checks_{args.stage}.json',{'report_sha256':sha(destination),'analysis_sha256':sha(OUT/f'analysis_{args.stage}.json'),'stage':args.stage})
    print(destination,flush=True)


if __name__=='__main__':main()
