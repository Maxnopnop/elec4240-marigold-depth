"""Bind the report prose and tables to the audited final result files."""
from pathlib import Path
import json

ROOT=Path(r'E:\Codex\2026-09-27\yo')
HERE=ROOT/'work/final-report-build'
P=ROOT/'outputs/marigold-depth/results/final_extension_v5'
read=lambda name:json.loads((P/name).read_text(encoding='utf-8'))

assert read('delivery_checks.json')['status']=='passed'
ext={r['group']:r for r in read('external_summary.json')}
nyu={r['group']:r for r in read('nyu31_summary.json')}
comparisons=read('external_comparisons.json')
budget={(r['mode'],r['resolution']):r for r in read('cost/summary.json')['budget']}
times={r['condition']:r for r in read('cost/summary.json')['inference']}
interaction=read('external_interaction.json')['crossed']
strata=read('failure/mixed_r512.json')['summary']

r={
'ABSTRACT_NEW':'The matched control outperforms mixed training at 256 inference on the observed external cohort, challenging a unique benefit from resolution mixing.',
'INTRO_NEW':'The added low-only control performs better at 256 on the observed external cohort, showing why the fixed-resolution alternatives matter.',
'LOW256':f"{ext['low256_r256']['abs_rel']:.5f}",
'LOW512':f"{ext['low256_r512']['abs_rel']:.5f}",
'LOW_DISCUSSION':(
 'On external200, low-only reaches AbsRel 0.09277 at 256 versus mixed 0.09632. '
 'The mixed-minus-low difference is +0.00355; nested/crossed Holm8 p-values are '
 '0.00150/0.01560, with location p=0.03660. Thus the observed low-resolution gain '
 'does not require mixing: the matched low-only control performs better on this cohort. '
 'At 512, mixed 0.07038 versus low-only 0.07140 is unconfirmed (crossed p=0.41038; '
 'location p=0.39578); this is not evidence of equivalence. On NYUv2-31, low-only '
 'scores 0.09905/0.08918 at 256/512, versus mixed 0.10128/0.08462. Neither '
 'mixed--low contrast passes the exploratory correction there (crossed p=0.59617 '
 'at both resolutions). These controls qualify the motivation for mixing without '
 'changing the earlier prospective results.'),
'INTERACTION':(
 f"On external200, the interaction estimate is {interaction['difference']:+.5f}, "
 f"with crossed percentile 95\\% interval [{interaction['ci_low']:+.5f}, {interaction['ci_high']:+.5f}] "
 f"and unadjusted $p={interaction['p_bootstrap']:.5f}$. "
 'The NYUv2-31 interaction remains unconfirmed (unadjusted p=0.17039).'),
'FAILURE_DISCUSSION':(
 f"Mixed512 boundary AbsRel is {strata['boundary']['aligned']:.5f} versus "
 f"{strata['interior']['aligned']:.5f} in the interior, across 200 contributing scenes. "
 f"In the near-depth bin, aligned error is {strata['0.1-1m']['aligned']:.5f} and "
 f"calibrated error {strata['0.1-1m']['calibrated']:.5f}, with only "
 f"{strata['0.1-1m']['scenes']} contributing scenes. The other depth-bin counts are "
 '190, 190 and 97. The mean aligned near-range error is slightly worse than the '
 'original model\'s 0.32195, despite improvements in some farther ranges. '
 'Aggregate accuracy therefore hides important local regressions.'),
'TIMING_DISCUSSION':(
 f"The separate benchmark measures original/mixed median latency of "
 f"{times['base_r256']['median_ms']:.2f}/{times['mixed_r256']['median_ms']:.2f}\\,ms at 256 "
 f"and {times['base_r512']['median_ms']:.2f}/{times['mixed_r512']['median_ms']:.2f}\\,ms at 512. "
 'All three adapter strategies are close: 73.41--73.59\\,ms at 256 and '
 '193.67--193.84\\,ms at 512. Changing inference resolution has a much larger '
 'latency effect than choosing among these unmerged adapters.'),
'BUDGET_DISCUSSION':(
 'Within the fixed-time diagnostic, low-only has the lowest mean error at 256 '
 '(0.08761 versus mixed 0.09041 and high-only 0.10431), while high-only has the '
 'lowest mean at 512 (0.06801 versus mixed 0.06963 and low-only 0.07132). '
 'Completed-update ranges overlap, so a fixed proportional speed advantage '
 'should not be assumed. These three-seed validation means again favor testing '
 'the deployment-matched schedule; they do not demonstrate general dominance.'),
'CONCLUSION_NEW':(
 'The matched low-only control is stronger at 256 on the observed external '
 'cohort, and the fixed-time validation diagnostic favors different fixed '
 'schedules at the two inference resolutions. Mixing is therefore not a '
 'uniformly superior adaptation rule.')}

labels={'mixed':'M','base':'O','high512':'H','low256':'L'}
rows=[]
for c in comparisons:
    cm,res=c['compared'].rsplit('_r',1);rm,_=c['reference'].rsplit('_r',1)
    rows.append(f"{labels[cm]}--{labels[rm]} ({res}) & "+' & '.join(f"{c[k]['p_holm8']:.5f}" for k in ['hierarchical','crossed','location'])+r'\\')
r['HOLM_ROWS']='\n'.join(rows)
rows=[]
for mode,name in [('low256','Low-only'),('high512','High-only'),('mixed','Mixed')]:
    a,b=budget[mode,256],budget[mode,512];u=a['updates']
    vals=[f"{v['abs_rel_mean']:.4f}".lstrip('0')+' ('+f"{v['abs_rel_sd']:.4f}".lstrip('0')+')' for v in [a,b]]
    rows.append(f"{name} & {min(u)}--{max(u)} & "+' & '.join(vals)+r'\\')
r['BUDGET_ROWS']='\n'.join(rows)
(HERE/'replacements.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
print('Report replacements generated:',len(r))
