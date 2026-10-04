"""Descriptive synthesis after the frozen V10 studies, with no new tests."""
import sys
sys.path.insert(0,r'E:\Codex\2026-09-27\yo\outputs\marigold-depth')
import html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnostic_v10.common import OUT,read,write,sha

a=read(OUT/'analysis.json');done=read(OUT/'complete.json')
assert sha(OUT/'analysis.json')==done['analysis_sha256']
g=read(OUT/'gradients.json')['records'];regions=read(OUT/'regional.json')['contrasts']
representation=read(OUT/'representation_audit.json')['summary'];quality=read(OUT/'independent_input_audit.json')['label_quality_degrees']
contrasts=[]
for condition in ['clean','heterogeneous_noise','smooth_bias']:
    for rule in ['weighted','shuffled','oracle']:
        for task in ['depth','normal']:
            def differences(c):
                left=a['noise'][c][rule]['seed_means'];right=a['noise'][c]['uniform']['seed_means']
                assert [r['seed'] for r in left]==[r['seed'] for r in right]
                return [l[task]-r[task] for l,r in zip(left,right)]
            delta=differences(condition);clean=differences('clean')
            contrasts.append(dict(condition=condition,rule=rule,task=task,difference=float(np.mean(delta)),
                                  seed_differences=delta,improving_seeds=sum(v<0 for v in delta),
                                  difference_in_differences_vs_clean=float(np.mean(np.array(delta)-clean))))
cosines=[r['cosines']['uniform_geometry__weighted_geometry'] for r in g]
summary=dict(contrasts=contrasts,geometry_direction_similarity_mean=float(np.mean(cosines)),
             geometry_direction_similarity_min=float(min(cosines)),
             scope='Post-hoc descriptive paired differences and difference-in-differences; two seeds only; no p-values or confirmation claim.',
             analysis_sha256=sha(OUT/'analysis.json'),source_sha256=sha(__file__))
write(OUT/'interpretation.json',summary)
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
conditions=['clean','heterogeneous_noise','smooth_bias'];rules=['uniform','weighted','shuffled','oracle']
colors=['#607d8b','#d47b27','#438c91','#8a5ba6'];x=np.arange(3)
for ax,task in zip(axes,['depth','normal']):
    for j,(rule,color) in enumerate(zip(rules,colors)):
        pos=x+(j-1.5)*.2
        ax.bar(pos,[a['noise'][c][rule][task] for c in conditions],width=.18,color=color,alpha=.65,label=rule)
        for index,c in enumerate(conditions):
            vals=[r[task] for r in a['noise'][c][rule]['seed_means']]
            ax.scatter([pos[index]-.025,pos[index]+.025],vals,color='black',s=18,zorder=3)
    ax.set_xticks(x,['Clean','Local noise','Smooth bias']);ax.set_ylabel('Depth AbsRel' if task=='depth' else 'Normal mean error (degrees)')
    ax.legend(fontsize=8)
fig.suptitle('Controlled-noise pilot: bars are means; black dots are the two training seeds\n64 updates, identical clean synthetic evaluation; no significance claim')
fig.savefig(OUT/'figures/noise_seed_detail.png',dpi=150);plt.close(fig)
rows=''.join(f'<tr><td>{r["condition"]}</td><td>{r["rule"]}</td><td>{r["task"]}</td><td>{r["difference"]:+.5f}</td><td>{r["improving_seeds"]}/2</td><td>{r["difference_in_differences_vs_clean"]:+.5f}</td></tr>' for r in contrasts)
prep=''.join(f'<tr><td>{c}</td><td>{r["normal_fullres_deg"]:.2f} → {r["normal_native_deg"]:.2f}</td><td>{100*r["depth_fullres_absrel"]:.3f}% → {100*r["depth_native_absrel"]:.3f}%</td><td>{r["latent_target_mse"]["depth"]:.5f}</td><td>{r["latent_target_mse"]["normal"]:.5f}</td></tr>' for c,r in representation.items())
text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>V10: interpreting the mechanism evidence</title><style>body{{font:16px/1.6 system-ui;max-width:1150px;margin:35px auto;padding:20px;color:#213c4f}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border:1px solid #ccd8df}}img{{width:100%}}.box{{background:#edf4f7;padding:18px}}</style>
<h1>What the three diagnostics establish—and what they do not</h1><p class="box">All three requested studies are completed. The frozen quantitative record is in the <a href="RESULTS.html">main report</a>. This page adds descriptive synthesis and preprocessing checks; it does not add hypothesis tests or select new training settings.</p>
<h2>Real-scene errors are heterogeneous</h2><p>Relative to uniform geometry, the V9 weighted model's depth gain is concentrated in the under2m bin: mean AbsRel difference -0.01322 over29 eligible scenes; the2–4m bin is near zero (+0.00025 over32), and the4m-plus bin is worse on average (+0.00382 over18). Normal error improves in the stable label-sensitivity quartile (-0.21724 degrees) but worsens in the sensitive quartile (+0.10437 degrees). These are retrospective region associations on observed development data, not causal explanations or new significant findings.</p>
<img src="figures/mechanism_detail.png" alt="Scene effects and gradient alignment">
<h2>The geometry loss is active, but weighting changes its aggregate direction modestly</h2><p>Across the six trained states, mean coefficient-scaled geometry-gradient norms are roughly39–59% of the supervised-gradient norm. This rules out a negligible geometry gradient on the sampled probes, not in all training steps. The mean uniform/weighted gradient cosine across all56 probes is {summary['geometry_direction_similarity_mean']:.4f}. Similar directions offer a plausible explanation for limited model differences, but do not establish causation.</p><p>Initial supervised/geometry gradients are negatively aligned on average; the trained-state means are positive. This suggests testing delayed geometry regularization as a future hypothesis. It does not prove that a warm-up schedule will help. The56 probes use only four fixed training images and two noise draws per state. The one BF16 precision flag and its FP32 reference are retained in the main report.</p>
<h2>Training sees preprocessed labels, not the raw error map</h2><table><tr><th>Condition</th><th>Normal error: full → native degrees</th><th>Depth error: full → native AbsRel</th><th>Depth latent MSE vs clean</th><th>Normal latent MSE vs clean</th></tr>{prep}</table><p>Native resolution is192×256. Latent MSE is a representation-space distance to the clean encoded target, not an angular error or metric depth error. Random depth noise is attenuated by resize; normal-label noise remains substantial. Smooth bias largely survives resizing. These measurements help interpret the intervention; they do not estimate real NYUv2 label noise.</p>
<h2>Actual model contrasts under controlled noise</h2><img src="figures/noise_seed_detail.png" alt="Two individual seed results per synthetic condition and method"><p>Each difference below is rule minus uniform on the same clean evaluation images, averaged across two seed-matched runs. Negative favors the rule. The last column subtracts the corresponding clean-training difference. This post-hoc difference-in-differences description separates a noise-specific change from a clean-condition difference; it has no significance test and does not establish an interaction.</p><table><tr><th>Training labels</th><th>Rule</th><th>Task</th><th>Mean difference</th><th>Seeds improving</th><th>Difference vs clean effect</th></tr>{rows}</table>
<h2>How to decide the next experiment</h2><p>If the noisy-label oracle helps consistently while the proxy does not, improve the reliability estimate. If neither helps, first inspect optimization and the fact that only geometric consistency is weighted: corrupted latent supervision is still unweighted. An oracle null result does not prove weighting is useless.</p><p>If depth improves while normals worsen, one testable explanation is that the same scalar weight protects depth from unreliable normals but also weakens useful depth-to-normal guidance. The current experiment does not isolate these directions. A new comparison could stop the gradient through one predicted branch at a time, alongside the unchanged bidirectional control. Shared LoRA parameters still couple future outputs even when one branch is detached, so both tasks must be evaluated. This is a proposed experiment, not a demonstrated fix or a novelty claim.</p><p>Direct normal-supervision weighting and delayed geometry are other untested alternatives. Do not extend the64-step pilot until a preferred p-value appears. Any larger study needs a new recorded protocol with a justified budget, more training seeds/draws and genuinely independent scene groups. This study has simplified synthetic RGB, two training seeds and shared generator families; it cannot establish real-world transfer or convergence.</p><p><a href="interpretation.json">Paired descriptive effects</a> | <a href="representation_audit.json">Preprocessing measurements</a> | <a href="independent_input_audit.json">Input audit</a> | <a href="RESULTS.html">Full quantitative report</a></p></html>'''
(OUT/'OBSERVATIONS.html').write_text(text,encoding='utf-8')
write(OUT/'observations_checks.json',dict(report_sha256=sha(OUT/'OBSERVATIONS.html'),interpretation_sha256=sha(OUT/'interpretation.json'),script_sha256=sha(__file__)))
print(OUT/'OBSERVATIONS.html')
