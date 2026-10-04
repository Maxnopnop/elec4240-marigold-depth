import html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multitask_v7.common import sample
from training_v9.analyze import depth_metrics,normal_metrics
from .common import *


def main():
    p=verify();regional=read(OUT/'regional.json');grad=read(OUT/'gradients.json');data=read(OUT/'noise_data.json')
    assert regional['verified_predictions']==1152 and len(grad['records'])==56 and grad['optimizer_updates']==0
    for name,digest in data['files'].items():assert sha(WORK/name)==digest,name
    summary={};checked=0;restored=0;updates=0;cost=0.
    for condition in CONDITIONS:
        summary[condition]={}
        for rule in RULES:
            values=[]
            for seed in SEEDS:
                name=f'{condition}_{rule}_seed{seed}';dest=OUT/'noise_runs'/name;local=WORK/'noise_runs'/name
                done=read(dest/'complete.json');stats=read(dest/'training.json');records=read(dest/'evaluation.json')
                assert done['protocol_sha256']==sha(OUT/'protocol.json')
                assert done['training_sha256']==sha(dest/'training.json') and done['evaluation_sha256']==sha(dest/'evaluation.json')
                assert done['checkpoint_sha256']==sha(local/'adapter.pt')
                assert done['exact_restored_tasks']==['depth','normal'];restored+=2
                assert stats['initialization_sha256']==read(OUT/f'initial_seed{seed}.json')['initialization_sha256']
                assert len(stats['history'])==64;updates+=64;cost+=stats['optimizer_seconds']
                rng=np.random.default_rng(seed)
                ids=[f'{kind}_{s}' for s in p['noise']['train_scene_seeds'] for kind in SURFACES]
                order=[ids[i] for _ in range(4) for i in rng.permutation(16)[::-1]]
                assert [r['id'] for r in stats['history']]==order
                for row in stats['history']:
                    assert abs(row['loss']-np.mean(list(row['task_losses'].values()))-.1*row['geometry'])<2e-5
                assert [r['id'] for r in records]==[r['id'] for r in data['evaluation']]
                for row,source in zip(records,data['evaluation']):
                    case=sample(WORK/source['file'])
                    for task,metrics in row['tasks'].items():
                        path=local/'predictions'/f"{row['id']}_{task}.npy";assert sha(path)==metrics['prediction_sha256']
                        pred=np.load(path)
                        actual=depth_metrics(pred[0],case['depth'],case['mask']) if task=='depth' else normal_metrics(pred,case['normal'],case['mask'])
                        for key,value in actual.items():assert np.isclose(value,metrics[key],rtol=1e-10,atol=1e-9),(name,row['id'],key)
                        checked+=1
                values.append(dict(seed=seed,depth=float(np.mean([r['tasks']['depth']['abs_rel'] for r in records])),
                                   normal=float(np.mean([r['tasks']['normal']['mean_deg'] for r in records]))))
            summary[condition][rule]=dict(seed_means=values,depth=float(np.mean([v['depth'] for v in values])),normal=float(np.mean([v['normal'] for v in values])))
    # Independently audit the two initial-model evaluation sets too.
    initial_means=[]
    for seed in SEEDS:
        rows=read(OUT/f'initial_seed{seed}.json')['records']
        initial_means.append(dict(seed=seed,depth=float(np.mean([r['tasks']['depth']['abs_rel'] for r in rows])),
                                 normal=float(np.mean([r['tasks']['normal']['mean_deg'] for r in rows]))))
        for row,source in zip(rows,data['evaluation']):
            case=sample(WORK/source['file'])
            for task,metrics in row['tasks'].items():
                path=WORK/f'initial_seed{seed}'/f"{row['id']}_{task}.npy";assert sha(path)==metrics['prediction_sha256']
                pred=np.load(path)
                actual=depth_metrics(pred[0],case['depth'],case['mask']) if task=='depth' else normal_metrics(pred,case['normal'],case['mask'])
                for key,value in actual.items():assert np.isclose(value,metrics[key],rtol=1e-10,atol=1e-9)
                checked+=1
    historical=read(WORK/'historical_hashes.json')
    assert all(sha(ROOT/name)==digest for name,digest in historical.items())
    assert checked==416 and restored==48 and updates==1536
    gs=[]
    for state in p['gradient']['states']:
        records=[r for r in grad['records'] if r['state']==state]
        row=dict(state=state,probes=len(records),depth_normal_cosine=float(np.mean([r['cosines']['depth__normal'] for r in records])),
                 conflicting_task_probes=sum(r['cosines']['depth__normal']<0 for r in records),
                 max_gradient_sum_discrepancy=max(r['sum_relative_error'] for r in records))
        for mode in ['uniform','weighted']:
            row[mode+'_ratio']=float(np.mean([r['geometry_to_supervised_ratio'][mode] for r in records]))
            row[mode+'_supervised_cosine']=float(np.mean([r['cosines']['supervised__'+mode+'_geometry'] for r in records]))
        gs.append(row)
    result=dict(noise=summary,initial_seed_means=initial_means,gradients=gs,verification=dict(new_noise_predictions=checked,restored_tasks=restored,optimizer_updates=updates,
                old_prediction_hashes=1152,gradient_probes=56,historical_files_unchanged=len(historical)),optimizer_seconds=cost,
                scope='Exploratory diagnosis, not an independent confirmation. No additional hypothesis tests. Synthetic labels have analytic truth; real normal labels remain depth-derived.')
    write(OUT/'analysis.json',result)
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    regions=['all','stable_quartile','sensitive_quartile','high_gradient','other_gradient','near_under2m','middle2to4m','far_over4m']
    for ax,task in zip(axes,['depth','normal']):
        rows=[next(x for x in regional['contrasts'] if x['task']==task and x['reference']=='uniform' and x['region']==region) for region in regions]
        ax.barh(regions,[r['difference'] for r in rows]);ax.axvline(0,color='black',linewidth=1);ax.set_title(task)
        ax.set_xlabel('Weighted minus uniform; lower favors weighted')
    fig.suptitle('Post-hoc regions on observed real development scenes; descriptive means')
    fig.savefig(figures/'regional.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained');x=np.arange(len(gs))
    axes[0].bar(x,[r['depth_normal_cosine'] for r in gs]);axes[0].axhline(0,color='black');axes[0].set_ylabel('Depth / normal gradient cosine')
    for j,mode in enumerate(['uniform','weighted']):axes[1].bar(x+(j-.5)*.3,[r[mode+'_ratio'] for r in gs],width=.3,label=mode)
    axes[1].set_ylabel('Norm(0.1 geometry gradient) / norm(supervised gradient)');axes[1].legend()
    for ax in axes:ax.set_xticks(x,[r['state'].replace('_seed','\nseed') for r in gs],rotation=30,ha='right')
    fig.suptitle('Fixed training-image probes; eight probes per state; zero updates')
    fig.savefig(figures/'gradients.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');x=np.arange(3)
    for ax,task in zip(axes,['depth','normal']):
        for j,rule in enumerate(RULES):ax.bar(x+(j-1.5)*.2,[summary[c][rule][task] for c in CONDITIONS],width=.2,label=rule)
        ax.set_xticks(x,['Clean','Heterogeneous noise','Smooth bias']);ax.set_ylabel('AbsRel' if task=='depth' else 'Normal error (degrees)');ax.legend(fontsize=8)
    fig.suptitle('Synthetic Marigold pilot: 16 train / 8 eval images; 64 updates; two seeds')
    fig.savefig(figures/'noise.png',dpi=150);plt.close(fig)
    noise_rows=''.join(f'<tr><td>{c}</td><td>{r}</td><td>{summary[c][r]["depth"]:.5f}</td><td>{summary[c][r]["normal"]:.3f}</td><td>{html.escape(str(summary[c][r]["seed_means"]))}</td></tr>' for c in CONDITIONS for r in RULES)
    gradient_rows=''.join(f'<tr><td>{r["state"]}</td><td>{r["depth_normal_cosine"]:.4f}</td><td>{r["conflicting_task_probes"]}/8</td><td>{r["uniform_ratio"]:.4f}</td><td>{r["weighted_ratio"]:.4f}</td></tr>' for r in gs)
    document=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>V10 mechanism diagnostics</title><style>body{{font:16px/1.6 system-ui;color:#193448;max-width:1150px;margin:35px auto;padding:20px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #ccd8df;padding:8px;text-align:left}}img{{width:100%}}.box{{background:#eaf2f6;padding:18px}}</style>
<h1>Scale-sensitive geometry: mechanism diagnostics</h1><p class="box">Three completed exploratory studies after V9: saved-prediction error regions, fixed training-image gradients, and a small controlled-noise Marigold training experiment. These studies explain mechanisms; they do not replace independent real-world confirmation.</p>
<h2>1. Where does the error change?</h2><p>All32 previously observed development scenes, all3 V9 seeds and all7 conditions. Regions are identical across models. Sensitivity quartiles use depth-derived labels only for retrospective interpretation; they were not supplied to the trained V9 models at inference. Scene errors are averaged equally. Depth-gradient regions are not semantic boundaries or verified planar surfaces.</p><img src="figures/regional.png"><p>No regional p-values or cherry-picked significance claims. Full per-scene means, pixel counts, p90 errors and paired scene differences: <a href="regional.json">regional.json</a>.</p>
<h2>2. Are gradients conflicting or too weak?</h2><p>Four predetermined training images and two noise draws per state. Shared LoRA gradients for depth supervision, normal supervision and uniform/weighted decoded geometry. Eight probes per state, no optimizer updates. Negative cosine indicates local disagreement; it does not prove validation harm. Ratios include the actual0.1 geometry coefficient.</p><img src="figures/gradients.png"><table><tr><th>State</th><th>Depth/normal cosine</th><th>Conflicting probes</th><th>Uniform ratio</th><th>Weighted ratio</th></tr>{gradient_rows}</table>
<h2>3. Controlled training-label corruption</h2><p>Actual Marigold with rank4 LoRA, fresh seed-matched initialization,16 analytic rendered training surfaces and8 evaluation surfaces with different generator seeds. Clean, heterogeneous-noise and smooth-bias training labels; all evaluation labels remain clean analytic geometry. RGB is always rendered from clean geometry. All rules share masks, targets, order and random noise. Only geometry consistency is spatially weighted. The privileged oracle uses true normal-label error; it is not deployable and is not a guaranteed performance upper bound.</p><img src="figures/noise.png"><table><tr><th>Corruption</th><th>Rule</th><th>Depth AbsRel</th><th>Normal degrees</th><th>Individual seed means</th></tr>{noise_rows}</table>
<p>Initial-model seed measurements (before adaptation): {html.escape(str(initial_means))}.</p><p>Two training seeds,64updates/run,24runs. This short synthetic-domain pilot is not evidence of convergence, adequate power or real-scene transfer. Failure to improve cannot establish that weighting is useless; an oracle null result can also arise from insufficient optimization or domain mismatch. Initial-model measurements and clean-label controls are retained. Noise conditions test mechanisms; they do not estimate how much such noise occurs in NYUv2.</p>
<h2>Verification</h2><pre>{html.escape(str(result['verification']))}</pre><p>Independent final and initial prediction metric checks, exact final checkpoint inference restoration, matched image-order checks, source/input hashes and historical-file preservation. Small mixed-precision differences between separate and combined gradient calculations are recorded; training is not claimed bitwise deterministic.</p><p><a href="protocol.json">Frozen protocol</a> | <a href="analysis.json">Summary data</a> | <a href="gradients.json">Gradient probes</a> | <a href="noise_data.json">Synthetic input provenance</a> | <a href="../reliability_training_v9/RESULTS.html">V9 results</a></p></html>'''
    (OUT/'RESULTS.html').write_text(document,encoding='utf-8')
    write(OUT/'complete.json',dict(analysis_sha256=sha(OUT/'analysis.json'),report_sha256=sha(OUT/'RESULTS.html'),protocol_sha256=sha(OUT/'protocol.json'),verification=result['verification']))
    print('V10_AUDIT_COMPLETE',result['verification'],flush=True)


if __name__=='__main__':main()
