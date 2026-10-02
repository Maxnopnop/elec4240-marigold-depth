"""Independent saved-prediction audit, paired exploratory analysis and figures."""
import hashlib
import html
import json
import subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,WORK,ASSETS,OUT,read,write,sha,sample


def independent_depth(pred,gt,mask):
    p=pred.astype(np.float64)[mask];d=gt.astype(np.float64)[mask]
    outside=float(np.mean((p<.1)|(p>10)));p=np.clip(p,.1,10)
    return {'abs_rel':float(np.mean(abs(p-d)/d)),'rmse_m':float(np.sqrt(np.mean((p-d)**2))),
            'delta1':float(np.mean(np.maximum(p/d,d/p)<1.25)),
            'prediction_outside_range_fraction':outside,'valid_pixels':int(mask.sum())}


def independent_normal(pred,gt,mask):
    p=pred[:,mask].astype(np.float64);g=gt[:,mask].astype(np.float64)
    p/=np.maximum(np.linalg.norm(p,axis=0,keepdims=True),1e-8)
    g/=np.maximum(np.linalg.norm(g,axis=0,keepdims=True),1e-8)
    degrees=np.arccos(np.clip(np.einsum('ij,ij->j',p,g),-1,1))*180/np.pi
    return {'mean_deg':float(degrees.mean()),'median_deg':float(np.median(degrees)),
            'within_11_25':float(np.mean(degrees<11.25)),'within_22_5':float(np.mean(degrees<22.5)),
            'within_30':float(np.mean(degrees<30)),'valid_pixels':int(mask.sum())}


def holm(ps):
    order=np.argsort(ps);values=np.empty(len(ps));maximum=0.
    for rank,i in enumerate(order):maximum=max(maximum,(len(ps)-rank)*ps[i]);values[i]=min(1.,maximum)
    return values.tolist()


def geometric_agreement(depth,normal,mask,camera):
    """Descriptive post-hoc diagnostic on saved 480x640 outputs, using NumPy.

    This has a different resolution from the native training loss. No smoothing,
    GT scale fit or target-normal values enter the comparison; only its mask.
    """
    h,w=depth.shape
    y,x=np.meshgrid(np.arange(1,h+1),np.arange(1,w+1),indexing='ij')
    ray=np.stack([(x-camera['cx_rgb'])/camera['fx_rgb'],
                  (y-camera['cy_rgb'])/camera['fy_rgb'],np.ones_like(x)],axis=-1)
    points=depth[...,None].astype(np.float64)*ray
    dx=points[1:-1,2:]-points[1:-1,:-2];dy=points[2:,1:-1]-points[:-2,1:-1]
    derived=np.cross(dx,dy)
    derived/=np.maximum(np.linalg.norm(derived,axis=-1,keepdims=True),1e-8)
    derived*=np.where(np.sum(derived*points[1:-1,1:-1],axis=-1,keepdims=True)>0,-1.,1.)
    predicted=np.moveaxis(normal.astype(np.float64),0,-1)[1:-1,1:-1]
    predicted/=np.maximum(np.linalg.norm(predicted,axis=-1,keepdims=True),1e-8)
    cosine=np.clip(np.sum(derived*predicted,axis=-1),-1,1)[mask[1:-1,1:-1]]
    return {'mean_cosine_disagreement':float(np.mean(1-cosine)),
            'mean_angle_deg':float(np.degrees(np.arccos(cosine)).mean())}


def main():
    protocol=read(OUT/'protocol.json');manifest=read(OUT/'manifest.json')
    modes=protocol['modes'];seeds=protocol['seeds'];rows=manifest['validation']
    assert sha(OUT/'manifest.json')==protocol['manifest_sha256']
    assert sha(OUT/'camera.json')==protocol['camera_sha256']
    train_scenes={r['scene'] for r in manifest['training']};validation_scenes={r['scene'] for r in rows}
    assert len(train_scenes)==128 and len(validation_scenes)==32 and not train_scenes&validation_scenes
    for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==digest,name
    assert np.allclose(holm([.01,.04,.03]),[.03,.06,.06])
    data={};predictions_checked=0;restorations=0;orders={};training={}
    for row in manifest['training']+rows:
        assert sha(ASSETS/'subset'/row['file'])==row['source_sha256']
        assert sha(WORK/'derived'/row['file'])==row['derived_sha256']
        z=sample(WORK/'derived'/row['file'])
        assert hashlib.sha256(z['raw_depth'].tobytes()).hexdigest()==row['raw_depth_array_sha256']
    truth={r['id']:(sample(ASSETS/'subset'/r['file']),sample(WORK/'derived'/r['file'])) for r in rows}
    for mode in modes:
        for seed in seeds:
            name=f'{mode}_seed{seed}';dest=OUT/'runs'/name
            mark=read(dest/'complete.json');records=read(dest/'validation.json');stats=read(dest/'training.json')
            assert mark['protocol_sha256']==sha(OUT/'protocol.json')
            assert mark['metrics_sha256']==sha(dest/'validation.json')
            assert mark['checkpoint_sha256']==sha(WORK/'checkpoints'/name/'adapter.pt')
            assert len(stats['history'])==protocol['steps']==stats['steps']
            assert [r['id'] for r in records]==[r['id'] for r in rows]
            expected_tasks=['depth'] if mode=='depth' else ['normal'] if mode=='normal' else ['depth','normal']
            assert stats['tasks']==expected_tasks and mark['checkpoint_restored_exact_tasks']==expected_tasks
            assert stats['trainable_parameters']==829952 and stats['train_images']==128
            assert all(set(r['tasks'])==set(expected_tasks) for r in records)
            orders[name]=[r['id'] for r in stats['history']]
            for step in stats['history']:
                expected=np.mean(list(step['task_losses'].values()))+stats['geometry_weight']*step['geometry_loss']
                assert abs(expected-step['loss'])<2e-5
            training[name]=stats
            restorations+=len(mark['checkpoint_restored_exact_tasks'])
            for record in records:
                original,derived=truth[record['id']]
                for task,values in record['tasks'].items():
                    path=WORK/'predictions'/name/f"{record['id']:04d}_{task}.npy"
                    assert sha(path)==values['prediction_sha256']
                    pred=np.load(path);assert np.isfinite(pred).all()
                    recomputed=independent_depth(pred[0],original['depth'],derived['depth_valid']) if task=='depth' else independent_normal(pred,derived['normal'],derived['normal_valid'])
                    if task=='depth':recomputed['raw_depth_abs_rel']=independent_depth(pred[0],derived['raw_depth'],derived['raw_valid'])['abs_rel']
                    for key,value in recomputed.items():assert np.isclose(value,values[key],rtol=1e-10,atol=1e-9),(name,record['id'],key)
                    predictions_checked+=1
            data[name]=records
    assert predictions_checked==576 and restorations==18
    assert len({t['checkpoint_sha256'] for t in training.values()})==12
    for seed in seeds:
        assert all(orders[f'{m}_seed{seed}']==orders[f'depth_seed{seed}'] for m in modes)
    baseline=read(OUT/'baselines.json');train_means=[]
    for row in manifest['training']:
        original=sample(ASSETS/'subset'/row['file']);derived=sample(WORK/'derived'/row['file'])
        train_means.append(float(original['depth'][derived['depth_valid']].mean()))
    assert abs(float(np.mean(train_means))-baseline['constant_depth_m'])<1e-12
    assert len(baseline['records'])==len(rows)
    for row,rec in zip(rows,baseline['records']):
        assert row['id']==rec['id'];original,derived=truth[row['id']]
        path=WORK/'predictions/expert'/f"{row['id']:04d}_depth.npy";assert sha(path)==rec['prediction_sha256']
        values=independent_depth(np.load(path),original['depth'],derived['depth_valid'])
        for key,value in values.items():assert np.isclose(value,rec['expert_native_metric'][key],rtol=1e-10,atol=1e-9)
        constant=np.full_like(original['depth'],baseline['constant_depth_m'])
        front=np.zeros_like(derived['normal']);front[2]=-1
        for label,values in [('constant_depth',independent_depth(constant,original['depth'],derived['depth_valid'])),
                             ('constant_normal',independent_normal(front,derived['normal'],derived['normal_valid']))]:
            for key,value in values.items():assert np.isclose(value,rec[label][key],rtol=1e-10,atol=1e-9)

    def matrix(mode,task,metric):
        return np.array([[r['tasks'][task][metric] for r in data[f'{mode}_seed{s}']] for s in seeds])
    summary={}
    for mode in modes:
        tasks=training[f'{mode}_seed{seeds[0]}']['tasks'];summ={}
        for task in tasks:
            keys=['abs_rel','rmse_m','delta1','raw_depth_abs_rel'] if task=='depth' else ['mean_deg','median_deg','within_11_25','within_22_5','within_30']
            summ[task]={}
            for key in keys:
                a=matrix(mode,task,key);per_seed=a.mean(1)
                summ[task][key]={'mean':float(a.mean()),'seed_sd':float(per_seed.std(ddof=1)),
                                 'per_seed':per_seed.tolist()}
        summ['training_seconds']=[training[f'{mode}_seed{s}']['training_seconds'] for s in seeds]
        summ['peak_allocated_mib']=[training[f'{mode}_seed{s}']['peak_allocated_mib'] for s in seeds]
        summ['inference_median_ms']=float(np.median([r['seconds_all_requested_tasks']*1000 for s in seeds for r in data[f'{mode}_seed{s}']]))
        summary[mode]=summ
    rng=np.random.default_rng(724240);B=20000
    seed_draw=rng.integers(0,len(seeds),(B,len(seeds)))
    scene_draw=rng.integers(0,len(rows),(B,len(rows)))
    comparisons=[]
    for new,old,key in protocol['primary_exploratory_contrasts']:
        task,metric=key.split('.');delta=matrix(new,task,metric)-matrix(old,task,metric)
        effect=float(delta.mean());boot=delta[seed_draw[:,:,None],scene_draw[:,None,:]].mean((1,2))
        p=float((1+np.count_nonzero(abs(boot-effect)>=abs(effect)))/(B+1))
        comparisons.append({'new':new,'old':old,'metric':key,'difference':effect,
                            'percentile95':np.quantile(boot,[.025,.975]).tolist(),'raw_p':p,
                            'per_seed_difference':delta.mean(1).tolist(),
                            'interpretation':'negative is lower error; exploratory on observed validation, not independent confirmation'})
    for c,p in zip(comparisons,holm([c['raw_p'] for c in comparisons])):c['holm4_p']=p
    # Descriptive direct comparisons complete the requested single-task controls.
    # They are not additional tests or additions to the frozen four-test family.
    direct=[]
    for mode in ['joint','joint_geometry']:
        dd=matrix(mode,'depth','abs_rel')-matrix('depth','depth','abs_rel')
        dn=matrix(mode,'normal','mean_deg')-matrix('normal','normal','mean_deg')
        dx=dd.mean(0);dy=dn.mean(0)
        direct.append({'mode':mode,'depth_abs_rel_change':float(dd.mean()),'normal_mean_deg_change':float(dn.mean()),
                       'depth_per_seed_change':dd.mean(1).tolist(),'normal_per_seed_change':dn.mean(1).tolist(),
                       'scene_counts':{'both_lower':int(np.sum((dx<0)&(dy<0))),
                                       'depth_lower_normal_higher':int(np.sum((dx<0)&(dy>=0))),
                                       'depth_higher_normal_lower':int(np.sum((dx>=0)&(dy<0))),
                                       'both_higher':int(np.sum((dx>=0)&(dy>=0)))},
                       'scope':'descriptive differences against separate single-task models; no extra hypothesis test'})
    base_summary={name:float(np.mean([r[name]['abs_rel'] for r in baseline['records']])) for name in ['constant_depth','expert_native_metric']}
    base_summary['constant_normal_mean_deg']=float(np.mean([r['constant_normal']['mean_deg'] for r in baseline['records']]))
    coverage={}
    for split in ['training','validation']:
        values=[r['normal_valid_pixels']/r['depth_valid_pixels'] for r in manifest[split]]
        coverage[split]={'mean_fraction':float(np.mean(values)),'min_fraction':float(min(values)),'max_fraction':float(max(values))}
    camera=read(OUT/'camera.json')['intrinsics'];agreement={}
    flat=np.full((480,640),3.,np.float32);front=np.zeros((3,480,640),np.float32);front[2]=-1
    for scale in [1.,2.]:
        assert geometric_agreement(flat*scale,front,np.ones_like(flat,dtype=bool),camera)['mean_angle_deg']<1e-7
    for label,md,mn in [('separate','depth','normal'),('joint','joint','joint'),('joint_geometry','joint_geometry','joint_geometry')]:
        seed_means=[]
        for seed in seeds:
            values=[]
            for row in rows:
                d=np.load(WORK/'predictions'/f'{md}_seed{seed}'/f"{row['id']:04d}_depth.npy")[0]
                n=np.load(WORK/'predictions'/f'{mn}_seed{seed}'/f"{row['id']:04d}_normal.npy")
                values.append(geometric_agreement(d,n,truth[row['id']][1]['normal_valid'],camera))
            seed_means.append({k:float(np.mean([v[k] for v in values])) for k in values[0]})
        agreement[label]={k:{'mean':float(np.mean([v[k] for v in seed_means])),
                             'seed_sd':float(np.std([v[k] for v in seed_means],ddof=1)),
                             'per_seed':[v[k] for v in seed_means]} for k in seed_means[0]}
    result={'scope':'observed development validation, one training draw, three seeds; not confirmatory',
            'summary':summary,'baselines':base_summary,'comparisons':comparisons,'direct_single_task_comparisons':direct,
            'normal_mask_coverage_relative_to_depth_mask':coverage,
            'post_hoc_geometry_diagnostic':{'scope':'descriptive only; saved full-resolution predictions, central differences, no smoothing, fixed normal mask; differs from native training resolution','results':agreement},
            'training_seconds_total':sum(t['training_seconds'] for t in training.values())}
    write(OUT/'analysis.json',result)
    # A single batched Git operation verifies prior results and delivered reports.
    tree=subprocess.check_output(['git','ls-tree','-rz',protocol['historical_baseline_commit'],'results','reports'],cwd=ROOT)
    expected=[]
    for item in tree.split(b'\0'):
        if item:
            meta,name=item.split(b'\t',1);expected.append((name.decode(),meta.split()[2].decode()))
    actual=subprocess.check_output(['git','hash-object','--stdin-paths'],cwd=ROOT,input=''.join(n+'\n' for n,h in expected),text=True).splitlines()
    assert len(actual)==len(expected) and all(h==v for (_,h),v in zip(expected,actual))
    write(OUT/'verification.json',{'status':'passed','data_pairs_hashed':160,'independently_recomputed_task_predictions':predictions_checked,
          'expert_metric_predictions_recomputed':32,'checkpoint_restored_exact_task_predictions':restorations,
          'constant_baselines_recomputed':64,'training_only_constant_calibration_verified':True,
          'training_validation_scene_overlap':0,
          'post_hoc_geometric_agreement_pairs':288,'geometry_diagnostic_front_plane_and_scale_check':True,
          'paired_training_image_orders_identical':True,'historical_result_report_blobs_unchanged':len(expected),
          'frozen_source_hashes_verified':len(protocol['source_sha256']),'protocol_sha256':sha(OUT/'protocol.json')})
    figures(result,rows,truth,matrix)
    report(result)
    print('V7_ANALYSIS_AUDIT_PASSED',predictions_checked,restorations,flush=True)
    print(json.dumps({'summary':summary,'comparisons':comparisons},indent=2),flush=True)


def figures(result,rows,truth,matrix):
    directory=OUT/'figures';directory.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    s=result['summary'];colors=['#64748b','#197c86','#c26735']
    labels=['Separate single-task models','Joint','Joint + geometry']
    xs=[s['depth']['depth']['abs_rel']['mean'],s['joint']['depth']['abs_rel']['mean'],s['joint_geometry']['depth']['abs_rel']['mean']]
    ys=[s['normal']['normal']['mean_deg']['mean'],s['joint']['normal']['mean_deg']['mean'],s['joint_geometry']['normal']['mean_deg']['mean']]
    costs=[np.mean(s['depth']['training_seconds'])+np.mean(s['normal']['training_seconds']),np.mean(s['joint']['training_seconds']),np.mean(s['joint_geometry']['training_seconds'])]
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for x,y,c,label,md,mn in zip(xs,ys,colors,labels,['depth','joint','joint_geometry'],['normal','joint','joint_geometry']):
        axes[0].scatter(s[md]['depth']['abs_rel']['per_seed'],s[mn]['normal']['mean_deg']['per_seed'],color=c,s=22,alpha=.35)
        axes[0].scatter(x,y,color=c,s=90,label=label)
    axes[0].set(xlabel='Metric depth AbsRel (lower is better)',ylabel='Normal mean angular error (degrees)',title='Observed validation: three-seed means');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    axes[1].bar(range(3),costs,color=colors);axes[1].set(xticks=range(3),xticklabels=['Separate\nmodels','Joint','Joint +\ngeometry'],ylabel='Optimizer seconds, mean per seed',title='Cost to train both tasks')
    for i,v in enumerate(costs):axes[1].text(i,v,f'{v:.0f}s',ha='center',va='bottom',fontsize=9)
    axes[1].set_ylim(0,max(costs)*1.15);fig.savefig(directory/'accuracy_cost.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,new,oldD,oldN in [(axes[0],'joint','depth','normal'),(axes[1],'joint_geometry','joint','joint')]:
        dx=(matrix(new,'depth','abs_rel')-matrix(oldD,'depth','abs_rel')).mean(0)
        dy=(matrix(new,'normal','mean_deg')-matrix(oldN,'normal','mean_deg')).mean(0)
        ax.scatter(dx,dy,color='#197c86',alpha=.75,s=22);ax.axhline(0,color='#677',lw=.8);ax.axvline(0,color='#677',lw=.8)
        ax.set(xlabel='Depth error change (negative is better)',ylabel='Normal error change in degrees',title='Joint vs separate' if new=='joint' else 'Geometry vs joint')
    fig.savefig(directory/'task_tradeoffs.png',dpi=170);plt.close(fig)
    for task in ['depth','normal']:
        fig,axes=plt.subplots(3,5,figsize=(12,7.2),layout='constrained')
        for i,row in enumerate(rows[:3]):
            original,derived=truth[row['id']];mask=derived['depth_valid' if task=='depth' else 'normal_valid']
            axes[i,0].imshow(original['image']);axes[i,0].set_ylabel(f"ID {row['id']}")
            truth_image=original['depth'] if task=='depth' else derived['normal']
            values=[truth_image]+[np.load(WORK/'predictions'/f'{mode}_seed17'/f"{row['id']:04d}_{task}.npy") for mode in [task,'joint','joint_geometry']]
            for j,value in enumerate(values,1):
                if task=='depth':
                    value=np.squeeze(value).copy();value[~mask]=np.nan
                    shown=axes[i,j].imshow(value,vmin=.1,vmax=10,cmap='viridis')
                else:
                    value=np.moveaxis((value+1)/2,0,-1).clip(0,1);value[~mask]=1
                    axes[i,j].imshow(value)
            for ax in axes[i]:ax.set_xticks([]);ax.set_yticks([])
        for ax,title in zip(axes[0],['RGB','Target','Single task','Joint','Joint + geometry']):ax.set_title(title,fontsize=11)
        fig.suptitle(('Metric depth (metres)' if task=='depth' else 'Camera-facing normals; derived targets')+' | first three validation scenes, seed 17',fontsize=12)
        if task=='depth':fig.colorbar(shown,ax=axes[:,1:],shrink=.65,label='Depth (m)')
        fig.savefig(directory/f'{task}_examples.png',dpi=160);plt.close(fig)


def report(result):
    s=result['summary']
    coverage=result['normal_mask_coverage_relative_to_depth_mask']['validation']
    label_quality=''
    if (OUT/'label_quality.json').exists():
        q=read(OUT/'label_quality.json');assert q['manifest_sha256']==sha(OUT/'manifest.json')
        v=q['summary']
        label_quality=f'''<h2>Training-only label-quality follow-up</h2><p>Visual inspection found noisy derived normal targets. A post-hoc check on all 128 training images gives {v['native_resolution_roundtrip_mean_deg']['mean']:.2f}° mean angular change after a target-only 192×256 downsample/upsample round trip. Changing the depth smoothing from sigma 1 to sigma 2 or 4 changes the labels by {v['sigma2_vs_frozen_sigma1_mean_deg']['mean']:.2f}° and {v['sigma4_vs_frozen_sigma1_mean_deg']['mean']:.2f}°, respectively, on the unchanged mask. These are sensitivity measurements, not model baselines, physical ground truth or error lower bounds. They do not explain away the observed negative transfer. No labels, models or frozen results were changed. See <a href="label_quality.json">the training-only records</a>.</p>'''
    def value(mode,task,key,precision=4):
        if task not in s[mode]:return '—'
        v=s[mode][task][key];return f"{v['mean']:.{precision}f} ± {v['seed_sd']:.{precision}f}"
    rows=''.join(f"<tr><td>{m}</td><td>{value(m,'depth','abs_rel')}</td><td>{value(m,'depth','rmse_m')}</td><td>{value(m,'depth','delta1')}</td><td>{value(m,'normal','mean_deg',2)}</td><td>{value(m,'normal','within_22_5')}</td><td>{np.mean(s[m]['training_seconds']):.1f}</td></tr>" for m in s)
    comparisons=''.join(f"<tr><td>{c['new']} − {c['old']}</td><td>{c['metric']}</td><td>{c['difference']:+.5f}</td><td>[{c['percentile95'][0]:+.5f}, {c['percentile95'][1]:+.5f}]</td><td>{c['holm4_p']:.5f}</td></tr>" for c in result['comparisons'])
    direct=''.join(f"<tr><td>{c['mode']}</td><td>{c['depth_abs_rel_change']:+.5f}</td><td>{c['normal_mean_deg_change']:+.2f}°</td><td>{c['scene_counts']['both_lower']}/32</td><td>{c['scene_counts']['depth_lower_normal_higher']+c['scene_counts']['depth_higher_normal_lower']}/32</td><td>{c['scene_counts']['both_higher']}/32</td></tr>" for c in result['direct_single_task_comparisons'])
    geometric=''.join(f"<tr><td>{mode}</td><td>{v['mean_cosine_disagreement']['mean']:.4f}</td><td>{v['mean_angle_deg']['mean']:.2f}°</td></tr>" for mode,v in result['post_hoc_geometry_diagnostic']['results'].items())
    peak=max(v for m in s.values() for v in m['peak_allocated_mib'])
    timing='; '.join(f"{m}: {v['inference_median_ms']:.1f} ms" for m,v in s.items())
    decisions=[]
    for c in result['comparisons']:
        direction='lower' if c['difference']<0 else 'higher'
        evidence='passes this exploratory four-test adjustment' if c['holm4_p']<.05 else 'does not pass this exploratory four-test adjustment'
        decisions.append(f"<li>{c['new']} has {direction} mean {c['metric']} than {c['old']}; the comparison {evidence}.</li>")
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Metric depth and normal adaptation — V7</title>
<style>body{{font:16px/1.65 Segoe UI,sans-serif;color:#20313f;max-width:1100px;margin:30px auto;padding:0 24px}}h1{{font-size:28px}}h2{{font-size:21px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:10px;border-bottom:1px solid #dce4e9;text-align:left}}th{{background:#edf4f7}}img{{width:100%;height:auto}}.note{{background:#fff6e6;border-left:4px solid #c18e29;padding:14px}}a{{color:#176387}}</style>
<h1>Metric depth → surface normals → geometric consistency</h1><p>ELEC4240 · 2026-10-02 · HO, Chun Wai 21053878; Tong, Man Hung 21064669; MA, Shenhan 21041382.</p>
<p class="note">Exploratory development study: one training draw, three seeds, 128 training scenes and 32 previously observed validation scenes. Normal targets are derived from depth and are not independent measured normal ground truth. No per-image ground-truth depth alignment is used.</p>
<h2>Implemented method</h2><p>A pinned Marigold Depth v1.1 backbone uses rank-4 attention LoRA (829,952 trainable parameters), frozen VAE/text encoder and two fixed CLIP task prompts. We train at 192×256 for 320 updates using the one-step, zero-SNR t=999 velocity objective. Depth is represented by a fixed log encoding over 0.1–10 metres; normals occupy all three decoded channels and are unit-normalized. This is an adaptation of a depth model, not a reproduction of Vision Banana's generalist or generation-preservation claims.</p>
<p>Joint updates use the same image and noise for both tasks, averaging the two supervised losses. Each task receives exactly 320 image exposures, matching its single-task control: only 2.5 passes over 128 images. This is a bounded pilot, not evidence of converged training. Joint models require two UNet forwards per update. The geometry variant adds 0.1 times cosine disagreement between the decoded normal and the camera-facing normal derived from decoded metric depth, with gradients through both branches. Its extra VAE decoding/backward cost is included.</p>
<h2>Data and geometry</h2><p>Source RGB and filled metric depth are unchanged NYUv2 arrays. Official toolbox RGB calibration uses 1-based pixel coordinates; resizing uses half-pixel-consistent rays. We treat projected depth as camera-z in the RGB pinhole approximation. Normal targets use sigma-1 smoothed filled depth, with a three-pixel eroded raw-depth agreement mask and exclusion around measured depth discontinuities. This filters unreliable labels but also excludes difficult regions; normal quality and coverage therefore constrain claims. Raw-depth AbsRel is retained as a secondary depth metric.</p>
<p>The normal mask retains an average {coverage['mean_fraction']:.1%} of depth-evaluation pixels per validation image (range {coverage['min_fraction']:.1%}–{coverage['max_fraction']:.1%}). Accuracy on excluded boundaries, missing-depth regions and the rest of the image is not established.</p>
<h2>Results</h2><p>Values are equal-scene means ± sample SD of the three seed means. All depth metrics use metre-valued predictions without alignment; AbsRel is dimensionless and RMSE is in metres. A common fixed 0.1–10m prediction clip applies to all depth methods. All checkpoints are the fixed final update, with no best-seed selection.</p>
<table><tr><th>Training</th><th>Depth AbsRel ↓</th><th>RMSE (m) ↓</th><th>δ1 ↑</th><th>Normal mean (°) ↓</th><th>Normal &lt;22.5° ↑</th><th>Train s</th></tr>{rows}</table>
<p>Training-only constant depth: AbsRel {result['baselines']['constant_depth']:.5f}; native Depth Anything V2 reference: {result['baselines']['expert_native_metric']:.5f}; constant camera-facing normal: {result['baselines']['constant_normal_mean_deg']:.2f}°. The specialist uses Hypersim metric fine-tuning and its official default processor, so prior supervision and input cost differ.</p>
<p>Across all 12 formal runs, optimizer time totals {result['training_seconds_total']/60:.2f} minutes; peak PyTorch-allocated training memory is {peak:.1f} MiB. Median inference time per image on this laptop: {timing}. Each joint timing produces both requested outputs; each single-task timing produces one.</p>
<img src="figures/accuracy_cost.png" alt="Accuracy of both tasks and training cost"><p>Large points show three-seed means; faint points show individual seeds. Separate single-task models use two adapters in total; joint variants share one. Optimizer times exclude model loading and the one-time shared target cache. Inference timings are descriptive sequential local measurements, not controlled latency benchmarks.</p>
<h2>Direct comparison with both single-task controls</h2><p>Negative changes indicate lower error. Scene counts average all three seeds before classifying direction, without significance or equivalence claims. Both joint variants are compared directly with the depth-only and normal-only models.</p><table><tr><th>Model</th><th>Depth AbsRel change</th><th>Normal angle change</th><th>Both improve</th><th>One improves, one worsens</th><th>Both worsen</th></tr>{direct}</table>
<h2>Task transfer and uncertainty</h2><table><tr><th>Comparison</th><th>Error metric</th><th>Mean change</th><th>Unadjusted percentile 95% interval</th><th>Holm4 p</th></tr>{comparisons}</table><ul>{''.join(decisions)}</ul>
<p>Paired crossed scene/seed bootstrap uses 20,000 replicates. Two-sided centered tests are adjusted over the four fixed contrasts. Only three seeds and an observed validation set are available; these approximate exploratory p-values do not establish independent generalization. A nonsignificant difference is not equivalence. Lower geometric disagreement alone would not prove better depth or normal accuracy.</p><img src="figures/task_tradeoffs.png" alt="Per-scene changes in depth and normal error"><p>Each point averages the three seeds for one validation scene. Lower-left indicates improvement in both tasks; upper-right indicates degradation in both.</p>
<h2>Post-hoc geometric agreement diagnostic</h2><p>This descriptive check compares normals derived from saved predicted depth with saved predicted normals on the fixed normal mask. It uses full-resolution central differences without smoothing, so its values are not the native-resolution training penalty. It adds no hypothesis tests to the frozen family. Target-normal values and GT scale alignment are not used; agreement does not imply accuracy.</p><table><tr><th>Model</th><th>Mean 1 − cosine ↓</th><th>Mean disagreement angle ↓</th></tr>{geometric}</table>
<h2>Fixed qualitative examples</h2><img src="figures/depth_examples.png" alt="Metric depth predictions on fixed examples"><img src="figures/normal_examples.png" alt="Surface normal predictions on fixed examples"><p>The examples are the first three manifest scenes and fixed seed17. White target/prediction areas denote excluded evaluation pixels. Their selection is independent of prediction quality.</p>
{label_quality}
<h2>Verification and reproduction</h2><p>See <a href="verification.json">verification.json</a> for independent recomputation of all 576 learned task predictions and 32 specialist predictions, checkpoint restoration and preservation of previous artifacts. <a href="protocol.json">The frozen protocol</a>, <a href="analysis.json">all seed-level summaries</a> and source scripts are retained. Local raw arrays, labels, checkpoints and model caches are excluded from Git. The previous final PDF remains unchanged.</p>
<h2>Research interpretation and next experiment</h2><p>Under this short training budget, the shared adapter worsens both tasks relative to their single-task controls in every seed. Geometric regularization reduces prediction-to-prediction disagreement, but its mean depth improvement over ordinary joint training does not pass the fixed exploratory adjustment, while its normal error increases. The geometry model also has worse three-seed mean errors than both separate controls and costs about 20% more optimizer time than ordinary joint training. These results support reporting a task tradeoff, not a successful improvement over the single-task methods.</p>
<p>The next stage should first establish stable metric-depth learning curves and better-validated normal targets. Then compare task-specific adapters with the shared adapter while accounting for their different parameter counts, and test delayed or confidence-weighted geometric regularization as separate ablations. These are future hypotheses; this experiment does not identify gradient conflict, adapter capacity, target noise or the fixed loss weight as the proven cause. A new versioned protocol should retain these results rather than replace them with a favorable setting.</p>
<h2>What remains</h2><p>This bounded study establishes an executable staged comparison and exposes positive or negative transfer under its constraints. Stronger claims need additional training draws, genuinely unobserved scenes and independently measured normal labels. Metric encoding is numerically invertible only within its fixed interval; neither the encoding nor depth-normal consistency identifies absolute scale without metric supervision.</p>
<p>Sources: <a href="https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html">NYUv2 data and toolbox</a>; <a href="https://arxiv.org/abs/2505.09358">Marigold 2025</a>; <a href="https://openaccess.thecvf.com/content_cvpr_2018/html/Qi_GeoNet_Geometric_Neural_CVPR_2018_paper.html">GeoNet: joint depth and normals</a>.</p></html>'''
    (OUT/'RESULTS.html').write_text(body,encoding='utf-8',newline='\n')


if __name__=='__main__':main()
