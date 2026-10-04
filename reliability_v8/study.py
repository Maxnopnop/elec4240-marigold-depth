"""Frozen, bounded validation of the proxy; no model accuracy claims."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr, rankdata
from multitask_v7.common import ROOT, ASSETS, WORK as V7WORK, OUT as V7OUT, read, write, sha, rays, sample
from .core import stability, original_support, native_weights, shuffled_weights, angles, weight_stats

OUT = ROOT/'results/reliability_v8'
WORK = V7WORK.parent/'reliability_v8'
SURFACES = ['plane', 'crease', 'step', 'curved']
CORRUPTIONS = ['clean', 'iid_noise', 'heterogeneous_noise', 'outliers', 'holes', 'smooth_bias']
DEV = [11,23,37,53]
HOLD = [101,103,107,109]


def freeze():
    path = OUT/'component_protocol.json'
    if path.exists():
        verify_protocol(); print('Existing frozen protocol verified', flush=True); return
    p = {'version':'reliability_v8_component_v1','created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
         'status':'frozen before component outcomes; engineering gate, not a confirmatory clinical/statistical trial',
         'dimensions':[480,640],'sigmas_fullres_pixels':[1.,2.,4.],'gaussian_truncate':2.,'temperature_deg':10.,
         'proxy':'mean of all three pairwise angular differences; exp(-disagreement_deg/10)',
         'surface_families':SURFACES,'corruptions':CORRUPTIONS,'development_seeds':DEV,'heldout_seeds':HOLD,
         'cases':len(SURFACES)*len(CORRUPTIONS)*len(DEV+HOLD),
         'corruption_parameters':{'iid_noise':'Gaussian std=1% camera-z','heterogeneous_noise':'std=.2% in left half,1.5% in right half',
            'outliers':'1% locations, random signed15% camera-z','holes':'8 random 9-25px rectangles, nearest-valid fill, exclude raw missing support',
            'smooth_bias':'multiplicative1+.08*ray_x+.12*ray_y^2; intentional stable-but-wrong counterexample'},
         'truth':'analytic normals of the clean pinhole-parametrized surfaces; exact crease/step line excluded by the existing jump/erosion mask as applicable',
         'mask':'V7 erosion3 and jump threshold >.1m AND >5% local depth, dilated2; also 10px outer border',
         'boundary_band':'12px either side of centre for crease/step, no band for other surfaces; report separately',
         'metrics':['mean normal target error','weight-normalized target error','spatially shuffled weighted error','Spearman sensitivity/error',
                    'AUROC error>10deg','top20/50/80% stable coverage error','effective pixel fraction','mean weight'],
         'interpretation':'Weighted errors characterize selection of supervision, NOT improved model predictions. Use cases/seeds as units, never pixels as independent trials.',
         'gate':{'heldout_noise_scope':['iid_noise','heterogeneous_noise','outliers','holes'],
                 'overall_weighted_error_relative_reduction_at_least':.10,
                 'weighted_vs_shuffled_relative_reduction_at_least':.05,
                 'families_with_nonpositive_weighted_minus_uniform_at_least':3,
                 'median_synthetic_effective_fraction_at_least':.25,
                 'max_clean_family_weighted_minus_uniform_deg':1.,
                 'real_train_fraction_with_effective_fraction_ge_0_25_at_least':.90,
                 'meaning':'predeclared engineering screen. Passing permits further pilot work; it does not demonstrate normal/depth model benefit.'},
         'real_data':'all128 frozen V7 training scenes only; no validation/test image loading; target consistency and weight distribution, no truth claim',
         'training_next':'stable single-task learning curves and weighted-loss gradient/profile smoke before a separately frozen 7-condition x3-seed ablation',
         'stop':'retain failures; no post-outcome temperature/scale tuning or full training under this version',
         'source_sha256':{str(f.relative_to(ROOT)).replace('\\','/'):sha(f) for f in sorted((ROOT/'reliability_v8').glob('*.py'))},
         'v7_source_sha256':read(V7OUT/'protocol.json')['source_sha256'],
         'manifest_sha256':sha(V7OUT/'manifest.json'),'camera_sha256':sha(V7OUT/'camera.json')}
    write(path,p)
    tracked=__import__('subprocess').check_output(['git','ls-files','results','reports'],cwd=ROOT,text=True).splitlines()
    write(WORK/'historical_hashes.json',{name:sha(ROOT/name) for name in tracked})
    print('PROTOCOL_FROZEN',sha(path),'historical files',len(tracked),flush=True)


def verify_protocol():
    p=read(OUT/'component_protocol.json')
    for name,value in {**p['source_sha256'],**p['v7_source_sha256']}.items():
        assert sha(ROOT/name)==value, name
    assert sha(V7OUT/'manifest.json')==p['manifest_sha256']
    assert sha(V7OUT/'camera.json')==p['camera_sha256']
    return p


def scene(kind, seed, ray):
    rng=np.random.default_rng(seed)
    x,y=ray[0,:2].numpy().astype(np.float64)
    base=float(rng.uniform(2.5,4.))
    a=float(rng.uniform(.2,.6)); b=float(rng.uniform(-.25,.25))
    if kind=='plane':
        denominator=1+a*x+b*y; z=base/denominator
        zx=-base*a/denominator**2; zy=-base*b/denominator**2
    elif kind=='crease':
        denominator=1+a*abs(x)+b*y; z=base/denominator
        zx=-base*a*np.sign(x)/denominator**2; zy=-base*b/denominator**2
    elif kind=='step':
        z=np.full_like(x,base)+(x>0)*.8; zx=np.zeros_like(x); zy=zx.copy()
    else:
        z=base+.9*(x*x+.8*y*y)+.15*x; zx=1.8*x+.15; zy=1.44*y
    n=np.stack([zx,zy,-z-x*zx-y*zy])
    n/=np.linalg.norm(n,axis=0,keepdims=True)
    edge=(abs(x)<(12/518.8579011745019)) if kind in ['crease','step'] else np.zeros_like(x,dtype=bool)
    return z.astype(np.float32),n.astype(np.float32),edge


def corrupt(depth, kind, seed, ray):
    rng=np.random.default_rng(seed+7000+100*CORRUPTIONS.index(kind))
    z=depth.copy(); valid=np.ones_like(depth,dtype=bool)
    if kind=='iid_noise':z+=rng.normal(size=z.shape)*.01*z
    elif kind=='heterogeneous_noise':
        std=np.where(ray[0,0].numpy()<0,.002,.015)
        z+=rng.normal(size=z.shape)*std*z
    elif kind=='outliers':
        flag=rng.random(z.shape)<.01
        z[flag]*=1+rng.choice([-1.,1.],flag.sum())*.15
    elif kind=='holes':
        h,w=z.shape
        for _ in range(8):
            cy,cx=int(rng.integers(40,h-40)),int(rng.integers(40,w-40))
            sy,sx=int(rng.integers(9,26)),int(rng.integers(9,26))
            valid[cy:cy+sy,cx:cx+sx]=False
        index=distance_transform_edt(~valid,return_distances=False,return_indices=True)
        z=z[tuple(index)]
    elif kind=='smooth_bias':
        x,y=ray[0,:2].numpy(); z*=1+.08*x+.12*y*y
    return z,valid


def summarize_pixels(error,u,w,mask,shuffle_seed):
    e=error[mask].astype(np.float64); v=u[mask].astype(np.float64); weights=w[mask].astype(np.float64)
    if e.size<100:return None
    shuffled=np.random.default_rng(shuffle_seed).permutation(weights)
    corr=float(spearmanr(v,e).statistic) if np.ptp(e)>1e-5 and np.ptp(v)>1e-5 else None
    bad=e>10
    auc=None
    if bad.any() and (~bad).any():
        rank=rankdata(v); np_=int(bad.sum()); nn=int((~bad).sum())
        auc=float((rank[bad].sum()-np_*(np_+1)/2)/(np_*nn))
    order=np.argsort(v,kind='stable')
    return {'pixels':int(e.size),'uniform_error_deg':float(e.mean()),
        'weighted_error_deg':float(np.dot(e,weights)/weights.sum()),
        'shuffled_error_deg':float(np.dot(e,shuffled)/shuffled.sum()),
        'spearman':corr,'bad10_auc':auc,
        'retained_error_deg':{str(f):float(e[order[:max(1,int(f*len(e)))]].mean()) for f in [.2,.5,.8]},**weight_stats(weights)}


def synthetic():
    p=verify_protocol(); torch.set_num_threads(2)
    ray=rays(*p['dimensions'],read(V7OUT/'camera.json')['intrinsics'])
    records=[]; examples=[]; start=time.perf_counter()
    with torch.inference_mode():
        for seed in DEV+HOLD:
            for kind in SURFACES:
                truth,ntrue,edge=scene(kind,seed,ray)
                for noise in CORRUPTIONS:
                    observed,valid=corrupt(truth,noise,seed,ray)
                    n,u,w=stability(observed,ray)
                    mask=original_support(observed,valid)
                    mask[:10]=False;mask[-10:]=False;mask[:,:10]=False;mask[:,-10:]=False
                    error=angles(n,ntrue)
                    scopes={'all':mask,'interior':mask&~edge,'boundary':mask&edge}
                    rec={'seed':seed,'split':'development' if seed in DEV else 'heldout','surface':kind,'corruption':noise,
                         'metrics':{name:summarize_pixels(error,u,w,m,seed+91000) for name,m in scopes.items()}}
                    records.append(rec)
                    if seed==HOLD[0] and (kind,noise) in [('plane','iid_noise'),('crease','clean'),('curved','smooth_bias')]:
                        examples.append({'name':kind+' / '+noise,'depth':observed,'error':error,'sensitivity':u,'weight':w,'mask':mask})
            print('SYNTHETIC_SEED_DONE',seed,len(records),'/',p['cases'],flush=True)
    write(OUT/'synthetic.json',{'protocol_sha256':sha(OUT/'component_protocol.json'),'seconds':time.perf_counter()-start,'records':records})
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,4,figsize=(13,8),layout='constrained')
    for row,example in enumerate(examples):
        for col,(key,label,cmap,limit) in enumerate([('depth','Observed depth (m)','viridis',None),('error','Target normal error (deg)','magma',(0,50)),('sensitivity','Scale disagreement (deg)','magma',(0,50)),('weight','Proposed weight','viridis',(0,1))]):
            a=axes[row,col]; values=np.where(example['mask'],example[key],np.nan)
            im=a.imshow(values,cmap=cmap,**({'vmin':limit[0],'vmax':limit[1]} if limit else {}))
            a.set_title(label if row==0 else '',fontsize=10);a.set_xticks([]);a.set_yticks([])
            if col==0:a.set_ylabel(example['name'],fontsize=10)
            fig.colorbar(im,ax=a,shrink=.7)
    fig.suptitle('Synthetic component test: label quality, not model accuracy',fontsize=13)
    (OUT/'figures').mkdir(parents=True,exist_ok=True);fig.savefig(OUT/'figures/synthetic_examples.png',dpi=150);plt.close(fig)


def real_training():
    verify_protocol();torch.set_num_threads(2)
    manifest=read(V7OUT/'manifest.json');ray=rays(480,640,read(V7OUT/'camera.json')['intrinsics'])
    rows=manifest['training']; records=[]; cache=WORK/'weights';cache.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    for i,row in enumerate(rows):
        path=ASSETS/'subset'/row['file'];derived=V7WORK/'derived'/row['file']
        assert sha(path)==row['source_sha256'] and sha(derived)==row['derived_sha256']
        source=sample(path);gt=sample(derived);n,u,w=stability(source['depth'],ray)
        assert np.max(abs(n-gt['normal']))<1e-6
        native,support=native_weights(w,gt['normal_valid'])
        original=torch.nn.functional.interpolate(torch.from_numpy(gt['normal_valid']).float()[None,None],size=(192,256),mode='area')>.9
        assert torch.equal(support,original)
        dest=cache/row['file'];np.savez_compressed(dest,weights=native.numpy(),mask=support.numpy())
        valid=gt['normal_valid']; grad=np.hypot(*np.gradient(source['depth']))
        cutoff=float(np.quantile(grad[valid],.9));high=valid&(grad>=cutoff)
        low=valid&~high
        records.append({'id':row['id'],'scene':row['scene'],'source_sha256':sha(path),'derived_sha256':sha(derived),'weight_sha256':sha(dest),
            'full_valid_pixels':int(valid.sum()),'native_valid_pixels':int(support.sum()),
            'sensitivity_mean_deg':float(u[valid].mean()),'fullres':weight_stats(w[valid]),'native':weight_stats(native[support].numpy()),
            'high_depth_gradient_mean_weight':float(w[high].mean()),'other_mean_weight':float(w[low].mean())})
        if (i+1)%16==0:print('TRAINING_WEIGHT_AUDIT',i+1,len(rows),flush=True)
    write(OUT/'training_audit.json',{'protocol_sha256':sha(OUT/'component_protocol.json'),
        'scope':'128 training scenes only; no normal-truth accuracy claims; no validation or test images accessed',
        'seconds':time.perf_counter()-start,'records':records})


def analyze():
    p=verify_protocol(); syn=read(OUT/'synthetic.json'); real=read(OUT/'training_audit.json')
    held=[r for r in syn['records'] if r['split']=='heldout']
    groups={}
    keys=['uniform_error_deg','weighted_error_deg','shuffled_error_deg','effective_fraction','mean_weight']
    for corruption in CORRUPTIONS:
        rows=[r['metrics']['all'] for r in held if r['corruption']==corruption]
        groups[corruption]={k:float(np.mean([r[k] for r in rows])) for k in keys}
    noisy=[r['metrics']['all'] for r in held if r['corruption'] in p['gate']['heldout_noise_scope']]
    uniform=np.mean([r['uniform_error_deg'] for r in noisy]);weighted=np.mean([r['weighted_error_deg'] for r in noisy]);shuffled=np.mean([r['shuffled_error_deg'] for r in noisy])
    clean={kind:float(np.mean([r['metrics']['all']['weighted_error_deg']-r['metrics']['all']['uniform_error_deg'] for r in held if r['corruption']=='clean' and r['surface']==kind])) for kind in SURFACES}
    goodfamilies=sum(groups[c]['weighted_error_deg']<=groups[c]['uniform_error_deg'] for c in p['gate']['heldout_noise_scope'])
    checks={'noise_relative_reduction':{'value':float(1-weighted/uniform),'required':'>=0.10','pass':bool(weighted<=.9*uniform)},
        'shuffled_relative_reduction':{'value':float(1-weighted/shuffled),'required':'>=0.05','pass':bool(weighted<=.95*shuffled)},
        'noise_families_nonworse':{'value':int(goodfamilies),'required':'>=3 of4','pass':bool(goodfamilies>=3)},
        'synthetic_median_effective_fraction':{'value':float(np.median([r['effective_fraction'] for r in noisy])),'required':'>=0.25','pass':bool(np.median([r['effective_fraction'] for r in noisy])>=.25)},
        'clean_surface_max_increase_deg':{'value':float(max(clean.values())),'required':'<=1.0','pass':bool(max(clean.values())<=1)},
        'real_scene_fraction_effective_ge_0_25':{'value':float(np.mean([r['native']['effective_fraction']>=.25 for r in real['records']])),'required':'>=0.90','pass':bool(np.mean([r['native']['effective_fraction']>=.25 for r in real['records']])>=.9)}}
    gate=all(v['pass'] for v in checks.values())
    old=read(WORK/'historical_hashes.json');changed=[n for n,v in old.items() if sha(ROOT/n)!=v]
    assert not changed,changed
    result={'protocol_sha256':sha(OUT/'component_protocol.json'),'synthetic_cases':len(syn['records']),
      'heldout_cases':len(held),'training_scenes':len(real['records']),'heldout_macro_by_corruption':groups,
      'checks':checks,'engineering_gate_passed':gate,'clean_surface_differences':clean,
      'real_native_mean_weight':float(np.mean([r['native']['mean_weight'] for r in real['records']])),
      'real_native_median_effective_fraction':float(np.median([r['native']['effective_fraction'] for r in real['records']])),
      'historical_files_unchanged':len(old),'training_audit_sha256':sha(OUT/'training_audit.json'),'synthetic_sha256':sha(OUT/'synthetic.json'),
      'decision':'Proceed to bounded integration and stable single-task pilot; full model study remains separately gated.' if gate else 'Do not launch full weighted model ablation. Diagnose failed proxy gate in a new version; preserve these results.',
      'limitations':['Synthetic weighted label error is not trained-model accuracy.','Heldout seeds share the same synthetic surface families and corruption generator.',
                    'No independent real-world normal ground truth.','No proof that label uncertainty caused V7 negative transfer.',
                    'Engineering thresholds are not significance tests or Holm-adjusted evidence.','Clean complex geometry and smooth bias must be reported even when noise averages improve.']}
    write(OUT/'analysis.json',result)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained');x=np.arange(len(groups))
    for j,(key,label) in enumerate([('uniform_error_deg','Uniform'),('weighted_error_deg','Scale-sensitive'),('shuffled_error_deg','Shuffled')]):
        ax.bar(x+(j-1)*.24,[v[key] for v in groups.values()],.24,label=label)
    ax.set_xticks(x,[c.replace('_','\n') for c in groups]);ax.set_ylabel('Selected supervision error (degrees)');ax.legend();ax.set_title('Heldout synthetic cases: not model prediction results')
    fig.savefig(OUT/'figures/proxy_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    axes[0].hist([r['native']['effective_fraction'] for r in real['records']],bins=20,color='#167D9A');axes[0].set_xlabel('Effective pixel fraction');axes[0].set_ylabel('Training scenes')
    axes[1].scatter([r['other_mean_weight'] for r in real['records']],[r['high_depth_gradient_mean_weight'] for r in real['records']],s=16,alpha=.7)
    axes[1].plot([0,1],[0,1],'--',color='gray');axes[1].set_xlabel('Other valid pixels: mean weight');axes[1].set_ylabel('High-depth-gradient pixels: mean weight')
    fig.suptitle('Training-only weight coverage: no real-label correctness claim');fig.savefig(OUT/'figures/real_weight_coverage.png',dpi=150);plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','synthetic','real','analyze']);args=a.parse_args()
    {'freeze':freeze,'synthetic':synthetic,'real':real_training,'analyze':analyze}[args.action]()


if __name__=='__main__':main()
