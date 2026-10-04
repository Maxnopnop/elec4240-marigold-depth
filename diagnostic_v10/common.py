import datetime
import numpy as np
import torch
from multitask_v7.common import ROOT, WORK as V7WORK, OUT as V7OUT, ASSETS, read, write, sha, rays
from training_v9.protocol import WORK as V9WORK, OUT as V9OUT, verify as verify_v9
from reliability_v8.study import scene, corrupt, SURFACES
from reliability_v8.core import stability, original_support, angles

OUT = ROOT / 'results/diagnostic_v10'
WORK = V7WORK.parent / 'diagnostic_v10'
CONDITIONS = ['clean', 'heterogeneous_noise', 'smooth_bias']
RULES = ['uniform', 'weighted', 'shuffled', 'oracle']
SEEDS = [17, 29]


def freeze():
    path = OUT / 'protocol.json'
    if path.exists():
        return verify()
    verify_v9('c')
    sources = {str(f.relative_to(ROOT)).replace('\\', '/'): sha(f)
               for f in sorted((ROOT / 'diagnostic_v10').glob('*.py'))}
    p = dict(version='diagnostic_v10_v1', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             interpretation='Post-V9 exploratory mechanism study. No new confirmatory p-values or significance-driven expansion.',
             regional=dict(scenes=32, seeds=[17,29,43], methods=['depth','normal','joint','uniform','weaker_uniform','weighted','shuffled'],
                           masks='Original task mask; per-image sensitivity quartiles and depth-gradient top10%; distance <2,2-4,>=4m. No learned selection.',
                           scope='Validation weights are computed for post-hoc interpretation only, never used to train V9.'),
             gradient=dict(states=['initial_seed17']+[f'{m}_seed{s}' for m in ['uniform','weighted'] for s in [17,29,43]],
                           training_indices=[0,32,64,96], noise_seeds=[91001,91002], optimizer_updates=0,
                           losses=['depth','normal','uniform_geometry','weighted_geometry'], geometry_coefficient=.1,
                           scope='Gradients on shared LoRA parameters, fixed training images. Cosine and norms are diagnostics, not causal proof.'),
             noise=dict(conditions=CONDITIONS, rules=RULES, seeds=SEEDS, train_scene_seeds=[201,211,223,227],
                        evaluation_scene_seeds=[401,409], surfaces=SURFACES, train_images=16, evaluation_images=8,
                        steps=64, runs=24, geometry_coefficient=.1, lr=1e-4, weight_decay=.01,
                        start='Fresh original Marigold plus new seed-matched rank4 LoRA, not a selected V9 checkpoint.',
                        labels='Both depth and derived sigma1 normals come from corrupted training depth. RGB always rendered from clean geometry.',
                        mask='Clean depth jump/erosion support and10px border fixed across corruptions/rules; differs intentionally from noisy-mask V8 component test.',
                        oracle='exp(-angular_error(noisy_sigma1_normal,analytic_clean_normal)/10); diagnostic privileged weight, not a deployable method or guaranteed upper bound.',
                        rule_scope='Only decoded geometry consistency is weighted. Supervised latent depth/normal objectives remain unweighted.',
                        evaluation='Clean analytic depth and normals; full480x640, identical fixed masks, no GT alignment; fixed final64step checkpoint.',
                        rendering='Lambertian shading plus fixed image-coordinate albedo and depth attenuation. Simplified synthetic domain, not photorealism.',
                        limitation='Only2training seeds and64updates. New generator seeds share the same surface families; no real-world generalization/convergence claim.'),
             source_sha256=sources,
             inputs={str((V9OUT/'protocol_c.json').relative_to(ROOT)).replace('\\','/'):sha(V9OUT/'protocol_c.json'),
                     str((V9OUT/'analysis_c.json').relative_to(ROOT)).replace('\\','/'):sha(V9OUT/'analysis_c.json')})
    historical={str(f.relative_to(ROOT)).replace('\\','/'):sha(f) for d in [ROOT/'results',ROOT/'reports']
                for f in d.rglob('*') if f.is_file() and OUT not in f.parents}
    write(WORK/'historical_hashes.json',historical)
    write(path,p)
    print('FROZEN_V10',sha(path),flush=True)
    return p


def verify():
    verify_v9('c')
    p=read(OUT/'protocol.json')
    for name,value in {**p['source_sha256'],**p['inputs']}.items():
        assert sha(ROOT/name)==value,name
    return p


def render(depth, normal):
    h,w=depth.shape
    yy,xx=np.meshgrid(np.linspace(0,1,h),np.linspace(0,1,w),indexing='ij')
    light=np.array([-.35,-.4,-1.]);light/=np.linalg.norm(light)
    shade=.3+.7*np.maximum(np.einsum('chw,c->hw',normal,light),0)
    albedo=np.stack([.6+.15*np.sin(xx*9),.55+.1*np.cos(yy*8),.5+.12*np.sin((xx+yy)*7)])
    color=albedo*shade[None]*np.exp(-.06*depth)[None]
    return np.moveaxis(np.round(np.clip(color,0,1)*255).astype(np.uint8),0,-1)


def synthetic_case(kind, seed, condition, ray):
    depth,normal,edge=scene(kind,seed,ray)
    observed,valid=corrupt(depth,condition,seed,ray)
    target,sensitivity,weight=stability(observed,ray)
    mask=original_support(depth,np.ones_like(valid))
    mask[:10]=False;mask[-10:]=False;mask[:,:10]=False;mask[:,-10:]=False
    oracle=np.exp(-angles(target,normal)/10).astype(np.float32)
    return dict(image=render(depth,normal),depth=depth,normal=normal,observed=observed,
                target_normal=target,weight=weight,oracle=oracle,mask=mask,sensitivity=sensitivity)


def vector_stats(vectors):
    v={name:np.asarray(x,dtype=np.float64) for name,x in vectors.items()}
    v['supervised']=.5*(v['depth']+v['normal'])
    norms={name:float(np.linalg.norm(x)) for name,x in v.items()}
    assert all(np.isfinite(list(norms.values())))
    def cosine(a,b):
        denominator=norms[a]*norms[b]
        return float(np.dot(v[a],v[b])/denominator) if denominator>1e-15 else None
    return dict(norms=norms,cosines={f'{a}__{b}':cosine(a,b) for a,b in [
        ('depth','normal'),('supervised','uniform_geometry'),('supervised','weighted_geometry'),
        ('depth','uniform_geometry'),('depth','weighted_geometry'),('normal','uniform_geometry'),
        ('normal','weighted_geometry'),('uniform_geometry','weighted_geometry')]},
        geometry_to_supervised_ratio={m:.1*norms[m+'_geometry']/max(norms['supervised'],1e-15)
                                      for m in ['uniform','weighted']})
