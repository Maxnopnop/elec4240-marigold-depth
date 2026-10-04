"""Independent saved-array verification and predeclared exploratory contrasts."""
import argparse,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multitask_v7.common import ROOT,ASSETS,WORK as V7WORK,OUT as V7OUT,read,write,sha,sample
from .protocol import OUT,WORK,SEEDS,JOINT,verify,gate


def depth_metrics(pred,gt,mask):
    raw=pred[mask].astype(np.float64);target=gt[mask].astype(np.float64);value=np.clip(raw,.1,10.)
    return {'abs_rel':float(np.mean(np.abs(value-target)/target)),'rmse_m':float(np.sqrt(np.mean((value-target)**2))),
        'delta1':float(np.mean(np.maximum(value/target,target/value)<1.25)),
        'prediction_outside_range_fraction':float(np.mean((raw<.1)|(raw>10.))),'valid_pixels':int(mask.sum())}


def normal_errors(pred,gt):
    p=pred.astype(np.float64);g=gt.astype(np.float64)
    p/=np.maximum(np.linalg.norm(p,axis=0,keepdims=True),1e-8)
    g/=np.maximum(np.linalg.norm(g,axis=0,keepdims=True),1e-8)
    return np.degrees(np.arccos(np.clip(np.einsum('ijk,ijk->jk',p,g),-1,1)))


def normal_metrics(pred,gt,mask):
    error=normal_errors(pred,gt)[mask]
    return {'mean_deg':float(error.mean()),'median_deg':float(np.median(error)),
        'within_11_25':float(np.mean(error<11.25)),'within_22_5':float(np.mean(error<22.5)),
        'within_30':float(np.mean(error<30)),'valid_pixels':int(mask.sum())}


def agreement(depth,normal,mask,camera):
    height,width=depth.shape;y,x=np.meshgrid(np.arange(1,height+1),np.arange(1,width+1),indexing='ij')
    ray=np.stack([(x-camera['cx_rgb'])/camera['fx_rgb'],(y-camera['cy_rgb'])/camera['fy_rgb'],np.ones_like(x)],axis=-1)
    points=depth.astype(np.float64)[...,None]*ray
    n=np.cross(points[1:-1,2:]-points[1:-1,:-2],points[2:,1:-1]-points[:-2,1:-1])
    n/=np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-8)
    n*=np.where(np.sum(n*points[1:-1,1:-1],axis=-1,keepdims=True)>0,-1.,1.)
    other=np.moveaxis(normal.astype(np.float64),0,-1)[1:-1,1:-1]
    other/=np.maximum(np.linalg.norm(other,axis=-1,keepdims=True),1e-8)
    angles=np.degrees(np.arccos(np.clip(np.sum(n*other,axis=-1),-1,1)))
    return float(angles[mask[1:-1,1:-1]].mean())


def holm(values):
    order=np.argsort(values);output=np.zeros(len(values));current=0.
    for rank,index in enumerate(order):
        current=max(current,(len(values)-rank)*values[index]);output[index]=min(1.,current)
    return output.tolist()


def bootstrap(differences,seed_index,scene_index):
    effect=float(differences.mean())
    resampled=differences[seed_index[:,:,None],scene_index[:,None,:]].mean(axis=(1,2))
    null=resampled-effect
    p=float((1+np.sum(abs(null)>=abs(effect)))/(len(null)+1))
    radius=float(np.quantile(abs(null),.95))
    return {'difference':effect,'unadjusted_symmetric_95_interval':[effect-radius,effect+radius],
            'raw_p_two_sided_centered':p,'negative_difference_favors_weighted':True}


def audit(stage):
    protocols={'b':verify('b')}
    if stage=='c':protocols['c']=verify('c')
    manifest=read(V7OUT/'manifest.json');rows=manifest['validation'];camera=read(V7OUT/'camera.json')['intrinsics']
    assert len({r['scene'] for r in rows})==32
    assert not {r['scene'] for r in rows}&{r['scene'] for r in manifest['training']}
    for row in manifest['training']+rows:
        assert sha(ASSETS/'subset'/row['file'])==row['source_sha256']
        assert sha(V7WORK/'derived'/row['file'])==row['derived_sha256']
    truth={r['id']:(sample(ASSETS/'subset'/r['file']),sample(V7WORK/'derived'/r['file'])) for r in rows}
    modes=['depth','normal']+(JOINT if stage=='c' else [])
    datasets={};training={};orders={};diagnostics={};checked=0;restored=0;checkpoints=0
    for mode in modes:
        s='b' if mode in ['depth','normal'] else 'c';p=protocols[s]
        for seed in SEEDS:
            name=f'{mode}_seed{seed}';dest=OUT/'runs'/name;local=WORK/'checkpoints'/name
            done=read(dest/'complete.json');stats=read(dest/'training.json');training[name]=stats
            assert done['protocol_sha256']==sha(OUT/f'protocol_{s}.json')
            assert done['training_sha256']==sha(dest/'training.json')
            assert len(stats['history'])==1280 and stats['trainable_parameters']==829952
            expected_tasks=[mode] if s=='b' else ['depth','normal']
            assert done['exact_restored_tasks']==expected_tasks and stats['tasks']==expected_tasks
            assert stats['task_examples']=={task:1280 for task in expected_tasks}
            restored+=len(expected_tasks)
            order=[r['id'] for r in stats['history']];orders[name]=order
            rng=np.random.default_rng(seed);expected=[]
            for _ in range(10):expected.extend([manifest['training'][i]['id'] for i in rng.permutation(128)[::-1]])
            assert order==expected
            for step in stats['history']:
                recomputed=np.mean(list(step['task_losses'].values()))+stats['geometry_weight']*step['geometry_loss']
                assert abs(recomputed-step['loss'])<2e-5 and np.isfinite(step['gradient_norm'])
            for step in p['evaluation_checkpoints']:
                records=read(dest/f'validation_{step}.json');mark=read(dest/f'checkpoint_{step}.json')
                assert [r['id'] for r in records]==[r['id'] for r in rows]
                assert sha(dest/f'validation_{step}.json')==done['metrics_sha256'][str(step)]==mark['metrics_sha256']
                assert sha(local/f'step{step}/adapter.pt')==mark['adapter_sha256']
                checkpoints+=1
                if step==1280:assert mark['adapter_sha256']==done['final_checkpoint_sha256']
                datasets[(mode,seed,step)]=records
                diag=[]
                for record in records:
                    original,derived=truth[record['id']];predictions={};entry={'id':record['id']}
                    for task,values in record['tasks'].items():
                        path=WORK/'predictions'/name/f'step{step}'/f"{record['id']:04d}_{task}.npy"
                        assert sha(path)==values['prediction_sha256']
                        pred=np.load(path);assert np.isfinite(pred).all();predictions[task]=pred
                        recomputed=depth_metrics(pred[0],original['depth'],derived['depth_valid']) if task=='depth' else normal_metrics(pred,derived['normal'],derived['normal_valid'])
                        if task=='depth':recomputed['raw_depth_abs_rel']=depth_metrics(pred[0],derived['raw_depth'],derived['raw_valid'])['abs_rel']
                        for key,value in recomputed.items():assert np.isclose(value,values[key],atol=1e-9,rtol=1e-10),(name,step,record['id'],key)
                        checked+=1
                        if step==1280:
                            mask=derived['depth_valid' if task=='depth' else 'normal_valid']
                            grad=np.hypot(*np.gradient(original['depth']));threshold=np.quantile(grad[mask],.9)
                            high=mask&(grad>=threshold);low=mask&~high
                            if task=='depth':error=abs(np.clip(pred[0].astype(np.float64),.1,10)-original['depth'])/np.maximum(original['depth'],.1)
                            else:error=normal_errors(pred,derived['normal'])
                            entry[task]={'high_gradient':float(error[high].mean()),'other_valid':float(error[low].mean())}
                    if step==1280 and len(predictions)==2:
                        entry['geometry_disagreement_deg']=agreement(predictions['depth'][0],predictions['normal'],derived['normal_valid'],camera)
                    if step==1280:diag.append(entry)
                if step==1280:diagnostics[name]=diag
            print('AUDIT_VERIFIED',name,'arrays',checked,flush=True)
    for seed in SEEDS:assert all(orders[f'{mode}_seed{seed}']==orders[f'depth_seed{seed}'] for mode in modes)
    historical=read(WORK/'historical_hashes.json')
    assert all(sha(ROOT/name)==value for name,value in historical.items())
    def matrix(mode,task,metric,step=1280):
        return np.array([[r['tasks'][task][metric] for r in datasets[mode,seed,step]] for seed in SEEDS])
    summary={}
    for mode in modes:
        summary[mode]={}
        for task in ([mode] if mode in ['depth','normal'] else ['depth','normal']):
            summary[mode][task]={}
            for metric in (['abs_rel','rmse_m','delta1','raw_depth_abs_rel'] if task=='depth' else ['mean_deg','median_deg','within_11_25','within_22_5','within_30']):
                m=matrix(mode,task,metric);means=m.mean(1)
                summary[mode][task][metric]={'mean':float(means.mean()),'seed_sd':float(means.std(ddof=1)),'seed_means':means.tolist()}
    comparisons=[]
    if stage=='c':
        rng=np.random.default_rng(934240);si=rng.integers(0,3,(20000,3));xi=rng.integers(0,32,(20000,32))
        for lhs,rhs,task,metric in protocols['c']['statistics']['family']:
            differences=matrix(lhs,task,metric)-matrix(rhs,task,metric)
            comparisons.append({'method':lhs,'reference':rhs,'task':task,'metric':metric,**bootstrap(differences,si,xi)})
        adjusted=holm([v['raw_p_two_sided_centered'] for v in comparisons])
        for row,value in zip(comparisons,adjusted):row['holm6_p']=value;row['passes_holm_0_05']=value<.05
    cost={mode:{'optimizer_seconds_total':sum(training[f'{mode}_seed{s}']['optimizer_seconds'] for s in SEEDS),
                'peak_allocated_mib':max(training[f'{mode}_seed{s}']['peak_allocated_mib'] for s in SEEDS)} for mode in modes}
    result={'stage':stage,'summary':summary,'comparisons':comparisons,'cost':cost,
        'verification':{'independently_recomputed_predictions':checked,'exact_restored_tasks':restored,'checkpoints':checkpoints,'data_pairs':160,
                        'history_steps':sum(len(v['history']) for v in training.values()),'historical_files_unchanged':len(historical)},
        'limitations':['All32evaluation scenes were previously observed.','One training draw and three seeds.','Normal targets are depth-derived, not independent ground truth.',
                       'Six contrasts form one exploratory Holm family. Intervals are unadjusted.','Nonsignificance does not establish equivalence or no harm.',
                       'Fixed1280updates do not prove convergence.','No new independent scenes or general superiority claim.']}
    write(OUT/f'analysis_{stage}.json',result);write(OUT/f'diagnostics_{stage}.json',diagnostics)
    return result


def figures(result):
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    curve=read(OUT/'baseline_gate.json')['learning_curves']
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,task in zip(axes,['depth','normal']):
        steps=[320,640,1280]
        for i,seed in enumerate(SEEDS):ax.plot(steps,[curve[task][str(s)]['seed_means'][i] for s in steps],'o-',label=f'Seed {seed}')
        ax.set_xlabel('Optimizer updates');ax.set_ylabel('Metric depth AbsRel' if task=='depth' else 'Normal mean error (deg)');ax.set_title(task+' only');ax.legend()
    fig.suptitle('Learning curves on previously observed development scenes');fig.savefig(OUT/'figures/learning_curves.png',dpi=150);plt.close(fig)
    if result['stage']=='c':
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for ax,task,key in zip(axes,['depth','normal'],['abs_rel','mean_deg']):
            modes=[m for m in result['summary'] if task in result['summary'][m]]
            vals=[result['summary'][m][task][key] for m in modes]
            ax.bar(range(len(modes)),[v['mean'] for v in vals],yerr=[v['seed_sd'] for v in vals],capsize=3,color=['#167d9a' if m!='weighted' else '#d58328' for m in modes])
            ax.set_xticks(range(len(modes)),[m.replace('_','\n') for m in modes],rotation=20);ax.set_ylabel(key);ax.set_title(task+' (lower is better)')
        fig.suptitle('Final1280-step development results; error bars are seed standard deviations')
        fig.savefig(OUT/'figures/final_metrics.png',dpi=150);plt.close(fig)


def main():
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['b','c'],required=True);args=a.parse_args()
    if args.stage=='b':gate()
    result=audit(args.stage);figures(result);print(result,flush=True)


if __name__=='__main__':main()
