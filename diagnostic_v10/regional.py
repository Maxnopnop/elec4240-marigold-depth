import numpy as np
import torch
from multitask_v7.common import sample
from training_v9.analyze import normal_errors
from reliability_v8.core import stability
from .common import *


def main():
    p=verify();torch.set_num_threads(2)
    manifest=read(V7OUT/'manifest.json');camera=read(V7OUT/'camera.json')['intrinsics']
    ray=rays(480,640,camera);records=[];checked=0
    for row in manifest['validation']:
        source=sample(ASSETS/'subset'/row['file']);target=sample(V7WORK/'derived'/row['file'])
        assert sha(ASSETS/'subset'/row['file'])==row['source_sha256']
        assert sha(V7WORK/'derived'/row['file'])==row['derived_sha256']
        _,sensitivity,weight=stability(source['depth'],ray)
        gradient=np.hypot(*np.gradient(source['depth']))
        for task in ['depth','normal']:
            valid=target[task+'_valid'];lo,hi=np.quantile(sensitivity[valid],[.25,.75]);g=np.quantile(gradient[valid],.9)
            regions={'all':valid,'stable_quartile':valid&(sensitivity<=lo),'sensitive_quartile':valid&(sensitivity>=hi),
                     'high_gradient':valid&(gradient>=g),'other_gradient':valid&(gradient<g),
                     'near_under2m':valid&(source['depth']<2),'middle2to4m':valid&(source['depth']>=2)&(source['depth']<4),
                     'far_over4m':valid&(source['depth']>=4)}
            methods=[task]+['joint','uniform','weaker_uniform','weighted','shuffled']
            for mode in methods:
                for seed in [17,29,43]:
                    path=V9WORK/'predictions'/f'{mode}_seed{seed}'/'step1280'/f"{row['id']:04d}_{task}.npy"
                    old=read(V9OUT/'runs'/f'{mode}_seed{seed}'/'validation_1280.json')
                    expected=next(x for x in old if x['id']==row['id'])['tasks'][task]['prediction_sha256']
                    assert sha(path)==expected
                    pred=np.load(path);checked+=1
                    error=abs(np.clip(pred[0].astype(float),.1,10)-source['depth'])/source['depth'] if task=='depth' else normal_errors(pred,target['normal'])
                    stats={name:dict(pixels=int(mask.sum()),mean=float(error[mask].mean()),p90=float(np.quantile(error[mask],.9)))
                           for name,mask in regions.items() if mask.sum()>=100}
                    records.append(dict(id=row['id'],scene=row['scene'],task=task,mode=mode,seed=seed,regions=stats))
        print('REGIONAL',row['id'],checked,flush=True)
    contrasts=[]
    for task in ['depth','normal']:
        for reference in ['uniform','weaker_uniform','shuffled','joint',task]:
            for region in regions:
                values=[]
                for row in manifest['validation']:
                    differences=[]
                    for seed in [17,29,43]:
                        def pick(mode):
                            return next(x for x in records if x['id']==row['id'] and x['task']==task and x['seed']==seed and x['mode']==mode)['regions'].get(region)
                        a,b=pick('weighted'),pick(reference)
                        if a and b:differences.append(a['mean']-b['mean'])
                    if differences:values.append(float(np.mean(differences)))
                if values:contrasts.append(dict(task=task,reference=reference,region=region,scene_count=len(values),
                                               difference=float(np.mean(values)),improved_scenes=int(np.sum(np.array(values)<0)),
                                               scene_differences=values))
    write(OUT/'regional.json',dict(records=records,contrasts=contrasts,verified_predictions=checked,
          interpretation='Descriptive post-hoc regions; scene means averaged equally after seed averaging. Pixel p90 describes within-scene tails, not uncertainty. No new p-values.'))


if __name__=='__main__':main()
