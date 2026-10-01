"""Separate exploratory eight-comparison family; no historical artifacts edited."""
from pathlib import Path
import numpy as np
from run_expanded import read,write
from run_final_control import OUT,SEEDS
from robustness_stats import CONTRASTS,bootstrap,holm
from crossed_bootstrap_sensitivity import crossed_bootstrap
from prospective_statistics import broad_location_bootstrap

def arrays(stage):
    n=200 if stage=='external' else 31
    data={}
    for mode in ['base','high512','mixed','low256']:
      for res in [256,512]:
        label=f'{mode}_r{res}';a=np.zeros((1,1,n)) if mode=='base' else np.zeros((3,5,n))
        for d in range(a.shape[0]):
          for s in range(a.shape[1]):
            rid='base' if mode=='base' else f'draw{d+1}_{mode}_seed{SEEDS[s]}'
            condition=rid+f'_r{res}'
            if mode=='low256':p=OUT/'evaluations'/stage/condition/'per_image.json'
            elif stage=='external':p=Path('results/prospective_v4/evaluations')/condition/'per_image.json'
            else:p=Path('results/robustness_v3/evaluations/test')/condition/'per_image.json'
            records=read(p)
            if mode!='low256' and stage!='external':records=[r for r in records if r['cohort']=='fresh31']
            assert len(records)==n;a[d,s]=[r['abs_rel'] for r in records]
        data[label]=a
    return data

def main():
    assert read(OUT/'verification.json')['status']=='passed'
    locations=[r['location_proxy'] for r in read('results/prospective_v4/manifest.json')['samples']]
    for stage in ['nyu31','external']:
        data=arrays(stage);rows=[]
        for c,r in CONTRASTS+[(f'mixed_r{x}',f'low256_r{x}') for x in [256,512]]:
            delta=data[c]-data[r]
            row={'compared':c,'reference':r,**bootstrap(delta,family=8),'crossed':crossed_bootstrap(delta,family=8)}
            if stage=='external':row['location']=broad_location_bootstrap(delta,locations,family=8)
            rows.append(row)
        for kind in ['hierarchical','crossed']+(['location'] if stage=='external' else []):
            for r,p in zip(rows,holm([x[kind]['p_bootstrap'] for x in rows])):r[kind]['p_holm8']=p
        interaction=(data['mixed_r256']-data['base_r256'])-(data['mixed_r512']-data['base_r512'])
        write(OUT/f'{stage}_comparisons.json',rows)
        write(OUT/f'{stage}_interaction.json',{'interpretation':'post-hoc exploratory difference of paired differences; separate single diagnostic, not original confirmation','crossed':crossed_bootstrap(interaction,family=1)})
        write(OUT/f'{stage}_summary.json',[{'group':g,'abs_rel':float(v.mean()),'draw_means':v.mean(axis=(1,2)).tolist()} for g,v in data.items()])
    print('EXPLORATORY_STATISTICS_COMPLETE',flush=True)

if __name__=='__main__':main()
