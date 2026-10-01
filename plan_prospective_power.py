"""Prospective planning from pilot paired data; no external outcomes are read.

Known-pilot-distribution approximation: simulate joint centered crossed-bootstrap
errors for four base contrasts; derive all six coherent contrasts, then apply
two-sided empirical tail tests and Holm in every simulated trial. Both nested
and crossed calibration distributions must reject, as in the intended analysis.
Independent calibration and trial resamples, each 20,000. This does NOT re-estimate a bootstrap
distribution inside every future trial, and does NOT guarantee actual test power.
"""
from pathlib import Path
import json, csv, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('results/robustness_v3')
OUT = Path('results/prospective_v4/power')
SEEDS = [17, 29, 43, 59, 71]
CONTRASTS = ['mixed256-base256', 'mixed512-base512', 'high256-base256', 'high512-base512', 'mixed256-high256', 'mixed512-high512']
TRANSFORM = np.vstack([np.eye(4), [1,0,-1,0], [0,1,0,-1]])

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def pilot(cohort):
    source = {}
    def values(label):
        p = ROOT/'evaluations/test'/label/'per_image.json'
        source[str(p)] = sha(p)
        rows = [r for r in read(p) if r['cohort'] == cohort]
        return np.array([r['abs_rel'] for r in rows])
    base = {r: values(f'base_r{r}') for r in [256,512]}
    x = np.empty((3,5,len(base[256]),4))
    for d in range(3):
        for s, seed in enumerate(SEEDS):
            for c, (mode,res) in enumerate([('mixed',256),('mixed',512),('high512',256),('high512',512)]):
                x[d,s,:,c] = values(f'draw{d+1}_{mode}_seed{seed}_r{res}')-base[res]
    return x, source

def errors(x, nd, ns, ni, rng, count=20000, kind='crossed'):
    # Multinomial weights exactly implement resampling each crossed factor.
    d,s,i,c = x.shape
    centered = x-x.mean(axis=(0,1,2),keepdims=True)
    result = []
    for start in range(0,count,500):
        b = min(500,count-start)
        if kind == 'crossed':
            wd = rng.multinomial(nd,np.ones(d)/d,size=b)/nd
            ws = rng.multinomial(ns,np.ones(s)/s,size=b)/ns
            weights = wd[:,:,None]*ws[:,None,:]
        else:
            draw_ids = rng.integers(d,size=(b,nd))
            seed_counts = rng.multinomial(ns,np.ones(s)/s,size=(b,nd))
            weights = np.zeros((b,d,s))
            for j in range(nd):
                np.add.at(weights,(np.arange(b),draw_ids[:,j]),seed_counts[:,j])
            weights /= nd*ns
        wi = rng.multinomial(ni,np.ones(i)/i,size=b)/ni
        intermediate = np.einsum('bds,dsic->bic',weights,centered,optimize=True)
        result.append(np.einsum('bi,bic->bc',wi,intermediate,optimize=True)@TRANSFORM.T)
    return np.concatenate(result)

def holm_rows(p):
    order = np.argsort(p,axis=1)
    ordered = np.take_along_axis(p,order,axis=1)
    adjusted = np.minimum(1,np.maximum.accumulate(ordered*np.arange(6,0,-1),axis=1))
    result = np.empty_like(p)
    np.put_along_axis(result,order,adjusted,axis=1)
    return result

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    rows, diagnostics, sources = [], [], {}
    sizes = [31,64,100,150,200,250,300,500,750,1000]
    for cohort_index, cohort in enumerate(['fresh31','observed64']):
        x, files = pilot(cohort); sources.update(files)
        original_mean = x.mean(axis=(0,1,2))
        for nd in [3,5,10]:
            for ni in sizes:
                rng = np.random.default_rng(4270000+cohort_index*100000+nd*1000+ni)
                calibration = errors(x,nd,5,ni,rng)
                nested_calibration = errors(x,nd,5,ni,rng,kind='nested')
                trials = errors(x,nd,5,ni,rng)
                for inflation in [1.,1.5]:
                    sorted_nulls = [np.sort(abs(c*inflation),axis=0) for c in [calibration,nested_calibration]]
                    for effect in [0.,.006,.009,.012]:
                        mean = original_mean*(effect/abs(original_mean[1]))@TRANSFORM.T
                        estimates = mean+trials*inflation
                        adjusted_methods = []
                        for sorted_null in sorted_nulls:
                            p = np.column_stack([(1+len(calibration)-np.searchsorted(sorted_null[:,j],abs(estimates[:,j]),side='left'))/(len(calibration)+1) for j in range(6)])
                            adjusted_methods.append(holm_rows(p))
                        adjusted = np.maximum(*adjusted_methods)
                        hit = (adjusted < .05)&(estimates < 0)
                        power = float(hit[:,1].mean())
                        rows.append({'pilot_cohort':cohort,'training_draws':nd,'training_seeds':5,'new_scenes':ni,
                                     'target_absrel_improvement':effect,'error_sd_multiplier':inflation,
                                     'target_power':power,'monte_carlo_se':float(np.sqrt(power*(1-power)/len(trials))),
                                     'family_any_rejection':float((adjusted.min(axis=1)<.05).mean()),
                                     **{f'power_{j}':float(hit[:,j].mean()) for j in range(6)}})
                diagnostics.append({'pilot':cohort,'draws':nd,'scenes':ni,'target_standard_error':float(calibration[:,1].std(ddof=1))})
            print('POWER_GRID_COMPLETE',cohort,'draws',nd,flush=True)
    metadata = {'method':__doc__,'source_sha256':sha(Path(__file__)), 'pilot_source_sha256':sources,
                'contrasts':CONTRASTS,'simulation_replicates_per_grid':20000,
                'effect_assumption':'All four base contrasts scale proportionally to the target mixed512 effect; no incoherent independent contrast shifts. Future mean errors follow crossed-factor pilot resampling; both nested and crossed test branches must reject.',
                'limits':['Only 3 observed training draws and 5 seeds; resampling cannot discover unobserved variability.',
                          'Pilot distribution reused as known planning distribution; actual future bootstrap test re-estimates uncertainty.',
                          'No external outcomes enter planning; domain shift can change means, variance, tails and correlations.',
                          'Candidate counts are distinct sampling units, not repeated frames from one room.',
                          'Monte Carlo SE is simulation precision only, not uncertainty in estimated population power.'],
                'minimum_candidates':[],'diagnostics':diagnostics}
    for cohort in ['fresh31','observed64']:
        for nd in [3,5,10]:
            for inflation in [1.,1.5]:
                for effect in [.006,.009,.012]:
                    selected = [r for r in rows if r['pilot_cohort']==cohort and r['training_draws']==nd and r['error_sd_multiplier']==inflation and r['target_absrel_improvement']==effect]
                    metadata['minimum_candidates'].append({'pilot':cohort,'draws':nd,'sd_multiplier':inflation,'effect':effect,
                        **{f'minimum_n_power_{int(threshold*100)}':next((r['new_scenes'] for r in selected if r['target_power']>=threshold),None) for threshold in [.8,.9]}})
    (OUT/'design_and_results.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    with (OUT/'power_grid.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row, inflation in enumerate([1.,1.5]):
        for col, effect in enumerate([.006,.009,.012]):
            ax=axes[row,col]
            for nd in [3,5,10]:
                selected=[r for r in rows if r['pilot_cohort']=='fresh31' and r['training_draws']==nd and r['error_sd_multiplier']==inflation and r['target_absrel_improvement']==effect]
                ax.plot([r['new_scenes'] for r in selected],[r['target_power'] for r in selected],marker='.',label=f'{nd} draws x 5 seeds')
            ax.axhline(.8,ls='--',color='gray');ax.set_ylim(0,1)
            ax.set(title=f'Effect {effect:.3f} | SD x{inflation:g}',xlabel='New independent scenes',ylabel='Estimated rejection probability')
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Planning approximation from fresh31 | six-test Holm | unknown domain shifts remain')
    fig.savefig(OUT/'power_sensitivity.png',dpi=150);plt.close(fig)
    lines=['# Prospective sample-size planning','',__doc__.strip(),'',
           'The pilot cohorts are analyzed separately. We do not pool their outcomes or treat simulations as new observations. '
           'Power is estimated for a negative mixed512-minus-base512 effect after Holm adjustment of all six contrasts. '
           'The proposed effects are assumptions, not promised improvements. Each simulated trial uses the complete joint contrast family.','',
           '| Pilot | Training draws | Error SD multiplier | Assumed AbsRel gain | First grid N for 80% | First grid N for 90% |',
           '|---|---:|---:|---:|---:|---:|']
    for r in metadata['minimum_candidates']:
        lines.append(f"| {r['pilot']} | {r['draws']} | {r['sd_multiplier']} | {r['effect']} | {r['minimum_n_power_80'] or '>1000'} | {r['minimum_n_power_90'] or '>1000'} |")
    lines+=['','![Power sensitivity](power_sensitivity.png)','','## Limits','']+['- '+x for x in metadata['limits']]
    lines+=['','Sample size must be frozen before external inference. A finite accessible cohort may be smaller than the planning target; report that shortfall instead of claiming adequate power. '
            'Never stop upon significance or add scenes because the fixed test failed. No future significant result establishes universal superiority.','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('POWER_ANALYSIS_READY',len(rows),flush=True)

if __name__=='__main__': run()
