"""Statistics fixed before external inference; all contrasts always reported."""
import numpy as np
from robustness_stats import CONTRASTS,bootstrap,holm,intervals
from crossed_bootstrap_sensitivity import crossed_bootstrap

def broad_location_bootstrap(delta,locations,replicates=20000,seed=4274,family=6):
    """Resample whole proxy locations, preserving equal weight per sampled scene."""
    delta=np.asarray(delta,dtype=np.float64)
    d,s,n=delta.shape;assert len(locations)==n
    groups=sorted(set(locations));index=np.array([groups.index(x) for x in locations]);ng=len(groups)
    rng=np.random.default_rng(seed);samples=[]
    for start in range(0,replicates,250):
        b=min(250,replicates-start)
        wd=rng.multinomial(d,np.ones(d)/d,size=b)/d
        ws=rng.multinomial(s,np.ones(s)/s,size=b)/s
        counts=rng.multinomial(ng,np.ones(ng)/ng,size=b)
        wi=counts[:,index].astype(np.float64);wi/=wi.sum(axis=1,keepdims=True)
        values=np.einsum('bd,bs,dsi->bi',wd,ws,delta,optimize=True)
        samples.extend(np.einsum('bi,bi->b',wi,values))
    result=intervals(float(delta.mean()),samples,family)
    result['location_groups']=ng
    return result

def analyze(data,locations):
    rows=[]
    for compared,reference in CONTRASTS:
        delta=data[compared]-data[reference]
        res=bootstrap(delta)
        rows.append({'compared':compared,'reference':reference,**res,
                     'crossed':crossed_bootstrap(delta),
                     'location':broad_location_bootstrap(delta,locations)})
    for method in ['scene','hierarchical','crossed','location']:
        for row,p in zip(rows,holm([r[method]['p_bootstrap'] for r in rows])):row[method]['p_holm']=p
    for row in rows:
        row['primary_improvement_supported']=bool(row['hierarchical']['difference']<0 and row['hierarchical']['p_holm']<.05 and row['crossed']['p_holm']<.05)
        row['location_sensitivity_supports_improvement']=bool(row['location']['difference']<0 and row['location']['p_holm']<.05)
    return rows
