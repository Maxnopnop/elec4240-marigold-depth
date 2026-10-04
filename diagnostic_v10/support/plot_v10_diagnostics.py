import sys
sys.path.insert(0,r'E:\Codex\2026-09-27\yo\outputs\marigold-depth')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnostic_v10.common import OUT,read,write,sha

regional=read(OUT/'regional.json');gradient=read(OUT/'gradients.json')
fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
for ax,task,regions in [(axes[0,0],'depth',['near_under2m','middle2to4m','far_over4m']),
                       (axes[0,1],'normal',['stable_quartile','sensitive_quartile'])]:
    labels=[]
    for index,region in enumerate(regions):
        row=next(r for r in regional['contrasts'] if r['reference']=='uniform' and r['task']==task and r['region']==region)
        values=row['scene_differences'];offset=np.linspace(-.15,.15,len(values))
        ax.scatter(index+offset,values,s=22,alpha=.65,color='#167d9a')
        ax.plot([index-.22,index+.22],[np.mean(values)]*2,color='#c46020',linewidth=3)
        labels.append(region.replace('_','\n')+f'\nn={len(values)}')
    ax.axhline(0,color='black',linewidth=1);ax.set_xticks(range(len(labels)),labels)
    ax.set_ylabel('Weighted - uniform: '+('AbsRel' if task=='depth' else 'degrees'))
    ax.set_title(task+': each dot is one scene, averaged over3seeds')
states=list(dict.fromkeys(r['state'] for r in gradient['records']))
for ax,keys,title in [(axes[1,0],['supervised__uniform_geometry','supervised__weighted_geometry'],'Supervised / geometry gradient cosine'),
                      (axes[1,1],['uniform_geometry__weighted_geometry'],'Uniform / weighted geometry gradient cosine')]:
    for j,key in enumerate(keys):
        means=[]
        for i,state in enumerate(states):
            values=[r['cosines'][key] for r in gradient['records'] if r['state']==state]
            offset=(j-(len(keys)-1)/2)*.22
            ax.scatter(np.full(len(values),i+offset),values,s=18,alpha=.45)
            means.append(np.mean(values))
        ax.plot(np.arange(len(states))+(j-(len(keys)-1)/2)*.22,means,'o-',label=key.replace('__',' / ').replace('_geometry',''))
    ax.axhline(0,color='black',linewidth=1);ax.set_ylim(-1.05,1.05);ax.set_title(title)
    ax.set_xticks(range(len(states)),[s.replace('_seed','\n') for s in states],rotation=30,ha='right',fontsize=8)
    ax.legend(fontsize=8,loc='lower right')
fig.suptitle('Exploratory mechanism evidence: scene heterogeneity and local gradient alignment\nOrange horizontal lines in top panels are means; no regional significance tests')
path=OUT/'figures/mechanism_detail.png';path.parent.mkdir(exist_ok=True);fig.savefig(path,dpi=150);plt.close(fig)
write(OUT/'mechanism_detail_provenance.json',dict(regional_sha256=sha(OUT/'regional.json'),gradients_sha256=sha(OUT/'gradients.json'),figure_sha256=sha(path),
      script_sha256=sha(__file__),scope='Descriptive plot of already completed fixed diagnostics. BF16 primary probes include one numerically flagged probe with separate FP32 reference.'))
print(path)
