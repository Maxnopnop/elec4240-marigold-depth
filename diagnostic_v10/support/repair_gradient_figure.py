"""Presentation-only fix: shorten the cropped gradient-axis label."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnostic_v10.common import OUT,read,write,sha

a=read(OUT/'analysis.json');rows=a['gradients'];x=np.arange(len(rows))
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
axes[0].bar(x,[r['depth_normal_cosine'] for r in rows]);axes[0].axhline(0,color='black',linewidth=1)
axes[0].set_ylabel('Depth / normal gradient cosine')
for j,mode in enumerate(['uniform','weighted']):
    axes[1].bar(x+(j-.5)*.3,[r[mode+'_ratio'] for r in rows],width=.3,label=mode)
axes[1].set_ylabel('Gradient norm ratio\n(0.1 geometry / supervision)');axes[1].legend()
for ax in axes:
    ax.set_xticks(x,[r['state'].replace('_seed','\nseed') for r in rows],rotation=30,ha='right',fontsize=10)
fig.suptitle('Fixed training-image probes: eight per state, zero optimizer updates')
path=OUT/'figures/gradients.png';fig.savefig(path,dpi=150);plt.close(fig)
write(OUT/'figure_layout_check.json',dict(analysis_sha256=sha(OUT/'analysis.json'),figure_sha256=sha(path),source_sha256=sha(__file__),
                                      change='Presentation only: shorter wrapped y-axis label and taller canvas; plotted values unchanged.'))
