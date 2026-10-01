"""Descriptive depth-range and boundary diagnostics on observed external scenes."""
import argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from run_expanded import read,write,sha
from run_final_control import ROOT,ASSETS,WORK,OUT,SEEDS
from prospective_metrics import valid_mask

def regions(gt):
    valid=valid_mask(gt);edge=np.zeros(gt.shape,bool)
    for dy,dx in [(1,0),(0,1)]:
        a=gt[dy:,dx:];b=gt[:gt.shape[0]-dy,:gt.shape[1]-dx]
        use=valid[dy:,dx:]&valid[:gt.shape[0]-dy,:gt.shape[1]-dx]
        jump=use&(abs(a-b)>.1)&(abs(a-b)>.05*np.minimum(a,b))
        edge[dy:,dx:]|=jump;edge[:gt.shape[0]-dy,:gt.shape[1]-dx]|=jump
    edge=binary_dilation(edge,iterations=2)&valid
    return {**{f'{lo:g}-{hi:g}m':valid&(gt>=lo)&(gt<hi) for lo,hi in [(.1,1),(1,2),(2,4),(4,10)]},'boundary':edge,'interior':valid&~edge}

def main():
    p=argparse.ArgumentParser();p.add_argument('--existing-only',action='store_true');p.add_argument('--examples-only',action='store_true');a=p.parse_args()
    rows=read('results/prospective_v4/manifest.json')['samples'];out=OUT/'failure';out.mkdir(parents=True,exist_ok=True)
    if a.examples_only:
        figdir=OUT/'figures';figdir.mkdir(exist_ok=True);plot_examples(rows,out,figdir);return
    for mode in ['base','mixed']+([] if a.existing_only else ['low256']):
      for res in [256,512]:
        group=f'{mode}_r{res}';dest=out/(group+'.json')
        if dest.exists():continue
        conditions=[f'base_r{res}'] if mode=='base' else [f'draw{d}_{mode}_seed{s}_r{res}' for d in [1,2,3] for s in SEEDS]
        prior=OUT if mode=='low256' else Path('results/prospective_v4');rawroot=WORK/'predictions/external' if mode=='low256' else ROOT/'prospective_v4/predictions'
        cal=read(OUT/'calibration.json') if mode=='low256' else read('results/robustness_v3/calibration.json')['conditions']
        records={c:read(prior/'evaluations'/('external' if mode=='low256' else '')/c/'per_image.json') for c in conditions}
        outputs=[]
        for i,row in enumerate(rows):
            path=ASSETS/'external_sun3d_v4'/row['file'];assert sha(path)==row['sha256']
            with np.load(path) as z:gt=z['depth'].astype(np.float64)
            masks=regions(gt);values={k:{'aligned':[],'calibrated':[]} for k,m in masks.items() if m.sum()>=100}
            for c in conditions:
                rec=records[c][i];assert rec['id']==row['id'];pred=np.load(rawroot/c/f"{row['id']:04d}.npy").astype(np.float64)
                aligned=np.clip(pred*rec['scale']+rec['shift'],.1,10)
                metric=np.clip(pred*cal[c]['scale']+cal[c]['shift'],.1,10)
                for k,v in values.items():
                    m=masks[k];v['aligned'].append(float(np.mean(abs(aligned[m]-gt[m])/gt[m])));v['calibrated'].append(float(np.mean(abs(metric[m]-gt[m])/gt[m])))
            outputs.append({'id':row['id'],'space':row['space'],'regions':{k:{'pixels':int(masks[k].sum()),**{m:float(np.mean(v)) for m,v in vals.items()}} for k,vals in values.items()}})
        summary={k:{'scenes':sum(k in r['regions'] for r in outputs),**{m:float(np.mean([r['regions'][k][m] for r in outputs if k in r['regions']])) for m in ['aligned','calibrated']}} for k in outputs[0]['regions'].keys()}
        # Include bins absent in the first frame as well.
        for k in sorted({k for r in outputs for k in r['regions']}):
            summary[k]={'scenes':sum(k in r['regions'] for r in outputs),**{m:float(np.mean([r['regions'][k][m] for r in outputs if k in r['regions']])) for m in ['aligned','calibrated']}}
        write(dest,{'group':group,'conditions':conditions,'summary':summary,'per_scene':outputs});print('FAILURE_GROUP_DONE',group,flush=True)
    if a.existing_only:return
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained');names=['0.1-1m','1-2m','2-4m','4-10m','boundary','interior']
    for ax,metric in zip(axes,['aligned','calibrated']):
        for mode,color in [('base','#64748b'),('low256','#059669'),('mixed','#d97706')]:
            data=read(out/f'{mode}_r512.json')['summary'];ax.plot(range(6),[data[k][metric] for k in names],'-o',label=mode,color=color)
        ax.set(xticks=range(6),xticklabels=names,ylabel='AbsRel',title=metric+' | inference 512');ax.tick_params(axis='x',rotation=30);ax.grid(alpha=.2);ax.legend(fontsize=10)
    fig.savefig(figdir/'failure_strata.png',dpi=180);plt.close(fig)
    plot_examples(rows,out,figdir)
    print('FAILURE_ANALYSIS_COMPLETE',flush=True)


def plot_examples(rows,out,figdir):
    base=read('results/prospective_v4/evaluations/base_r512/per_image.json');mixed=np.mean([[r['abs_rel'] for r in read(Path('results/prospective_v4/evaluations')/f'draw{d}_mixed_seed{s}_r512/per_image.json')] for d in [1,2,3] for s in SEEDS],axis=0)
    delta=mixed-np.array([r['abs_rel'] for r in base]);worst=np.argsort(-delta,kind='stable')[:3].tolist()
    write(out/'example_selection.json',{'rule':'three largest scene-mean mixed512-minus-base512 aligned errors; post-hoc error-selected','indices':worst,'ids':[rows[i]['id'] for i in worst],'differences':[float(delta[i]) for i in worst]})
    for selection,title,name in [(list(range(3)),'Fixed first three manifest examples','fixed_examples'),(worst,'Error-selected failures: largest mean mixed512 regressions','selected_failures')]:
        fig=plt.figure(figsize=(12,7.5),layout='constrained')
        grid=fig.add_gridspec(4,5,height_ratios=[1,1,1,.06])
        axes=np.array([[fig.add_subplot(grid[j,k]) for k in range(5)] for j in range(3)])
        for j,i in enumerate(selection):
            row=rows[i]
            with np.load(ASSETS/'external_sun3d_v4'/row['file']) as z:rgb=z['image'];gt=z['depth']
            mask=valid_mask(gt);preds=[]
            for label in ['base_r512','draw1_mixed_seed17_r512']:
                rec=read(Path('results/prospective_v4/evaluations')/label/'per_image.json')[i]
                p=np.load(ROOT/'prospective_v4/predictions'/label/f"{row['id']:04d}.npy")
                preds.append(np.clip(p*rec['scale']+rec['shift'],.1,10))
            fields=[rgb,gt,*preds,abs(preds[1]-gt)]
            for k,field in enumerate(fields):
                ax=axes[j,k]
                if k==0:ax.imshow(field)
                else:im=ax.imshow(np.where(mask,field,np.nan),cmap='inferno' if k==4 else 'magma_r',vmin=0 if k==4 else .1,vmax=1 if k==4 else 7)
                ax.set_xticks([]);ax.set_yticks([])
                if j==0:ax.set_title(['RGB','Measured GT','Original aligned','Mixed aligned','Mixed abs. error'][k],fontsize=13)
                if k==0:ax.set_ylabel(f"ID {row['id']}\nmean delta\n{delta[i]:+.3f}",fontsize=12,rotation=0,ha='right',va='center',labelpad=10)
        depthbar=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(.1,7),cmap='magma_r'),cax=fig.add_subplot(grid[3,1:4]),orientation='horizontal',ticks=[.1,2,4,7])
        depthbar.set_label('Depth (m)',fontsize=12)
        errorbar=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap='inferno'),cax=fig.add_subplot(grid[3,4]),orientation='horizontal',ticks=[0,.5,1])
        errorbar.set_label('Absolute error (m)',fontsize=12)
        fig.suptitle(title+'\ndraw1/seed17 | white: invalid GT | row delta: mean mixed-minus-original AbsRel',fontsize=14)
        fig.set_dpi(160);fig.canvas.draw();renderer=fig.canvas.get_renderer()
        for artist in fig.findobj(matplotlib.text.Text):
            if artist.get_visible() and artist.get_text():
                box=artist.get_window_extent(renderer)
                assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.width+1 and box.y1<=fig.bbox.height+1,artist.get_text()
        fig.savefig(figdir/(name+'.png'),dpi=160);plt.close(fig)
    print('QUALITATIVE_EXAMPLES_COMPLETE',flush=True)

if __name__=='__main__':main()
