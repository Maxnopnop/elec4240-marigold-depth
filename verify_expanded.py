"""Audit expanded data, all measurements, and optionally checkpoint restoration."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import torch
from experiment import load_sample,evaluate_depth,load_pipe,infer_marigold
from run_expanded import read,sha,write,matrix,restore_adapter
from prepare_subset import SPLIT_SHA256

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--work',required=True)
    p.add_argument('--results',default='results/expanded_v1');p.add_argument('--check-restoration',action='store_true');a=p.parse_args()
    assets=Path(a.assets);work=Path(a.work);out=Path(a.results)
    manifest=read(out/'split_manifest.json');rows=manifest['samples'];byid={r['id']:r for r in rows}
    assert len(rows)==len(byid)==len({r['scene'] for r in rows})==200
    assert {g:sum(r['split']==g for r in rows) for g in ['train','val','test']}=={'train':64,'val':16,'test':120}
    assert sum(r.get('cohort')=='fresh96' for r in rows)==96
    assert sum(r.get('cohort')=='pilot24' for r in rows)==24
    assert sha(assets/'splits.mat')==SPLIT_SHA256
    split=loadmat(assets/'splits.mat')
    train=set(split['trainNdxs'].ravel().tolist());test=set(split['testNdxs'].ravel().tolist())
    assert sha(out/'selection_plan.json')==manifest['selection_plan_sha256']
    pilot=read(out.parent/'split_manifest.json');assert sha(out.parent/'split_manifest.json')==manifest['pilot_manifest_sha256']
    for row in pilot['samples']:
        assert byid[row['id']]['split']==row['split'] and byid[row['id']]['scene']==row['scene']
    assert [r['id'] for r in rows if r['split']=='train'][:32]==[r['id'] for r in pilot['samples'] if r['split']=='train']
    assert not {r['scene'] for r in pilot['samples']}&{r['scene'] for r in rows if r.get('cohort')=='fresh96'}
    depths={}
    for row in rows:
        assert row['id'] in (test if row['split']=='test' else train)
        assert sha(assets/'subset'/row['file'])==row['sha256']
        rgb,dep=load_sample(assets,row)
        assert rgb.shape==(480,640,3) and rgb.dtype==np.uint8 and dep.shape==(480,640)
        assert np.isfinite(dep).all()
        if row['split']!='train':depths[row['id']]=dep
    protocol=read(out/'protocol.json')
    assert protocol['manifest_sha256']==sha(out/'split_manifest.json')
    assert protocol['protocol_sha256']==sha(Path(__file__).with_name('EXPANDED_PROTOCOL.md'))
    for filename,checksum in protocol['source_sha256'].items():assert sha(Path(__file__).with_name(filename))==checksum
    predictions=0;checkpoints=[];baseline_replays=0
    for cfg in matrix():
        rid=cfg['run_id'];rd=out/'runs'/rid;config=read(rd/'config.json')
        assert config['manifest_sha256']==protocol['manifest_sha256']
        assert config['inference_seed_offset']==17
        if cfg['mode'] not in ['base','expert']:
            stats=read(rd/'training.json');checkpoint=work/'checkpoints'/rid/'adapter.pt'
            assert stats['checkpoint_sha256']==sha(checkpoint)
            assert stats['config_sha256']==sha(rd/'config.json')
            assert stats['seed']==cfg['train_seed'] and stats['train_images']==cfg['train_images']
            assert stats['trainable_parameters']==(829952 if cfg['mode']=='lora' else 11524)
            assert len(stats['history'])==stats['steps']==160
            assert stats['train_ids']==[r['id'] for r in rows if r['split']=='train'][:cfg['train_images']]
            assert all(np.isfinite(h['loss']) and np.isfinite(h['grad_norm']) for h in stats['history'])
            checkpoints.append(stats['checkpoint_sha256'])
        labels=[rid] if cfg['mode']=='expert' else [f'{rid}_s1',f'{rid}_s4']
        for label in labels:
            path=rd/f'{label}_per_image.json';records=read(path);complete=read(rd/f'{label}_complete.json')
            previous={r['id']:r for r in read(out.parent/f'{label}_per_image.json')} if cfg['mode']=='base' else {}
            assert complete['metrics_sha256']==sha(path) and complete['config_sha256']==sha(rd/'config.json')
            assert len(records)==len(depths)==136 and {r['id'] for r in records}==set(depths)
            for r in records:
                sample=byid[r['id']]
                assert r['cohort']==sample.get('cohort','val16') and r['train_seed']==cfg['train_seed']
                assert r['scene']==sample['scene'] and r['split']==sample['split']
                pred=np.load(work/'predictions'/label/f"{r['id']:04d}.npy")
                metrics,_,_=evaluate_depth(pred,depths[r['id']])
                for key in ['abs_rel','rmse_m','delta1','scale','shift']:
                    assert np.isclose(metrics[key],r[key],rtol=0,atol=1e-10),(label,r['id'],key)
                    if r['id'] in previous:
                        assert np.isclose(previous[r['id']][key],r[key],rtol=0,atol=1e-10),'Pilot baseline regression'
                if r['id'] in previous:baseline_replays+=1
                assert r['inference_seconds']>0 and r['peak_allocated_mib']>0
                predictions+=1
            print('VERIFIED',label,predictions,flush=True)
    assert len(checkpoints)==len(set(checkpoints))==9
    restored=[]
    if a.check_restoration:
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
        sample=next(r for r in rows if r['split']=='val');rgb,_=load_sample(assets,sample)
        for rid,mode in [('lora64_seed29','lora'),('head64_seed29','head')]:
            pipe=load_pipe(read(assets/'model_path.json')['path'])
            restore_adapter(pipe,mode,work/'checkpoints'/rid/'adapter.pt')
            pred,_,_=infer_marigold(pipe,rgb,4,256,17+sample['id'])
            saved=np.load(work/'predictions'/f'{rid}_s4'/f"{sample['id']:04d}.npy")
            difference=float(np.max(np.abs(pred-saved)))
            assert np.allclose(pred,saved,rtol=0,atol=1e-6),(rid,difference)
            restored.append({'run_id':rid,'frame_id':sample['id'],'max_absolute_prediction_difference':difference})
            del pipe;gc.collect();torch.cuda.empty_cache()
    report={'status':'passed','sample_hashes_checked':200,'unique_scenes':200,'fresh_test_scenes':96,
            'official_split_membership_checked':True,'pilot_roles_preserved':True,
            'prediction_metrics_recomputed':predictions,'evaluation_conditions':21,
            'pilot_baseline_metric_records_reproduced':baseline_replays,
            'finite_160_step_training_runs':9,'distinct_checkpoint_hashes':len(set(checkpoints)),
            'checkpoint_restoration_checks':restored}
    write(out/'verification.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
