"""Verify data provenance, split isolation and saved predictions vs reported metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from experiment import evaluate_depth, load_sample
from summarize import METHODS
from prepare_subset import SPLIT_SHA256

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True)
    p.add_argument('--work',required=True);p.add_argument('--results',default='results')
    a=p.parse_args();assets=Path(a.assets);results=Path(a.results)
    split_path=assets/'splits.mat'
    assert hashlib.sha256(split_path.read_bytes()).hexdigest()==SPLIT_SHA256
    original=loadmat(split_path)
    official_train=set(original['trainNdxs'].ravel().tolist());official_test=set(original['testNdxs'].ravel().tolist())
    samples=json.loads((results/'split_manifest.json').read_text())['samples']
    assert len(samples)==64 and len({r['id'] for r in samples})==64
    assert len({r['scene'] for r in samples})==64
    assert {g:sum(r['split']==g for r in samples) for g in ['train','val','test']}=={'train':32,'val':8,'test':24}
    byid={r['id']:r for r in samples}
    for r in samples:
        assert r['id'] in (official_test if r['split']=='test' else official_train)
        assert hashlib.sha256((assets/'subset'/r['file']).read_bytes()).hexdigest()==r['sha256']
        image,depth=load_sample(assets,r)
        assert image.dtype==np.uint8 and image.shape==(480,640,3)
        assert depth.shape==(480,640) and np.isfinite(depth).all()
    expected={r['id'] for r in samples if r['split']!='train'}
    checked=0
    for method in METHODS:
        records=json.loads((results/f'{method}_per_image.json').read_text())
        assert len(records)==32 and {r['id'] for r in records}==expected
        for r in records:
            sample=byid[r['id']];assert r['split']==sample['split'] and r['scene']==sample['scene']
            _,gt=load_sample(assets,sample)
            pred=np.load(Path(a.work)/'predictions'/method/f"{r['id']:04d}.npy")
            recomputed,_,_=evaluate_depth(pred,gt)
            for metric in ['abs_rel','rmse_m','delta1','scale','shift']:
                assert np.isclose(recomputed[metric],r[metric],rtol=0,atol=1e-10),(method,r['id'],metric)
            assert r['inference_seconds']>0 and r['peak_allocated_mib']>0
            checked+=1
    for method in ['lora8','lora32','head32']:
        training=json.loads((results/f'{method}_training.json').read_text())
        assert len(training['history'])==training['steps']==80
        assert all(np.isfinite(r['loss']) and np.isfinite(r['grad_norm']) for r in training['history'])
    report={'status':'passed','samples_hash_checked':len(samples),'unique_scenes':len(samples),
            'official_split_membership_checked':True,'prediction_metrics_recomputed':checked,
            'adaptation_runs_with_80_finite_steps':3}
    (results/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
