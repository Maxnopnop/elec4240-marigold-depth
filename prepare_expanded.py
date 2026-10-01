"""Extend the pilot without reassigning any existing scene to a new split."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.io import loadmat
from prepare_subset import RangeFile, extract_frame, SPLIT_SHA256
from download_assets import NYU_URL, SPLIT_URL

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True)
    p.add_argument('--pilot',default='results/split_manifest.json')
    p.add_argument('--results',default='results/expanded_v1');a=p.parse_args()
    assets=Path(a.assets);out=Path(a.results);out.mkdir(parents=True,exist_ok=True)
    pilot_path=Path(a.pilot);pilot=json.loads(pilot_path.read_text())
    pilot_sha=hashlib.sha256(pilot_path.read_bytes()).hexdigest()
    assert hashlib.sha256((assets/'splits.mat').read_bytes()).hexdigest()==SPLIT_SHA256
    split=loadmat(assets/'splits.mat')
    plan_path=out/'selection_plan.json'
    if plan_path.exists():
        plan=json.loads(plan_path.read_text());assert plan['pilot_manifest_sha256']==pilot_sha
    else:
        groups={g:[dict(r) for r in pilot['samples'] if r['split']==g] for g in ['train','val','test']}
        for r in groups['test']:r['cohort']='pilot24'
        seen={r['scene'] for r in pilot['samples']}
        scenes={r['id']-1:r['scene'] for r in pilot['samples']}
        remote=RangeFile(NYU_URL,assets/'range_cache')
        with h5py.File(remote,'r') as f:
            refs=f['scenes'][0]
            def scene(idx):
                idx=int(idx)
                if idx not in scenes:scenes[idx]=''.join(chr(int(c)) for c in f[refs[idx]][()].ravel())
                return scenes[idx]
            def extend(group,indices,target,seed):
                for idx in np.random.default_rng(seed).permutation(indices):
                    name=scene(idx)
                    if name in seen:continue
                    seen.add(name);row={'id':int(idx)+1,'scene':name,'split':group,'file':f'{int(idx)+1:04d}.npz'}
                    if group=='test':row['cohort']='fresh96'
                    groups[group].append(row)
                    print('SELECT',group,len(groups[group]),row['id'],name,flush=True)
                    if len(groups[group])==target:return
                raise ValueError('Insufficient distinct scenes')
            extend('train',split['trainNdxs'].ravel()-1,64,4241)
            extend('val',split['trainNdxs'].ravel()-1,16,4242)
            extend('test',split['testNdxs'].ravel()-1,120,4243)
        plan={'dataset':'NYU Depth V2 labeled','source_url':NYU_URL,'split_url':SPLIT_URL,
              'split_sha256':SPLIT_SHA256,'pilot_manifest_sha256':pilot_sha,
              'selection_seeds':{'train':4241,'val':4242,'test':4243},
              'one_frame_per_scene':True,'counts':{'train':64,'val':16,'test':120},
              'primary_test_cohort':'fresh96','samples':sum(groups.values(),[])}
        plan_path.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    rows=plan['samples'];assert len(rows)==len({r['scene'] for r in rows})==len({r['id'] for r in rows})==200
    with ProcessPoolExecutor(max_workers=4) as ex:
        extracted=[]
        for row in ex.map(extract_frame,[(str(assets),dict(r)) for r in rows]):
            extracted.append(row)
            print('EXTRACT',len(extracted),'/200',row['split'],row['id'],flush=True)
    result={**plan,'samples':extracted,'selection_plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest()}
    target=out/'split_manifest.json'
    if target.exists():assert json.loads(target.read_text())==result,'Manifest changed; refuse overwrite'
    else:target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('EXPANDED_DATA_READY',flush=True)

if __name__=='__main__':main()
