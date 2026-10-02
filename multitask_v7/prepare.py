"""Acquire raw-depth validity and derive explicitly non-independent normal targets."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import re
import hashlib
import h5py
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
from prepare_subset import RangeFile
from download_assets import NYU_URL
from experiment import mask_for
from .common import ASSETS,WORK,OUT,ROOT,read,write,sha,rays,depth_to_normals


def extract(row):
    torch.set_num_threads(2)
    path=WORK/'derived'/row['file']
    if not path.exists():
        with np.load(ASSETS/'subset'/row['file']) as z:depth=z['depth'].copy()
        remote=RangeFile(NYU_URL,ASSETS/'range_cache')
        with h5py.File(remote,'r') as f:raw=f['rawDepths'][row['id']-1].T.copy()
        assert raw.shape==depth.shape==(480,640)
        valid=mask_for(depth)&(depth>=.1)
        raw_valid=np.isfinite(raw)&(raw>=.1)&(raw<10)
        raw_agrees=raw_valid&(abs(raw-depth)<(.03+.01*depth))
        jumps=np.zeros_like(valid)
        for axis in [0,1]:
            diff=np.diff(depth,axis=axis)
            a=depth[1:,:] if axis==0 else depth[:,1:]
            b=depth[:-1,:] if axis==0 else depth[:,:-1]
            jump=(abs(diff)>.1)&(abs(diff)>.05*np.minimum(a,b))
            if axis==0:jumps[1:,:]|=jump;jumps[:-1,:]|=jump
            else:jumps[:,1:]|=jump;jumps[:,:-1]|=jump
        normal_valid=binary_erosion(valid&raw_agrees,iterations=3)&~binary_dilation(jumps,iterations=2)
        smooth=gaussian_filter(depth,sigma=1.,truncate=2.)
        camera=read(OUT/'camera.json')['intrinsics']
        with torch.no_grad():normal=depth_to_normals(torch.from_numpy(smooth)[None,None],rays(480,640,camera))[0].numpy()
        assert normal_valid.sum()>100, (row['id'],int(normal_valid.sum()))
        assert np.isfinite(normal).all()
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.with_suffix('.partial').open('wb') as f:
            np.savez_compressed(f,normal=normal.astype(np.float32),normal_valid=normal_valid,
                                depth_valid=valid,raw_depth=raw,raw_valid=raw_valid&valid)
        path.with_suffix('.partial').replace(path)
    with np.load(path) as z:
        meta={**row,'source_sha256':sha(ASSETS/'subset'/row['file']),'derived_sha256':sha(path),
              'normal_valid_pixels':int(z['normal_valid'].sum()),
              'depth_valid_pixels':int(z['depth_valid'].sum()),
              'raw_valid_pixels':int(z['raw_valid'].sum()),
              'raw_depth_array_sha256':hashlib.sha256(z['raw_depth'].tobytes()).hexdigest()}
    assert meta['source_sha256']==row['sha256']
    return meta


def main():
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=4);args=a.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    text=(WORK/'camera_params.m').read_text()
    intrinsics={key:float(re.search(r'\b'+key+r'\s*=\s*([^;]+);',text).group(1))
                for key in ['fx_rgb','fy_rgb','cx_rgb','cy_rgb']}
    write(OUT/'camera.json',{'intrinsics':intrinsics,'image_shape':[480,640],
          'pixel_convention':'1-based toolbox coordinates; half-pixel resize for align_corners=False',
          'normal_convention':'camera coordinates x right, y down, z forward; normals face camera',
          'projection_approximation':'provided projected depth treated as camera-z using RGB pinhole calibration',
          'source':read(WORK/'toolbox_provenance.json')})
    old=read(ROOT/'results/robustness_v3/split_manifest.json')
    train=old['draws']['draw1'];validation=old['validation']
    assert not {r['scene'] for r in train}&{r['scene'] for r in validation}
    meta=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,row in enumerate(pool.map(extract,train+validation)):
            meta.append(row)
            print('DERIVED_LABELS',i+1,len(train+validation),row['id'],row['normal_valid_pixels'],flush=True)
    write(OUT/'manifest.json',{'version':'metric_multitask_v7','training':meta[:len(train)],
          'validation':meta[len(train):],'source_manifest_sha256':sha(ROOT/'results/robustness_v3/split_manifest.json'),
          'camera_sha256':sha(OUT/'camera.json'),'raw_depth_source':NYU_URL,
          'normal_target':'derived from sigma1/truncate2 smoothed inpainted depth; eroded raw-agreement mask and excluded depth jumps',
          'normal_evaluation_is_independent_ground_truth':False,
          'evaluation_status':'observed development validation; no test scenes used in this stage'})
    print('DATA_PREPARATION_PASSED',len(meta),flush=True)


if __name__=='__main__':main()
