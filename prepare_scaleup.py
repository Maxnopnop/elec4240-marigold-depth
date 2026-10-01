"""Expand training/validation and reserve entirely new test scenes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import h5py
import numpy as np
from scipy.io import loadmat
from prepare_subset import RangeFile, extract_frame, SPLIT_SHA256
from download_assets import NYU_URL
from run_expanded import read, write, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assets', required=True, type=Path)
    p.add_argument('--results', default='results/scaleup_v2', type=Path)
    a = p.parse_args()
    prior_path = Path('results/expanded_v1/split_manifest.json')
    prior = read(prior_path)
    assert sha(a.assets / 'splits.mat') == SPLIT_SHA256
    split = loadmat(a.assets / 'splits.mat')
    plan_path = a.results / 'selection_plan.json'
    if plan_path.exists():
        plan = read(plan_path)
        assert plan['prior_manifest_sha256'] == sha(prior_path)
    else:
        groups = {s: [dict(r) for r in prior['samples'] if r['split'] == s] for s in ['train', 'val']}
        groups['test'] = []
        seen = {r['scene'] for r in prior['samples']}
        known = {r['id']-1: r['scene'] for r in prior['samples']}
        with h5py.File(RangeFile(NYU_URL, a.assets / 'range_cache'), 'r') as f:
            refs = f['scenes'][0]
            def scene(idx):
                idx = int(idx)
                if idx not in known:
                    known[idx] = ''.join(chr(int(c)) for c in f[refs[idx]][()].ravel())
                return known[idx]
            for s, key, total, seed in [('train', 'trainNdxs', 128, 4251), ('val', 'trainNdxs', 32, 4252), ('test', 'testNdxs', 64, 4253)]:
                for idx in np.random.default_rng(seed).permutation(split[key].ravel()-1):
                    name = scene(idx)
                    if name in seen:
                        continue
                    seen.add(name)
                    groups[s].append({'id': int(idx)+1, 'scene': name, 'split': s, 'file': f'{int(idx)+1:04d}.npz'})
                    if len(groups[s]) == total:
                        break
                assert len(groups[s]) == total, (s, len(groups[s]), 'insufficient unseen scenes')
                print('SELECTED', s, total, flush=True)
        plan = {'version': 'scaleup_v2', 'prior_manifest_sha256': sha(prior_path),
                'selection_seeds': [4251, 4252, 4253], 'source_url': NYU_URL,
                'counts': {'train': 128, 'val': 32, 'test': 64},
                'excluded_previous_test_ids': [r['id'] for r in prior['samples'] if r['split'] == 'test'],
                'samples': sum(groups.values(), [])}
        write(plan_path, plan)
    rows = plan['samples']
    assert len(rows) == len({r['scene'] for r in rows}) == len({r['id'] for r in rows}) == 224
    previous_scenes = {r['scene'] for r in prior['samples']}
    assert not {r['scene'] for r in rows if r['split'] == 'test'} & previous_scenes
    for s, key in [('train', 'trainNdxs'), ('val', 'trainNdxs'), ('test', 'testNdxs')]:
        assert {r['id'] for r in rows if r['split'] == s} <= set(split[key].ravel())
    with ProcessPoolExecutor(max_workers=4) as pool:
        extracted = []
        for row in pool.map(extract_frame, [(str(a.assets), r) for r in rows]):
            extracted.append(row)
            if len(extracted) % 16 == 0:
                print('EXTRACTED', len(extracted), '/224', flush=True)
    result = {**plan, 'samples': extracted, 'selection_plan_sha256': sha(plan_path)}
    dest = a.results / 'split_manifest.json'
    if dest.exists():
        assert read(dest) == result
    else:
        write(dest, result)
    print('SCALEUP_DATA_READY', flush=True)


if __name__ == '__main__':
    main()
