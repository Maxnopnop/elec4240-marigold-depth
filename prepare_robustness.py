"""Freeze independent scene-level training draws and the remaining unseen test scenes."""
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.io import loadmat
from prepare_subset import extract_frame, SPLIT_SHA256
from run_expanded import read, write, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assets', required=True, type=Path)
    p.add_argument('--census', required=True, type=Path)
    p.add_argument('--results', default='results/robustness_v3', type=Path)
    a = p.parse_args()
    out = a.results
    oldpaths = [Path('results/expanded_v1/split_manifest.json'), Path('results/scaleup_v2/split_manifest.json')]
    old = [read(x) for x in oldpaths]
    census = read(a.census)
    census_dest = out/'scene_census.json'
    if census_dest.exists():
        assert census_dest.read_bytes() == a.census.read_bytes()
    else:
        census_dest.parent.mkdir(parents=True, exist_ok=True)
        census_dest.write_bytes(a.census.read_bytes())
    assert sha(a.assets/'splits.mat') == SPLIT_SHA256
    split = loadmat(a.assets/'splits.mat')
    for name, key in [('train', 'trainNdxs'), ('test', 'testNdxs')]:
        assert {r['id'] for r in census if r['official_split'] == name} == set(split[key].ravel())
    validation = [dict(r, split='validation', cohort='validation') for r in old[1]['samples'] if r['split'] == 'val']
    observed = [dict(r, split='test', cohort='observed64') for r in old[1]['samples'] if r['split'] == 'test']
    seen = {r['scene'] for x in old for r in x['samples']}
    valnames = {r['scene'] for r in validation}
    groups = {s: {} for s in ['train', 'test']}
    for r in sorted(census, key=lambda r: r['id']):
        groups[r['official_split']].setdefault(r['scene'], r['id'])
    assert len(groups['train']) == 249 and len(groups['test']) == 215
    assert not set(groups['train']) & set(groups['test'])
    pool = sorted(set(groups['train']) - valnames)
    assert len(pool) == 217
    def row(scene, official, role, cohort):
        idx = groups[official][scene]
        return {'id': idx, 'scene': scene, 'file': f'{idx:04d}.npz', 'split': role, 'cohort': cohort}
    draws = {}
    for name, seed in [('draw1', 4261), ('draw2', 4262), ('draw3', 4263)]:
        scenes = np.random.default_rng(seed).choice(pool, 128, replace=False).tolist()
        draws[name] = [row(s, 'train', 'train', name) for s in scenes]
    fresh = [row(s, 'test', 'test', 'fresh31') for s in sorted(set(groups['test'])-seen)]
    assert len(fresh) == 31
    unique = {r['id']: dict(r) for rows in [*draws.values(), validation, observed, fresh] for r in rows}
    plan = {'version': 'robustness_v3', 'source_manifests': {str(p): sha(p) for p in oldpaths},
            'census_sha256': sha(a.census), 'selection_seeds': [4261, 4262, 4263],
            'counts': {'train_per_draw': 128, 'draws': 3, 'validation': 32, 'observed64': 64, 'fresh31': 31},
            'draws': draws, 'validation': validation, 'observed64': observed, 'fresh31': fresh,
            'training_scene_intersections': {f'{x}-{y}': len({r['scene'] for r in draws[x]} & {r['scene'] for r in draws[y]})
                                              for x in draws for y in draws if x < y}}
    dest = out/'selection_plan.json'
    if dest.exists():
        assert read(dest) == plan
    else:
        write(dest, plan)
    print('SELECTION_FROZEN', plan['counts'], plan['training_scene_intersections'], 'unique_frames', len(unique), flush=True)
    extracted = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, r in enumerate(pool.map(extract_frame, [(str(a.assets), r) for r in unique.values()]), 1):
            extracted[r['id']] = r['sha256']
            if i % 16 == 0:
                print('EXTRACTED', i, '/', len(unique), flush=True)
    for rows in [*draws.values(), validation, observed, fresh]:
        for r in rows:
            r['sha256'] = extracted[r['id']]
    manifest = {**plan, 'selection_plan_sha256': sha(dest), 'unique_frames': len(unique)}
    dest = out/'split_manifest.json'
    if dest.exists():
        assert read(dest) == manifest
    else:
        write(dest, manifest)
    print('ROBUSTNESS_DATA_READY', flush=True)


if __name__ == '__main__':
    main()
