"""Recompute v3 metrics, freeze calibration, and gate test inference."""
import gc
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.io import loadmat
from run_robustness import (arguments, matrix, conditions, rows_for, fingerprint, complete,
                            fit_calibration, fixed_metrics, DRAWS, SEEDS, MODES)
from run_expanded import read, write, sha, restore_adapter
from experiment import load_sample, evaluate_depth, load_pipe, infer_marigold
from prepare_subset import SPLIT_SHA256


def main():
    a = arguments()
    assert read(a.results/'protocol.json') == fingerprint(a.results)
    manifest = read(a.results/'split_manifest.json')
    addendum = read(a.results/'statistical_addendum.json')
    assert addendum['pre_test_status']['stage'] == 'validation'
    assert addendum['test_prediction_directory_absent'] is True
    for name, checksum in addendum['source_sha256'].items():
        assert sha(name) == checksum
    census_path = a.results/'scene_census.json'
    assert sha(census_path) == manifest['census_sha256']
    census = read(census_path)
    old = []
    for name, checksum in manifest['source_manifests'].items():
        assert sha(name) == checksum
        old += read(name)['samples']
    validation, fresh, observed = [manifest[k] for k in ['validation', 'fresh31', 'observed64']]
    assert [len(x) for x in [validation, fresh, observed]] == [32, 31, 64]
    assert not {r['scene'] for r in fresh} & {r['scene'] for r in old}
    heldout = validation+fresh+observed
    assert len({r['scene'] for r in heldout}) == 127
    allrows = heldout + sum(manifest['draws'].values(), [])
    unique = {r['id']: r for r in allrows}
    assert len(unique) == manifest['unique_frames']
    assert sha(a.assets/'splits.mat') == SPLIT_SHA256
    official = loadmat(a.assets/'splits.mat')
    for r in unique.values():
        assert r['id'] in official['testNdxs' if r['split'] == 'test' else 'trainNdxs']
        assert sha(a.assets/'subset'/r['file']) == r['sha256']
    for rows in manifest['draws'].values():
        assert len(rows) == len({r['scene'] for r in rows}) == 128
        assert not {r['scene'] for r in rows} & {r['scene'] for r in heldout}
    canonical = {}
    for row in sorted(census, key=lambda r: r['id']):
        canonical.setdefault(row['scene'], row['id'])
    pool = sorted({r['scene'] for r in census if r['official_split'] == 'train'} - {r['scene'] for r in validation})
    assert len(pool) == 217
    for draw, seed in zip(DRAWS, [4261, 4262, 4263]):
        expected = np.random.default_rng(seed).choice(pool, 128, replace=False).tolist()
        assert [r['scene'] for r in manifest['draws'][draw]] == expected
        assert [r['id'] for r in manifest['draws'][draw]] == [canonical[s] for s in expected]
    expected_fresh = sorted({r['scene'] for r in census if r['official_split'] == 'test'} - {r['scene'] for r in old})
    assert [r['scene'] for r in fresh] == expected_fresh
    assert [r['id'] for r in fresh] == [canonical[s] for s in expected_fresh]
    artifacts, checkpoints, tensor_hashes = {}, {}, {}
    def record(path):
        artifacts[path.relative_to(a.results).as_posix()] = sha(path)
    for cfg in matrix():
        rd = a.results/'runs'/cfg['run_id']
        assert read(rd/'config.json') == {**cfg, 'protocol_sha256': sha(a.results/'protocol.json')}
        record(rd/'config.json')
        if cfg['mode'] not in MODES:
            continue
        s = read(rd/'training.json')
        assert s['steps'] == len(s['history']) == 320
        assert s['seed'] == cfg['seed'] and s['mode'] == cfg['mode']
        assert s['trainable_parameters'] == 829952 and s['train_images'] == 128
        assert s['training_ids'] == [r['id'] for r in manifest['draws'][cfg['draw']]]
        for i, h in enumerate(s['history']):
            assert h['step'] == i+1
            assert h['resolution'] == (512 if cfg['mode'] == 'high512' else [256, 512][i % 2])
            assert all(np.isfinite(h[k]) for k in ['loss', 'supervised_loss', 'gradient_norm', 'prior_loss'])
            assert h['prior_loss'] == 0 and h['loss'] == h['supervised_loss']
        cp = a.work/'checkpoints'/cfg['run_id']/'adapter.pt'
        assert s['checkpoint_sha256'] == sha(cp) and s['config_sha256'] == sha(rd/'config.json')
        tensors = torch.load(cp, map_location='cpu', weights_only=True)
        assert sum(t.numel() for t in tensors.values()) == 829952
        assert all(torch.isfinite(t).all() for t in tensors.values())
        digest = hashlib.sha256()
        for name, tensor in sorted(tensors.items()):
            digest.update(name.encode('utf-8'))
            digest.update(tensor.contiguous().numpy().tobytes())
        tensor_hashes[cfg['run_id']] = digest.hexdigest()
        checkpoints[cfg['run_id']] = sha(cp)
        record(rd/'training.json')
    assert len(checkpoints) == len(set(checkpoints.values())) == 30
    assert len(set(tensor_hashes.values())) == 30
    for d in DRAWS:
        for seed in SEEDS:
            histories = [read(a.results/'runs'/f'{d}_{m}_seed{seed}'/'training.json')['history'] for m in MODES]
            assert [[(r['sample_index'], r['timestep']) for r in h] for h in histories][0] == [
                (r['sample_index'], r['timestep']) for r in histories[1]]
    # Cached GT is immutable and only used here for scoring/calibration.
    ground_truth = {r['id']: load_sample(a.assets, r)[1] for r in heldout}
    calibration = {'protocol_sha256': sha(a.results/'protocol.json'),
                   'validation_ids': [r['id'] for r in validation], 'conditions': {}}
    for cfg in matrix():
        for label, _, _ in conditions(cfg, 'validation'):
            pairs = [(np.load(a.work/'predictions'/'validation'/label/f"{r['id']:04d}.npy"), ground_truth[r['id']])
                     for r in validation]
            calibration['conditions'][label] = fit_calibration(pairs)
            del pairs
    if (a.results/'calibration.json').exists():
        assert read(a.results/'calibration.json') == calibration
    else:
        assert a.stage == 'validation'
        write(a.results/'calibration.json', calibration)
    stages = ['validation'] if a.stage == 'validation' else ['validation', 'test']
    if a.stage == 'corruption':
        stages.append('corruption')
    total = 0
    for stage in stages:
        selected = rows_for(manifest, stage)
        stage_args = SimpleNamespace(**{**vars(a), 'stage': stage})
        for cfg in matrix():
            for label, res, corruption in conditions(cfg, stage):
                assert complete(stage_args, label, selected)
                dest = a.results/'evaluations'/stage/label
                records = read(dest/'per_image.json')
                clean = label if corruption == 'clean' else label.rsplit('_', 1)[0]
                for row, rec in zip(selected, records):
                    assert all(rec[k] == v for k, v in cfg.items())
                    assert rec['resolution'] == res and rec['corruption'] == corruption
                    assert rec['scene'] == row['scene'] and rec['cohort'] == row['cohort']
                    pred = np.load(a.work/'predictions'/stage/label/f"{row['id']:04d}.npy")
                    gt = ground_truth[row['id']]
                    metrics, _, _ = evaluate_depth(pred, gt)
                    if stage != 'validation':
                        c = calibration['conditions'][clean]
                        metrics.update({'calibrated_'+k: v for k, v in fixed_metrics(pred, gt, c['scale'], c['shift']).items()})
                        if cfg['mode'] == 'expert':
                            metrics.update({'native_'+k: v for k, v in fixed_metrics(pred, gt).items()})
                    assert all(np.isclose(rec[k], v, atol=1e-10, rtol=0) for k, v in metrics.items())
                    assert np.isfinite(rec['inference_seconds']) and rec['inference_seconds'] > 0
                    total += 1
                record(dest/'complete.json'); record(dest/'per_image.json')
        print('AUDITED', stage, total, flush=True)
    restores = []
    if a.stage == 'validation':
        torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
        row = validation[0]
        image, _ = load_sample(a.assets, row)
        for mode in MODES:
            rid = f'draw2_{mode}_seed59'
            pipe = load_pipe(read(a.assets/'model_path.json')['path'])
            restore_adapter(pipe, 'lora', a.work/'checkpoints'/rid/'adapter.pt')
            for res in [256, 512]:
                pred, _, _ = infer_marigold(pipe, image, 1, res, 17+row['id'])
                saved = np.load(a.work/'predictions'/'validation'/f'{rid}_r{res}'/f"{row['id']:04d}.npy")
                error = float(np.max(abs(pred-saved)))
                assert error == 0
                restores.append({'run_id': rid, 'frame_id': row['id'], 'resolution': res, 'max_difference': error})
            del pipe
            gc.collect(); torch.cuda.empty_cache()
        assert total == 2048
    else:
        restores = read(a.results/'validation_gate.json')['checkpoint_restorations']
        assert total == (9182 if a.stage == 'corruption' else 8128)
    # Extra reproducibility check where the previous local raw arrays are available.
    prior_root = a.work.parent/'scaleup_v2'/'predictions'
    previous_reproductions = {'available': prior_root.exists(), 'arrays_compared': 0,
                             'maximum_absolute_difference': None}
    if prior_root.exists():
        largest = 0.
        for stage in ['validation'] + (['test'] if a.stage != 'validation' else []):
            selected = validation if stage == 'validation' else observed
            for current, previous in [('base_r256', 'base_r256'), ('base_r512', 'base_r512'), ('expert256', 'expert')]:
                for row in selected:
                    filename = f"{row['id']:04d}.npy"
                    new = np.load(a.work/'predictions'/stage/current/filename)
                    old = np.load(prior_root/stage/previous/filename)
                    diff = float(np.max(abs(new-old)))
                    largest = max(largest, diff)
                    assert np.array_equal(new, old), (stage, current, row['id'], diff)
                    previous_reproductions['arrays_compared'] += 1
        previous_reproductions['maximum_absolute_difference'] = largest
    report = {'status': 'passed', 'protocol_sha256': sha(a.results/'protocol.json'),
              'calibration_sha256': sha(a.results/'calibration.json'), 'sample_hashes_verified': len(unique),
              'training_runs': 30, 'updates_per_run': 320, 'metrics_recomputed': total,
              'checkpoint_restorations': restores, 'paired_training_schedules_verified': True,
              'checkpoint_tensor_sha256': tensor_hashes,
              'previous_baseline_reproductions': previous_reproductions,
              'checkpoint_sha256': checkpoints, 'audited_artifact_sha256': artifacts}
    write(a.results/('validation_gate.json' if a.stage == 'validation' else 'verification.json'), report)
    print('ROBUSTNESS_AUDIT_PASSED', a.stage, total, flush=True)


if __name__ == '__main__':
    main()
