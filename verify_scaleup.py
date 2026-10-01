"""Audit larger experiment and issue the technical gate for fresh test inference."""
import gc
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from run_scaleup import arguments, matrix, labels, fingerprint, complete
from run_expanded import read, write, sha, restore_adapter
from experiment import load_sample, evaluate_depth, load_pipe, infer_marigold
from train_scaleup import MODES
from prepare_subset import SPLIT_SHA256


def main():
    a = arguments()
    assert read(a.results/'protocol.json') == fingerprint(a.results)
    manifest = read(a.results/'split_manifest.json')
    rows = manifest['samples']
    assert len(rows) == len({r['id'] for r in rows}) == len({r['scene'] for r in rows}) == 224
    prior_path = Path('results/expanded_v1/split_manifest.json')
    assert manifest['prior_manifest_sha256'] == sha(prior_path)
    prior = read(prior_path)['samples']
    assert not {r['scene'] for r in rows if r['split'] == 'test'} & {r['scene'] for r in prior}
    for split in ['train', 'val']:
        assert {r['id'] for r in prior if r['split'] == split} <= {r['id'] for r in rows if r['split'] == split}
    assert sha(a.assets/'splits.mat') == SPLIT_SHA256
    official = loadmat(a.assets/'splits.mat')
    for row in rows:
        key = 'testNdxs' if row['split'] == 'test' else 'trainNdxs'
        assert row['id'] in official[key]
        assert sha(a.assets/'subset'/row['file']) == row['sha256']
    checkpoints, artifacts, restores = {}, {}, []
    paths_to_hash = []
    for cfg in matrix():
        rd = a.results/'runs'/cfg['run_id']
        assert read(rd/'config.json') == {**cfg, 'protocol_sha256': sha(a.results/'protocol.json')}
        paths_to_hash.append(rd/'config.json')
        if cfg['mode'] in MODES:
            stats = read(rd/'training.json')
            assert stats['steps'] == len(stats['history']) == 320
            assert stats['seed'] == cfg['seed'] and stats['mode'] == cfg['mode']
            assert stats['training_ids'] == [r['id'] for r in rows if r['split'] == 'train']
            assert stats['trainable_parameters'] == 829952 and stats['train_images'] == 128
            for i, rec in enumerate(stats['history']):
                expected = 256 if cfg['mode'] == 'low256' else 512 if cfg['mode'] == 'high512' else [256, 512][i % 2]
                assert rec['step'] == i+1 and rec['resolution'] == expected
                assert all(np.isfinite(rec[k]) for k in ['loss', 'supervised_loss', 'prior_loss', 'gradient_norm'])
                assert abs(rec['loss']-rec['supervised_loss']-.1*rec['prior_loss']) < 1e-5
            if cfg['mode'] == 'mixed_prior':
                assert max(r['prior_loss'] for r in stats['history']) > 0, 'Inactive teacher objective'
            else:
                assert all(r['prior_loss'] == 0 for r in stats['history'])
            checkpoint = a.work/'checkpoints'/cfg['run_id']/'adapter.pt'
            assert stats['checkpoint_sha256'] == sha(checkpoint)
            assert stats['config_sha256'] == sha(rd/'config.json')
            tensors = torch.load(checkpoint, map_location='cpu', weights_only=True)
            assert sum(v.numel() for v in tensors.values()) == 829952
            assert all(torch.isfinite(v).all() for v in tensors.values()), 'Non-finite checkpoint'
            del tensors
            checkpoints[cfg['run_id']] = stats['checkpoint_sha256']
            paths_to_hash.append(rd/'training.json')
    assert len(checkpoints) == len(set(checkpoints.values())) == 12
    # Validate paired data order/timesteps independently of resolution and teacher overhead.
    for seed in [17, 29, 43]:
        schedules = [[(h['sample_index'], h['timestep']) for h in read(a.results/'runs'/f'{m}_seed{seed}'/'training.json')['history']] for m in MODES]
        assert all(s == schedules[0] for s in schedules)
    stages = ['validation'] if a.stage == 'validation' else ['validation', 'test']
    total = 0
    for stage in stages:
        selected = [r for r in rows if r['split'] == ('val' if stage == 'validation' else 'test')]
        for cfg in matrix():
            for label, res in labels(cfg):
                assert complete(a.results, a.work, stage, label, [r['id'] for r in selected])
                rd = a.results/'evaluations'/stage/label
                records = read(rd/'per_image.json')
                for row, rec in zip(selected, records):
                    assert all(rec[k] == v for k, v in cfg.items())
                    assert rec['resolution'] == res and rec['scene'] == row['scene'] and rec['split'] == row['split']
                    pred = np.load(a.work/'predictions'/stage/label/f"{row['id']:04d}.npy")
                    _, gt = load_sample(a.assets, row)
                    metrics, _, _ = evaluate_depth(pred, gt)
                    assert all(np.isclose(rec[k], v, atol=1e-10, rtol=0) for k, v in metrics.items())
                    assert np.isfinite(rec['inference_seconds']) and rec['inference_seconds'] > 0
                    total += 1
                paths_to_hash.extend([rd/'complete.json', rd/'per_image.json'])
    if a.stage == 'validation':
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        row = next(r for r in rows if r['split'] == 'val')
        image, _ = load_sample(a.assets, row)
        for mode in MODES:
            rid = f'{mode}_seed29'
            pipe = load_pipe(read(a.assets/'model_path.json')['path'])
            restore_adapter(pipe, 'lora', a.work/'checkpoints'/rid/'adapter.pt')
            pred, _, _ = infer_marigold(pipe, image, 1, 512, 17+row['id'])
            saved = np.load(a.work/'predictions'/'validation'/f'{rid}_r512'/f"{row['id']:04d}.npy")
            difference = float(np.max(abs(pred-saved)))
            assert difference == 0
            restores.append({'run_id': rid, 'frame_id': row['id'], 'resolution': 512, 'max_difference': difference})
            del pipe
            gc.collect(); torch.cuda.empty_cache()
        assert total == 864
    else:
        assert total == 2592
        restores = read(a.results/'validation_gate.json')['checkpoint_restorations']
    for path in paths_to_hash:
        artifacts[path.relative_to(a.results).as_posix()] = sha(path)
    report = {'status': 'passed', 'protocol_sha256': sha(a.results/'protocol.json'),
              'sample_hashes_verified': 224, 'unique_scenes': 224, 'fresh_test_scenes': 64,
              'training_runs': 12, 'updates_per_run': 320, 'metrics_recomputed': total,
              'checkpoint_restorations': restores, 'paired_training_schedules_verified': True,
              'checkpoint_sha256': checkpoints, 'audited_artifact_sha256': artifacts}
    write(a.results/('validation_gate.json' if a.stage == 'validation' else 'verification.json'), report)
    print('SCALEUP_AUDIT_PASSED', a.stage, total, flush=True)


if __name__ == '__main__':
    main()
