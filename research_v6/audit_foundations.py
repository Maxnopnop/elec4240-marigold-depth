"""Read-only review of data roles, exposure, and metric-depth codec feasibility.

No test images or model predictions are loaded. Codec checks establish numerical
invertibility only; they do not establish learned metric-depth accuracy.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT.parents[1] / 'work/marigold-local/assets'
OUT = ROOT / 'research_v6'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode_metric(depth, dmin=.1, dmax=10.):
    """Fixed-range log-depth grayscale; preserves scale inside this interval."""
    return np.log(np.clip(depth, dmin, dmax) / dmin) / np.log(dmax / dmin)


def decode_metric(encoded, dmin=.1, dmax=10.):
    return dmin * np.exp(np.clip(encoded, 0., 1.) * np.log(dmax / dmin))


def main():
    manifest_path = ROOT / 'results/robustness_v3/split_manifest.json'
    manifest = read(manifest_path)
    train_scenes = {r['scene'] for rows in manifest['draws'].values() for r in rows}
    val_scenes = {r['scene'] for r in manifest['validation']}
    test_scenes = {r['scene'] for cohort in ['observed64', 'fresh31'] for r in manifest[cohort]}
    assert not train_scenes & val_scenes
    assert not train_scenes & test_scenes
    assert not val_scenes & test_scenes
    # Fixed outcome-independent selection; only training arrays are inspected.
    selected = sorted(manifest['draws']['draw1'], key=lambda x: x['id'])[:8]
    samples = []
    for row in selected:
        path = ASSETS / 'subset' / row['file']
        assert sha(path) == row['sha256']
        with np.load(path) as data:
            depth = data['depth']
            samples.append({'id': row['id'], 'scene': row['scene'],
                            'keys': list(data.files), 'shape': list(depth.shape),
                            'finite_fraction': float(np.isfinite(depth).mean()),
                            'depth_range_m': [float(np.min(depth)), float(np.max(depth))]})
    exposures = []
    for path in sorted((ROOT / 'results/robustness_v3/runs').glob('draw*_mixed_seed*/training.json')):
        record = read(path)
        history = record['history']
        by_res = {r: Counter(v['sample_index'] for v in history if v['resolution'] == r)
                  for r in [256, 512]}
        exposures.append({'run': path.parent.name, 'steps': record['steps'],
                          'updates': {str(r): sum(c.values()) for r, c in by_res.items()},
                          'unique_images': {str(r): len(c) for r, c in by_res.items()},
                          'images_seen_at_both': len(set(by_res[256]) & set(by_res[512])),
                          'training_images': record['train_images']})
    assert len(exposures) == 15
    grid = np.geomspace(.1, 10., 10001)
    recovered = decode_metric(encode_metric(grid))
    assert np.allclose(recovered, grid, rtol=1e-12, atol=1e-12)
    # A simple affine-normalization counterexample: 1--2 m and 2--4 m look identical.
    first = np.linspace(1., 2., 100)
    second = 2 * first
    norm = lambda x: (x - np.quantile(x, .02)) / (np.quantile(x, .98) - np.quantile(x, .02))
    assert np.allclose(norm(first), norm(second))
    # Do not confuse 8-bit RGB-file quantization with floating-point training targets.
    q = np.round(encode_metric(grid) * 255) / 255
    result = {
        'status': 'passed', 'scope': 'training arrays and existing metadata only; no new model evaluations',
        'manifest_sha256': sha(manifest_path), 'source_sha256': sha(__file__),
        'training_scene_union': len(train_scenes), 'validation_scenes': len(val_scenes),
        'scene_role_overlap': False, 'training_sample_audit': samples,
        'local_second_task_labels_present': False, 'local_camera_intrinsics_in_npz': False,
        'mixed_training_exposure': exposures,
        'mean_images_seen_at_both_resolutions': float(np.mean([r['images_seen_at_both'] for r in exposures])),
        'metric_codec': {'range_m': [.1, 10.], 'mapping': 'log(d/0.1)/log(100)',
                         'float_max_abs_error_m': float(np.max(abs(recovered-grid))),
                         'eight_bit_max_relative_error': float(np.max(abs(decode_metric(q)-grid)/grid)),
                         'outside_range': 'clipped; not invertible outside the fixed interval',
                         'learned_prediction_accuracy_tested': False},
        'per_image_normalization_loses_metric_scale_counterexample': True,
        'notes': ['NYUv2 depths are inpainted metric labels; missing raw validity masks matter for derived normals.',
                  'Original official test cohorts and external200 have already been observed.',
                  'Normal labels derived from depth are correlated supervision, not an independent geometry measurement.',
                  'Normal RGB decoding must preserve three channels; existing depth pipeline averages channels.',
                  'Affine-invariant ensembling must be disabled/replaced for metric outputs.']}
    (OUT / 'foundation_audit.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','training_scene_union','validation_scenes',
                       'mean_images_seen_at_both_resolutions','metric_codec']}, indent=2))


if __name__ == '__main__':
    main()
