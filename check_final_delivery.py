"""Independent integrity checks for the exploratory final extension.

The experiment scripts already recompute all control metrics. This delivery audit
checks their saved artifact/raw hashes, redoes every budget metric, and protects
the complete historical results tree. It does not change historical artifacts.
"""
import json, subprocess
from pathlib import Path
import numpy as np
from experiment import load_sample, evaluate_depth
from run_expanded import read, write, sha
from run_final_control import ROOT, WORK, OUT, ASSETS
from analyze_final_v5 import arrays


def main():
    old = 'd14701c251ac1df58c9bfb9025287c476b127d3a'
    tree = subprocess.check_output(['git', 'ls-tree', '-r', old, '--', 'results'], text=True)
    paths, hashes = [], []
    for line in tree.splitlines():
        meta, name = line.split('\t', 1)
        paths.append(name); hashes.append(meta.split()[2])
    actual = subprocess.check_output(['git', 'hash-object', '--stdin-paths'],
                                    input='\n'.join(paths)+'\n', text=True).splitlines()
    assert actual == hashes, 'Historical result blobs changed'
    proto = read(OUT/'protocol.json')
    for name, digest in proto['sources'].items(): assert sha(name) == digest, name
    cost_proto = read(OUT/'cost/protocol.json')
    for name, digest in cost_proto['source_sha256'].items(): assert sha(name) == digest, name
    old_proto = read('results/prospective_v4/protocol.json')['fingerprint']
    for name, digest in old_proto['source_sha256'].items(): assert sha(name) == digest, name
    audit = read(OUT/'verification.json')
    assert audit['status'] == 'passed' and audit['metrics_recomputed'] == 7890
    assert audit['checkpoint_restorations'] == 30 and audit['protocol_sha256'] == sha(OUT/'protocol.json')
    for name, digest in audit['artifacts'].items(): assert sha(OUT/name) == digest, name
    restored = read(OUT/'restoration.json'); assert len(restored) == 15
    n = 0
    for mark_path in (OUT/'evaluations').glob('*/*/complete.json'):
        stage, label = mark_path.parent.parent.name, mark_path.parent.name
        mark = read(mark_path); recs = read(mark_path.parent/'per_image.json')
        assert sha(mark_path.parent/'per_image.json') == mark['metrics_sha256']
        for name, digest in mark['prediction_sha256'].items():
            assert sha(WORK/'predictions'/stage/label/name) == digest
            n += 1
        assert len(recs) == len(mark['prediction_sha256'])
        assert all(np.isfinite(r['abs_rel']) for r in recs)
    assert n == 7890
    for tr_path in (OUT/'runs').glob('*/training.json'):
        tr = read(tr_path); label = tr_path.parent.name
        assert tr['steps'] == 320 and tr['protocol_sha256'] == sha(OUT/'protocol.json')
        assert sha(WORK/'checkpoints'/label/'adapter.pt') == tr['checkpoint_sha256']
        assert restored[label]['resolutions'] == [256, 512] and restored[label]['max_abs_difference'] == 0

    manifest = read('results/robustness_v3/split_manifest.json')
    for stage, source in [('nyu31', manifest['fresh31']),
                          ('external', read('results/prospective_v4/manifest.json')['samples'])]:
        ids = [r['id'] for r in source]
        for p in (OUT/'evaluations'/stage).glob('*/per_image.json'):
            assert [r['id'] for r in read(p)] == ids
        prior = Path('results/prospective_v4/evaluations') if stage == 'external' else Path('results/robustness_v3/evaluations/test')
        for p in prior.glob('*/per_image.json'):
            recs = read(p)
            if stage == 'nyu31': recs = [r for r in recs if r['cohort'] == 'fresh31']
            assert [r['id'] for r in recs] == ids, str(p)
        data = arrays(stage)
        for row in read(OUT/f'{stage}_summary.json'):
            assert np.isclose(data[row['group']].mean(), row['abs_rel'], atol=1e-12, rtol=0)
        assert len(read(OUT/f'{stage}_comparisons.json')) == 8

    timed = 0
    for p in (OUT/'cost/inference').glob('*.json'):
        obj = read(p); mode, res = obj['condition'].split('_r')
        prior = WORK if mode == 'low256' else ROOT/'robustness_v3'
        label = ('base' if mode == 'base' else f'draw1_{mode}_seed17') + '_r' + res
        assert [r['id'] for r in obj['records']] == [r['id'] for r in manifest['validation'][:8]]
        for row in obj['records']:
            ref = prior/'predictions/validation'/label/f"{row['id']:04d}.npy"
            assert row['exact_prediction_match'] and sha(ref) == row['reference_sha256']
            assert 0 < row['seconds'] < 60 and row['peak_mib'] > 0
            timed += 1
    assert timed == 320
    budget_n = 0
    for p in (OUT/'cost/budget').glob('*/complete.json'):
        mark = read(p); label = p.parent.name
        assert sha(p.parent/'evaluation.json') == mark['evaluation_sha256']
        tr = read(p.parent/'training.json')
        assert sha(WORK/'budget_checkpoints'/label/'adapter.pt') == tr['checkpoint_sha256']
        assert 120 <= tr['training_seconds'] < 130
        for name, digest in mark['predictions_sha256'].items(): assert sha(WORK/name) == digest
        recs = read(p.parent/'evaluation.json'); assert len(recs) == 64
        for r in recs:
            row = next(x for x in manifest['validation'] if x['id'] == r['id'])
            _, gt = load_sample(ASSETS, row)
            pred = np.load(WORK/'budget_predictions'/label/f"r{r['resolution']}"/f"{r['id']:04d}.npy")
            recomputed = evaluate_depth(pred, gt)[0]
            assert all(np.isclose(r[k], v, atol=1e-10, rtol=0) for k, v in recomputed.items())
            budget_n += 1
    assert budget_n == 576
    assert read(OUT/'cost/verification.json')['status'] == 'passed'
    tracked = subprocess.check_output(['git', 'ls-files'], text=True).splitlines()
    assert not any(Path(p).suffix.lower() in {'.npy','.npz','.pt','.pth','.safetensors','.mat'} for p in tracked)
    result = {'status':'passed', 'historical_result_blobs_unchanged':len(paths),
              'control_raw_prediction_hashes_verified':n, 'control_checkpoint_restorations':30,
              'timing_records_and_reference_hashes_verified':timed,
              'budget_metrics_independently_recomputed':budget_n,
              'all_comparison_image_orders_match':True, 'no_weights_or_raw_arrays_tracked':True}
    write(OUT/'delivery_checks.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
