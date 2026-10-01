"""Validation-only diagnostic of resolution-dependent LoRA strength."""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from peft.tuners.tuners_utils import BaseTunerLayer
from experiment import load_pipe, load_sample, infer_marigold, evaluate_depth, seed_all
from run_expanded import read, write, sha, restore_adapter


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assets', required=True, type=Path)
    p.add_argument('--expanded-work', required=True, type=Path)
    p.add_argument('--prior-work', required=True, type=Path)
    p.add_argument('--work', required=True, type=Path)
    p.add_argument('--results', type=Path, default='results/scaling_v1')
    a = p.parse_args()
    manifest = Path('results/expanded_v1/split_manifest.json')
    rows = [r for r in read(manifest)['samples'] if r['split'] == 'val']
    assert len(rows) == 16
    for r in rows:
        assert sha(a.assets / 'subset' / r['file']) == r['sha256']
    checkpoint = a.expanded_work / 'checkpoints/lora64_seed17/adapter.pt'
    assert sha(checkpoint) == read('results/expanded_v1/runs/lora64_seed17/training.json')['checkpoint_sha256']
    protocol = {'checkpoint_sha256': sha(checkpoint), 'validation_ids': [r['id'] for r in rows],
                'manifest_sha256': sha(manifest), 'resolutions': [256, 512],
                'scales': [0, .25, .5, .75, 1], 'denoising_steps': 1, 'inference_seed_offset': 17,
                'source_sha256': {n: sha(Path(n)) for n in ['explore_scaling.py', 'experiment.py', 'run_expanded.py']},
                'protocol_sha256': sha(Path('SCALING_PROTOCOL.md'))}
    if (a.results / 'protocol.json').exists():
        assert read(a.results / 'protocol.json') == protocol
    else:
        write(a.results / 'protocol.json', protocol)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    seed_all(17)
    pipe = None
    summaries, endpoint_diffs, checked = [], [], 0
    by_condition = {}
    for resolution in protocol['resolutions']:
        for strength in protocol['scales']:
            label = f'r{resolution}_a{int(strength*100):03d}'
            rd = a.results / 'runs' / label
            pred_dir = a.work / 'predictions' / label
            if not (rd / 'complete.json').exists():
                if pipe is None:
                    pipe = load_pipe(read(a.assets / 'model_path.json')['path'])
                    restore_adapter(pipe, 'lora', checkpoint)
                pipe.unet.set_adapters('default', weights=strength)
                for module in pipe.unet.modules():
                    if isinstance(module, BaseTunerLayer):
                        assert module.scaling['default'] == strength
                image, _ = load_sample(a.assets, rows[0])
                infer_marigold(pipe, image, 1, resolution, 17 + rows[0]['id'])
                pred_dir.mkdir(parents=True, exist_ok=True)
                records, hashes = [], {}
                for row in rows:
                    image, gt = load_sample(a.assets, row)
                    pred, seconds, memory = infer_marigold(pipe, image, 1, resolution, 17 + row['id'])
                    metrics, _, _ = evaluate_depth(pred, gt)
                    filename = f"{row['id']:04d}.npy"
                    np.save(pred_dir / filename, pred)
                    hashes[filename] = sha(pred_dir / filename)
                    records.append({'id': row['id'], 'scene': row['scene'], 'split': 'val',
                                    'resolution': resolution, 'strength': strength, **metrics,
                                    'inference_seconds': seconds, 'peak_allocated_mib': memory})
                write(rd / 'per_image.json', records)
                write(rd / 'complete.json', {'metrics_sha256': sha(rd / 'per_image.json'),
                                            'protocol_sha256': sha(a.results / 'protocol.json'),
                                            'prediction_sha256': hashes})
            records = read(rd / 'per_image.json')
            mark = read(rd / 'complete.json')
            assert mark['metrics_sha256'] == sha(rd / 'per_image.json')
            assert mark['protocol_sha256'] == sha(a.results / 'protocol.json')
            assert [r['id'] for r in records] == protocol['validation_ids']
            for row, rec in zip(rows, records):
                assert rec['resolution'] == resolution and rec['strength'] == strength and rec['split'] == 'val'
                filename = f"{row['id']:04d}.npy"
                path = pred_dir / filename
                assert sha(path) == mark['prediction_sha256'][filename]
                pred = np.load(path)
                _, gt = load_sample(a.assets, row)
                metrics, _, _ = evaluate_depth(pred, gt)
                assert all(np.isclose(rec[k], v, atol=1e-10, rtol=0) for k, v in metrics.items())
                checked += 1
                if strength in [0, 1]:
                    mode = 'base' if strength == 0 else 'unmerged'
                    old = np.load(a.prior_work / 'predictions' / f'round1_{mode}_r{resolution}_s1' / filename)
                    difference = float(np.max(np.abs(pred.astype(float) - old.astype(float))))
                    assert difference == 0, (label, row['id'], difference)
                    endpoint_diffs.append(difference)
            by_condition[(resolution, strength)] = np.array([r['abs_rel'] for r in records])
            summaries.append({'resolution': resolution, 'strength': strength,
                              **{k: float(np.mean([r[k] for r in records])) for k in ['abs_rel', 'rmse_m', 'delta1']}})
            print('VERIFIED', label, f"AbsRel={summaries[-1]['abs_rel']:.6f}", flush=True)
    assert checked == 160 and len(endpoint_diffs) == 64
    write(a.results / 'summary.json', summaries)
    with (a.results / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    boot = np.random.default_rng(4245).integers(0, 16, size=(10000, 16))
    comparisons = []
    for resolution in protocol['resolutions']:
        for strength in protocol['scales'][1:]:
            delta = by_condition[(resolution, strength)] - by_condition[(resolution, 0)]
            low, high = np.quantile(delta[boot].mean(axis=1), [.025, .975])
            comparisons.append({'resolution': resolution, 'strength': strength,
                                'abs_rel_difference_vs_base': float(delta.mean()),
                                'scene_ci_low': float(low), 'scene_ci_high': float(high)})
    write(a.results / 'comparisons.json', comparisons)
    minima = {str(res): min([r for r in summaries if r['resolution'] == res], key=lambda r: r['abs_rel'])
              for res in protocol['resolutions']}
    common = [{'strength': strength, 'equal_resolution_mean_abs_rel': float(np.mean(
        [by_condition[(res, strength)].mean() for res in protocol['resolutions']]))} for strength in protocol['scales']]
    write(a.results / 'descriptive_selection.json', {'per_resolution_grid_minima': minima,
          'common_strength_grid_minimum': min(common, key=lambda r: r['equal_resolution_mean_abs_rel']),
          'warning': 'Selected and scored on the same observed validation scenes; descriptive only, not a held-out gain.'})
    write(a.results / 'verification.json', {'status': 'passed', 'validation_data_hashes': 16,
          'prediction_hashes_and_metrics': checked, 'endpoint_predictions_reproduced': len(endpoint_diffs),
          'maximum_endpoint_difference': max(endpoint_diffs), 'new_training_runs': 0, 'test_predictions_generated': 0})
    fig, ax = plt.subplots(figsize=(7, 4.3), layout='constrained')
    for res, color in [(256, '#2563eb'), (512, '#059669')]:
        ax.plot(protocol['scales'], [by_condition[(res, strength)].mean() for strength in protocol['scales']],
                marker='o', color=color, label=f'{res}-pixel long side')
    ax.set(xlabel='LoRA update strength (0 = original model, 1 = full adapter)',
           ylabel='GT-affine-aligned AbsRel (lower is better)', xticks=protocol['scales'],
           title='Validation diagnostic: resolution and adapter strength\n16 scenes; one training checkpoint; one-step inference')
    ax.legend(); ax.grid(alpha=.2)
    fig.savefig(a.results / 'strength_curve.png', dpi=160); plt.close(fig)
    lines = ['# LoRA strength diagnostic on validation scenes', '',
             'All ten predefined settings completed. No training or test inference was performed. '
             'Scale-zero and scale-one predictions exactly reproduced 64 prior validation outputs. '
             'All 160 raw prediction hashes and recomputed metrics passed.', '',
             '| Long side | LoRA strength | AbsRel | RMSE (m) | Delta1 |',
             '|---|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['resolution']} | {r['strength']:.2f} | {r['abs_rel']:.6f} | {r['rmse_m']:.4f} | {r['delta1']:.4f} |")
    lines += ['', '![Adapter strength curve](strength_curve.png)', '',
              '## Interpretation', '',
              f"The observed grid minimum is strength {minima['256']['strength']:.2f} at resolution 256 and "
              f"{minima['512']['strength']:.2f} at resolution 512. These settings are selected and evaluated on the same 16 already observed "
              'validation scenes. They are diagnostic descriptions, not independently validated gains. '
              'The per-resolution gate also has more selection freedom than a single fixed strength.', '',
              'This interpolation is related to existing weight-ensembling methods such as [WiSE-FT](https://arxiv.org/abs/2109.01903). '
              'It is a baseline and a motivation for future work, not an original algorithm. '
              'A future method could combine mixed-resolution training, cross-resolution consistency and confidence-weighted preservation '
              'of the pretrained predictor. Such a method would need controlled ablations, several seeds and new unseen evaluation scenes.', '',
              'See [the frozen protocol](../../SCALING_PROTOCOL.md), `comparisons.json` for paired scene intervals, '
              'and `verification.json`. Timings recorded per image are incidental single-pass diagnostics, not a speed benchmark.']
    (a.results / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('SCALING_DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
