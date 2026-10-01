"""Recompute every validation prediction, then report paired quality and timing."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from explore_inference import arguments, inputs, configurations
from experiment import load_sample, evaluate_depth
from run_expanded import read, write, sha


def csv_out(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    a = arguments()
    rows, checkpoint, manifest = inputs(a)
    out = a.results
    protocol = read(out / 'protocol.json')
    assert protocol['checkpoint_sha256'] == sha(checkpoint)
    assert protocol['manifest_sha256'] == sha(manifest)
    assert protocol['protocol_sha256'] == sha(Path('EXPLORATION_PROTOCOL.md'))
    for name, checksum in protocol['source_sha256'].items():
        assert sha(Path(name)) == checksum, name
    ids = [r['id'] for r in rows]
    assert protocol['validation_ids'] == ids
    configs = configurations()
    all_records, grouped, repeated, restoration = [], {}, [], []
    summary = []
    # Keep only first-round arrays in RAM (12 x 16 x 480 x 640).
    predictions = {}
    for cfg in configs:
        label = cfg['label']
        rounds = []
        for round_index in range(1, 4):
            run = f'round{round_index}_{label}'
            rd = out / 'runs' / run
            records = read(rd / 'per_image.json')
            mark = read(rd / 'complete.json')
            assert mark['protocol_sha256'] == sha(out / 'protocol.json')
            assert mark['metrics_sha256'] == sha(rd / 'per_image.json')
            assert [r['id'] for r in records] == ids and len(records) == 16
            assert (mark['adapter_wrappers_after'] > 0) == (cfg['mode'] == 'unmerged')
            for row, record in zip(rows, records):
                assert record['split'] == 'val' and record['round'] == round_index
                assert all(record[k] == v for k, v in cfg.items())
                filename = f"{row['id']:04d}.npy"
                path = a.work / 'predictions' / run / filename
                assert sha(path) == mark['prediction_sha256'][filename]
                pred = np.load(path)
                _, gt = load_sample(a.assets, row)
                metrics, _, _ = evaluate_depth(pred, gt)
                for key, value in metrics.items():
                    assert np.isclose(value, record[key], atol=1e-10, rtol=0), (run, key)
                assert record['inference_seconds'] > 0 and record['peak_allocated_mib'] > 0
                if round_index == 1:
                    predictions[(label, row['id'])] = pred
                    if cfg['resolution'] == 256 and cfg['mode'] in ['base', 'unmerged']:
                        old_label = ('base' if cfg['mode'] == 'base' else 'lora64_seed17') + f"_s{cfg['steps']}"
                        old = np.load(a.expanded_work / 'predictions' / old_label / filename)
                        diff = float(np.max(np.abs(pred.astype(float) - old.astype(float))))
                        assert diff == 0, (label, row['id'], diff)
                        restoration.append(diff)
                else:
                    diff = float(np.max(np.abs(pred.astype(float) - predictions[(label, row['id'])].astype(float))))
                    repeated.append(diff)
            rounds.append(records)
            all_records.extend(records)
        grouped[label] = {idx: {key: float(np.mean([batch[j][key] for batch in rounds]))
                               for key in ['abs_rel', 'rmse_m', 'delta1', 'inference_seconds']}
                          for j, idx in enumerate(ids)}
        times = [np.mean([r['inference_seconds'] for r in batch]) for batch in rounds]
        summary.append({**cfg, 'validation_scenes': 16, 'timing_rounds': 3,
                        **{key: float(np.mean([v[key] for v in grouped[label].values()]))
                           for key in ['abs_rel', 'rmse_m', 'delta1']},
                        'seconds_mean': float(np.mean(times)), 'seconds_round_std': float(np.std(times, ddof=1)),
                        'peak_allocated_mib': max(r['peak_allocated_mib'] for batch in rounds for r in batch)})
    assert len(all_records) == 576
    write(out / 'verification.json', {'status': 'passed', 'validation_hashes_checked': 16,
          'prediction_hashes_and_metrics_checked': 576, 'configurations': 12, 'timing_rounds': 3,
          'expanded_validation_predictions_reproduced': len(restoration),
          'maximum_restoration_difference': max(restoration),
          'repeated_predictions_compared': len(repeated), 'maximum_repeat_difference': max(repeated),
          'test_predictions_generated': 0, 'checkpoint_and_frozen_sources_verified': True})
    csv_out(out / 'summary.csv', summary)
    write(out / 'summary.json', summary)
    rng = np.random.default_rng(4244)
    bootstrap_indices = rng.integers(0, 16, size=(10000, 16))
    comparisons = []
    pairs = [(f'merged_r{res}_s{s}', f'unmerged_r{res}_s{s}', 'merge')
             for res in [256, 512] for s in [1, 4]]
    pairs += [(f'{m}_r512_s{s}', f'{m}_r256_s{s}', 'resolution')
              for m in ['base', 'unmerged', 'merged'] for s in [1, 4]]
    lookup = {r['label']: r for r in summary}
    for compared, reference, factor in pairs:
        delta = np.array([grouped[compared][idx]['abs_rel'] - grouped[reference][idx]['abs_rel'] for idx in ids])
        low, high = np.quantile(delta[bootstrap_indices].mean(axis=1), [.025, .975])
        comparisons.append({'factor': factor, 'compared': compared, 'reference': reference,
                            'abs_rel_difference': float(delta.mean()), 'scene_ci_low': float(low),
                            'scene_ci_high': float(high),
                            'latency_ratio': lookup[compared]['seconds_mean'] / lookup[reference]['seconds_mean']})
    csv_out(out / 'paired_comparisons.csv', comparisons)
    merge_differences = []
    for res in [256, 512]:
        for steps in [1, 4]:
            for idx in ids:
                diff = np.abs(predictions[(f'merged_r{res}_s{steps}', idx)].astype(float) -
                              predictions[(f'unmerged_r{res}_s{steps}', idx)].astype(float))
                merge_differences.append({'resolution': res, 'steps': steps, 'id': idx,
                                          'raw_prediction_max_abs_difference': float(diff.max()),
                                          'raw_prediction_mean_abs_difference': float(diff.mean())})
    csv_out(out / 'merge_prediction_differences.csv', merge_differences)
    figdir = out / 'figures'
    figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'base': '#64748b', 'unmerged': '#2563eb', 'merged': '#059669'}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout='constrained')
    for ax, steps in zip(axes, [1, 4]):
        for mode in ['base', 'unmerged', 'merged']:
            selected = [lookup[f'{mode}_r{res}_s{steps}'] for res in [256, 512]]
            ax.errorbar([r['seconds_mean'] * 1000 for r in selected], [r['abs_rel'] for r in selected],
                        xerr=[r['seconds_round_std'] * 1000 for r in selected], capsize=3,
                        color=colors[mode], label=mode.capitalize())
            for r in selected:
                ax.scatter(r['seconds_mean'] * 1000, r['abs_rel'], color=colors[mode],
                           marker='o' if r['resolution'] == 256 else 's', s=35, zorder=4)
        ax.set(xlabel='Mean inference latency (ms/image)', ylabel='Aligned AbsRel (lower is better)',
               title=f'{steps} denoising step' + ('s' if steps > 1 else ''))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Validation only: resolution and LoRA merging\n16 scenes; three timing rounds | circles: 256 pixels; squares: 512 pixels', fontsize=12)
    fig.savefig(figdir / 'quality_latency.png', dpi=160)
    plt.close(fig)
    # Fixed first two validation frames, not selected for visual improvement.
    labels = ['base_r256_s1', 'base_r512_s1', 'unmerged_r256_s1', 'unmerged_r512_s1', 'merged_r512_s1']
    titles = ['RGB', 'Ground truth', 'Base 256', 'Base 512', 'LoRA 256', 'LoRA 512', 'Merged LoRA 512']
    fig, axes = plt.subplots(2, 7, figsize=(17.5, 4.8), layout='constrained')
    for index, row in enumerate(rows[:2]):
        image, gt = load_sample(a.assets, row)
        axes[index, 0].imshow(image)
        axes[index, 1].imshow(gt, cmap='magma_r', vmin=0, vmax=7)
        for column, label in enumerate(labels, 2):
            metrics, aligned, _ = evaluate_depth(predictions[(label, row['id'])], gt)
            axes[index, column].imshow(aligned, cmap='magma_r', vmin=0, vmax=7)
            axes[index, column].set_xlabel(f"AbsRel {metrics['abs_rel']:.4f}", fontsize=9)
        for column in range(7):
            axes[index, column].set_xticks([])
            axes[index, column].set_yticks([])
            if index == 0:
                axes[index, column].set_title(titles[column], fontsize=10)
        axes[index, 0].set_ylabel(f"Frame {row['id']}", fontsize=10)
    fig.suptitle('Fixed first two validation examples | GT-affine-aligned | 1 step | shared depth colors: 0-7 m', fontsize=12)
    fig.savefig(figdir / 'validation_examples.png', dpi=145)
    plt.close(fig)
    lines = ['# Validation exploration: LoRA merging and inference resolution', '',
             'Twelve predefined configurations completed three timing rounds on 16 validation scenes (576 predictions). '
             'All use the same fixed LoRA64 seed-17 checkpoint trained at long side 256; there is no additional training. '
             'Pretrained and LoRA results at resolution 256 reproduce all 64 corresponding expanded validation predictions exactly. '
             'The expanded test results were not changed or re-evaluated.', '',
             '**These are exploratory validation results, not new held-out test results.** '
             'All scores use GT affine alignment and measure relative depth structure.', '',
             '## Findings', '',
             'Merging reduced measured latency by 9.9% (one step) and 11.6% (four steps) at resolution 256, '
             'with smaller 2.8% and 4.6% reductions at 512. Mean AbsRel changes were below 0.00009; '
             'all four paired scene intervals crossed zero. This is a useful local speed improvement with small observed mean accuracy changes, '
             'not proof of numerical equivalence or a universal no-loss guarantee.', '',
             'Increasing resolution from 256 to 512 reduced one-step unmerged LoRA AbsRel from 0.09676 to 0.07527 (22.2%), '
             'while latency increased from 75.6 to 198.7 ms (2.63x). All six resolution comparisons had negative paired scene intervals. '
             'Higher resolution benefits the pretrained baseline too: its one-step AbsRel reaches 0.07362, below the LoRA result of 0.07527. '
             'At four steps, pretrained and unmerged LoRA are nearly tied (0.07877 and 0.07869). '
             'The low-resolution adaptation advantage therefore does not automatically transfer to higher resolution.', '',
             'A motivated follow-up is to compare training resolution, training strength and preservation of pretrained behavior. '
             'That requires a new validation protocol and unseen scenes for any final selected configuration; it is not claimed as established by this study.', '',
             '## Measurements', '',
             'Latency SD is the sample SD of three round means, not training-seed variation. '
             'Each condition is warmed up twice; loading/merging time is excluded. Peak memory is allocated CUDA memory, not total device use.', '',
             '| Model | Long side | Steps | AbsRel | RMSE (m) | Delta1 | Latency (ms, mean +/- SD) | Peak MiB |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['mode']} | {r['resolution']} | {r['steps']} | {r['abs_rel']:.5f} | {r['rmse_m']:.4f} | {r['delta1']:.4f} | {r['seconds_mean']*1000:.1f} +/- {r['seconds_round_std']*1000:.1f} | {r['peak_allocated_mib']:.0f} |")
    lines += ['', '## Paired comparisons', '',
              'AbsRel difference is compared minus reference; negative means lower error. '
              'Intervals resample the 16 scenes after averaging timing repeats, with 10,000 draws and seed 4244. '
              'They do not account for checkpoint selection, other datasets, hardware or multiple comparisons. '
              'Latency ratio below 1 means the compared configuration is faster.', '',
              '| Compared | Reference | AbsRel difference | 95% scene interval | Latency ratio |',
              '|---|---|---:|---:|---:|']
    for r in comparisons:
        lines.append(f"| {r['compared']} | {r['reference']} | {r['abs_rel_difference']:+.5f} | [{r['scene_ci_low']:+.5f}, {r['scene_ci_high']:+.5f}] | {r['latency_ratio']:.3f} |")
    lines += ['', '## Numerical checks', '',
              f"All 576 prediction hashes and recomputed metric records passed. The maximum raw-prediction difference across timing repeats was {max(repeated):.8f}. "
              f"The largest merged-versus-unmerged raw-prediction difference was {max(r['raw_prediction_max_abs_difference'] for r in merge_differences):.6f} "
              '(model output units before alignment). Fusion is therefore assessed as a numerical change as well as a runtime optimization. '
              'Adapter wrappers were confirmed absent after merging. Per-image differences are recorded in `merge_prediction_differences.csv`.', '',
              '## Figures', '', '![Validation quality and latency](figures/quality_latency.png)', '',
              '![Fixed validation examples](figures/validation_examples.png)', '',
              '## Interpretation limits', '',
              'One checkpoint and 16 validation scenes cannot establish a general accuracy improvement. '
              'Resolution changes the latent shape and sampled noise, even with the same numeric seed. '
              'The model was trained at 256, so 512 tests inference-resolution transfer. '
              'Three timing rounds on one working laptop support local engineering comparisons, not hardware-independent speed claims. '
              'Use these findings to motivate a future validation study; independent unseen evaluation is needed for newly selected configurations.', '',
              'See the frozen [protocol](../../EXPLORATION_PROTOCOL.md), `verification.json`, and the CSV/JSON files for reproducibility.']
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('VERIFIED_AND_SUMMARIZED', len(all_records), 'predictions', flush=True)
    for r in summary:
        print(r['label'], f"AbsRel={r['abs_rel']:.5f}", f"time={r['seconds_mean']*1000:.1f}ms", flush=True)


if __name__ == '__main__':
    main()
