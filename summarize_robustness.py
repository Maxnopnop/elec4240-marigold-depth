"""English report of the predefined replication, including negative findings."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_robustness import arguments, matrix, conditions, DRAWS, SEEDS, fixed_metrics
from run_expanded import read, write
from experiment import load_sample
from robustness_stats import CONTRASTS, bootstrap, holm
from analyze_scaleup_multiplicity import analyze
from crossed_bootstrap_sensitivity import crossed_bootstrap


def csv_out(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    a = arguments()
    out = a.results
    assert read(out/'verification.json')['metrics_recomputed'] == 9182
    manifest = read(out/'split_manifest.json')
    groups = ['base_r256', 'high512_r256', 'mixed_r256', 'base_r512', 'high512_r512', 'mixed_r512', 'expert256', 'expert504']
    metrics = ['abs_rel', 'rmse_m', 'delta1', 'calibrated_abs_rel', 'calibrated_rmse_m', 'calibrated_delta1']
    data = {}
    per_run, costs = [], []
    for cfg in matrix():
        if cfg['mode'] in ['high512', 'mixed']:
            s = read(out/'runs'/cfg['run_id']/'training.json')
            costs.append({**cfg, **{k: s[k] for k in ['training_seconds', 'latent_cache_seconds', 'peak_allocated_mib']}})
        for label, res, _ in conditions(cfg, 'test'):
            allrows = read(out/'evaluations'/'test'/label/'per_image.json')
            group = cfg['run_id'] if cfg['mode'] == 'expert' else f"{cfg['mode']}_r{res}"
            for cohort in ['fresh31', 'observed64']:
                rows = [r for r in allrows if r['cohort'] == cohort]
                assert [r['id'] for r in rows] == [r['id'] for r in manifest[cohort]]
                for metric in metrics + (['native_abs_rel', 'native_rmse_m', 'native_delta1'] if cfg['mode'] == 'expert' else []):
                    key = (cohort, group, metric)
                    if key not in data:
                        shape = (3, 5, len(rows)) if cfg['mode'] in ['high512', 'mixed'] else (1, 1, len(rows))
                        data[key] = np.full(shape, np.nan)
                    d = DRAWS.index(cfg['draw']) if cfg['draw'] else 0
                    s = SEEDS.index(cfg['seed']) if cfg['seed'] else 0
                    data[key][d, s] = [r[metric] for r in rows]
                per_run.append({'cohort': cohort, 'group': group, 'label': label, 'draw': cfg['draw'], 'seed': cfg['seed'],
                                **{m: float(np.mean([r[m] for r in rows])) for m in metrics},
                                'inference_seconds': float(np.mean([r['inference_seconds'] for r in rows]))})
    assert all(np.isfinite(v).all() for v in data.values())
    summary = []
    for cohort in ['fresh31', 'observed64']:
        for group in groups:
            arr = data[(cohort, group, 'abs_rel')]
            item = {'cohort': cohort, 'group': group, 'scenes': arr.shape[-1], 'training_draws': arr.shape[0], 'seeds': arr.shape[1]}
            for metric in metrics:
                v = data[(cohort, group, metric)]
                item[metric] = float(v.mean())
                item[metric+'_run_sd'] = float(v.mean(axis=2).std(ddof=1)) if v.size > v.shape[-1] else None
            item['draw_abs_rel_means'] = arr.mean(axis=(1, 2)).tolist()
            if group.startswith('expert'):
                for m in ['native_abs_rel', 'native_rmse_m', 'native_delta1']:
                    item[m] = float(data[(cohort, group, m)].mean())
            summary.append(item)
    write(out/'group_summary.json', summary)
    csv_out(out/'per_run_summary.csv', per_run); csv_out(out/'training_cost.csv', costs)
    comparisons = []
    for cohort in ['fresh31', 'observed64']:
        for metric in ['abs_rel', 'calibrated_abs_rel']:
            family = []
            for compared, reference in CONTRASTS:
                delta = data[(cohort, compared, metric)]-data[(cohort, reference, metric)]
                result = bootstrap(delta)
                result['crossed'] = crossed_bootstrap(delta)
                family.append({'cohort': cohort, 'metric': metric, 'compared': compared, 'reference': reference, **result})
            for kind in ['scene', 'hierarchical', 'crossed']:
                for r, p in zip(family, holm([r[kind]['p_bootstrap'] for r in family])):
                    r[kind]['p_holm'] = p
            comparisons.extend(family)
    write(out/'paired_comparisons.json', comparisons)
    flat = [{'cohort': r['cohort'], 'metric': r['metric'], 'compared': r['compared'], 'reference': r['reference'],
             'uncertainty': kind, **r[kind]} for r in comparisons for kind in ['scene', 'hierarchical', 'crossed']]
    csv_out(out/'paired_comparisons.csv', flat)
    posthoc = analyze(out/'scaleup_posthoc_multiplicity.json')
    c = next(iter(read(out/'calibration.json')['conditions'].values()))['constant_depth_m']
    constant = []
    for cohort in ['fresh31', 'observed64']:
        for row in manifest[cohort]:
            _, gt = load_sample(a.assets, row)
            constant.append({'cohort': cohort, 'id': row['id'], 'constant_depth_m': c,
                             **fixed_metrics(np.full(gt.shape, c), gt)})
    write(out/'constant_baseline.json', constant)
    corruption_rows = []
    for cfg in matrix():
        for label, _, corruption in conditions(cfg, 'corruption'):
            rows = read(out/'evaluations'/'corruption'/label/'per_image.json')
            clean_label = label.rsplit('_', 1)[0]
            clean = [r for r in read(out/'evaluations'/'test'/clean_label/'per_image.json') if r['cohort'] == 'fresh31']
            assert [r['id'] for r in rows] == [r['id'] for r in clean]
            corruption_rows.append({'mode': cfg['mode'], 'run_id': cfg['run_id'], 'corruption': corruption,
                                    **{m: float(np.mean([r[m] for r in rows])) for m in metrics},
                                    'aligned_change': float(np.mean([r['abs_rel']-c['abs_rel'] for r, c in zip(rows, clean)])),
                                    'calibrated_change': float(np.mean([r['calibrated_abs_rel']-c['calibrated_abs_rel'] for r, c in zip(rows, clean)])),
                                    'native_abs_rel': float(np.mean([r['native_abs_rel'] for r in rows])) if cfg['mode'] == 'expert' else None})
    csv_out(out/'corruption_summary.csv', corruption_rows)
    corruption_comparisons = []
    for kind in ['dark', 'blur']:
        for metric in ['abs_rel', 'calibrated_abs_rel']:
            scene_values = {}
            for mode in ['base', 'mixed', 'expert']:
                batches = []
                for cfg in matrix():
                    if cfg['mode'] != mode:
                        continue
                    for label, _, corruption in conditions(cfg, 'corruption'):
                        if corruption == kind:
                            rr = read(out/'evaluations'/'corruption'/label/'per_image.json')
                            batches.append([r[metric] for r in rr])
                scene_values[mode] = np.mean(batches, axis=0)
            for reference in ['base', 'expert']:
                delta = scene_values['mixed']-scene_values[reference]
                corruption_comparisons.append({'corruption': kind, 'metric': metric, 'compared': 'mixed',
                                               'reference': reference, 'mean_difference': float(delta.mean()),
                                               'scene_win_fraction': float(np.mean(delta < 0))})
    write(out/'corruption_comparisons.json', corruption_comparisons)
    # Figures: display all draws and uncertainty, without selecting a winning run.
    figdir = out/'figures'; figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for ax, res in zip(axes, [256, 512]):
        for d in range(3):
            vals = [data[('fresh31', f'{m}_r{res}', 'abs_rel')][d].mean() for m in ['high512', 'mixed']]
            err = [data[('fresh31', f'{m}_r{res}', 'abs_rel')][d].mean(axis=1).std(ddof=1) for m in ['high512', 'mixed']]
            ax.errorbar([0, 1], vals, yerr=err, marker='o', capsize=4, label=f'Training draw {d+1}')
        ax.axhline(data[('fresh31', f'base_r{res}', 'abs_rel')].mean(), color='#475569', ls='--', label='Pretrained')
        ax.set(xticks=[0, 1], xticklabels=['Train512', 'Mixed'], ylabel='GT-aligned AbsRel', title=f'Inference {res} | fresh 31 scenes')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Three independent training draws | five seeds per method | error bars: seed SD')
    fig.savefig(figdir/'training_draws.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5.2), layout='constrained')
    primary = [r for r in comparisons if r['cohort'] == 'fresh31' and r['metric'] == 'abs_rel']
    labels = []
    for i, r in enumerate(primary):
        h = r['hierarchical']; labels.append(f"{r['compared']} - {r['reference']}\nHolm p={h['p_holm']:.4f}")
        ax.plot([h['bonferroni_ci_low'], h['bonferroni_ci_high']], [i, i], color='#94a3b8', lw=3)
        ax.plot([h['ci_low'], h['ci_high']], [i, i], color='#2563eb', lw=5)
        ax.scatter(h['difference'], i, color='#0f172a', zorder=3)
    ax.axvline(0, color='black', ls='--', lw=1)
    ax.set(yticks=range(len(primary)), yticklabels=labels, xlabel='Aligned AbsRel difference (negative favors first method)',
           title='Fresh 31 scenes | paired training-draw / seed / scene bootstrap')
    ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
    fig.suptitle('Blue: 95% interval | gray: Bonferroni interval for six comparisons', fontsize=10)
    fig.savefig(figdir/'corrected_comparisons.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    show = ['base_r512', 'high512_r512', 'mixed_r512', 'expert256', 'expert504']
    names = ['Original512', 'Train512', 'Mixed512', 'Expert336', 'Expert504']
    for ax, metric, title in zip(axes, ['abs_rel', 'calibrated_abs_rel'], ['GT alignment per test image', 'Fixed validation-only calibration']):
        vals = [float(data[('fresh31', g, metric)].mean()) for g in show]
        ax.bar(range(5), vals, color=['#64748b', '#059669', '#d97706', '#a78bfa', '#7c3aed'])
        ax.set(xticks=range(5), xticklabels=names, ylabel='AbsRel (lower is better)', title=title)
        ax.tick_params(axis='x', labelrotation=20); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        for i, v in enumerate(vals):
            ax.annotate(f'{v:.3f}', (i, v), xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
        ax.set_ylim(0, max(vals)*1.2)
    fig.suptitle('Fresh test scenes | aligned structure and calibrated metric depth are different evaluations')
    fig.savefig(figdir/'alignment_vs_calibration.png', dpi=150); plt.close(fig)
    lookup = {(r['cohort'], r['group']): r for r in summary}
    lines = ['# Robustness replication: training draws, uncertainty, and depth calibration', '',
             'This study tests whether the earlier mixed-resolution result survives new training samples, more seeds, '
             'multiplicity adjustment, and evaluation without fitting to test depths. All protocol choices were fixed before the new runs.', '',
             '## Design and audit', '',
             '- Three new scene-uniform training draws of 128 from 217 eligible scenes; five seeds (17,29,43,59,71); '
             'high512 and mixed; 30 fresh trainings of 320 updates.',
             '- Training draws share 76, 68, and 68 scenes pairwise. They are independent random draws from a fixed pool, '
             'not disjoint datasets or three different domains.',
             '- Fresh primary cohort: all 31 previously unused official test scenes. The already observed 64-scene cohort is secondary. '
             'The other 120 previously tested scenes were not rerun. All 215 official test scenes have now been observed across project phases.',
             '- Validation-only calibration uses 32 scenes. All 9,182 saved predictions, 335 input hashes, 30 checkpoints, paired '
             'training schedules and four restoration checks passed the audit. No test result selected a checkpoint or hyperparameter.', '',
             '## Fresh-scene scores', '',
             'AbsRel, RMSE and delta1 below use the same crop/range. Each adapted entry averages three training draws and five seeds. '
             'Aligned metrics fit one affine transform using each test image\'s GT. Calibrated metrics use a fixed transform learned on validation only.', '',
             '| Method | Aligned AbsRel | Aligned RMSE (m) | Aligned delta1 | Calibrated AbsRel | Calibrated RMSE (m) | Calibrated delta1 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for group in groups:
        r = lookup[('fresh31', group)]
        lines.append(f"| {group} | {r['abs_rel']:.5f} | {r['rmse_m']:.4f} | {r['delta1']:.4f} | {r['calibrated_abs_rel']:.5f} | {r['calibrated_rmse_m']:.4f} | {r['calibrated_delta1']:.4f} |")
    lines += ['', 'Native metric-depth references (no fitted scale or shift):', '', '| Reference | AbsRel | RMSE (m) | delta1 |', '|---|---:|---:|---:|']
    for group in ['expert256', 'expert504']:
        r = lookup[('fresh31', group)]
        lines.append(f"| {group} native | {r['native_abs_rel']:.5f} | {r['native_rmse_m']:.4f} | {r['native_delta1']:.4f} |")
    cc = [r for r in constant if r['cohort'] == 'fresh31']
    lines.append(f"| Constant {c:.3f} m from validation | {np.mean([r['abs_rel'] for r in cc]):.5f} | {np.mean([r['rmse_m'] for r in cc]):.4f} | {np.mean([r['delta1'] for r in cc]):.4f} |")
    lines += ['', '![Training draw replication](figures/training_draws.png)', '', '![Alignment versus calibration](figures/alignment_vs_calibration.png)', '',
              '## Predefined fresh-scene comparisons', '',
              'Negative differences favor the first method. Intervals below resample training draws, seeds within each draw, and paired test scenes. '
              'Holm p-values cover the six aligned comparisons as one family. Bootstrap p-values and interval coverage are approximate with only three training draws. '
              'The CSV also reports scene-only intervals and conservative Bonferroni intervals.', '',
              'The pre-test [sensitivity addendum](../../STATISTICAL_SENSITIVITY_V3.md) also resamples seed indices jointly across training draws '
              'because numerical seeds share initialization/schedules. Both analyses are shown; no favorable analysis is selected.', '',
              '| Comparison | Difference | Nested 95% interval | Nested Holm p | Crossed 95% interval | Crossed Holm p |', '|---|---:|---|---:|---|---:|']
    for r in primary:
        h = r['hierarchical']
        x = r['crossed']
        lines.append(f"| {r['compared']} - {r['reference']} | {h['difference']:+.5f} | [{h['ci_low']:+.5f}, {h['ci_high']:+.5f}] | {h['p_holm']:.4f} | [{x['ci_low']:+.5f}, {x['ci_high']:+.5f}] | {x['p_holm']:.4f} |")
    lines += ['', '![Multiplicity-aware comparisons](figures/corrected_comparisons.png)', '', '## Previously observed 64-scene cohort', '',
              'This cohort is a replication/sensitivity check for new trainings; it is not newly unseen test data.', '',
              '| Method | Aligned AbsRel | Calibrated AbsRel |', '|---|---:|---:|']
    for group in groups:
        r = lookup[('observed64', group)]
        lines.append(f"| {group} | {r['abs_rel']:.5f} | {r['calibrated_abs_rel']:.5f} |")
    lines += ['', '## Controlled image perturbations', '',
              'Fresh 31 scenes only. Mixed averages all 15 checkpoints; calibration stays fixed from clean validation. '
              'These descriptive checks test two synthetic perturbations within NYUv2, not cross-dataset generalization.', '',
              '| Perturbation | Method | Aligned AbsRel | Change from clean | Calibrated AbsRel |', '|---|---|---:|---:|---:|']
    for corruption in ['dark', 'blur']:
        for mode in ['base', 'mixed', 'expert']:
            rr = [r for r in corruption_rows if r['corruption'] == corruption and r['mode'] == mode]
            lines.append(f"| {corruption} | {mode} | {np.mean([r['abs_rel'] for r in rr]):.5f} | {np.mean([r['aligned_change'] for r in rr]):+.5f} | {np.mean([r['calibrated_abs_rel'] for r in rr]):.5f} |")
    lines += ['', 'The per-scene win fractions and paired descriptive changes are in `corruption_comparisons.json`; '
              'the specialist\'s native metric scores under each perturbation are in `corruption_summary.csv`.', '']
    lines += ['', '## Retrospective multiple-comparison sensitivity', '',
              'All 21 originally published scaleup_v2 contrasts are included in a post-hoc Holm family. This reanalysis averages the '
              'original three training seeds first and resamples scenes only; it does not retroactively make the original study preregistered.', '',
              '| Earlier comparison | Difference | Holm p |', '|---|---:|---:|']
    for r in posthoc['comparisons']:
        lines.append(f"| {r['compared']} - {r['reference']} | {r['difference']:+.5f} | {r['p_holm']:.4f} |")
    lines += ['', '## Computational cost and limits', '',
              f"- New training total: {sum(r['training_seconds'] for r in costs)/60:.2f} minutes, excluding latent caching, model loading, evaluation and data extraction; peak allocated training VRAM {max(r['peak_allocated_mib'] for r in costs):.1f} MiB.",
              '- Three draws and five seeds strengthen the evidence within this finite NYUv2 pool, but do not establish universal performance. The fresh cohort has only 31 scenes and is deliberately separate from the previously observed cohort.',
              '- The specialist has different prior supervision, precision, architecture and compute. Its 378x504 input is approximately matched to Marigold 384x512, not an architecture-controlled comparison. Its prior NYUv2 supervision can include our validation scenes.',
              '- Calibration removes test-target fitting but adds a simple validation-trained postprocessor. It does not convert the original Marigold architecture into a natively metric model; conclusions concern the measured calibrated system and indoor split.',
              '- Mixed-prior and low256 were not retrained in this phase. No conclusion about their across-training-set robustness or equivalence is implied.',
              '- Centered bootstrap p-values are approximate; small numbers of training draws constrain uncertainty estimates. Report effect sizes, both interval types, all seed/draw scores and the adjusted comparisons together.',
              '- Raw weights/data/predictions remain local. Code, hashes, coefficients, scores and figures are published in the private course repository.', '']
    (out/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('ROBUSTNESS_REPORT_READY', flush=True)
    for r in primary:
        print(r['compared'], r['reference'], r['hierarchical'], flush=True)


if __name__ == '__main__':
    main()
