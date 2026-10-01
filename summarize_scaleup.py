"""Report the fixed larger-scale matrix, including seed and scene uncertainty."""
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_scaleup import arguments, matrix, labels
from run_expanded import read, write
from experiment import load_sample, evaluate_depth
from train_scaleup import MODES


def csv_out(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    a = arguments()
    assert a.stage == 'test'
    assert read(a.results/'verification.json')['metrics_recomputed'] == 2592
    out = a.results
    order = [f'{m}_r{r}' for m in ['base']+MODES for r in [256, 512]]+['expert']
    names = {'base': 'Original', 'low256': 'Train256', 'high512': 'Train512',
             'mixed': 'Mixed', 'mixed_prior': 'Mixed + prior', 'expert': 'Depth Anything V2'}
    batches, training, per_run = defaultdict(list), [], []
    for cfg in matrix():
        if cfg['mode'] in MODES:
            s = read(out/'runs'/cfg['run_id']/'training.json')
            training.append({**cfg, **{k: s[k] for k in ['training_seconds', 'latent_cache_seconds', 'peak_allocated_mib', 'trainable_parameters']}})
        for stage in ['validation', 'test']:
            for label, resolution in labels(cfg):
                group = 'expert' if resolution is None else f"{cfg['mode']}_r{resolution}"
                records = read(out/'evaluations'/stage/label/'per_image.json')
                batches[(stage, group)].append(records)
                per_run.append({'stage': stage, 'group': group, 'label': label, 'seed': cfg['seed'],
                                **{k: float(np.mean([r[k] for r in records])) for k in
                                   ['abs_rel', 'rmse_m', 'delta1', 'inference_seconds']},
                                'peak_allocated_mib': max(r['peak_allocated_mib'] for r in records)})
    summaries, scenes = [], {}
    for stage in ['validation', 'test']:
        for group in order:
            runs = [r for r in per_run if r['stage'] == stage and r['group'] == group]
            item = {'stage': stage, 'group': group, 'runs': len(runs), 'scenes': 32 if stage == 'validation' else 64}
            for metric in ['abs_rel', 'rmse_m', 'delta1', 'inference_seconds', 'peak_allocated_mib']:
                vals = [r[metric] for r in runs]
                item[metric+'_mean'] = float(np.mean(vals))
                item[metric+'_seed_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
            summaries.append(item)
            ids = [r['id'] for r in batches[(stage, group)][0]]
            scenes[(stage, group)] = {idx: float(np.mean([batch[j]['abs_rel'] for batch in batches[(stage, group)]])) for j, idx in enumerate(ids)}
    csv_out(out/'per_run_summary.csv', per_run); csv_out(out/'group_summary.csv', summaries)
    csv_out(out/'training_cost.csv', training); write(out/'group_summary.json', summaries)
    pairs = [(f'{m}_r{r}', f'base_r{r}') for m in MODES for r in [256, 512]]
    for r in [256, 512]:
        pairs += [(f'high512_r{r}', f'low256_r{r}'), (f'mixed_r{r}', f'low256_r{r}'),
                  (f'mixed_r{r}', f'high512_r{r}'), (f'mixed_prior_r{r}', f'mixed_r{r}')]
    pairs += [(f'{m}_r512', f'{m}_r256') for m in ['base']+MODES]
    contrasts = []
    ids = list(scenes[('test', 'base_r256')])
    boot = np.random.default_rng(4254).integers(0, 64, size=(10000, 64))
    for compared, reference in pairs:
        delta = np.array([scenes[('test', compared)][idx]-scenes[('test', reference)][idx] for idx in ids])
        low, high = np.quantile(delta[boot].mean(axis=1), [.025, .975])
        contrasts.append({'compared': compared, 'reference': reference, 'abs_rel_difference': float(delta.mean()),
                          'scene_ci_low': float(low), 'scene_ci_high': float(high)})
    csv_out(out/'paired_comparisons.csv', contrasts)
    lookup = {(r['stage'], r['group']): r for r in summaries}
    figdir = out/'figures'; figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#64748b', '#2563eb', '#059669', '#d97706', '#9333ea']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for ax, res in zip(axes, [256, 512]):
        groups = [f'{m}_r{res}' for m in ['base']+MODES]
        means = [lookup[('test', g)]['abs_rel_mean'] for g in groups]
        errors = [lookup[('test', g)]['abs_rel_seed_std'] or 0 for g in groups]
        ax.bar(range(5), means, yerr=errors, capsize=4, color=colors)
        ax.axhline(lookup[('test', 'expert')]['abs_rel_mean'], ls='--', color='#111827', label='Specialist reference')
        ax.set_xticks(range(5), [names[m] for m in ['base']+MODES], rotation=20, ha='right')
        ax.set(title=f'Inference long side {res}', ylabel='Aligned AbsRel (lower is better)')
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        for i, value in enumerate(means):
            ax.text(i, value+errors[i]+.0015, f'{value:.4f}', ha='center', va='bottom', fontsize=8)
        ax.set_ylim(0, max(max(means), lookup[('test', 'expert')]['abs_rel_mean'])*1.18)
        ax.legend(fontsize=8)
    fig.suptitle('64 previously unused test scenes | 128 training scenes | error bars: training-seed SD', fontsize=12)
    fig.savefig(figdir/'test_ablation.png', dpi=140); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
    for m, color in zip(MODES, colors[1:]):
        runs = [r for r in training if r['mode'] == m]
        times = [r['training_seconds'] for r in runs]
        score = lookup[('test', f'{m}_r512')]
        axes[0].errorbar(np.mean(times), score['abs_rel_mean'], xerr=np.std(times, ddof=1),
                         yerr=score['abs_rel_seed_std'], marker='o', capsize=3, color=color, label=names[m])
        histories = [read(out/'runs'/r['run_id']/'training.json')['history'] for r in runs]
        losses = np.array([[h['supervised_loss'] for h in hist] for hist in histories])
        smooth = np.convolve(losses.mean(axis=0), np.ones(20)/20, mode='valid')
        axes[1].plot(np.arange(20, 321), smooth, color=color, label=names[m])
    axes[0].set(xlabel='Training seconds (cache/loading excluded)', ylabel='Test AbsRel at inference 512', title='Accuracy and training cost (bars: run SD)')
    axes[1].set(xlabel='Optimizer update', ylabel='Supervised velocity MSE', title='Mean over seeds; 20-update moving average')
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(figdir/'training_cost.png', dpi=140); plt.close(fig)
    rows = [r for r in read(out/'split_manifest.json')['samples'] if r['split'] == 'test'][:3]
    columns = [('base_r512', 'Original512'), ('low256_seed17_r512', 'Train256'),
               ('high512_seed17_r512', 'Train512'), ('mixed_seed17_r512', 'Mixed'),
               ('mixed_prior_seed17_r512', 'Mixed + prior')]
    fig, axes = plt.subplots(3, 8, figsize=(18, 7), layout='constrained')
    for i, row in enumerate(rows):
        image, gt = load_sample(a.assets, row)
        axes[i, 0].imshow(image); axes[i, 1].imshow(gt, cmap='magma_r', vmin=0, vmax=7)
        for j, (label, title) in enumerate(columns, 2):
            pred = np.load(a.work/'predictions'/'test'/label/f"{row['id']:04d}.npy")
            metric, aligned, mask = evaluate_depth(pred, gt)
            axes[i, j].imshow(aligned, cmap='magma_r', vmin=0, vmax=7)
            axes[i, j].set_xlabel(f"AbsRel {metric['abs_rel']:.3f}", fontsize=8)
        axes[i, 7].imshow(np.where(mask, abs(aligned-gt), np.nan), cmap='inferno', vmin=0, vmax=1)
        for j in range(8):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title((['RGB', 'GT']+[c[1] for c in columns]+['Prior error (0-1m)'])[j], fontsize=10)
        axes[i, 0].set_ylabel(f"Frame {row['id']}")
    fig.suptitle('Fixed first three new test scenes | inference 512, seed 17 | GT affine alignment | shared depth range 0-7m', fontsize=12)
    fig.savefig(figdir/'fixed_test_examples.png', dpi=125); plt.close(fig)
    lines = ['# Larger local experiment: training resolution and denoiser preservation', '',
             'Completed 12 independent LoRA training runs (four methods x three seeds), each with 128 training scenes and 320 updates. '
             'Validation uses 32 scenes; the primary test uses 64 scenes never selected in earlier phases. '
             'All 27 predefined conditions were evaluated on both splits, giving 2,592 predictions. '
             'The full fixed matrix was evaluated after a technical validation audit; no winner was selected to determine test entries.', '',
             '**All metrics use per-image ground-truth affine alignment: results concern relative depth geometry, not uncalibrated metric distance.**', '',
             '## Main findings', '',
             'At 512-pixel inference, mixed-resolution training reduces mean test AbsRel from 0.06661 to 0.05968 (10.4%). '
             'The paired difference is -0.00693, with a scene-bootstrap interval of [-0.01102, -0.00357]. '
             'Training only at 512 reaches 0.06075; the mixed-versus-high512 interval crosses zero, so the smaller mixed mean '
             'does not establish a reliable advantage over high-resolution-only training.', '',
             'The best observed low-resolution result comes from training at 256: AbsRel 0.08149 versus 0.11161 for the original model. '
             'Mixed training reaches 0.08545 at that inference resolution. It therefore trades some low-resolution accuracy '
             'for stronger high-resolution performance; it does not dominate single-resolution adaptation at every operating point.', '',
             'Uniform denoiser preservation adds no convincing high-resolution gain: 0.05967 versus 0.05968, '
             'with a paired difference interval of [-0.00045, +0.00058]. At 256, its mean is slightly worse '
             '(difference +0.00060; unadjusted interval [+0.00002, +0.00124]). '
             'Mean training time increases from 105.5 to 137.0 seconds, approximately 30%, excluding latent preparation. '
             'The lower observed high-resolution seed SD with preservation is based on only three seeds and is insufficient '
             'to establish a general stability benefit. These results do not support adding this particular penalty for improved mean accuracy.', '',
             'The specialist reference reaches 0.08877 AbsRel at 22.7 ms/image. The better high-resolution Marigold scores '
             'come with much slower inference and different input processing; this is not evidence of general superiority over specialist depth models.', '',
             '## Method and boundaries', '',
             '`low256` and `high512` train at a single resolution; `mixed` alternates 256/512 updates. '
             '`mixed_prior` adds 0.1 times masked student-versus-original velocity MSE at identical noisy depth/RGB latents and timesteps. '
             'The original denoiser is obtained with the adapter disabled and no teacher gradients. '
             'This implements uniform denoiser-output preservation, not confidence weighting or cross-resolution geometric consistency. '
             'All adapters update 829,952 parameters; the same per-seed image order and timestep schedule were verified across methods.', '',
             '## Primary fresh64 test results', '',
             '| Training method | Inference long side / size | AbsRel mean +/- seed SD | RMSE (m) | Delta1 | Inference ms |',
             '|---|---:|---:|---:|---:|---:|']
    for group in order:
        r = lookup[('test', group)]
        mode, res = group.rsplit('_r', 1) if group != 'expert' else ('expert', '252x336')
        std = f" +/- {r['abs_rel_seed_std']:.5f}" if r['runs'] > 1 else ''
        lines.append(f"| {names[mode]} | {res} | {r['abs_rel_mean']:.5f}{std} | {r['rmse_m_mean']:.4f} | {r['delta1_mean']:.4f} | {r['inference_seconds_mean']*1000:.1f} |")
    lines += ['', 'The Depth Anything V2 reference has different prior NYUv2 supervision, preprocessing and precision. '
              'It is not a controlled architecture comparison; its validation subset is not known to be unseen during prior training.', '',
              '## Predefined paired test comparisons', '',
              'Differences are compared minus reference; negative favors the compared method. '
              'Intervals bootstrap 64 scenes after averaging per-scene metrics over training seeds (10,000 resamples, seed 4254). '
              'They describe scene variation, not training-seed variation, and are not corrected for multiple comparisons.', '',
              '| Compared | Reference | AbsRel difference | 95% scene interval |', '|---|---|---:|---:|']
    for r in contrasts:
        lines.append(f"| {r['compared']} | {r['reference']} | {r['abs_rel_difference']:+.5f} | [{r['scene_ci_low']:+.5f}, {r['scene_ci_high']:+.5f}] |")
    lines += ['', '## Training cost', '', '| Method | Seed | Training seconds | Cache seconds | Peak training MiB |', '|---|---:|---:|---:|---:|']
    for r in training:
        lines.append(f"| {r['mode']} | {r['seed']} | {r['training_seconds']:.1f} | {r['latent_cache_seconds']:.1f} | {r['peak_allocated_mib']:.0f} |")
    lines += ['', 'Equal optimizer updates are not equal compute. The teacher requires an additional forward pass, while higher resolution changes compute per update. '
              'These results cannot establish superiority under an equal-time budget. Timing excludes loading/downloads and is measured on one working laptop.', '',
              '## Figures', '', '![Test ablation](figures/test_ablation.png)', '',
              '![Training cost and losses](figures/training_cost.png)', '', '![Fixed examples](figures/fixed_test_examples.png)', '',
              '## Verification and limitations', '',
              'All 224 sample hashes, official split membership, scene separation, 12 distinct finite checkpoints and all 2,592 prediction metrics were verified. '
              'Each method\'s seed-29 checkpoint reproduced a saved 512-pixel validation prediction exactly. '
              'Training/validation scene roles were preserved, and all 64 test scenes were excluded from previous manifests. '
              'See `verification.json` and `validation_gate.json` for provenance.', '',
              'The study covers one 128-scene training subset, three training seeds, a fixed 320-update budget, one teacher weight and one inference seed per image. '
              'Comparisons with earlier phases also change data, schedules and test cohorts; they cannot isolate the effect of scale alone. '
              'This is not the full NYUv2 benchmark. Subsequent tuning must not use these test outcomes as an unseen final evaluation.', '',
              'Reproduction: follow [SCALEUP_PROTOCOL.md](../../SCALEUP_PROTOCOL.md) and `run_scaleup.ps1`. Raw arrays and weights remain outside Git.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    for r in summaries:
        if r['stage'] == 'test':
            print(r['group'], r['abs_rel_mean'], r['abs_rel_seed_std'], flush=True)
    print('SCALEUP_SUMMARY_READY', flush=True)


if __name__ == '__main__':
    main()
