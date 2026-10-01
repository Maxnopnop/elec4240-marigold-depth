"""Paired bootstrap estimators and Holm adjustment, fixed before test evaluation."""
import numpy as np

CONTRASTS = [(f'{m}_r{r}', f'base_r{r}') for m in ['mixed', 'high512'] for r in [256, 512]] + [
    (f'mixed_r{r}', f'high512_r{r}') for r in [256, 512]]


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1., np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    return adjusted.tolist()


def intervals(observed, samples, family):
    samples = np.asarray(samples)
    lo, hi = np.quantile(samples, [.025, .975])
    blo, bhi = np.quantile(samples, [.025/family, 1-.025/family])
    # Center the resampling distribution at the null. Approximate, not exact.
    p = (1+np.count_nonzero(np.abs(samples-observed) >= abs(observed)))/(len(samples)+1)
    return {'difference': float(observed), 'ci_low': float(lo), 'ci_high': float(hi),
            'bonferroni_ci_low': float(blo), 'bonferroni_ci_high': float(bhi), 'p_bootstrap': float(p)}


def bootstrap(delta, replicates=20000, seed=4264, family=6):
    """delta is [training draw, training seed, paired evaluation scene]."""
    delta = np.asarray(delta, dtype=np.float64)
    assert delta.ndim == 3 and np.isfinite(delta).all()
    draws, seeds, scenes = delta.shape
    rng = np.random.default_rng(seed)
    scene_means = delta.mean(axis=(0, 1))
    scene_boot, hierarchical = [], []
    for start in range(0, replicates, 250):
        b = min(250, replicates-start)
        ii = rng.integers(scenes, size=(b, scenes))
        dd = rng.integers(draws, size=(b, draws))
        ss = rng.integers(seeds, size=(b, draws, seeds))
        scene_boot.extend(scene_means[ii].mean(axis=1))
        sampled = delta[dd[:, :, None, None], ss[:, :, :, None], ii[:, None, None, :]]
        hierarchical.extend(sampled.mean(axis=(1, 2, 3)))
    observed = float(delta.mean())
    return {'scene': intervals(observed, scene_boot, family),
            'hierarchical': intervals(observed, hierarchical, family),
            'replicates': replicates, 'draw_means': delta.mean(axis=(1, 2)).tolist(),
            'seed_run_means': delta.mean(axis=2).tolist()}
