"""Crossed draw/seed/scene sensitivity, fixed before v3 test inference."""
import numpy as np
from robustness_stats import intervals


def crossed_bootstrap(delta, replicates=20000, seed=4266, family=6):
    delta = np.asarray(delta, dtype=np.float64)
    assert delta.ndim == 3 and np.isfinite(delta).all()
    draws, seeds, scenes = delta.shape
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, replicates, 250):
        b = min(250, replicates-start)
        dd = rng.integers(draws, size=(b, draws))
        ss = rng.integers(seeds, size=(b, seeds))
        ii = rng.integers(scenes, size=(b, scenes))
        values = delta[dd[:, :, None, None], ss[:, None, :, None], ii[:, None, None, :]]
        samples.extend(values.mean(axis=(1, 2, 3)))
    return intervals(float(delta.mean()), samples, family)
