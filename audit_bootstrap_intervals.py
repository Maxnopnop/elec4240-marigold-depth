"""Post-hoc interval display audit; replay frozen bootstrap p-values exactly.

Percentile intervals are not inversions of centered absolute-deviation tests.
Symmetric intervals below use the same centered absolute-deviation quantiles.
They add a diagnostic display, without changing the predefined tests or decisions.
"""
import numpy as np


def matched_intervals(delta, kind, expected_p, replicates=20000, family=6):
    delta = np.asarray(delta, dtype=np.float64)
    nd, ns, ni = delta.shape
    rng = np.random.default_rng(4264 if kind == 'hierarchical' else 4266)
    samples = []
    for start in range(0, replicates, 250):
        b = min(250, replicates-start)
        if kind == 'hierarchical':
            ii = rng.integers(ni, size=(b, ni))
            dd = rng.integers(nd, size=(b, nd))
            ss = rng.integers(ns, size=(b, nd, ns))
            values = delta[dd[:, :, None, None], ss[:, :, :, None], ii[:, None, None, :]]
        else:
            assert kind == 'crossed'
            dd = rng.integers(nd, size=(b, nd))
            ss = rng.integers(ns, size=(b, ns))
            ii = rng.integers(ni, size=(b, ni))
            values = delta[dd[:, :, None, None], ss[:, None, :, None], ii[:, None, None, :]]
        samples.extend(values.mean(axis=(1, 2, 3)))
    observed = float(delta.mean())
    errors = np.abs(np.asarray(samples)-observed)
    p = (1+np.count_nonzero(errors >= abs(observed)))/(replicates+1)
    assert p == expected_p, (kind, p, expected_p)
    q95, qfamily = np.quantile(errors, [.95, 1-.05/family])
    return {'difference': observed, 'ci_low': float(observed-q95), 'ci_high': float(observed+q95),
            'bonferroni_ci_low': float(observed-qfamily), 'bonferroni_ci_high': float(observed+qfamily),
            'p_bootstrap_reproduced': float(p)}
