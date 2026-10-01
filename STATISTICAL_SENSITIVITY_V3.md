# Pre-test statistical sensitivity addendum

Added during the training/validation stage, before any robustness_v3 test or
corruption predictions exist. The original protocol, estimands, comparisons,
training matrix and primary analysis remain unchanged.

The same five numerical RNG seeds are used in every training draw. This shares
initial adapter weights and random-number schedules across draws. In addition to
the planned nested draw/seed bootstrap, report a crossed-factor sensitivity:
resample three draw indices, five seed indices shared across sampled draws, and
paired scene indices shared across all trained models. Use 20,000 replicates and
RNG 4266. Apply Holm correction to the same complete six-contrast family for each
cohort/metric, separately from the primary calculation. Report both intervals and
adjusted p-values; regard a significance statement as robust to this choice only
if both calculations support it. Do not select whichever calculation is favorable.

This is an approximate sensitivity analysis, not an exact test or a guarantee of
population coverage. Three training draws remain a small sample of possible
training sets. The held-out scene factor is crossed with trained models in both
analyses; pixels and model-scene predictions are not independent replicates.

Implementation: `crossed_bootstrap_sensitivity.py`; its hashes and the pre-test
status snapshot are saved in `results/robustness_v3/statistical_addendum.json`.
