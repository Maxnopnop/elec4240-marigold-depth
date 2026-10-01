# Robustness v3: training draws, multiplicity, and metric calibration

Frozen before new training or evaluation. This is a targeted replication of the
mixed-resolution claim, not a claim of general superiority or a new model search.

## Data and design

- NYUv2 official 795/654 frame split has 249 training and 215 test scenes.
- Preserve the scaleup_v2 32 validation scenes. Uniformly sample 128 of the
  remaining 217 training scenes without replacement, independently with seeds
  4261, 4262, 4263. Within a scene use its smallest official training frame ID.
  These are three NEW training sets, not the previous nested training set.
  Training draws may overlap; report scene intersection sizes. No validation or
  test scene can enter a training draw.
- Fresh primary test: ALL 31 official test scenes unused in any previous study,
  one smallest official frame ID per scene, sorted by scene. This exhausts the
  previously unused scenes in the labeled NYUv2 test split. The previous
  scaleup_v2 64 test scenes form a separately reported, already observed cohort.
- Keep all 32 previous validation images for calibration. Do not select methods,
  checkpoints, regularization strengths or resolutions from these new results.

## Fixed methods

- Two adapted methods: high512 and mixed, using the unchanged train_scaleup.py,
  320 updates, LoRA rank/alpha 4, 829952 parameters, AdamW 1e-4, batch 1.
- Three independent training draws x seeds 17,29,43,59,71 x two methods = 30
  completely new trainings. Pair data order/timesteps by seed across methods.
- Evaluate original and every trained checkpoint at 256 and 512, one denoising
  step, inference seed 17+frame ID, ensemble one. No model selection.
- Evaluate the fixed Depth Anything V2 Metric Indoor Small checkpoint at its
  previous processor setting (requested 256, actual 252x336), and at explicitly
  resized 378x504 input with processor resizing disabled. The latter approximates
  Marigold's 384x512 input within the specialist's patch-size constraint. Prior
  supervision, model scale and precision remain uncontrolled differences.
- Low256 and mixed_prior are NOT retrained here. Their earlier conclusions get a
  retrospective multiplicity analysis, not a multi-training-set replication.

## Metrics without test-target fitting

Report both (1) the existing GT per-image affine-aligned relative-depth scores
and (2) fixed global affine calibration fitted ONLY on the 32 validation images.
Fit a separate scale/shift for each checkpoint/resolution using equal weight per
validation image and pixel-mean sufficient statistics. Clip final predictions
to [0.001,10] m, and score the unchanged Eigen crop and finite 0.001<GT<10 mask.
Test GT is used only for scoring in (2). Marigold's normalized raw output is not
called meters. Report the specialist's raw metric predictions as an additional
native-metric reference. Also report a constant-depth baseline equal to the
validation mean depth. Calibration is a simple diagnostic, not a trained metric
depth head, and a poor result does not prove metric prediction impossible.

## Uncertainty and multiple comparisons

Primary fresh-31 family contains six fixed aligned-AbsRel contrasts:
mixed-base at 256/512, high512-base at 256/512, mixed-high512 at 256/512.
Average each training draw/seed/scene equally. Separately report:

1. Paired scene bootstrap after averaging training runs (20,000 replicates).
2. Paired hierarchical bootstrap resampling training draws, seeds within each
   sampled draw, and test scenes shared across draws (20,000 replicates, RNG4264).
   It captures observed training and scene variability, but three draws and five
   seeds still limit precision. Dataset overlap and the fixed finite scene pool
   prevent claims of population-wide coverage or independence of all pixels.
3. Approximate centered-bootstrap two-sided p-values and Holm adjustment within
   the complete six-contrast family, plus conservative Bonferroni percentile
   intervals at 1-0.05/6. Bootstrap inference is approximate, especially with only
   three training draws. A CI crossing zero is not proof of equivalence.

Report the already-observed 64-scene cohort separately; do not combine it with
fresh scenes for a confirmatory claim. Apply the same six-contrast adjustment as
a secondary analysis. Metric-calibration comparisons form a separate six-contrast
secondary family. Raw scores, seed SDs, per-draw means and all comparisons remain
visible regardless of sign. Also retrospectively apply scene-centered-bootstrap
p-values and Holm adjustment to ALL 21 comparisons in scaleup_v2; label this
post-hoc sensitivity analysis, not a preregistered confirmation.

## Controlled robustness probes

On fresh test scenes only, repeat 512 inference for original, all 15 mixed
checkpoints and the 378x504 specialist under two fixed image changes: brightness
times 0.5 (rounded/clipped uint8), and PIL GaussianBlur radius 2. Use identical
inference seeds and clean-validation calibration; never recalibrate on test.
These test two indoor image perturbations, not cross-dataset or outdoor transfer.
Report degradation, absolute metrics and win fractions descriptively; no tuning
or additional significance claims. Native-metric specialist remains a reference.

## Technical gate and artifacts

Freeze protocol, source hashes and sample manifest before full runs. Complete
training and validation, verify all checkpoint/data hashes, finite logs, scene
separation, paired schedules, saved metrics, and representative checkpoint
restoration BEFORE new test inference. Fit and freeze calibration before test.
Recompute every saved test metric and verify prediction hashes after completion.
Keep arrays, weights and caches outside Git; publish code, manifests, per-image
scores, calibration coefficients, audit, plots and English report to the existing
private repository. Never edit or overwrite earlier result artifacts.
