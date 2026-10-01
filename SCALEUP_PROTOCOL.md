# Larger local experiment: scaleup_v2

Fixed before the reported training runs. Preflight uses two existing training images only, four updates per mode, and its checkpoints are never evaluated as reported models.

## Data and scope

Expand the training subset from 64 to 128 scene-disjoint frames and validation from 16 to 32, preserving all previous roles. Select 64 new test scenes, excluding every scene in the previous 200-frame manifest. Selection seeds are 4251/4252/4253. Continue using the official NYUv2 train/test split, one frame per scene and checksum-verified extraction. Prior test images are excluded from this study. The test files may be downloaded in advance, but no prediction or performance analysis is allowed before the validation-stage technical audit.

## Fixed matrix

| Training condition | Resolution schedule | Teacher penalty | Seeds | Updates |
|---|---|---:|---|---:|
| low256 | Always 256 | 0 | 17,29,43 | 320 |
| high512 | Always 512 | 0 | 17,29,43 | 320 |
| mixed | Alternate 256/512 | 0 | 17,29,43 | 320 |
| mixed_prior | Alternate 256/512 | 0.1 | 17,29,43 | 320 |

All twelve runs start independently from the pinned original Marigold v1.1. Use the same 128 training frames, rank4 attention LoRA, alpha4, AdamW learning rate1e-4, weight decay0.01, batch1, gradient clipping1, cached VAE latent modes, frozen VAE/text encoder, BF16 autocast and FP32 trainable parameters. Each seed shares image order and sampled timesteps across conditions; per-update noise uses generator seed `training_seed + 1000 * zero_based_step`. Different resolutions have different noise shapes. The mixed schedule alternates resolutions per update, not simultaneous multiscale processing of the same sample.

The `mixed_prior` objective is masked supervised velocity MSE plus 0.1 times masked student-versus-base velocity MSE at the same noisy ground-truth depth latent, timestep and RGB latent. Obtain the teacher in no-grad/eval mode with LoRA disabled; re-enable LoRA for the student and backward pass. This is **uniform denoiser-output preservation**, not confidence-weighted depth distillation or cross-resolution consistency. Those more elaborate proposals are not implemented in this phase. Equal updates do not imply equal training compute, which must be reported explicitly.

Evaluate every model at both 256 and 512 processing long sides, one denoising step, ensemble1, inference seed17+frame ID. Include original Marigold at both resolutions and the pinned Depth Anything V2 Metric Indoor Small reference. There are27 evaluation conditions on32 validation and64 new test scenes:2592 predictions. The specialist retains its existing252x336 processor/FP32 configuration and different prior NYUv2 supervision; it is not a controlled architecture ablation.

## Staging and fixed decisions

1. Profile all four training paths on training data only, check finite gradients, restoration and peak allocated VRAM below6800MiB. Freeze the final source/protocol in Git before the full runs.
2. Run all twelve trainings and every validation evaluation. Audit all training records, sample/source/checkpoint hashes and recomputed validation metrics. Audit four seed29 checkpoint restorations at512.
3. Unlock test evaluation only after that technical audit passes. Evaluate the full predefined matrix, without selecting a winner, changing hyperparameters or inspecting intermediate test outcomes to revise models. Fresh test results are an independent evaluation of the fixed matrix, not a reusable future tuning set.
4. Recompute every saved test prediction metric, produce reports/figures and publish code plus compact results. Keep all arrays/checkpoints outside Git. Preserve all prior result artifacts unchanged.

## Evaluation and analysis

Use the same Eigen crop, finite valid GT0.001<depth<10m, per-image GT least-squares affine alignment, and clipping as earlier studies. Metrics: AbsRel, RMSE, delta1; these measure relative depth structure rather than uncalibrated metric distance. Record synchronized warm-start inference time, training time, latent-cache time, trainable parameters and peak allocated CUDA memory. Loading and downloads are excluded from training/inference timing.

Report means and sample SD over three training seeds for each adaptation condition. For paired scene intervals, average metrics over seeds per scene, then bootstrap64 test scenes10000 times with seed4254. Report all methods, including negative results. Predefined comparisons at each inference resolution: every adapted group versus resolution-matched original; high512 versus low256; mixed versus low256 and high512; mixed_prior versus mixed. Also compare512 versus256 inference within each group. Intervals describe scene variation, are unadjusted for multiplicity, and are separate from training-seed SD. Latency is a single pass per trained run, not a repeated performance benchmark.

Qualitative figures use the first three new test frames in the frozen manifest, irrespective of performance. This remains a subset study, not the full NYUv2 benchmark. One training-data selection and three seeds cannot establish universal superiority.
