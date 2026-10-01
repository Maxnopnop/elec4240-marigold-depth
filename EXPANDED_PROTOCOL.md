# Expanded evaluation v1

This protocol was fixed after the initial 24-scene pilot and before expanded experiments. The pilot motivated the choices below; this is not a claim that the project was designed without seeing any test results. Existing pilot outputs remain unchanged.

## Data and evaluation cohorts

Keep the original 32 training frames, 8 validation frames and 24 test frames in their existing roles. Add one frame per new scene to reach 64 training, 16 validation and 120 test frames. New sampling uses seeds 4241, 4242 and 4243 respectively, excludes every already selected scene, and stays inside the conventional official train/test split. Validation scenes come from the official training split. Preserve the nested original32 training pool within train64.

The **96 previously unused test scenes** are the primary evaluation cohort. Report the combined120 and original pilot24 separately as secondary cohorts. Each scene contributes one frame. Save selection before extraction and save hashes after extraction. Never use evaluation scores to change this matrix or select a checkpoint. The fixed final checkpoint is evaluated; this phase does not search hyperparameters. These are expanded research results, not a reserved test set for later model selection.

## Fixed experiment matrix

| Condition | Train images | Training seeds | Updates | Denoising steps |
|---|---:|---|---:|---|
| Pretrained Marigold | 0 | not applicable | 0 | 1, 4 |
| Attention LoRA rank 4 | 32 | 17, 29, 43 | 160 | 1, 4 |
| Attention LoRA rank 4 | 64 | 17, 29, 43 | 160 | 1, 4 |
| Output convolution only | 64 | 17, 29, 43 | 160 | 1, 4 |
| Depth Anything V2 Metric Indoor Small | 0 in this project | not applicable | 0 | not applicable |

Nine independent adaptation runs, each followed by both inference settings on the same final checkpoint. This produces 21 evaluation conditions, each covering val16 and test120. All adaptation runs use equal optimizer-update budgets, not equal epochs: approximately 5 passes over 32 frames or 2.5 over 64. Compared with the pilot, the update budget changes from 80 to 160; do not attribute pilot-to-expanded differences solely to data size.

Reuse the pinned models, mask, Eigen crop, original-resolution target pixels, per-image GT affine alignment, AdamW learning rate 1e-4, weight decay 0.01, batch size 1, gradient clipping 1.0, BF16 frozen weights, FP32 trainable parameters, cached deterministic VAE latents, and 256-pixel Marigold long side from PROTOCOL.md. Keep adapters unmerged for consistency. LoRA initialization, data order, training timestep and noise depend on the training seed. **Inference noise is fixed to 17 + frame ID for every run**, independently of the training seed. This isolates training variability while keeping paired evaluation fair; it does not estimate inference-noise variability.

The specialist uses its prescribed processor (252x336 here), FP32 inference, and different prior NYUv2 supervision. It is a reference, not a controlled architecture comparison; val16 is not a clean held-out set for that pretrained specialist. All methods share the same target pixels and oracle affine alignment. Claims concern relative depth, not uncalibrated metric distances.

## Analysis and records

Report per-image AbsRel, RMSE, delta1, latency and peak allocated memory. Aggregate adapted conditions as mean and sample standard deviation over three training seeds. For paired scene-bootstrap intervals, first average each scene across training seeds, then resample scenes (10,000 resamples, seed 4240). Such intervals describe scene variation; training-seed variation is reported separately, and three seeds do not establish broad statistical superiority. Include all conditions and negative results. Separate one-step and four-step comparisons and do not select the best seed.

Run GPU workloads sequentially. Inference timings include preprocessing and device-to-host output, exclude model loading and a warm-up, and remain single-pass laptop measurements. Differences between identically shaped models can reflect thermal/system variation. Save training time, latent preparation time, trainable parameter count, hardware/package metadata, protocol/manifest hashes and resumable status. Resume only matching configurations with complete records; partial evaluations may be rerun. Keep data, raw predictions and checkpoints in local work storage, outside Git.

Verify scene isolation, original split membership, all sample hashes, finite training losses/gradients, checkpoint provenance and every saved prediction against its reported metrics before publishing. Qualitative examples are the first three fresh test frames in the frozen manifest, irrespective of score.
