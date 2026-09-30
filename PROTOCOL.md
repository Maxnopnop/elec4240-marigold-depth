# Predefined pilot protocol

## Purpose

Test whether local inference and limited-parameter adaptation of Marigold fit an 8 GB laptop GPU, and record initial quality/cost measurements. This pilot cannot establish a general improvement or reproduce Vision Banana's multi-task findings.

## Data

NYUv2 official labeled MAT with inpainted depth in meters. Conventional split: 795 train, 654 test. With NumPy RNG seed 4240, randomly select one frame per distinct scene: 32 training frames, 8 validation frames from other official training scenes, and 24 official test frames. Assert selected scene sets do not overlap. The 8-frame training subset is the first eight entries of the fixed 32-frame selection. Save frame IDs, scene names and extracted-file hashes in `results/split_manifest.json`.

## Fixed conditions

- Marigold Depth v1.1, pinned revision recorded in REFERENCES.md.
- Image processing long side: 256 pixels (192x256 for NYUv2). Original 480x640 resolution used for evaluation after bilinear Marigold output resizing.
- Inference: DDIM, 4 denoising steps, ensemble size 1. Additional pretrained 1-step inference measures the speed/accuracy tradeoff.
- Three adaptation configurations: LoRA rank 4 with 8 frames; LoRA rank 4 with 32 frames; output-convolution-only training with 32 frames.
- 80 optimizer steps each, batch size 1, learning rate 1e-4, AdamW weight decay 0.01, gradient norm clipping at 1.0, training seed 17. Same optimizer-step budget, not the same number of data epochs.
- Freeze VAE and text encoder. Cache deterministic VAE mode latents. Train LoRA in attention query/key/value/output projections. For the head condition, freeze all but `unet.conv_out`.
- FP32 trainable parameters under bfloat16 autocast. Frozen model weights bfloat16. LoRA gradient checkpointing enabled.
- Image preprocessing and antialiased output resizing use FP32 before returning to model precision, to avoid an unsupported BF16 resize kernel on Windows. The text encoder remains on CUDA for reliable pipeline device detection.
- Fixed final-step checkpoint. No choosing a winner or changing hyperparameters on test results. Validation is reported separately; this first pilot does not conduct a hyperparameter search.
- Per-image Marigold inference random seed = 17 + original one-based frame ID, shared across methods.

## Evaluation

For every model, use the same ground-truth valid mask (`0.001 < depth < 10 m`) and Eigen crop `[45:471, 41:601]`. Fit scalar `a,b` to minimize the squared error of `a * predicted_depth + b` on those valid pixels. Clip aligned predictions to [0.001, 10] m. Report mean per-image AbsRel, RMSE and delta1 (ratio < 1.25). No ground-truth information is used for model inference, training selection or hyperparameter tuning; ground truth **is** used for this explicitly oracle affine evaluation.

These aligned scores measure relative depth structure. They do not demonstrate that the model predicts absolute metric depth without calibration. Ground-truth-aligned visualizations must also be labeled.

Depth Anything V2 Metric Indoor Small is an additional specialist reference. It has NYUv2 training supervision and a different architecture/pretraining history, so it is not a controlled architecture ablation. Its model-specific image processor preserves aspect ratio and rounds dimensions to its patch size: the requested 256x256 target produces 252x336 inputs, compared with Marigold's 192x256 inputs. Its native metric outputs undergo the same affine alignment for the comparison. Different prescribed input processing and output interpolation are disclosed; all metrics use the same original-resolution target pixels. The eight validation frames are drawn from the official training split, so they should not be treated as held-out validation data for the pretrained specialist.

## Cost measurement

Synchronize CUDA before and after inference. Include image preprocessing and the final GPU-to-CPU prediction transfer; exclude model loading and one warm-up call. Process one model at a time. Record PyTorch peak allocated memory (not whole-device VRAM, which also includes CUDA context and other apps). Training timing excludes model loading and cached latent preparation; record latent-cache preparation separately.

## Interpretation limits

Only 24 test scenes, one training seed, low resolution, 80 steps, and 8/32 adaptation images. Results are a feasibility pilot, not a full NYUv2 benchmark or evidence of statistical superiority. Failed or worse adaptation results must be reported. Dataset/model assets and checkpoints are kept locally, outside the Git repository.
