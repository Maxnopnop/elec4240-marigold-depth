# Metric depth and surface normals: staged local experiment

This module implements the requested progression: metric depth alone, normals alone, shared joint training, and joint training with decoded geometric consistency. The frozen configuration and data manifest are in `results/metric_multitask_v7/`. All formal comparisons use the **previously observed validation set**; this is an exploratory development experiment.

## Implementation

- One Marigold Depth v1.1 backbone, rank-4 attention LoRA, 829,952 trainable parameters. Base weights, VAE and text encoder remain frozen.
- Fixed depth/normal CLIP prompts distinguish tasks. The joint conditions share the same adapter, rather than switching between independently trained models.
- One-step DDIM at zero-SNR timestep 999. The velocity target is the negative clean task latent. This is a single-step supervised adaptation variant, not a full reproduction of the original multi-timestep Marigold trainer.
- Fixed metric log-depth encoding on 0.1–10 metres; no image-wise quantile normalization, affine depth alignment or scale-invariant ensembling. All methods receive the same fixed prediction clipping range for depth metrics.
- Camera-facing normal targets retain all three decoded channels. Targets are derived from smoothed filled NYUv2 depths under the documented RGB-pinhole approximation. Raw-depth agreement, erosion and discontinuity masks restrict supervision/evaluation. These are correlated, imperfect targets, not independently measured ground-truth normals.
- Each single-task run receives 320 image exposures (2.5 passes over 128 training images; this does not establish convergence). Each joint run receives 320 paired exposures **per task** and uses twice as many UNet forwards as one single-task run. Comparing against both separate single-task models accounts for their combined cost and two adapters.
- The geometry loss is cosine disagreement between the predicted normal map and normals derived from predicted metric depth. Gradients flow through both predictions and frozen VAE decoders. Its coefficient is fixed at 0.1.

## Local commands and prerequisites

Use the existing CUDA experiment environment from the repository root:

```powershell
& 'E:\Codex\2026-09-27\yo\work\marigold-local\.venv\Scripts\python.exe' -m multitask_v7.test_geometry
& 'E:\Codex\2026-09-27\yo\work\marigold-local\.venv\Scripts\python.exe' -m multitask_v7.prepare
& 'E:\Codex\2026-09-27\yo\work\marigold-local\.venv\Scripts\python.exe' -m multitask_v7.smoke
& 'E:\Codex\2026-09-27\yo\work\marigold-local\.venv\Scripts\python.exe' -m multitask_v7.run
& 'E:\Codex\2026-09-27\yo\work\marigold-local\.venv\Scripts\python.exe' -m multitask_v7.label_quality
& 'E:\Codex\2026-09-27\yo\work\marigold-local\.venv\Scripts\python.exe' -m multitask_v7.analyze
```

The existing project layout is intentional: `common.py` resolves retained assets under `work/marigold-local/assets`, and new raw arrays/checkpoints under `work/marigold-local/metric_multitask_v7`. This implementation uses a Windows process-scoped keep-awake request during training, released in `finally`; it does not schedule shutdown.

Required local assets are the pinned Marigold and Depth Anything models, `model_path.json`, the 160 source RGB-D files listed in the manifest, and access to NYUv2 raw-depth data or cached byte ranges. Preparation also reads `camera_params.m` and `toolbox_provenance.json` from the V7 work directory. These came from the [official NYUv2 toolbox](https://cs.nyu.edu/~fergus/datasets/toolbox_nyu_depth_v2.zip); `camera.json` records the source ZIP/member hashes and extracted calibration. Do not replace calibration with guessed camera values.

A Git clone alone does **not** contain these data, model weights, derived arrays or checkpoints. Archived completion records are not evidence that a fresh clone has local outputs: a resumed run checks checkpoint and metrics hashes, and full verification checks saved predictions. Preserve archived results and use a separate experiment directory/configuration when adapting this implementation to another environment. Do not delete completion records to force a replay over published outputs.

The smoke stage uses eight training images only. Its before/after results are implementation checks, not held-out accuracy. The formal protocol was frozen and published before its validation predictions; all methods use fixed final checkpoints and all three seeds are reported.

## Evidence and limits

`label_quality.py` is a post-hoc, training-only follow-up prompted by visible noise in the derived targets. It measures target resolution-round-trip and smoothing sensitivity without model inference or optimization. It does not regenerate any labels, select settings or add hypothesis tests. The resulting angular changes are not physical ground truth or lower bounds on model error.

`analyze.py` independently recomputes saved prediction metrics, verifies task exposure, checks frozen hashes and historical artifacts, and reports four predeclared exploratory comparisons with paired scene/seed bootstrap intervals and Holm-adjusted two-sided tests. Only one training draw and three seeds are present. Normal accuracy is conditional on derived labels and their retained pixels. Negative transfer is reported rather than hidden; nonsignificance does not establish equivalence. Both joint variants also receive direct descriptive comparisons with the separate single-task controls. A clearly post-hoc consistency diagnostic compares saved full-resolution predicted depth/normals using central differences and the fixed normal mask; it adds no hypothesis tests and is not numerically identical to the native-resolution training penalty.

The method draws on [Marigold](https://arxiv.org/abs/2505.09358), [LoRA](https://arxiv.org/abs/2106.09685), and established [joint depth/normal geometry](https://openaccess.thecvf.com/content_cvpr_2018/html/Qi_GeoNet_Geometric_Neural_CVPR_2018_paper.html). Metric coding, joint prediction and geometric consistency are not claimed as new concepts. Current code does not reproduce Vision Banana, demonstrate generation-capability retention, or establish a generalist foundation model.
