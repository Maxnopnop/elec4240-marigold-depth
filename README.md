# ELEC4240: Low-cost adaptation of Marigold for indoor depth

A local course-project pilot that measures the accuracy and compute cost of adapting a diffusion-based depth model on an 8 GB laptop GPU. The starting point is the public **Marigold Depth v1.1** checkpoint. The project is inspired by transferring image generators to perception, including Vision Banana, but does not implement Vision Banana.

## What was run

- **Data:** 32 training, 8 validation and 24 test frames from NYU Depth V2. One frame per scene; selected scenes are disjoint across splits. The 8-image adaptation set is nested within the 32-image set (25% versus 100% of this pilot's training pool, not of the full dataset).
- **Controlled comparisons:** pretrained Marigold; rank-4 attention LoRA with 8 or 32 images; output-convolution-only adaptation with 32 images. All adaptations use 80 optimizer steps and one training seed.
- **Additional comparisons:** pretrained Marigold at 1 versus 4 denoising steps; Depth Anything V2 Metric Indoor Small as a specialist reference.
- **Hardware:** NVIDIA GeForce RTX 5060 Laptop GPU, approximately 8 GB VRAM. Processing long side is 256 pixels for Marigold.
- **Outputs:** per-image metrics, validation/test summaries, training curves, measured time and peak allocated GPU memory, and fixed qualitative examples.

Read the [recorded results](results/RESULTS.md), [predefined protocol](PROTOCOL.md), and [sources and implementation attribution](REFERENCES.md). Exact package versions are in [environment.json](results/environment.json).

**Evaluation uses ground-truth affine alignment for every method.** It measures relative depth structure; the scores are not evidence of accurate absolute distances from an uncalibrated model. Depth Anything has different pretraining and NYUv2 supervision, so it is a useful reference but not a controlled architecture ablation. This 24-scene, single-seed, low-resolution pilot is not the full NYUv2 benchmark.

![Quality and inference time](results/figures/quality_and_latency.png)

## Pilot findings

On the fixed 24-scene test subset, rank-4 LoRA with 32 training images reduced aligned AbsRel from **0.1353 to 0.0900**, a 33.5% reduction relative to the original four-step model. Eight-image LoRA reached **0.0988**. Updating only the output convolution reached **0.1354**, essentially unchanged from the original model. LoRA updated **829,952 parameters (0.096% of the UNet)**; the 32-image run needed **33.2 seconds** for 80 updates plus **1.9 seconds** for latent preparation, with **2,416 MiB** peak allocated training memory. Model loading and downloads are excluded from those times.

The additional baselines qualify that result: the **unadapted one-step model reached 0.1019 AbsRel at 0.075 s/image**, outperforming the four-step setting on this small subset. The **Depth Anything V2 specialist reached 0.0811 at 0.021 s/image**. Therefore the evidence supports feasibility of cheap adaptation, not superiority over specialist models or a general claim that more denoising steps are better. The specialist receives 252x336 processed inputs and different prior supervision; Marigold receives 192x256 inputs.

LoRA inference was measured at 0.216–0.264 s/image with the adapters left unmerged. The two LoRA conditions have identical parameter counts and inference structure; their observed timing difference should be treated as runtime variation, not a cost caused by training on more images. These are single sequential timing passes on a working laptop, not a rigorous performance benchmark.

All reported values were checked against **192 saved predictions**, and **64 extracted sample hashes**, official split membership and scene isolation were verified. Three evaluation unit tests passed. See [verification.json](results/verification.json). Paired bootstrap intervals in the result report describe this subset's scene variation only. More data, several seeds and repeated timing runs are needed before making broad claims.

## Reproduce on Windows

Use Python 3.12 and install a CUDA-enabled PyTorch build that supports your GPU. The recorded run used PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128 and CUDA 12.8. A GPU with BF16 support is required by this implementation. Start with a fresh virtual environment and install `requirements.txt` after installing the appropriate PyTorch build.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest test_protocol -v
.\run_pilot.ps1 -Python "$PWD\.venv\Scripts\python.exe" -WorkDir 'E:\elec4240-marigold-work'
```

The work directory should be outside this repository. It stores model weights, selected dataset arrays, HTTP cache, raw predictions and adapter checkpoints. The downloaders use pinned model revisions. Dataset extraction reads selected frames from the official labeled NYUv2 MAT file through HTTPS byte ranges; the conventional 795/654 split file is checksum verified. Downloads can be slow, but cached blocks are reused. See the upstream references for model/data terms. Results and figures are written under `results/` and will be replaced by a rerun.

To run one condition after downloading assets and preparing the manifest:

```powershell
.\.venv\Scripts\python.exe experiment.py --assets 'E:\elec4240-marigold-work\assets' --work 'E:\elec4240-marigold-work\runs' --results results --method lora32
```

Supported methods: `base`, `lora8`, `lora32`, `head32`, `expert`. The default denoising budget is 4; use `--denoise-steps 1` for the pretrained speed comparison. Recreate the report with `summarize.py` using the same three path arguments. All six runs must be complete before summarizing. GPU timings depend on hardware, thermals and concurrent applications and should not be expected to match exactly.

## Repository map

| File | Purpose |
|---|---|
| `download_assets.py`, `download_expert.py` | Obtain pinned public model assets |
| `prepare_subset.py` | Select and extract reproducible scene-disjoint frames |
| `experiment.py` | Inference, limited-parameter training and aligned evaluation |
| `test_protocol.py` | Check affine recovery, crop and invalid inputs |
| `verify_results.py` | Verify 64 hashes, scene isolation and all 192 saved predictions against metrics |
| `summarize.py` | Aggregate metrics, paired scene bootstrap and figures |
| `run_pilot.ps1` | Sequential end-to-end experiment runner |
| `results/` | Recorded measurements, manifests and figures |

Weights, dataset arrays, raw predictions, environments and checkpoints are excluded from GitHub. The qualitative figure contains only the fixed first three test examples, with dataset attribution in REFERENCES.md. The local output checkpoints contain trainable parameter tensors; they are research artifacts rather than a packaged inference product.

## Course-project interpretation

The original work here is the constrained experiment: controlled trainable-parameter and data-budget comparisons, consistent evaluation, and a measured quality/cost analysis. Merely running a pretrained model would be insufficient. The current pilot establishes an executable starting point; a stronger final project should expand scene coverage, repeat adaptation with several seeds, use validation for a planned learning-rate/step search, and evaluate once on a fixed larger test set. Claims of novelty or general improvement require additional evidence.

All model and method sources are acknowledged in [REFERENCES.md](REFERENCES.md). Academic claims should cite the original papers as well as describe this pilot's departures from their training recipes.
