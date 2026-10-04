# Reliability-weighted geometric supervision: executable experiment plan

Date: 2 October 2026. This is an extension of the Marigold course project, not a reproduction of Vision Banana. Existing reports, V7 targets, checkpoints and results remain frozen.

## Research question and mechanism

Does the sensitivity of depth-derived normal labels to their construction scale identify where geometric regularization is useful during low-cost depth/normal adaptation? The hypothesis is that downweighting unstable locations can reduce harmful transfer through the extra geometry term. Neither that causal explanation nor a trained-model benefit has been demonstrated.

From the same full-resolution training depth, compute normals using Gaussian sigma 1, 2 and 4 pixels (truncate 2). Define pixel sensitivity as the mean of the three pairwise angular disagreements, in degrees. The fixed first candidate is `w = exp(-sensitivity / 10 degrees)`. The temperature is an engineering starting value, not an optimized result.

Keep the existing V7 binary validity mask and both supervised latent objectives. Area-resample `w * valid` and divide by area-resampled validity; use exactly the existing native support `valid_fraction > 0.9`. Normalize the geometry penalty by total valid weight. Weights are detached and computed from training labels only. The inference architecture does not change.

## Stage A: validate the component before expensive training

This stage is being executed locally. Its machine-readable configuration is `results/reliability_v8/component_protocol.json`; code hashes are frozen before outcomes. It is a predeclared engineering screen, not a formal claim of statistical significance.

- 192 synthetic cases: four surfaces (slanted plane, crease, depth step, smooth curve), six conditions (clean, independent noise, heterogeneous noise, outliers, filled holes, smooth systematic bias), and eight scene/noise seeds.
- Four development seeds and four previously unused component-evaluation seeds. They share the same generator and surface families, so they are not a new real-world domain.
- Analytic normals of clean geometry provide known truth. Existing erosion and jump masks remain in place. Report the full retained mask plus boundary and interior subsets.
- Compare uniform, proposed and spatially shuffled weights. Report weight-normalized label error, error-versus-coverage, rank correlation, error-detection AUROC where defined, effective pixel fraction and mean weight.
- Smooth systematic bias is an intentional stable-but-wrong counterexample. Clean curves and boundaries test whether the method suppresses genuine structure.
- Audit all 128 existing training scenes, without loading validation or test images. Reproduce the frozen sigma-1 normal labels exactly, preserve native masks, cache training weights and measure coverage. Real-label stability is not independent evidence of correctness.

The gate requires at least 10% mean weighted-label error reduction over uniform weighting in the four predeclared noisy conditions, 5% over shuffled weights, non-worse means in at least three of those four families, median synthetic effective pixel fraction at least 0.25, no clean-family mean error increase above 1 degree, and effective pixel fraction at least 0.25 in at least 90% of real training scenes. These screening thresholds do not guarantee downstream benefit and do not apply a significance test to pixels.

If the gate fails, retain the results and do not launch a large weighted-model comparison under this version. Any revised proxy gets a new protocol and a new component evaluation cohort. If it passes, proceed to the next bounded stage; do not describe weighted label selection as improved model prediction.

## Stage B: verify integration and stabilize the baselines

The current bounded GPU check uses three training images and four weighting conditions, for twelve forward/backward passes and **zero optimizer updates**. It verifies equivalence to the original V7 objective for uniform weights, finite nonzero adapter gradients, detached weights and memory fit. This is implementation evidence only.

Next, run depth-only and normal-only learning curves on the existing 128-image training draw with seeds 17, 29 and 43. A proposed first budget is 1,280 updates per run, inspecting predeclared checkpoints at 320, 640 and 1,280 on the previously observed 32-scene development set. This budget is four times V7 and is not assumed to establish convergence. Investigate persistent loss trends or unstable task accuracy before locking the main comparison. If training settings change, record them before starting candidate comparisons and rerun affected controls.

## Stage C: matched model ablation

Use seven conditions, three seeds each, with a fixed final checkpoint and common training images, order, image exposure, resolution, initialization, inference seed and evaluation mask:

| Condition | Proposed geometry coefficient | Purpose |
|---|---:|---|
| Depth only | 0 | Metric-depth single-task control |
| Normals only | 0 | Normal single-task control |
| Shared joint model | 0 | Negative-transfer control |
| Uniform geometry | 0.10 | Existing geometric method |
| Weaker uniform geometry | 0.03 | Control for simply reducing regularization |
| Scale-sensitive geometry | 0.10 | Proposed spatial selection |
| Shuffled scale-sensitive weights | 0.10 | Same weight distribution and sum, different locations |

The candidate model budget is 1,280 updates, subject to Stage B findings and a separate freeze. Single-task controls can be reused only when their finalized settings exactly match. Do not select the best seed. Joint training gives one exposure per task per update and costs more than either single task; report task exposure and actual compute.

Primary development comparisons are weighted versus uniform, weaker uniform and shuffled weighting, evaluated separately for depth AbsRel and normal mean angular error: six planned contrasts, with paired scene/seed effects, uncertainty intervals and Holm adjustment. Comparisons against joint-only and single-task models are also required for interpreting transfer and should be reported descriptively unless included in the frozen hypothesis family.

Evaluate metric depth without per-image ground-truth alignment: AbsRel, RMSE in metres and delta1. Evaluate normals with angular errors and threshold accuracies, explicitly acknowledging derived targets. Add boundary/interior breakdowns, depth-normal consistency, memory, optimizer time and inference time. Low geometric disagreement alone is not a success criterion.

A non-significant normal difference is not evidence of no harm. A formal non-inferiority claim requires a scientifically justified degradation margin, its confidence-bound analysis and an appropriate multiplicity plan, all frozen before the relevant evaluation. This design does not invent such a claim after results.

## Stage D: independent evidence

All previously evaluated NYUv2 and SUN3D cohorts remain development or retrospective evidence for the new method. Select the method using development data, then estimate sample size for the exact planned contrasts and acquire genuinely unobserved scene groups. Account for room/building clustering and independent training runs; pixels are not independent observations. A real-world surface-normal claim needs trusted independent normal labels or high-quality verified geometry. New test data must not be used to tune temperature, smoothing or stopping time.

## Execution and resource choice

The current RTX 5060 Laptop has 8 GB VRAM. Stage A needs CPU only; the bounded integration check uses the existing CUDA environment. Colab is optional for rerunning the synthetic component study. The portable notebook and source bundle do not contain model weights, private credentials or RGB-D image arrays. Their Colab execution has not been performed by this local run.

Local commands, from the repository root:

```powershell
$python = 'E:\Codex\2026-09-27\yo\work\marigold-local\.venv\Scripts\python.exe'
& $python -m reliability_v8.test_core
& $python -m reliability_v8.study freeze
& $python -m reliability_v8.study synthetic
& $python -m reliability_v8.study real
& $python -m reliability_v8.study analyze
& $python -m reliability_v8.gpu_check
```

The frozen script verifies hashes before each stage. A fresh clone lacks the local V7 assets needed for `real` and `gpu_check`. Preserve published outputs when reproducing; use the clean notebook bundle or a separate checkout/work directory. Raw weight maps remain under `work/marigold-local/reliability_v8`, outside Git.

## Prior art and claim boundary

[GeoWizard](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03265.pdf) already studies joint diffusion-based depth and normal prediction. [Li et al., MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/1214_paper.pdf) use distance-based uncertainty for depth-normal consistency in laparoscopic images. The proposed distinction is training-label construction-scale sensitivity within a fixed parameter-efficient generative adaptation pipeline. A stronger publication novelty claim needs a direct comparison against relevant reliability rules, including local plane-residual weighting; changing the application alone is insufficient.

Stage A can support a claim about supervision selection under its synthetic assumptions. Only controlled model training can support a claim about adaptation performance. Neither stage alone establishes broad superiority or reliable real-world distance measurement.
