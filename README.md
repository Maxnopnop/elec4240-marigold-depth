# ELEC4240: Low-cost adaptation of Marigold for indoor depth

A local course-project study that measures the accuracy and compute cost of adapting a diffusion-based depth model on an 8 GB laptop GPU. The starting point is the public **Marigold Depth v1.1** checkpoint. The project is inspired by transferring image generators to perception, including Vision Banana, but does not implement Vision Banana.

**Provenance correction (2026-10-01):** the fixed Depth Anything V2 indoor reference is metric-fine-tuned on **Hypersim**, not NYUv2. Earlier NYUv2-supervision wording and the overlap inference derived from it are withdrawn; measured results are unchanged. See [the correction and pinned primary sources](CORRECTIONS.md).

## Research review and prerequisites for the next stage

The [project review and five English proposal answers](research_v6/project_review.html) distinguish the completed relative-depth study from proposed metric-depth and multi-task extensions. A read-only [foundation audit](research_v6/foundation_audit.json) finds that only 80 of 128 training images on average appear at both resolutions in the existing mixed runs. A [training-only gradient probe](research_v6/gradient_results.json) completes 256 backward passes without updating weights: all eight grouped mean cosines are positive, with 12 negative pairs out of 128. This small diagnostic does not establish widespread gradient conflict. Exposure-matched schedules and consistent preprocessing take priority over more complex optimization.

The review lists the missing metric encoding, camera calibration/label masks, task conditioning and decoding, single-task controls, memory profiling, and genuinely new evaluation scenes needed for a depth-plus-normal extension. No new metric or multi-task model has been trained. The delivered final report and its source/code archives remain unchanged. The diagnostic scripts require the retained local assets and checkpoints documented in the existing reproduction guide; their protocol is in `research_v6/gradient_protocol.json`.

## Completed final report and matched controls

The [final findings](FINAL_FINDINGS.md) integrate **24 additional trainings**, **7,890 matched-control predictions**, **576 fixed-time validation predictions**, and **320 controlled timing matches**. All **939 historical result blobs** remain unchanged. The [independent audit](results/final_extension_v5/delivery_checks.json) passed.

The added low-only control reaches **0.09277 aligned AbsRel** at external 256 inference, versus mixed **0.09632** (crossed Holm8 p=**0.01560**, location p=**0.03660**, favoring low-only). The mixed-versus-low difference at 512 remains unconfirmed. In the separate 120-second validation diagnostic, low-only has the best mean at 256 and high-only at 512. These findings qualify the benefit of mixing; the extension is exploratory and does not replace the original frozen six-test conclusions.

Deliverables: [eight-page English PDF](reports/final/final_report.pdf), [Overleaf source](reports/final/overleaf_source.zip), [code supplement below 10 MB](reports/final/supplementary_code.zip), [complete extension tables and figures](results/final_extension_v5/RESULTS.md), and [reproduction instructions](FINAL_REPRODUCE.md). The PDF lists all three team members. No Canvas submission was made.

## Prospective external validation: prospective_v4

The [sample-size study](results/prospective_v4/power/RESULTS.md) explores 480 planning scenarios using only earlier NYUv2 results. Under a hypothetical 0.009 AbsRel gain and fresh31-like error variation, 200 new groups yield approximately **84.3%** estimated power; inflating the error SD by 1.5 lowers it to **38.5%**. These are conditional planning approximations with uncertainty in the pilot distribution, not guarantees of significance.

The [frozen external protocol](PROSPECTIVE_PROTOCOL.md) selects **200 SUN3D-source space groups**, one image per group, from the official SUNRGBD archive. All NYUv2-containing branches are excluded. Source grouping yields 63 broader location proxies for a clustering sensitivity. The complete matrix reuses all 30 v3 adapters, original Marigold at two resolutions, and a descriptive specialist reference: **63 conditions / 12,600 predictions**. No additional training, external calibration, checkpoint selection, or significance-based stopping is allowed. The sample manifest and code were committed as `1fa376f` before external prediction.

After official-host connectivity failures, a pinned public mirror supplied archive members checked against the official central-directory names, lengths and CRC32 values. The original failed availability attempt and final successful acquisition audit are retained. All 200 selected groups passed the original first-choice selection and QC rules; no result-driven substitution occurred.

**Completed findings:** all **12,600 predictions** passed audit. At **256 inference**, mixed training reduces aligned AbsRel from **0.11106 to 0.09632 (13.3%)**, supported by nested/crossed Holm p=0.00030/0.00030 and broader-location p=0.00150. It also beats high512-only training at 256 (all three adjusted p=0.00030). At **512 inference**, mixed reaches **0.07038 versus 0.07224 (2.6% lower)**, but the primary tests remain nonsignificant (Holm p=0.29684/0.30778; location p=0.61617). The actual mean gain of 0.00186 is much smaller than the assumed 0.009 planning gain. The study supports a low-resolution benefit on this cohort; it does not establish a resolution interaction or universal superiority.

Fixed-calibration mixed512 AbsRel is **0.37886**, versus specialist **0.18231**; absolute-distance accuracy remains a limitation. The earlier fresh31 NYUv2 result is unchanged. Read the [complete English report](results/prospective_v4/RESULTS.md), [prediction audit](results/prospective_v4/verification.json), and [delivery checks](results/prospective_v4/delivery_checks.json). Three figures show the scores, complete comparison family, and examples fixed before outcome inspection. No new training was required; measured inference totaled **32.10 minutes**, peaking at **2,946.7 MiB** allocated VRAM (loading and audit excluded).

Exact local resume/audit uses the retained v3 checkpoints and old validation predictions for restoration checks, plus the pinned assets and new raw dataset/predictions outside Git:

```powershell
.\run_prospective.ps1
# To recompute/check all saved predictions without new inference:
.\run_prospective.ps1 -AuditOnly
```

This wrapper replays the frozen local study; a clone alone lacks excluded weights and raw arrays. Regenerate the earlier experiments and acquire the new data before attempting inference. Strict checkpoint/restoration checks intentionally reject mismatched training artifacts. This external subset is not the standard full SUNRGBD benchmark, and its raw-depth full-image mask differs from the earlier NYUv2 crop.

## Robustness replication: robustness_v3

The [frozen replication design](ROBUSTNESS_PROTOCOL.md) directly examines the limitations of the earlier single-training-set result. It uses **three new random draws of 128 training scenes**, **five training seeds**, and two fixed methods (high512 and mixed): 30 new training runs. The primary test contains **all 31 previously unused official test scenes**. The earlier 64-scene cohort is reported separately as an already observed replication cohort; the two cohorts are not described as 95 unseen scenes.

The analysis includes paired scene uncertainty, training-draw/seed/scene uncertainty, Holm adjustment of six predefined comparisons, and a [pre-test crossed-seed sensitivity](STATISTICAL_SENSITIVITY_V3.md). It also measures depth using **fixed validation-only scale/shift calibration**, compares a native metric-depth specialist at two input sizes, and probes brightness reduction and blur. GT per-image alignment and fixed calibration answer different questions and are reported separately. Three draws from one indoor dataset cannot establish universal superiority or cross-dataset generalization.

The protocol and source hashes were committed before new training. The statistical sensitivity addendum was committed during validation, before test inference. Earlier result directories are preserved; old low256 and preservation-loss results receive a retrospective 21-comparison adjustment but are not retrained in this replication.

**Completed findings:** on fresh31, mixed512 reduces aligned AbsRel from **0.09687 to 0.08462 (12.6%)**, but the predefined Holm-adjusted tests do **not** confirm the improvement at 0.05 (nested p=0.1088; crossed p=0.1094). Mixed training does outperform high512-only training at 256 inference (**0.10128 vs 0.12696**, adjusted p=0.0042/0.0084), while no advantage is established at 512. On already observed64, mixed512 reaches 0.06008 versus 0.06661, with adjusted p=0.0102/0.0174; this is a separate replication on previously examined scenes.

Fixed validation-only calibration yields fresh31 AbsRel **0.31261** and RMSE **0.9220 m** for mixed512, compared with calibrated specialist AbsRel 0.14752 at the approximately matched input size. A separately specified, **post-hoc official-default preprocessing diagnostic** gives specialist native AbsRel 0.25369 and calibrated AbsRel 0.15036, using larger 518x686 inputs. These results distinguish relative structure from meter-valued accuracy and show why preprocessing and metric choice matter.

Read the [complete English results and figures](results/robustness_v3/RESULTS.md), [main verification](results/robustness_v3/verification.json), and [official-default diagnostic](results/expert_default_diagnostic_v1/RESULTS.md). All **9,182 main plus 127 diagnostic predictions** were verified. The 30 new trainings used **52.40 minutes** of optimizer time and peaked at **2,512 MiB** allocated training VRAM; wall times are descriptive, not a controlled speed benchmark. The report retains the original percentile intervals and adds a clearly marked interval/test consistency display audit; significance statements follow the unchanged predefined Holm tests.

With the pinned assets and environment prepared, use **new external work and results directories** for a fresh reproduction (published completion markers require raw arrays that are intentionally not in Git):

```powershell
.\run_robustness.ps1 -Python 'E:\elec4240\.venv\Scripts\python.exe' -Assets 'E:\elec4240\assets' -Work 'E:\elec4240\replication-work' -Results 'E:\elec4240\replication-results' -PrepareOnly
# Remove -PrepareOnly to run training, validation, test, perturbations, audit and report.
```

The preparation-only path verifies the frozen scene selection and input hashes without training. Resume a local run with the same work/results pair. Model/data caches, adapters and raw prediction arrays remain local; the repository contains reproducibility records, metrics and figures. After data preparation, reproduce the separate [post-hoc diagnostic](EXPERT_DEFAULT_DIAGNOSTIC.md) from the repository with the prepared Python environment:

```powershell
python diagnose_expert_default.py --assets 'E:\elec4240\assets' --work 'E:\elec4240\expert-default-work' --results 'E:\elec4240\expert-default-results'
```

## Larger local experiment: scaleup_v2

The larger study uses **128 training scenes, 32 validation scenes and 64 new test scenes** that were excluded from all earlier selections. It compares LoRA trained at 256, at 512, with alternating 256/512 updates, and with alternating resolutions plus a uniform original-denoiser preservation penalty. Each condition has three training seeds and 320 optimizer updates, followed by one-step inference at both resolutions. Original Marigold and the Depth Anything V2 specialist remain references: 27 evaluation conditions and 2,592 predictions in total.

The added loss is masked student-versus-original **velocity MSE**, weighted by 0.1, at identical noisy training inputs. It is a lightweight prototype; confidence-weighted depth preservation and explicit cross-resolution consistency remain future methods. Equal updates do not imply equal compute. Read the [frozen protocol](SCALEUP_PROTOCOL.md), [full results](results/scaleup_v2/RESULTS.md), and [technical verification](results/scaleup_v2/verification.json). The complete fixed matrix enters fresh test evaluation only after the training/validation technical gate passes, with no model selection between stages.

**Completed findings:** at 512-pixel inference, mixed-resolution LoRA reaches **0.05968 +/- 0.00119 AbsRel**, versus **0.06661** for original Marigold (10.4% lower). The scene-bootstrap interval for the paired difference excludes zero. High-resolution-only training reaches 0.06075; its difference from mixed training remains uncertain. Uniform denoiser preservation reaches 0.05967 and provides no reliable extra mean-accuracy gain while increasing measured training time by about30%. At 256-pixel inference, low-resolution-only training remains strongest among the tested adaptation modes. All **2,592** predictions and **224** data hashes passed verification; peak allocated training memory was **2,512 MiB** on the local RTX5060 Laptop GPU.

With the pinned assets and Python environment already prepared:

```powershell
.\run_scaleup.ps1 -Python "$PWD\.venv\Scripts\python.exe" -WorkDir 'E:\elec4240-marigold-work'
```

The wrapper prepares the larger subset, runs or resumes matching completed conditions, audits validation, evaluates the fixed fresh test matrix, recomputes metrics and generates the report. Local preflight checked all four training paths and checkpoint restoration before full runs. Earlier result directories remain unchanged.

## Expanded evaluation

The completed expanded phase preserves the pilot and extends it to **64 training, 16 validation and 120 test scenes**, with **96 previously unused test scenes as the primary evaluation**. It repeats LoRA32, LoRA64 and output-head64 adaptation with training seeds 17, 29 and 43, at 160 optimizer steps. Each trained checkpoint is evaluated at both 1 and 4 denoising steps. Inference noise stays fixed independently of the training seed. There are nine adaptation runs and 21 evaluation conditions, including the original model and the specialist reference.

See the [frozen expanded protocol](EXPANDED_PROTOCOL.md), [expanded results](results/expanded_v1/RESULTS.md), and [expanded verification](results/expanded_v1/verification.json). Mean and sample SD across training seeds are reported separately from paired scene-bootstrap intervals. Pilot results below remain the original measurements; comparisons between phases also change the optimization budget and evaluation cohort.

On the fresh 96 test scenes, one-step LoRA64 reached **0.10697 +/- 0.00108 AbsRel** across three training seeds, versus **0.13585** for the one-step pretrained model (21.3% lower). LoRA32 reached **0.10913 +/- 0.00052**; the small LoRA64-versus-LoRA32 difference has a paired scene interval crossing zero. One-step inference outperformed four-step inference in this setup. The specialist reference remained best at **0.10089**, with different prior supervision. All **2,856** saved predictions, **200** data hashes and all nine training records were checked; LoRA and output-head checkpoint restoration reproduced saved predictions exactly on a fixed validation image.

Run or resume the expanded phase with the same environment and external work directory:

```powershell
.\run_expanded.ps1 -Python "$PWD\.venv\Scripts\python.exe" -WorkDir 'E:\elec4240-marigold-work'
```

`run_expanded.py` also accepts `--only lora64_seed29` to run a particular predefined condition. Completed evaluations are skipped only when their configuration, completion hash and local prediction files match. Dataset selection, source-code hashes and training checkpoint hashes are retained for auditing. Changes to frozen training/evaluation code are rejected on resume; use a new versioned experiment directory and protocol for a changed study.

## Validation exploration

The follow-up [inference study](results/validation_exploration_v1/RESULTS.md) tests LoRA merging and 256/512 processing resolution on **validation16 only**, using the fixed LoRA64 seed-17 checkpoint. Twelve configurations each receive three randomized timing rounds (576 predictions); this adds no training or test evaluation. See the [predefined exploration protocol](EXPLORATION_PROTOCOL.md).

Resolution materially changes the interpretation: one-step unmerged LoRA improves from **0.09676** at 256 to **0.07527** at 512, but pretrained Marigold at 512 reaches **0.07362** on the same validation scenes. Thus the low-resolution adaptation advantage does not establish an advantage at higher resolution. BF16 LoRA fusion changes predictions slightly; its measured quality and latency effects are reported separately. These validation findings motivate future experiments and do not replace the frozen expanded test results.

After completing the expanded run, reproduce this study from the repository directory:

```powershell
python explore_inference.py --assets 'E:\elec4240-marigold-work\assets' --expanded-work 'E:\elec4240-marigold-work\expanded_v1' --work 'E:\elec4240-marigold-work\validation_exploration_v1'
python summarize_exploration.py --assets 'E:\elec4240-marigold-work\assets' --expanded-work 'E:\elec4240-marigold-work\expanded_v1' --work 'E:\elec4240-marigold-work\validation_exploration_v1'
```

For a proposed method extension, see [resolution-consistent, prior-preserving adaptation](METHOD_DIRECTIONS.md). A completed [adapter-strength diagnostic](results/scaling_v1/RESULTS.md) checks five strengths at each resolution on validation16. Its descriptive grid minima differ by resolution, but the apparent high-resolution gain over the original model has a scene interval crossing zero. Scalar interpolation is an existing-method baseline; the more substantial training modifications in the method note remain proposals.

## Original pilot setup

- **Data:** 32 training, 8 validation and 24 test frames from NYU Depth V2. One frame per scene; selected scenes are disjoint across splits. The 8-image adaptation set is nested within the 32-image set (25% versus 100% of this pilot's training pool, not of the full dataset).
- **Controlled comparisons:** pretrained Marigold; rank-4 attention LoRA with 8 or 32 images; output-convolution-only adaptation with 32 images. All adaptations use 80 optimizer steps and one training seed.
- **Additional comparisons:** pretrained Marigold at 1 versus 4 denoising steps; Depth Anything V2 Metric Indoor Small as a specialist reference.
- **Hardware:** NVIDIA GeForce RTX 5060 Laptop GPU, approximately 8 GB VRAM. Processing long side is 256 pixels for Marigold.
- **Outputs:** per-image metrics, validation/test summaries, training curves, measured time and peak allocated GPU memory, and fixed qualitative examples.

Read the [recorded results](results/RESULTS.md), [predefined protocol](PROTOCOL.md), and [sources and implementation attribution](REFERENCES.md). Exact package versions are in [environment.json](results/environment.json).

**The original pilot evaluation uses ground-truth affine alignment for every method.** It measures relative depth structure; the scores are not evidence of accurate absolute distances from an uncalibrated model. Depth Anything has different pretraining and Hypersim metric-depth fine-tuning, so it is a useful reference but not a controlled architecture ablation. This 24-scene, single-seed, low-resolution pilot is not the full NYUv2 benchmark.

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

The original work here is the constrained experiment: controlled trainable-parameter and data-budget comparisons, consistent evaluation, and a measured quality/cost analysis. The expanded phase now includes more scenes and three training seeds, with separate validation exploration of inference cost and resolution. A useful next research question is whether adapting at higher resolution preserves or improves the already strong pretrained high-resolution predictions. A future training search should use validation data and reserve new unseen evaluation scenes for any selected configuration. Claims of broad novelty or general improvement require additional evidence.

All model and method sources are acknowledged in [REFERENCES.md](REFERENCES.md). Academic claims should cite the original papers as well as describe this pilot's departures from their training recipes.
