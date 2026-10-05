# Complete experiment archive — 4 October 2026

This repository preserves the full research history, including negative and inconclusive findings. The original depth final report is a historical deliverable, not a report of the later referring-expression segmentation experiments.

## Reports and experimental stages

| Stage | Entry point |
|---|---|
| Proposal | [PDF](reports/proposal/ELEC4240_Marigold_Proposal.pdf) |
| Milestone | [PDF](reports/milestone/ELEC4240_Milestone.pdf), [LaTeX](reports/milestone/milestone.tex) |
| Relative-depth final report | [PDF](reports/final/final_report.pdf), [LaTeX](reports/final/final_report.tex) |
| Vision Banana analysis | [English](reports/vision_banana_analysis/Vision_Banana_Research_Analysis_EN.pdf), [Chinese](reports/vision_banana_analysis/Vision_Banana_Research_Analysis_ZH.pdf) |
| Early experiments, scale-up and statistical validation | [Historical results](results), [README](README.md) |
| V7 metric depth and normals | [Report](results/metric_multitask_v7/RESULTS.html) |
| V8 reliability component study | [Report](results/reliability_v8/RESULTS.html) |
| V9 reliability training | [Report](results/reliability_training_v9/RESULTS.html) |
| V10 error/gradient/noise diagnostics | [Results](results/diagnostic_v10) |
| V11 generative segmentation baseline and loss pilot | [Results](results/generative_ris_v11), [Training protocol](generative_ris_v11/TRAINING.html) |
| V12 five-method, three-seed comparison | [Report](results/generative_ris_v12/RESULTS.html), [Analysis](results/generative_ris_v12/analysis.json) |
| V12 condition reliability, budget and correction/harm | [Report](results/generative_ris_v12/conditions/RESULTS.html) |

GitHub shows HTML source; download the repository and open the HTML files locally to view reports with their figures.

## Large experimental artifacts

Generated historical predictions, trained adapters/checkpoints and execution logs are published separately in the [all-experiments release](https://github.com/Maxnopnop/elec4240-marigold-depth/releases/tag/all-experiments-2026-10-04). The release manifest lists every included file with its SHA-256 and the checksum of every archive part. Large ZIPs are split into ordered `.zip.part001`, `.zip.part002`, ... files: concatenate the parts in binary order before extracting. Do not unzip individual parts. Smaller archives use a single `.zip` file.

Third-party raw datasets/annotations, downloaded pretrained models, Python environments, recomputable top-level training caches and unrelated projects are excluded. Dataset selections, source URLs and provenance hashes remain in the experiment protocols/manifests. These releases contain project-generated research artifacts, not a redistributed dataset. Existing local delivery ZIPs remain untouched; their original completion receipts describe those local packages, not the newly packaged release.

V12 completed 15 runs of 2048 updates and its registered auxiliary experiments. Its completion audit recomputed 17,040 primary predictions and 12,672 auxiliary records. The final automatic shutdown failed with Windows error 203; `failed_no_shutdown` records that shutdown failure, not an incomplete training matrix. Completion does not by itself establish method superiority or novelty. GQA auxiliary evaluation uses boxes and derived queries, not ground-truth segmentation masks. RefCOCO variants share COCO imagery.

Published checkpoints are research artifacts. Read the recorded protocols and initialize a separate work/results directory for reproduction; do not overwrite published results or automatically execute historical shutdown scripts.


Upload verified on 2026-10-04: 17 historical experiment work directories, 45,896 generated files, 197 release assets including the inventory, 18,880,926,293 archive bytes. All GitHub SHA-256 digests matched. [Upload receipt](delivery/2026-10-04/COMPLETION.json).
