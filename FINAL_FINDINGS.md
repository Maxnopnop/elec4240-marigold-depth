# Final findings and interpretation

The final extension is complete. It adds **15 fixed-low-resolution control trainings** and **9 fixed-time diagnostic trainings**. The original frozen external study is preserved; the additions below are exploratory because their evaluation cohorts were already observed.

## What the missing control changes

On external200 at 256 inference, low-only training obtains **0.09277 aligned AbsRel**, versus mixed **0.09632**. The mixed-minus-low difference is **+0.00355**. Its separate eight-test family gives nested/crossed Holm p=**0.00150/0.01560**, and broader-location p=**0.03660**. Thus the observed low-resolution adaptation benefit does not uniquely support mixing resolutions: the matched low-only control performs better on this cohort.

At 512, mixed obtains **0.07038** versus low-only **0.07140**. The difference remains unconfirmed (crossed Holm8 p=**0.41038**, location p=**0.39578**). Failure to detect a difference does not establish equivalence. On NYUv2-31, neither mixed-versus-low comparison passes the exploratory correction (crossed Holm8 p=**0.59617** at both resolutions).

The original six-comparison conclusions remain unchanged: mixed beats the original model and high-only training at external 256 inference; its advantage over the original at 512 is not confirmed. The [complete extension report](results/final_extension_v5/RESULTS.md) prints all eight exploratory comparisons, including negative results.

## Resolution interaction and failure patterns

The separately specified post-hoc interaction, `(mixed256-original256) - (mixed512-original512)`, is **-0.01288**, with crossed percentile95% interval **[-0.01800, -0.00808]** and unadjusted p=**0.00005** on external200. The same diagnostic is unconfirmed on NYUv2-31 (p=**0.17039**). This describes a cohort-dependent effect pattern, not a prospective confirmation or an identified causal mechanism.

Mixed512 aligned boundary AbsRel is **0.14719**, compared with **0.06639** in the interior. Its near-depth bin has aligned error **0.33324** and fixed-calibrated error **1.07609**, with **89** contributing scenes. The original near-bin aligned mean is **0.32195**. Aggregate mean improvements can therefore coexist with local regressions. Bins have different scene membership and measured-depth boundaries are imperfect; no extra significance tests are claimed for these descriptive strata.

The first three manifest examples remain outcome-independent examples. The three largest mean mixed512 regressions are IDs **125, 196 and 58**, with differences **+0.04374, +0.04224 and +0.03712**. Their displayed adapter is fixed to draw1/seed17, while selection averages all 15 runs. These selected images illustrate doorway-boundary errors, smoothed chair structure and a cluttered computer room with missing depth; they do not estimate failure frequency.

## Accuracy and local cost

Five randomized condition rounds, eight fixed validation images and three warmups produce **320 exact prediction-reference matches**. Original/mixed median latencies are **66.71/73.41 ms at 256** and **188.75/193.75 ms at 512**. The three unmerged adapter strategies are close (73.41–73.59 ms at 256, 193.67–193.84 ms at 512). Resolution changes dominate these small between-adapter timing differences on this machine.

The fixed-time comparison gives each schedule **120 seconds of optimizer-loop wall time**, with three seeds on draw1. Low-only has the best validation mean at 256 (**0.08761**, mixed **0.09041**, high-only **0.10431**); high-only has the best mean at 512 (**0.06801**, mixed **0.06963**, low-only **0.07132**). Update ranges are low-only **405–454**, high-only **405–422**, mixed **390–426**. These means do not establish general equal-compute superiority: the diagnostic uses one training draw and an observed validation set, and excludes loading, caching and warmup.

The uncontrolled low-only execution contains isolated extremely long wall-time readings. They remain in the raw records and are not used to infer speed. The separate controlled measurements passed the delivery checks.

## Audit and deliverables

The [independent delivery audit](results/final_extension_v5/delivery_checks.json) verifies **7,890** control raw-prediction hashes, **30** checkpoint restoration comparisons, **320** timing reference records and **576** independently recomputed budget-validation metrics. All **939** historical result blobs at `d14701c` remain unchanged. Model weights and raw datasets/predictions remain outside Git.

The [eight-page English final report](reports/final/final_report.pdf) uses the unchanged official CVPR2022 style. The [Overleaf source](reports/final/overleaf_source.zip) and [separate code supplement](reports/final/supplementary_code.zip) are included. The supplement excludes weights, datasets, raw arrays and installed frameworks; both its compressed and uncompressed size are below 10 MB. See [FINAL_REPRODUCE.md](FINAL_REPRODUCE.md) for replay requirements and limitations. No Canvas submission has been performed.
