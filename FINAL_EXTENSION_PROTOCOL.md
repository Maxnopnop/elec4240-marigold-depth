# Exploratory completion study for the final report

The original prospective_v4 six-comparison study remains unchanged. Its 200
external groups and the NYUv2 test cohorts are now observed. Everything below
is a post-hoc extension, not an independently confirmed new method selection.

## Missing fixed-low-resolution control
Train low256 LoRA on the exact three robustness_v3 draws, with seeds
17/29/43/59/71, 320 updates and unchanged train_scaleup.py settings. Evaluate all
15 adapters at 256/512 on validation32, formerly-fresh31, and external200.
Use the original image-specific inference seeds and masks. Fit each new
adapter's global scale/shift using only validation32. Reuse all existing
high512/mixed/base scores without changing their artifacts. Save/check every
raw prediction, checkpoint and input hash, and verify checkpoint restoration.

Report the original six plus mixed-low256 at both resolutions as one separate
eight-comparison exploratory family, with 20,000 nested/crossed bootstraps and
Holm correction; external broader-location sensitivity also uses all eight.
Do not reinterpret the historical six-test adjusted p-values. Report one
exploratory mixed-versus-base resolution interaction using crossed paired
resampling, labeled separately from the eight-comparison family.

## Fixed descriptive failure analysis
For external original/mixed/low256 at each inference resolution, score every
adapter separately and then average equally over scenes and training runs.
Depth bins: (0.1,1), [1,2), [2,4), [4,10) meters. An image contributes to a bin
only with >=100 valid pixels; report contributing scene counts. Depth edges
are adjacent valid GT pairs with absolute jump >0.1m and relative jump >5%
of the smaller depth, dilated by two pixels and intersected with the measured
valid mask. Interior is the remaining mask. Compare aligned and fixed-calibrated
AbsRel. These strata are exploratory diagnostics, not extra confirmatory tests.
Examples: retain first three manifest IDs; additionally show the three largest
scene-mean mixed512-minus-base512 errors, explicitly labeled error-selected.

## Cost controls
Inference: validation's first eight images, three warm-up calls, five randomized
condition rounds (seed4285), original/high512/mixed/low256 at both resolutions;
adapter checkpoint draw1/seed17. Report median/p10/p90 and every round; measure
preprocessing through resized CPU prediction with CUDA synchronization, excluding
model loading. Check predictions against saved references, record GPU telemetry.

Compute budget diagnostic: draw1 training scenes; seeds17/29/43; low256/high512/
mixed; 120 seconds of synchronized optimizer-loop wall time per run, stop at
the first completed update reaching the budget. Warm-up passes take no optimizer
step. Cache/loading/warm-up excluded and separately reported. Evaluate only
validation32 at256/512. No checkpoint selection or external test inference.
This small one-draw validation diagnostic cannot establish general efficiency.

## Delivery and stopping
Finish the fixed matrices regardless of score. Keep weights/data/raw arrays
outside Git. Audit old result blobs against commit d14701c, publish new code,
metrics and figures to the authorized private repository, write a 6-8 page
English CVPR2022 final report including references, editable source and an
independent code supplement below10MB. Visually check every PDF page. User
explicitly requests Windows shutdown only after all deliverables are saved,
verified and publication checked; no Canvas submission is authorized.
