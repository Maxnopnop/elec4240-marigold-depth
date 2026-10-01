# Prospective external confirmation after sample-size planning

This protocol is specified before any model prediction on the external cohort.
Earlier NYUv2 results remain exploratory evidence for planning, not new tests.

## Planning and fixed budget

The planning simulation uses both old cohorts separately, effect assumptions of
0.006/0.009/0.012 aligned AbsRel improvement, error SD multipliers 1/1.5, three/five/
ten training draws and five seeds, and 31--1000 candidate independent scene units.
Each simulated trial applies the complete six-contrast Holm procedure; both
nested and crossed branches must reject. These are known-pilot-distribution
approximations, not guarantees or exact prospective power calculations.

Fix N=200 new scene groups and retain ALL 30 robustness_v3 checkpoints (three
draws x five seeds x two methods). The fresh31 pilot gives approximately 80%
planning power at N=200 for a 0.009 gain without variance inflation. More adverse
assumptions require substantially larger cohorts. Increasing training repetitions
cannot substitute for independent test scenes. No extra training or checkpoint
selection occurs in this external test.

## Independent data acquisition and exclusions

Use only SUNRGBD/xtion/sun3ddata in the official SUNRGBD archive. Exclude all NYUv2,
B3DO, Kinect2 and RealSense branches. The metadata census has 3,090 RGB frames
across 207 first-path-component space groups. Use one image per such group, never
treat multiple sequences or frames in a group as independent observations. This
is an external SUN3D-source indoor test, not additional unseen NYUv2 scenes or a
standard full SUNRGBD benchmark. Broad image-pretraining overlap cannot be ruled
out; no claim of independence from every model's pretraining is made.

Rank groups by SHA256('4271:'+space). Within each group rank candidate RGB paths
by SHA256('4272:'+path). Before prediction, try up to the first three candidates
in each group. Accept a pair only if RGB is decodable, raw depth is uint16 and
dimension-matched, and at least 10% of pixels have 0.1<depth<10 meters. Decode
depth by rotating uint16 right by three bits and dividing by 1000, following
official SUN3D depthRead.m. Use the archive's associated raw depth PNG, not the
filled depth_bfx map. Exclude an exact decoded-RGB duplicate of any earlier local
NYUv2 study input or accepted external frame. Record exclusions before prediction.

Take the first 200 eligible groups in the fixed ranking; the remaining seven are
availability/QC reserves only, never result-driven replacements. If fewer than
200 groups pass, stop acquisition and document the shortfall before any inference.
Freeze the final IDs, source members, CRCs, data hashes, and exclusions in Git.
Scene groups can share buildings or institutions. Report their distribution and
a conservative broader-location clustering sensitivity; 200 groups are not 200
independent buildings. Selection/QC uses no model output or error score.

## Fixed evaluation

Evaluate original Marigold and every high512/mixed checkpoint at processing
resolution256/512, BF16, one step, ensemble1, inference seed427200+sampleID.
Add the same pinned Hypersim-fine-tuned Depth Anything V2 indoor specialist using
its unchanged processor defaults as a descriptive reference. Total63 conditions,
12,600 predictions. Restore each condition on the old fixed validation image and
require exact equality to its saved previous prediction before new evaluation.

Primary metric: per-image GT-affine-aligned AbsRel, averaged equally across
selected groups and training runs. Use all finite measured pixels with
0.1<GT<10m; clip predictions to[0.1,10]. Do not transfer the NYUv2 Eigen crop to
this external dataset. GT alignment measures relative structure, not native
absolute distances. Also report RMSE and delta1 descriptively.

Secondary diagnostic: reuse each condition's ALREADY FIXED NYUv2-validation
global affine calibration, without any external recalibration. Report calibrated
AbsRel/RMSE/delta1 and specialist native metric scores. No external GT enters
calibration, training, checkpoint selection or hyperparameter choices.

## Frozen inference and decision rule

Six aligned-AbsRel contrasts: mixed-base at256/512; high512-base at256/512;
mixed-high512 at256/512. Use unchanged centered two-sided bootstrap tests:
20,000 nested draw/seed/paired-scene replicates(seed4264) and20,000 crossed
draw/seed/scene replicates(seed4266). Apply Holm separately to the full six-test
family in each analysis. A directional improvement requires a negative mean
difference and BOTH adjusted p-values<0.05. Report all six comparisons regardless
of sign. No equivalence inference from a nonsignificant test.

Report percentile95% intervals alongside p-values, explicitly noting these are
not mathematical inverses of the centered tests. Also show symmetric intervals
matched to centered absolute deviations. No confidence-interval cherry-picking.
The broader-location sensitivity resamples proxy location blocks with all their
selected groups, plus crossed training draws/seeds; it remains conditional on
imperfect location metadata. Qualified conclusions must disclose any disagreement.

Stop only when all63 conditions have completed on the frozen200 groups. Do not
inspect aggregate hypothesis tests until all predictions and metric audits pass.
Do not add scenes, seeds, models or tuning after observing the outcome. Resume
technical interruptions only with identical frozen source/data/checkpoint hashes.
Publish negative as well as positive results, costs and verification records.

## Sources

- SUNRGBD official data: https://rgbd.cs.princeton.edu/ and https://rgbd.cs.princeton.edu/data/README.txt
- Song, Lichtenberg and Xiao. SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite. CVPR2015.
- SUN3D source and sequence list: https://sun3d.cs.princeton.edu/ and https://sun3d.cs.princeton.edu/SUN3Dv1.txt
- Xiao, Owens and Torralba. SUN3D: A Database of Big Spaces Reconstructed using SfM and Object Labels. ICCV2013.
- Official decoder: https://github.com/PrincetonVision/SUN3Dsfm/blob/master/depthRead.m
- Holm implementation reference: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html
