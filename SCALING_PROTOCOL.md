# Validation diagnostic: resolution and LoRA strength

This protocol is fixed before generating new predictions. It follows the completed validation resolution study and is exploratory, not an independent test.

- Question: does the useful LoRA strength differ between processing resolutions?
- Fixed checkpoint: expanded `lora64_seed17`, trained at 256 for 160 updates. No training or checkpoint search.
- Use all 16 existing validation scenes, one denoising step, long side 256 and 512, scale in {0, 0.25, 0.5, 0.75, 1}. Ten configurations, 160 predictions. No test access.
- Apply the scalar to the existing LoRA update through Diffusers `unet.set_adapters`, without fusion. This is weight-update interpolation, not blending depth outputs. It is a known technique, not claimed as an original algorithm.
- Same inference seed 17 + frame ID, mask/crop/GT affine alignment and metrics. Store every prediction and recompute every metric. Verify scale 0 and 1 against the previous pretrained and unmerged predictions; scale 0 may retain adapter computation and is not a deployment latency baseline.
- Select nothing for deployment. Report the entire curve, per-resolution grid minima and the minimum of an equally weighted 256/512 mean. Minima selected and evaluated on the same validation scenes are optimistically biased. No final-test or generalization claim.
- Include paired scene-bootstrap intervals for every nonzero strength versus scale 0, using 10,000 resamples and seed 4245; report no multiplicity-adjusted significance. One checkpoint is a limitation.

This diagnostic informs possible future training modifications: mixed-resolution training, cross-resolution geometric consistency and confidence-weighted preservation of the original depth predictor. These modifications are proposals, not implemented or validated by this diagnostic.

Related work: [WiSE-FT](https://arxiv.org/abs/2109.01903) studies weight interpolation for robust fine-tuning; [ResAdapter](https://arxiv.org/abs/2403.02084) studies diffusion image-generation resolution adaptation; [Iris](https://arxiv.org/abs/2603.16340) studies real-world priors and consistency for diffusion depth estimation. None is directly reproduced here.
