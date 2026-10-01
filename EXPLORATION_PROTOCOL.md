# Validation-only inference exploration v1

Frozen before running the exploration. Expanded test results remain fixed.

## Questions

1. Does merging rank-4 LoRA into the BF16 UNet reduce inference latency, and how much does rounding change predictions or aligned accuracy?
2. Does increasing the processing long side from 256 to 512 improve aligned depth quality on validation scenes, and at what latency/memory cost?

## Fixed design

- Use only the 16 validation scenes in `results/expanded_v1/split_manifest.json`. No training or test predictions are generated in this phase.
- Use the first predefined training seed, `lora64_seed17`, after 160 updates at resolution 256. This is not selected by its test performance. No further training or checkpoint selection.
- Full factorial: pretrained, unmerged LoRA, merged LoRA; processing resolution 256 or 512; 1 or 4 denoising steps. Twelve configurations.
- Three timing rounds with independently shuffled configuration order from one NumPy RNG seeded 4244. These are timing repetitions, not independent training runs. Sixteen images per configuration per round: 576 timed predictions.
- Fresh pipeline per configuration/round, sequential GPU execution. Two validation-image warmups excluded from timing. Loading, LoRA restoration and merging are excluded from inference latency. Same synchronized inference helper and peak allocated memory measurement as expanded evaluation.
- BF16 pipeline. Restore the same adapter for both LoRA variants. For the merged variant call `unet.fuse_lora(safe_fusing=True)` then `unet.unload_lora()`, assert all adapter wrappers removed. Fusion may change numerical results; exact equivalence is not assumed.
- Input size 192x256 or 384x512 before output resizing to the original 480x640. Inference seed 17 + frame ID, ensemble size 1, unchanged fixed scheduler. Different resolutions use different latent shapes, so the experiment tests the complete inference configuration rather than pixel count alone.
- Preserve the existing valid-depth mask, Eigen crop, per-image GT affine alignment and clipping to [0.001, 10] m. This measures relative depth structure, not calibrated metric distance.

## Analysis and verification

Report mean AbsRel, RMSE and delta1, mean latency and sample SD across the three round means, and peak allocated VRAM. Recompute all stored metrics from saved raw predictions. Check validation IDs, data hashes, source hashes and checkpoint SHA. Compare repeated predictions for numerical reproducibility. At resolution 256, compare pretrained and unmerged predictions against the saved expanded validation predictions.

For merging, report raw maximum/mean absolute prediction differences and paired scene-level aligned metric differences. For resolution, compare 512 minus 256 for each method and step budget. Use paired bootstrap intervals over the 16 scenes (10,000 resamples, seed 4244), averaging repeats first. Intervals describe scene variation only; no multiplicity adjustment. Timing round variation is separate. These are exploratory validation findings, not a new held-out test claim. Keep positive and negative findings.

## Reproduction

Run `explore_inference.py --assets <work>/assets --expanded-work <work>/expanded_v1 --work <work>/validation_exploration_v1`, then `summarize_exploration.py` with the same arguments. Checkpoints and raw arrays stay outside Git. Configuration and source hashes reject changed resumes.
