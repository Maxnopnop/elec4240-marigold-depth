# Method extensions motivated by the validation experiments

**Historical direction:** this document records the earlier resolution-adaptation proposal. For the latest direction motivated by the completed metric-depth and normal study, see [the method innovation plan](METHOD_INNOVATION_PLAN.md). That new weighting method remains proposed and untested.

## Recommended research question

Can a small depth adapter retain low-resolution adaptation gains while preserving the pretrained model's stronger high-resolution geometry?

Working project title: **Resolution-Consistent, Prior-Preserving LoRA for Efficient Monocular Depth Estimation**.

The complete design below remains a proposed course-project extension; the partial prototype is distinguished in the status update below. The underlying components are related to established methods; neither the title nor the combination establishes publication-level novelty.

Implementation status update: `scaleup_v2` now implements and tests mixed-resolution LoRA and a lightweight uniform denoiser-velocity preservation penalty. See [the fixed larger experiment](SCALEUP_PROTOCOL.md). This is a partial prototype: confidence-weighted depth preservation and explicit cross-resolution geometric consistency below remain proposed, unimplemented extensions. Do not describe the lightweight velocity objective as the complete proposed method.

## Evidence and the simplest control

At 256 pixels, the seed-17 adapter improves validation AbsRel; at 512 pixels, its full-strength one-step prediction is slightly worse on average than the original model. These are 16-scene validation observations, not evidence of catastrophic forgetting or a demonstrated causal mechanism. Limited training data, optimization strength and resolution mismatch are competing explanations.

The companion [strength diagnostic](results/scaling_v1/RESULTS.md) evaluates five scalar strengths at each resolution. Scaling the update is a necessary simple control before proposing a more complex training method. It introduces no new trainable parameters and can later be fused for a fixed configuration. A grid minimum selected on these same validation scenes is not a confirmed generalization gain.

The completed diagnostic found grid minima at strength 1.0 for 256 pixels and 0.5 for 512 pixels. At 512, half-strength LoRA achieved AbsRel 0.072963 versus 0.073620 for the original model and 0.075273 for full strength. The half-strength-minus-original mean difference was -0.000657, with a paired 95% scene-bootstrap interval of [-0.003301, +0.002159]. The interval includes zero, and selection used the same validation scenes, so this is motivation for further investigation rather than evidence of a reliable gain. All 160 predictions were verified, including exact reproduction of all 64 endpoint predictions.

## Proposed training changes

1. **Mixed-resolution adaptation.** Cache or construct both 256- and 512-pixel versions of the same training images. Sample resolutions with a fixed balanced schedule while updating the same rank-4 LoRA. This tests whether training/inference resolution mismatch explains the observed behavior.
2. **Cross-resolution consistency.** For a training image, encourage the normalized 256-pixel depth prediction to agree with a downsampled normalized 512-pixel prediction. Compare geometry after removing global scale/shift; do not force equality of unaligned depth values. Predictions use a controlled noise policy. This regularizer should supplement ground-truth supervision, because agreement alone can produce consistently incorrect depths.
3. **Confidence-weighted preservation of the original predictor.** Use the frozen original Marigold as a teacher. Penalize unnecessary deviations primarily where its training-image geometry is reliable. One reproducible confidence proxy is its affine-aligned error against training ground truth; another is stability across image transformations, which avoids extra labels but does not guarantee correctness. Fix one proxy and tune its parameters using validation only. Do not preserve teacher mistakes indiscriminately. Teacher targets can be cached; no teacher or ground truth is required at deployment.

An initial objective is

`L = L_denoising + lambda_consistency * L_cross_resolution + lambda_preserve * L_teacher`.

The original supervised diffusion loss remains the anchor. For the additional terms, explicitly define depth normalization, masks, scale matching and teacher confidence before implementation. Depth-domain regularization needs differentiable decoding and must be profiled on the 8 GB GPU. A latent-space approximation is cheaper but is a different objective and should be described as such. No memory feasibility claim has yet been established for this proposed training objective.

## Minimum useful ablation

| Condition | Purpose |
|---|---|
| Original model at 256 and 512 | Resolution-matched baseline |
| Current single-resolution LoRA | Established adaptation baseline |
| Scalar-strength interpolation | Test whether a simpler existing method suffices |
| Mixed-resolution LoRA | Isolate training-resolution exposure |
| Mixed resolution + consistency | Isolate geometric consistency |
| Mixed resolution + teacher preservation | Isolate prior preservation |
| Mixed resolution + both regularizers | Test the combined proposal |

Use the same 64 training scenes, freeze candidate settings before each validation study and repeat promising conditions with three training seeds. Match update budgets and report actual wall time/VRAM; higher-resolution or multi-forward-pass training costs more per update, so equal steps are not equal compute. If claiming compute efficiency, include a matched-compute comparison. Compare at both inference resolutions with the same number of denoising steps. Report relative-depth metrics and paired scene intervals, all seeds, and failures.

The current test scenes are already observed. They cannot validate a method selected from the current exploration as if it were chosen independently. Reserve new scene-disjoint official test scenes before evaluating any selected final method. Avoid promising better-than-baseline results in the proposal.

## Other directions

- **Boundary-weighted geometry loss:** emphasize genuine depth discontinuities identified from training depths. Add a boundary-specific metric so this direction has a testable goal. RGB texture edges are not reliable substitutes for depth boundaries. This is a useful extension, but edge-aware losses are established techniques.
- **Image-dependent adapter strength:** predict a scalar from inexpensive image or resolution features. Start with fixed per-resolution strengths, then ask whether image-dependent control adds a reproducible gain. A learned gate has additional selection capacity and can overfit 16 validation scenes. Its inference must not use ground-truth depths or oracle errors.
- **Robustness under corruptions:** evaluate fixed brightness, blur and noise changes to test whether adaptation harms robustness. This provides a different contribution through analysis; it should use a predefined corruption protocol and not be labeled a new architecture.

## Closest background and limits on novelty

- [WiSE-FT: Robust Fine-Tuning of Zero-Shot Models](https://arxiv.org/abs/2109.01903) studies interpolation between pretrained and fine-tuned weights. Scalar LoRA interpolation is not an original algorithm in this project.
- [ResAdapter: Domain Consistent Resolution Adapter for Diffusion Models](https://arxiv.org/abs/2403.02084) studies resolution adaptation for diffusion image generation. It is relevant background, not a depth-estimation method reproduced here.
- [Iris: Bringing Real-World Priors into Diffusion Model for Monocular Depth Estimation](https://arxiv.org/abs/2603.16340) studies prior transfer and spectral consistency for diffusion depth. It is particularly important prior work for any proposal claiming consistency or preservation as a new concept.

The defensible course contribution is an explicit, affordable adaptation modification motivated by a measured failure mode, plus controlled ablations establishing when it helps. A claim of a globally new method would require substantially broader prior-art analysis.

## Proposal-ready method paragraph

We propose to investigate resolution-consistent, prior-preserving LoRA adaptation for monocular depth estimation. Motivated by preliminary evidence that low-resolution adaptation gains may diminish at higher inference resolutions, we will study mixed-resolution training and a cross-resolution geometric consistency loss. We will further explore confidence-weighted regularization toward the original pretrained predictor to limit harmful changes while retaining useful task adaptation. Scalar adapter-strength interpolation will serve as a simple baseline. Controlled ablations will separate the effects of training resolution, consistency and prior preservation, with accuracy and computational cost evaluated across multiple training seeds. These modifications are proposed research directions; their benefits remain to be established.
