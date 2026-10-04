# Method innovation plan for reliable geometric supervision

Updated: 2 October 2026.

**Status update: the candidate weighting component is now implemented and has completed its initial synthetic/training-label audit and GPU gradient checks. It has not been trained or evaluated for adapted-model accuracy.** See the [executable design](reliability_v8/EXPERIMENT_PLAN.md) and [V8 component evidence](results/reliability_v8/RESULTS.html). The design below records the initial proposal; this document is not a frozen experimental protocol. The completed [V7 experiment](results/metric_multitask_v7/RESULTS.html), its labels and its results remain unchanged.

The next question is whether the reliability of depth-derived normal labels can guide geometric regularization and reduce negative transfer during parameter-efficient depth and normal adaptation. This is motivated by measured failures, but neither the cause of those failures nor the proposed remedy has been established.

## Current project position

The project has a substantial course-report foundation: reproducible depth adaptation, controlled comparisons, external evaluation, cost measurements and audited results. The [existing eight-page report](reports/final/final_report.pdf) covers the earlier relative-depth study; it does not include V7 or this proposal. Incorporating the new direction would require revising and shortening that report within the course page limit.

The main risk is now scientific scope and evidence, rather than whether the local implementation runs. V7 used 128 training scenes, 32 previously observed validation scenes, three seeds and 320 updates per run. Its 12 formal runs used 34.76 minutes of optimizer time and peaked at 3,164.2 MiB of allocated training memory. This excludes preparation, loading and auditing and is not a prediction of larger-scale cost.

| Completed V7 setting | Metric depth AbsRel, lower is better | Derived-normal mean error, lower is better |
|---|---:|---:|
| Depth only | 0.37068 | — |
| Normals only | — | 39.83° |
| Shared joint adaptation | 0.41819 | 42.08° |
| Joint adaptation with geometry | 0.39685 | 45.07° |

Ordinary joint adaptation worsened both tasks in every paired seed. Adding geometry improved mean depth relative to joint adaptation, but this difference did not pass the exploratory Holm adjustment (p = 0.19344); normal degradation did (p = 0.00020). Prediction-to-prediction angular disagreement fell from 48.54° to 31.11°, demonstrating that greater agreement can coexist with worse agreement with the normal targets.

The [training-only label audit](results/metric_multitask_v7/label_quality.json) found that changing depth smoothing from sigma 1 to sigma 2 or 4 changed derived normal directions by an average of 9.97° or 20.87°. These are label-sensitivity measurements, not proof that one smoothing scale is correct. The current short training budget, shared adapter capacity and optimization are alternative explanations for the observed failures. Neither label noise nor gradient conflict has been isolated as the cause.

Sources for these observations: [V7 analysis](results/metric_multitask_v7/analysis.json), [frozen protocol](results/metric_multitask_v7/protocol.json), and [verification](results/metric_multitask_v7/verification.json). Earlier affine-aligned relative-depth scores must not be compared directly with the metric scores above.

## Candidate method

**Working hypothesis:** applying a uniform geometric penalty to uncertain derived-normal supervision can propagate unreliable constraints. Weighting that penalty by a validated reliability proxy may improve the accuracy tradeoff.

The current method already excludes unreliable pixels and depth discontinuities through a binary mask. The proposed difference is a continuous, spatially varying weight within the existing mask.

1. Compute candidate normal labels from the same training depth using several specified neighborhood sizes or smoothing scales. Sigma values 1, 2 and 4 are candidate starting points from the existing diagnostic, not validated optimal settings.
2. At each valid position, measure disagreement between the candidate normal directions.
3. Give consistent positions higher weights and unstable positions lower weights.
4. Retain both original supervised objectives and apply the weights only to the additional geometric term. Keep the task architecture unchanged for this first study.

One candidate definition is:

$$
u_i=\frac{1}{|\mathcal P|}\sum_{(a,b)\in\mathcal P}
\arccos\!\left(\operatorname{clip}(\langle n_i^{(a)},n_i^{(b)}\rangle,-1,1)\right),
\qquad w_i=m_i\exp(-u_i/\tau).
$$

Here, the normals are unit vectors in the same camera coordinate system, angles are in radians, and the pairs in $\mathcal P$ compare different label-construction scales. The mask $m_i$ retains the existing validity conditions. The temperature $\tau>0$ controls how strongly disagreement reduces the weight.

The candidate training objective is:

$$
L=\tfrac12 L_D+\tfrac12 L_N+
\lambda\frac{\sum_i w_i\left[1-\langle n(\hat D)_i,\hat N_i\rangle\right]}
{\sum_i w_i+\epsilon}.
$$

$L_D$ and $L_N$ are the existing supervised latent objectives for the fixed metric-depth encoding and normal encoding. The operator $n(\hat D)$ derives camera-facing normals from decoded metric depth using the calibrated geometry. All normal vectors are normalized before evaluating cosine disagreement.

Weights would be computed from training data, cached, and detached from optimization. They are reliability proxies, not calibrated correctness probabilities. The model cannot reduce the loss by learning to set all weights to zero. Weight construction, resampling to the geometry-loss grid, valid support and handling of near-zero total weight must be specified and tested before training.

No ground-truth depth or weight map is required at inference. The prediction architecture remains unchanged, although preparation and training costs still need measurement. Geometry remains scale-invariant in the ideal pinhole model; metric supervision is still required to learn absolute distances.

## Validate the reliability proxy first

Stable labels can be consistently wrong. Before using the proxy to train a model:

- Test known synthetic planes and intersecting surfaces with controlled depth noise, missing pixels and discontinuities. Measure whether larger disagreement corresponds to larger angular error against the known normals.
- Check flat, slanted and boundary regions separately. A proxy that merely selects easy flat surfaces may reduce coverage without identifying useful supervision.
- Examine representative training scenes and compare weighting against the existing binary mask.
- Keep this component validation separate from model generalization claims. Synthetic geometry alone does not establish the accuracy of real-world normal labels.

If the proxy does not identify less reliable supervision, revise or abandon it before claiming a useful regularizer. Do not change evaluation labels to favor the candidate.

## Minimum informative ablation

| Condition | Question |
|---|---|
| Joint adaptation without geometry | What is the shared-model baseline? |
| Current uniform geometric penalty | What does the existing penalty change? |
| Smaller uniform geometric weight | Is any gain explained by weaker regularization? |
| Reliability-weighted geometry | Does the proposed spatial rule help? |
| Spatially shuffled reliability weights | Does their spatial placement matter? |

For the shuffled control, permute weights within each image's valid support with fixed recorded seeds. Preserve the weight distribution and total weight while disrupting its association with individual locations.

Also retain corresponding depth-only and normal-only controls. Match initialization, training data, image exposure, processing resolution and checkpoint selection across the new comparisons. If the training budget or label construction changes, rerun the affected controls; historical V7 scores are not automatically matched controls for a revised protocol.

Use at least three seeds for development, report every seed and record actual optimizer time, memory and parameter counts. If making an efficiency claim, include a matched-compute comparison. Use a common evaluation mask for all methods and report whole-mask scores; high- and low-reliability subsets are supplementary diagnostics.

Before independent evaluation, specify the primary contrasts, multiplicity correction, uncertainty analysis and any acceptable degradation margin. Failure to find a significant normal difference is not evidence of no harm: a non-inferiority claim needs a predefined margin and suitable confidence-bound analysis. The already observed validation and test cohorts remain development or retrospective evidence for this new method.

## Order of work and open decisions

1. Establish more stable single-task learning curves and verify normal-label quality. The V7 budget is only 2.5 passes over the training images.
2. Validate the reliability proxy on controlled geometry and training data.
3. Implement only the weighted geometric term and run its implementation checks.
4. Freeze a new, versioned development protocol and run the ablations above.
5. If the evidence supports further work, freeze the selected method before obtaining genuinely unobserved evaluation scenes.

The temperature, loss-weight candidates, longer training budget, label scales, weighting-grid construction, confirmatory sample size and non-inferiority margin remain open. They must be resolved before the corresponding experiments, without repeatedly extending testing until a favorable result appears.

Task-specific adapters, delayed activation of geometry and one-way gradient propagation are possible later controls. Adding them together now would obscure which change helped. With shared trainable parameters, stopping gradients through one output does not guarantee that its predictions remain unchanged when another branch updates those parameters.

## Novelty and related work

This is a candidate course-project modification, not a claim of a globally new algorithm. The contribution would be a precise reliability rule motivated by observed failure, its implementation, and controlled evidence showing when it improves the task tradeoff.

- [Marigold: Affordable Adaptation of Diffusion-Based Image Generators for Image Analysis](https://arxiv.org/abs/2505.09358) already studies adaptation to depth, surface normals and other dense analysis tasks. Adding normals alone does not establish novelty.
- [Unsupervised Learning of Geometry with Edge-aware Depth-Normal Consistency](https://arxiv.org/abs/1711.03665) establishes relevant depth-normal and edge-aware consistency techniques.
- [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115) studies uncertainty-based task weighting. Its task-level uncertainty is different from the proposed per-location label-stability proxy, but it is important background.
- [MTLoRA: A Low-Rank Adaptation Approach for Efficient Multi-Task Learning](https://arxiv.org/abs/2403.20320) already combines task-agnostic and task-specific adaptation. Such components should be credited and evaluated as related methods.

A broader prior-art review is still needed before a publication-level novelty claim. More training, a different learning rate, or a renamed combination of established components is not sufficient evidence of method novelty. Successful ablations would support a specific claim about this design and setting; negative results should be retained with the same clarity.
