# V10: error regions, gradient diagnostics and controlled label noise

This is a bounded exploratory follow-up to V9, not another attempt to obtain a significant V9 result. Old protocols, predictions and reports are preserved. No shutdown or cloud upload is performed.

The protocol is frozen before the new analyses. A preliminary one-image GPU check validates finite gradients and approximate linearity of separately computed gradient components; it is an implementation check, not a selected scientific outcome.

1. Recompute region errors for all 32 observed development scenes and all V9 seeds. Report sensitivity quartiles, depth-gradient regions, and camera-depth distance bins. Regions use identical masks across methods. Average scenes equally and retain missing-region counts. No new regional hypothesis tests.
2. Probe initialization and the six final uniform/weighted adapters on four fixed training images with two inference-noise draws each. Compute depth/normal and supervised/geometry gradient cosines and actual coefficient-scaled gradient norm ratios. Do not update parameters. Local conflict does not establish a causal explanation of test errors.
3. Train actual Marigold rank-4 LoRA on simplified RGB renderings of analytic geometry. Use 16 training images and eight images from different generator seeds, three training-label conditions, four geometry-weight rules and two training seeds. Each run has 64 updates. Both depth and normal training labels are corrupted; evaluation is always against clean analytic truth. Original camera and codec remain fixed. The masks are derived from clean geometry and identical across conditions, intentionally isolating weighting from noisy-mask changes.

The oracle rule weights geometry consistency using known normal-label angular error. It does not repair the corrupted latent supervision and is not an upper bound on achievable accuracy. A null result could reflect the proxy, the objective, short optimization, or synthetic-domain mismatch. It cannot on its own falsify weighting in real scenes.

The synthetic train/evaluation seeds differ, but share surface families and rendering. This is not independent real-world confirmation. No checkpoint/seed/temperature selection or significance-based continuation is allowed within this protocol.

Run using the existing CUDA Python environment, from the repository root:

```powershell
python -m unittest -v diagnostic_v10.test_core
python -m diagnostic_v10.pipeline
```

`results/diagnostic_v10/status.json` records finite pipeline progress. Every synthetic run saves an optimizer/adapter/RNG checkpoint every 16 updates, and final inference is checked after loading the saved adapter into a new model. The final report independently recalculates 416 initial/final synthetic task predictions and checks the 24 run histories, exact-restoration records, frozen source/input hashes and all old report/result bytes. Raw inputs, latent caches, optimizer states and predictions stay outside Git in `work/marigold-local/diagnostic_v10`.

## Numerical diagnostic amendment

The original gradient module stopped when one BF16 probe exceeded its 2% component-sum consistency screen. The original protocol, module and failure record remain unchanged. `numerical_amendment.json` pins `gradients_precision_v2.py` and `resume_precision_v2.py`: all BF16 probes are retained; flagged probes also receive FP32 (TF32 disabled) references with 0.1% consistency tolerance. Only diagnostic numerics changed; all noise-training settings remain frozen. The continuation command for this recorded run is `python -m diagnostic_v10.resume_precision_v2`. Do not start it while `pipeline.lock` has a live owner. Full per-probe flags and references are in `gradients.json`.
