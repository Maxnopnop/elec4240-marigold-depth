# Official-default specialist preprocessing diagnostic

This post-hoc diagnostic was added after the clean robustness_v3 results were observed. It evaluates the pinned Depth Anything V2 indoor model with its unchanged processor defaults. Actual input is 518x686, larger than Marigold 384x512. It is a descriptive reference with a different pixel budget, not an added confirmatory comparison.

The model is metric-fine-tuned on Hypersim. Fixed calibration uses only the same 32validation scenes and is saved before test prediction. All 127 predictions and recomputed metrics passed the audit.

| Cohort | GT-aligned AbsRel | Native AbsRel | Native RMSE (m) | Calibrated AbsRel | Calibrated RMSE (m) |
|---|---:|---:|---:|---:|---:|
| fresh31 | 0.09205 | 0.25369 | 0.7567 | 0.15036 | 0.5069 |
| observed64 | 0.07300 | 0.20571 | 0.6091 | 0.13817 | 0.4762 |

These numbers do not replace the predefined custom-input comparisons. Compare all input settings, distinguish native from calibrated outputs, and do not attribute preprocessing differences solely to model architecture. No cross-dataset or universal superiority claim follows.
