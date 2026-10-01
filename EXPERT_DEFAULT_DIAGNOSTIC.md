# Official-default specialist preprocessing: post-hoc diagnostic

This diagnostic was specified after the clean robustness_v3 test scores were
observed. It is explicitly exploratory and is not part of the six predefined
comparisons or an additional confirmatory test.

The fixed specialist's native metric error changed substantially between custom
252x336 and 378x504 inputs. To avoid attributing preprocessing effects to the
model, evaluate the pinned processor with **no size override**, as in its model
card's usage example. Its saved configuration requests size518, preserves aspect
ratio and enforces multiples of14; record the actual input size. This uses more
pixels than Marigold512 and is a default-setting reference, not equal compute.

Use exactly the same32validation images and31fresh/64previously observed test
images as robustness_v3. No training, augmentation, parameter search or method
selection. Fit the same global affine diagnostic on validation only, freeze its
coefficients before test prediction, and report native metric, calibrated metric
and GT-affine-aligned scores. The specialist's metric fine-tuning source is
Hypersim, as corrected in CORRECTIONS.md. Preserve every prior result.

Save all127predictions locally; recompute every score and hash. Publish metadata,
per-image scores, coefficients and an English descriptive report. Do not report
new significance claims or replace the original primary analysis with this result.
