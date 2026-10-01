# Corrections to earlier experiment descriptions

## 2026-10-01: specialist training provenance

Earlier project prose and generated metadata described
`depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` as NYUv2-supervised or
NYUv2-trained. **That description was incorrect.** The exact model card at our
pinned revision `8078d68a9c75a972131914f6afd0c1723be0da7f` identifies synthetic
**Hypersim** as its indoor metric-depth fine-tuning dataset. The official metric
depth repository independently identifies the indoor models with Hypersim.

Sources:

- [Pinned model card](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf/blob/8078d68a9c75a972131914f6afd0c1723be0da7f/README.md)
- [Official metric-depth training description](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/README.md)
- Retrieved-source metadata and SHA256: [expert provenance correction](results/robustness_v3/expert_provenance.json).

The associated suggestion that this checkpoint's NYUv2 supervision might overlap
our validation scenes is withdrawn: its stated metric fine-tuning source does not
support that claim. This does not establish an exhaustive no-overlap guarantee
for every image in the model's broader representation-learning history.

This correction changes **the interpretation of the reference model's training
history**, not any measured prediction, metric, model revision or test membership.
Our LoRA models receive NYUv2 labels in their local adaptation, while this fixed
reference has its own different pretraining and Hypersim metric fine-tuning.
Architecture, training data, precision and compute remain uncontrolled differences.
Neither method's score establishes universal model superiority.

Historical frozen reports, configuration JSON and hashed scripts are retained
byte-for-byte for provenance; any NYUv2-trained wording in those artifacts is
superseded by this correction. This includes the original pilot metadata,
expanded/scaling/inference reports and scaleup_v2. Current README and the new
robustness report use the corrected description. The training/inference code and
all experiment choices remain unchanged.
