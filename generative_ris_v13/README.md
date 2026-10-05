# V13: data and optimization scale in generative referring segmentation

This independent module continues matching V11 seed-specific step-1000 adapters.
It does not edit V11/V12 implementations, original COCO pixels, or RefCOCOg labels.
The three arms are the frozen V12 pixel, always-pair, and readiness-pair objectives.
No superiority of readiness weighting has been established by V12.

The registered matrix has 18 runs: 3 objectives × 2048/8192 nested training images
× seeds 17/29/43, with 8192 updates per run and fixed 2048/8192 checkpoints.
At 2048 updates the large pool has exposed only 2048 distinct images. At 8192
updates the small pool has four epochs and the large pool has one. These are
equal-update comparisons, not equal-epoch or convergence comparisons.

Fresh320 evaluation is blocked until every run and both checkpoints are verified.
The report recomputes native mask IoU and object selection from saved PNG files.
There are 13 fixed contrasts × 2 metrics in one Holm family. Confidence intervals
are pointwise image-cluster bootstrap intervals conditional on three fixed seeds.
This is an enriched same-category-pair COCO subset, not a full official benchmark.

## Execution

Run modules from the repository directory. `ELEC4240_ROOT` can override the
workspace root; original manifest paths must resolve (moving to Colab requires
an audited path relocation, not data substitution).

1. `python -m generative_ris_v13.check_resume --root <workspace>`: real training-only
   checkpoint/reconstructed-pipeline/AdamW/RNG next-update equivalence test.
2. `python -m generative_ris_v13.test_protocol`: exposure and statistical-family tests.
3. `python -m generative_ris_v13.benchmark`: deterministic timing on old training images.
4. `python -m generative_ris_v13.prepare_cache`: integrity recheck and resumable
   training-only shards; records measured overhead and revised budget, preserving
   the original failed rough budget estimate.
5. `python -m generative_ris_v13.run`: refuses a failed budget, freezes sources and
   inputs, trains the finite matrix, evaluates, audits, and reports.

An existing `formal.run.lock` requires checking PID, creation time, and command
before any manual recovery. Do not start a competing runner. A `PAUSE` marker in
the V13 work directory stops at the next status boundary; the most recent atomic
128-update checkpoint is retained. No shutdown code is present. All fresh results,
including null/negative findings, are retained without adaptive tuning.

The initial nondeterministic next-update test failed despite matching forward
loss. Deterministic CUDA/cuDNN settings passed all three objectives exactly; the
failure and passing receipts remain in the work directory. Formal timing uses
these same settings. Completed engineering checks are not effectiveness results.
