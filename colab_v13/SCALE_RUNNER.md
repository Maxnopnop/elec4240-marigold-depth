# Finite cloud scaling study

The resource-registered design uses pixel, pair_always and pair_ready, nested
2048/4096 training images, seeds17/29, and4096 updates per run. Checkpoints2048
and4096 are fixed. All12 runs must pass their checkpoint barrier before any
fresh320 holdout inference. Original labels and images remain unchanged.

`prepare_scale_cache.py` builds cloud BF16 training caches and verifies durable
Drive shards. `scale_queue.py` waits for that existing process, verifies the
source bundle, runs `scale_preflight.py`, then invokes `scale_run.py`.
Preflight exercises the production update/cache/checkpoint path for each arm,
with a fresh process restoring durable step16 to reproduce step17 exactly.
It uses training images only, including inference timing.

The runner refuses to start if the measured budget exceeds50 hours. It retains
the registered20% optimization margin and an8-hour minimum overhead allowance.
The execution ledger includes cache/preflight and a conservative prior-engineering
allowance. Reservations are persisted before compute; interrupted reservations
remain charged. A watchdog ends the process if a reservation expires. Automatic
resumption after Colab runtime destruction is not claimed: reconnect, verify
the pinned environment/data/cache, and inspect the saved error first.

Each128-update state contains adapter, optimizer, CPU/CUDA RNG, history and
protocol hash. Immutable generations are copied and rehashed before updating
the latest pointer. No checkpoint deletion is automated. Drive destination:
`MyDrive/ELEC4240/v13-cloud-scale-2026-10-05/scale_experiment`.
Place a `PAUSE` file in the project Drive directory to stop at the next bounded
chunk. No OS shutdown or local GPU execution is requested.

Evaluation saves native masks and per-image completion records, permitting
restart without rerunning completed images. Reporting reopens and rechecks all
15360 expression predictions and corresponding empty-prompt masks, performs
26 predeclared Holm-adjusted tests, and reports descriptive correction/harm
and data-by-step interactions without adding tests. Statistical units are
images averaged over the two fixed seeds. This is not full-benchmark or
cross-domain evidence; two seeds do not establish broad training stability.

Run CPU failure-path and report-fixture tests with `python -m unittest
test_scale_cpu -v` from this directory. The synthetic report fixture is test
data, not an experiment result. Actual cloud preflight, protocol, budget,
progress and completion receipts are authoritative; source existence alone
does not establish successful formal training.

Budget amendment authorized by user on 2026-10-05: 48h -> 50h. Matrix and margins unchanged; cumulative ledger explicitly uses frozen protocol cap.
