# Execution of longer single-task training and reliability ablation

This module executes the next stages of the [V8 design](../reliability_v8/EXPERIMENT_PLAN.md). The component study is preserved. Sources and inputs are hashed separately for Stage B and Stage C before their outcomes.

## Stage B

Six runs: depth-only and normal-only, seeds 17/29/43, 1,280 updates per run. Evaluate fixed checkpoints at 320/640/1,280 on the existing 32-scene development set. The engineering gate requires finite completed runs, exact final-checkpoint inference restoration, and mean final error no more than 5% worse than mean step-320 error for each task. This does not establish convergence.

## Stage C

After inspecting the baseline gate, freeze five joint variants for the same three seeds and 1,280 updates: no geometry, uniform geometry at 0.1, weaker uniform at 0.03, scale-sensitive weights at 0.1, and spatially shuffled weights at 0.1. Reuse the Stage B single-task controls only because the training configuration is matched. Final 1,280-step checkpoints determine all comparisons; intermediate baseline observations do not select a best checkpoint or seed.

Primary exploratory family: weighted versus uniform/weaker/shuffled, separately for depth AbsRel and normal mean angle (six paired comparisons). Use 20,000 crossed seed/scene bootstrap samples, two-sided centered tests and Holm correction. Report unadjusted symmetric 95% intervals explicitly. Other comparisons are descriptive. Failure to reject a difference is not a no-harm or equivalence result.

## Run

Use the existing local CUDA Python environment from the repository root:

```powershell
$python = 'E:\Codex\2026-09-27\yo\work\marigold-local\.venv\Scripts\python.exe'
& $python -m training_v9.test_resume
& $python -m training_v9.protocol freeze_b
& $python -m training_v9.run --stage b
& $python -m training_v9.analyze --stage b
& $python -m training_v9.test_analysis
& $python -m training_v9.protocol freeze_c
& $python -m training_v9.run --stage c
& $python -m training_v9.analyze --stage c
```

Every training run saves adapter weights, AdamW state, local sampler state and global RNG state every 160 updates. Interrupted checkpoint evaluations are rerun from the matching saved state. Completion markers and resumed states must pass hashes; do not remove them to force an overwrite. GPU training is sequential, with a process-scoped Windows sleep-prevention request released in `finally`. No shutdown is scheduled.

## Finite background pipeline

`python -m training_v9.pipeline` executes the full sequence from Stage B through its independent audit/report, checks the baseline gate, freezes Stage C, runs all joint variants, and audits/reports the final results. A running instance holds a lock under the local work directory. Do not start a duplicate while a pipeline is active.

For this execution, the pipeline was attached to the already-running Stage B process using `--wait-baseline-pid`; that PID is recorded in `results/reliability_training_v9/pipeline_status.json`. The attachment avoids duplicating training. `status.json` records the current optimizer progress every 80 updates, and `pipeline_status.json` records the overall stage, completion or failure. Standard output/error logs are retained under `work/marigold-local/reliability_training_v9`.

The pipeline stops on a failed baseline gate or an execution/integrity error and records the reason. It does not change hyperparameters, retry a failed experiment under a modified protocol, publish to GitHub, spend cloud credits or schedule a shutdown. It is a single finite background job. The computer must remain powered on for execution to continue.

Raw checkpoints/predictions remain in `work/marigold-local/reliability_training_v9`. Source metadata and scores are saved to `results/reliability_training_v9`. A Git clone alone does not contain the required V7 model/data/cache or V8 weight maps.

## Reproducibility boundary

The resume serialization test establishes exact parameters, losses, sampling and RNG on its small CUDA test model. It does not prove bitwise deterministic diffusion-model training. A diagnostic of the first new depth run found identical V7 input order and first forward loss, but a small first-backward gradient difference and different subsequent predictions. The cause has not been isolated; mixed-precision backward execution or accumulation differences are plausible. Therefore, old V7 scores are historical context, not matched controls for the new study. All seven current methods are run under the same new execution configuration. Checkpoint inference restoration is tested separately and exactly.

The new study remains development evidence: one training draw, three seeds, previously observed scenes and depth-derived normal targets. No absolute-distance guarantee, independent normal-ground-truth claim, global model superiority or fresh-data confirmation follows from it.
