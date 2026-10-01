# Reproducing the final report

The supplement contains project-authored code, protocols, selected manifests and
summary results. It does not include a Python environment, framework source trees,
datasets, checkpoints, or raw predictions. The private Git repository retains all
per-image numerical records. Its full result tree is larger than the course's
supplementary-code limit, so the ZIP is a deliberately smaller source package.

## Environment and assets

Use the recorded versions in `results/environment.json`; install a CUDA PyTorch
build compatible with your GPU, then `requirements.txt`. The measured machine was
an RTX 5060 Laptop GPU with approximately 8 GB VRAM. The pinned model revisions and
dataset URLs are in `REFERENCES.md`. `CORRECTIONS.md` corrects the specialist's
training-data provenance. No access credentials are included or required by code.

Read `README.md` and run the preparation/pilot/expanded/scale-up stages in order to
recreate their input manifests. The replication wrapper accepts explicit paths:

```powershell
.\run_robustness.ps1 -Python 'E:\elec4240\.venv\Scripts\python.exe' -Assets 'E:\elec4240\assets' -Work 'E:\elec4240\robustness_v3' -Results 'E:\elec4240\replication-results' -PrepareOnly
```

Remove `-PrepareOnly` for the complete replication. Do not point a fresh run at
published completion markers without their matching excluded raw files. Exact
checkpoint restoration is intentional: retraining on another GPU or software
version can change weights, so a fresh reproduction needs its own provenance and
may not satisfy historical bitwise checks. Keep the historical study unchanged.

The external acquisition uses `prepare_prospective.py` and the pinned transport
recovery in `resume_prospective_download.py`. The original selection/QC rules and
archive-member checks are documented in `PROSPECTIVE_PROTOCOL.md`. Acquisition
requires network access and substantial additional storage. The mirror does not
change which groups or frames are selected. A frozen external replay also requires
all v3 checkpoints, validation references, calibration records and raw arrays.

## Final extension with retained upstream artifacts

`final_extension_runner.py` redirects only storage/result paths; it does not edit
the frozen experiment sources. Run it from the repository directory. The storage
root must contain `assets/`, `robustness_v3/`, and `prospective_v4/`. Earlier result
records must be present under the repository's `results/` directory. An empty
clone/supplement alone is therefore not an immediately runnable historical replay.

```powershell
$pythonPath = 'E:\elec4240\.venv\Scripts\python.exe'
$storageRoot = 'E:\elec4240'
$newResults = 'E:\elec4240\final-results'
& $pythonPath final_extension_runner.py control --root $storageRoot --out $newResults --check-paths
# Freeze only in a new result directory, before training or timing:
& $pythonPath final_extension_runner.py control --root $storageRoot --out $newResults --freeze
& $pythonPath final_extension_runner.py cost --root $storageRoot --out $newResults --freeze
& $pythonPath final_extension_runner.py control --root $storageRoot --out $newResults
& $pythonPath final_extension_runner.py cost --root $storageRoot --out $newResults
& $pythonPath final_extension_runner.py analysis --root $storageRoot --out $newResults
& $pythonPath final_extension_runner.py failure --root $storageRoot --out $newResults
& $pythonPath final_extension_runner.py summary --root $storageRoot --out $newResults
```

Run GPU stages sequentially; competing workloads invalidate the intended timing
conditions. The final `audit` action additionally requires the full Git history
containing the archived `d14701c` results tree. The source ZIP alone has no Git
database. Recompute saved control metrics with action `control --audit-only`.

## What was frozen, and what was exploratory

- Original external six-comparison family: frozen before its first prediction.
- Fixed-low control: 15 trainings, 7,890 predictions; specified after examining
  the external outcomes, therefore exploratory.
- New eight-comparison family: old six contrast definitions plus mixed versus
  low-only at two resolutions; it does not replace the original family.
- Resolution interaction, strata and selected failures: post-hoc diagnostics.
- Cost measurement: five randomized rounds on eight validation images, three
  warmups, 320 exact prediction-reference matches.
- Equal-time training: one training draw, three seeds per schedule, 120 seconds
  of optimizer time per run, 576 validation predictions; no unseen test claim.

The report distinguishes GT-aligned relative depth from fixed-calibrated or native
metric depth. A favorable aligned score is not a demonstration of absolute-distance
accuracy. Negative corrected tests and unsuccessful ablations are retained.
