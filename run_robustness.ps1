param(
    [string]$Python = 'E:\Codex\2026-09-27\yo\work\marigold-local\.venv\Scripts\python.exe',
    [string]$Assets = 'E:\Codex\2026-09-27\yo\work\marigold-local\assets',
    [string]$Work = 'E:\Codex\2026-09-27\yo\work\marigold-local\robustness_v3'
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONUNBUFFERED = '1'
& $Python run_robustness.py --assets $Assets --work $Work --stage validation
if ($LASTEXITCODE -ne 0) { throw 'Validation/training failed' }
& $Python verify_robustness.py --assets $Assets --work $Work --stage validation
if ($LASTEXITCODE -ne 0) { throw 'Validation gate failed' }
& $Python run_robustness.py --assets $Assets --work $Work --stage test
if ($LASTEXITCODE -ne 0) { throw 'Test inference failed' }
& $Python run_robustness.py --assets $Assets --work $Work --stage corruption
if ($LASTEXITCODE -ne 0) { throw 'Corruption inference failed' }
& $Python verify_robustness.py --assets $Assets --work $Work --stage corruption
if ($LASTEXITCODE -ne 0) { throw 'Final audit failed' }
& $Python summarize_robustness.py --assets $Assets --work $Work --stage corruption
if ($LASTEXITCODE -ne 0) { throw 'Summary failed' }
