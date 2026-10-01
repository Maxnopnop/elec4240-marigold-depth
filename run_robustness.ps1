param(
    [string]$Python = 'E:\Codex\2026-09-27\yo\work\marigold-local\.venv\Scripts\python.exe',
    [string]$Assets = 'E:\Codex\2026-09-27\yo\work\marigold-local\assets',
    [string]$Work = 'E:\Codex\2026-09-27\yo\work\marigold-local\robustness_v3',
    [string]$Results = 'results/robustness_v3',
    [switch]$PrepareOnly
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONUNBUFFERED = '1'
if (Test-Path -LiteralPath (Join-Path $Work 'checkpoints')) {
    foreach ($checkpoint in Get-ChildItem -LiteralPath (Join-Path $Work 'checkpoints') -Filter adapter.pt -Recurse -File) {
        $record = Join-Path $Results ('runs/' + $checkpoint.Directory.Name + '/training.json')
        if (-not (Test-Path -LiteralPath $record)) { throw 'Existing checkpoints require matching results. Choose a new Work directory for a fresh reproduction.' }
        $stats = Get-Content -LiteralPath $record -Raw | ConvertFrom-Json
        if ((Get-FileHash -LiteralPath $checkpoint.FullName -Algorithm SHA256).Hash -ine $stats.checkpoint_sha256) { throw 'Existing checkpoint does not match its training record.' }
    }
}
& $Python prepare_robustness.py --assets $Assets --census (Join-Path $PSScriptRoot 'results/robustness_v3/scene_census.json') --results $Results
if ($LASTEXITCODE -ne 0) { throw 'Data preparation failed' }
$addendumSource = Join-Path $PSScriptRoot 'results/robustness_v3/statistical_addendum.json'
$addendumTarget = Join-Path $Results 'statistical_addendum.json'
if (Test-Path -LiteralPath $addendumTarget) {
    if ((Get-FileHash -LiteralPath $addendumSource).Hash -ne (Get-FileHash -LiteralPath $addendumTarget).Hash) { throw 'Statistical addendum does not match the frozen source.' }
} else {
    Copy-Item -LiteralPath $addendumSource -Destination $addendumTarget
}
if ($PrepareOnly) { Write-Output 'ROBUSTNESS_PREPARATION_VERIFIED'; return }
& $Python run_robustness.py --assets $Assets --work $Work --results $Results --stage validation
if ($LASTEXITCODE -ne 0) { throw 'Validation/training failed' }
& $Python verify_robustness.py --assets $Assets --work $Work --results $Results --stage validation
if ($LASTEXITCODE -ne 0) { throw 'Validation gate failed' }
& $Python run_robustness.py --assets $Assets --work $Work --results $Results --stage test
if ($LASTEXITCODE -ne 0) { throw 'Test inference failed' }
& $Python run_robustness.py --assets $Assets --work $Work --results $Results --stage corruption
if ($LASTEXITCODE -ne 0) { throw 'Corruption inference failed' }
& $Python verify_robustness.py --assets $Assets --work $Work --results $Results --stage corruption
if ($LASTEXITCODE -ne 0) { throw 'Final audit failed' }
& $Python summarize_robustness.py --assets $Assets --work $Work --results $Results --stage corruption
if ($LASTEXITCODE -ne 0) { throw 'Summary failed' }
