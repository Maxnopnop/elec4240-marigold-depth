param(
    [string]$Python = 'python',
    [Parameter(Mandatory=$true)][string]$WorkDir
)
$ErrorActionPreference = 'Stop'
$repoDir = $PSScriptRoot
$assetsDir = Join-Path $WorkDir 'assets'
$runDir = Join-Path $WorkDir 'runs'
$resultsDir = Join-Path $repoDir 'results'
$env:HF_HUB_DISABLE_XET = '1'
Push-Location $repoDir
try {
    & $Python download_assets.py --root $assetsDir --asset model
    if ($LASTEXITCODE -ne 0) { throw 'Model download failed' }
    & $Python prepare_subset.py --root $assetsDir --manifest (Join-Path $resultsDir 'split_manifest.json')
    if ($LASTEXITCODE -ne 0) { throw 'Data preparation failed' }
    & $Python download_expert.py --root $assetsDir
    if ($LASTEXITCODE -ne 0) { throw 'Expert download failed' }
    foreach ($methodName in @('base','lora8','lora32','head32','expert')) {
        & $Python experiment.py --assets $assetsDir --work $runDir --results $resultsDir --method $methodName
        if ($LASTEXITCODE -ne 0) { throw "Experiment failed: $methodName" }
    }
    & $Python experiment.py --assets $assetsDir --work $runDir --results $resultsDir --method base --denoise-steps 1
    if ($LASTEXITCODE -ne 0) { throw 'One-step experiment failed' }
    & $Python verify_results.py --assets $assetsDir --work $runDir --results $resultsDir
    if ($LASTEXITCODE -ne 0) { throw 'Result verification failed' }
    & $Python summarize.py --assets $assetsDir --work $runDir --results $resultsDir
    if ($LASTEXITCODE -ne 0) { throw 'Summary failed' }
} finally { Pop-Location }
