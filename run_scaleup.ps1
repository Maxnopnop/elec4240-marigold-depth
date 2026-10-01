param(
    [string]$Python = 'python',
    [Parameter(Mandatory=$true)][string]$WorkDir
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $assetsDir = Join-Path $WorkDir 'assets'
    $runDir = Join-Path $WorkDir 'scaleup_v2'
    & $Python prepare_scaleup.py --assets $assetsDir
    if ($LASTEXITCODE -ne 0) { throw 'Data preparation failed' }
    # Preflight is required before freezing/running the full matrix.
    if (-not (Test-Path 'results/scaleup_v2/preflight.json')) {
        & $Python profile_scaleup.py --assets $assetsDir --work (Join-Path $WorkDir 'scaleup_preflight')
        if ($LASTEXITCODE -ne 0) { throw 'Preflight failed' }
    }
    & $Python run_scaleup.py --assets $assetsDir --work $runDir --stage validation
    if ($LASTEXITCODE -ne 0) { throw 'Training/validation failed; matching runs are resumable' }
    & $Python verify_scaleup.py --assets $assetsDir --work $runDir --stage validation
    if ($LASTEXITCODE -ne 0) { throw 'Validation audit failed; fresh test remains locked' }
    & $Python run_scaleup.py --assets $assetsDir --work $runDir --stage test
    if ($LASTEXITCODE -ne 0) { throw 'Fresh test evaluation failed' }
    & $Python verify_scaleup.py --assets $assetsDir --work $runDir --stage test
    if ($LASTEXITCODE -ne 0) { throw 'Final prediction audit failed' }
    & $Python summarize_scaleup.py --assets $assetsDir --work $runDir --stage test
    if ($LASTEXITCODE -ne 0) { throw 'Summary failed' }
} finally { Pop-Location }
