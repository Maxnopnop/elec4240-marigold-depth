param(
    [string]$Python = 'python',
    [Parameter(Mandatory=$true)][string]$WorkDir
)
$ErrorActionPreference = 'Stop'
$repoDir = $PSScriptRoot
$assetsDir = Join-Path $WorkDir 'assets'
$runDir = Join-Path $WorkDir 'expanded_v1'
$resultDir = Join-Path $repoDir 'results\expanded_v1'
$env:HF_HUB_DISABLE_XET = '1'
Push-Location $repoDir
try {
    & $Python download_assets.py --root $assetsDir --asset model
    if ($LASTEXITCODE -ne 0) { throw 'Model download failed' }
    & $Python download_expert.py --root $assetsDir
    if ($LASTEXITCODE -ne 0) { throw 'Reference model download failed' }
    & $Python prepare_expanded.py --assets $assetsDir --pilot (Join-Path $repoDir 'results\split_manifest.json') --results $resultDir
    if ($LASTEXITCODE -ne 0) { throw 'Expanded data preparation failed' }
    & $Python -m unittest test_protocol test_expanded -v
    if ($LASTEXITCODE -ne 0) { throw 'Protocol tests failed' }
    & $Python run_expanded.py --assets $assetsDir --work $runDir --results $resultDir
    if ($LASTEXITCODE -ne 0) { throw 'Expanded experiments failed; matching completed runs can be resumed' }
    & $Python verify_expanded.py --assets $assetsDir --work $runDir --results $resultDir --check-restoration
    if ($LASTEXITCODE -ne 0) { throw 'Expanded result verification failed' }
    & $Python summarize_expanded.py --assets $assetsDir --work $runDir --results $resultDir
    if ($LASTEXITCODE -ne 0) { throw 'Expanded summary failed' }
} finally { Pop-Location }
