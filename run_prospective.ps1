param(
    [string]$Python = 'E:\Codex\2026-09-27\yo\work\marigold-local\.venv\Scripts\python.exe',
    [string]$Assets = 'E:\Codex\2026-09-27\yo\work\marigold-local\assets',
    [string]$PriorWork = 'E:\Codex\2026-09-27\yo\work\marigold-local\robustness_v3',
    [string]$Data = 'E:\Codex\2026-09-27\yo\work\marigold-local\assets\external_sun3d_v4',
    [string]$Work = 'E:\Codex\2026-09-27\yo\work\marigold-local\prospective_v4',
    [string]$Results = 'results/prospective_v4',
    [switch]$AuditOnly
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONUNBUFFERED = '1'
if (-not (Test-Path -LiteralPath (Join-Path $Results 'protocol.json'))) {
    throw 'Freeze the audited data manifest and protocol before running external inference.'
}
$evaluationArgs = @('run_prospective.py', '--assets', $Assets, '--prior-work', $PriorWork,
    '--data', $Data, '--work', $Work, '--results', $Results)
if ($AuditOnly) { $evaluationArgs += '--audit-only' }
& $Python @evaluationArgs
if ($LASTEXITCODE -ne 0) { throw 'External evaluation/audit failed. Preserve frozen artifacts and inspect the technical error.' }
& $Python summarize_prospective.py --results $Results --data $Data --work $Work
if ($LASTEXITCODE -ne 0) { throw 'External report generation failed' }
