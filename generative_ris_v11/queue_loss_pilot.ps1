$ErrorActionPreference = 'Stop'
$taskRoot = 'E:\Codex\2026-09-27\yo\outputs\marigold-depth'
$taskPython = 'E:\Codex\2026-09-27\yo\work\marigold-local\.venv\Scripts\python.exe'
$taskOutput = Join-Path $taskRoot 'results\generative_ris_v11\training\loss_pilot'
try {
    $baselineProcess = Get-Process -Id 43240 -ErrorAction SilentlyContinue
    if ($baselineProcess) {
        if ($baselineProcess.StartTime.ToString('yyyy-MM-dd HH:mm:ss') -ne '2026-10-03 20:54:29') {
            throw 'Baseline process identity changed; refusing to wait on unrelated process.'
        }
        @{stage='waiting_for_baseline'; baseline_pid=43240; queue_pid=$PID} | ConvertTo-Json | Set-Content (Join-Path $taskOutput 'queue_status.json')
        $baselineProcess | Wait-Process
    }
    Set-Location -LiteralPath $taskRoot
    @{stage='starting_pilot'; queue_pid=$PID} | ConvertTo-Json | Set-Content (Join-Path $taskOutput 'queue_status.json')
    & $taskPython -m generative_ris_v11.loss_pilot
    if ($LASTEXITCODE -ne 0) { throw "Pilot failed with exit code $LASTEXITCODE" }
    @{stage='complete'; queue_pid=$PID; shutdown=$false} | ConvertTo-Json | Set-Content (Join-Path $taskOutput 'queue_status.json')
} catch {
    @{stage='error'; detail=$_.Exception.Message; shutdown=$false} | ConvertTo-Json | Set-Content (Join-Path $taskOutput 'queue_status.json')
    throw
}
