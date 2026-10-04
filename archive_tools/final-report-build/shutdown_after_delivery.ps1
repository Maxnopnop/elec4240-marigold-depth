$ErrorActionPreference = 'Stop'
$recordPath = 'E:\Codex\2026-09-27\yo\outputs\elec4240-final-report\completion.json'
$logPath = Join-Path $PSScriptRoot 'shutdown.status.txt'
Start-Sleep -Seconds 60
try {
    $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
    if (-not $record.all_deliverables_complete) { throw 'Delivery completion gate is not satisfied.' }
    if (-not $record.shutdown.authorized) { throw 'Shutdown authorization is missing.' }
    $record.shutdown.status = 'shutdown_requested'
    $record.shutdown | Add-Member -NotePropertyName requested_utc -NotePropertyValue ([DateTime]::UtcNow.ToString('o')) -Force
    $record | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $recordPath -Encoding UTF8
    Add-Content -LiteralPath $logPath -Value ('Normal shutdown requested at ' + (Get-Date -Format o))
    # A zero-second timeout without /f preserves normal application shutdown handling.
    & "$env:SystemRoot\System32\shutdown.exe" /s /t 0
    if ($LASTEXITCODE -ne 0) { throw ('shutdown.exe exit code ' + $LASTEXITCODE) }
} catch {
    Add-Content -LiteralPath $logPath -Value ('Shutdown failed: ' + $_.Exception.Message)
    throw
}
