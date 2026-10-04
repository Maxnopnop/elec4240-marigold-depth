$ErrorActionPreference = 'Stop'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class FinalReportPower {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint state);
}
'@
$stopPath = Join-Path $PSScriptRoot 'keepawake.stop'
$recordPath = Join-Path $PSScriptRoot 'keepawake.status.txt'
try {
    $previousState = [FinalReportPower]::SetThreadExecutionState([uint32]2147483649)
    if ($previousState -eq 0) { throw 'Could not request temporary system keep-awake.' }
    Set-Content -LiteralPath $recordPath -Value ('active PID=' + $PID + ' started=' + (Get-Date -Format o))
    while (-not (Test-Path -LiteralPath $stopPath)) { Start-Sleep -Seconds 15 }
} finally {
    [void][FinalReportPower]::SetThreadExecutionState([uint32]2147483648)
    Add-Content -LiteralPath $recordPath -Value ('released=' + (Get-Date -Format o))
}
