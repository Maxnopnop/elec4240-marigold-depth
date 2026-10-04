param([string]$InputDocx, [string]$OutputPdf)
$ErrorActionPreference = 'Stop'
$wordApp = $null
$proposalDoc = $null
$createdEmptyInstance = $false
try {
    $wordApp = New-Object -ComObject Word.Application
    $createdEmptyInstance = $wordApp.Documents.Count -eq 0
    if ($createdEmptyInstance) {
        $wordApp.Visible = $false
        $wordApp.DisplayAlerts = 0
    }
    $proposalDoc = $wordApp.Documents.Open($InputDocx, $false, $true, $false)
    $proposalDoc.Repaginate()
    $proposalDoc.ExportAsFixedFormat($OutputPdf, 17)
    Write-Output ('Rendered pages: ' + $proposalDoc.ComputeStatistics(2))
} finally {
    if ($null -ne $proposalDoc) {
        $proposalDoc.Close(0)
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($proposalDoc)
    }
    if ($null -ne $wordApp) {
        if ($createdEmptyInstance) { $wordApp.Quit() }
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($wordApp)
    }
}
