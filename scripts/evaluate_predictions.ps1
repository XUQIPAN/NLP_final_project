param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string[]]$Predictions,

    [string]$OutputDirectory = "results"
)

$ErrorActionPreference = "Stop"
$PackageRoot = Split-Path -Parent $PSScriptRoot
Push-Location $PackageRoot
try {
    python benchmark_eval.py evaluate `
        --labels data/updated_list_identity.txt `
        --predictions $Predictions `
        --output-dir $OutputDirectory
}
finally {
    Pop-Location
}
