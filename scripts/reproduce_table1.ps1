$ErrorActionPreference = "Stop"

$PackageRoot = Split-Path -Parent $PSScriptRoot
Push-Location $PackageRoot
try {
    python benchmark_eval.py summary `
        --labels data/updated_list_identity.txt `
        --output-dir results
}
finally {
    Pop-Location
}
