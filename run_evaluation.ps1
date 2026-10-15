# Run from any directory. The private virtual environment and dataset cache
# stay in this checkout; benchmark artifacts stay under reports/benchmark_runs.
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$CliArgs)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$venv = Join-Path $PSScriptRoot '.venv-eval'
$python = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    # The Windows Python launcher may know only a newer Python even when a
    # compatible Conda environment is already installed. Inspect both sources.
    $candidates = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($line in (& py -0p 2>$null)) {
            if ($line -match '-V:3\.(10|11|12)\s+(.+)$') {
                $candidates += $Matches[2].Trim()
            }
        }
    }
    if (Get-Command conda -ErrorAction SilentlyContinue) {
        try {
            $condaInfo = (& conda env list --json | ConvertFrom-Json)
            foreach ($prefix in $condaInfo.envs) {
                $candidates += (Join-Path $prefix 'python.exe')
            }
        } catch { Write-Warning "Could not inspect Conda environments: $_" }
    }
    $systemPython = Get-Command python -ErrorAction SilentlyContinue
    if ($systemPython) { $candidates += $systemPython.Source }
    $selected = $null
    foreach ($minor in 10, 11, 12) {
        foreach ($candidate in ($candidates | Select-Object -Unique)) {
            if (-not (Test-Path -LiteralPath $candidate)) { continue }
            $version = & $candidate -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>$null
            if ($LASTEXITCODE -eq 0 -and $version -eq "3.$minor") {
                $selected = $candidate
                break
            }
        }
        if ($selected) { break }
    }
    if (-not $selected) { throw 'Install Python 3.10-3.12, then retry.' }
    Write-Host "Creating evaluation environment with $selected"
    & $selected -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python virtual environment. Install Python 3.10 and retry.' }
}
$requirements = Join-Path $PSScriptRoot 'requirements-eval.txt'
$marker = Join-Path $venv 'requirements-eval.sha256'
$hash = (Get-FileHash -LiteralPath $requirements -Algorithm SHA256).Hash
if (-not (Test-Path -LiteralPath $marker) -or (Get-Content -LiteralPath $marker -Raw).Trim() -ne $hash) {
    & $python -m pip install --quiet --disable-pip-version-check -r $requirements
    if ($LASTEXITCODE -ne 0) { throw 'Evaluation dependency installation failed.' }
    Set-Content -LiteralPath $marker -Value $hash
}
if (-not $CliArgs -or $CliArgs.Count -eq 0) {
    $CliArgs = @('run', '--suites', 'integrity', '--profile', 'smoke')
}
if ($CliArgs[0] -eq 'dashboard') {
    $dashboardArgs = @()
    if ($CliArgs.Count -gt 1) { $dashboardArgs = $CliArgs[1..($CliArgs.Count - 1)] }
    & $python -m benchmarks.evaluation_dashboard @dashboardArgs
    exit $LASTEXITCODE
}
& $python -m benchmarks.run @CliArgs
exit $LASTEXITCODE
