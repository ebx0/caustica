<#
.SYNOPSIS
Run this machine's GPU evidence in one command: the gpu-marked tests, then the
on-device gate suite, both landing in one stamped folder under
benchmarks/reports/gpu_gates/<device>-<date>/.

.DESCRIPTION
The PowerShell twin of scripts/run_gpu_local.sh. The two take the same
decisions and write the same files, so a Windows and a Linux box produce
comparable folders.

Exit code: 0 both legs green; 1 a gpu-marked test failed; 64 an empty -Out;
69 no in-tree interpreter; otherwise the gate suite's own code (2 no usable
GPU, 4 a gate failed or is incomplete). An unknown parameter, or -Out with
nothing after it, is refused by PowerShell's own parameter binding before this
script runs, with its own non-zero code; the shell twin answers both of those
with 64.

.PARAMETER Quick
Small ladder rungs (targets 1 and 2 GiB) instead of the ones this device's
free VRAM would pick. Every gate the device can close is still measured; only
the rungs shrink. The time gate is the one it weakens, see the note beside the
targets below.

.PARAMETER Out
Report root. Default: benchmarks/reports/gpu_gates.

.PARAMETER SkipPytest
Gate suite only.

.PARAMETER SkipGates
Gpu-marked tests only.

.EXAMPLE
.\scripts\run_gpu_local.ps1 -Quick
#>
[CmdletBinding()]
param(
    [switch]$Quick,
    [string]$Out = "benchmarks/reports/gpu_gates",
    [switch]$SkipPytest,
    [switch]$SkipGates
)

# The two values the twins must agree on, one assignment each so the test that
# compares them reads a value instead of searching the file for a phrase. The
# third, the default report root, is the -Out default above.
$marker = 'gpu and not kwave and not network'
# 1 and 2 GiB, not smaller: a rung whose solve is under about two seconds is
# dominated by its own warmup, and the time gate then grades the start-up cost
# rather than the model. Measured on an RTX 5050 laptop with the card to
# itself: a 0.5 GiB rung (162^3) solves in 1.60 s and misses its plan by
# -31.2% against a +/-25% tolerance, while the preset's own rungs stayed
# inside it in all six quick runs of this task, the 1 GiB rung (216^3) at
# -5.7 to -13.6% and the 2 GiB rung (270^3) at +4.6 to +18.1%. These are
# absolute VRAM targets, so they are sized for a laptop-class card; on a
# faster device the same two rungs fall back into the warmup-dominated regime
# and only the other four gates mean anything.
$quickTargets = '1.0,2.0'

# A native command's stderr becomes ErrorRecords when it is merged into the
# pipeline; with 'Stop' that would end the run on the first warning pytest
# prints, which is not a failure.
$ErrorActionPreference = 'Continue'

# Not [ValidateNotNullOrEmpty()]: that fails in PowerShell's parameter binder,
# which exits 1 and cannot be given a code of its own, and the shell twin
# answers an empty --out with 64.
if ([string]::IsNullOrWhiteSpace($Out)) {
    Write-Host "run_gpu_local: -Out needs a directory"
    exit 64
}

$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    # caustica is installed in the in-tree .venv and in no other interpreter: a
    # bare python collects dozens of ModuleNotFoundError and reads as a broken
    # tree. Spell the interpreter out rather than trusting the active shell.
    $py = Join-Path $root '.venv\Scripts\python.exe'
    if (-not (Test-Path $py)) { $py = Join-Path $root '.venv/bin/python' }
    if (-not (Test-Path $py)) {
        # 69 (EX_UNAVAILABLE), not 2: 2 is the gate suite's "no usable GPU", and
        # a CI step cannot tell a missing card from a missing environment if
        # they share a code.
        Write-Host "run_gpu_local: no in-tree interpreter under $root\.venv"
        Write-Host "  fix: python -m venv .venv ; .venv\Scripts\pip install -e '.[dev,gpu]'"
        exit 69
    }

    # Windows PowerShell's redirection operators and -Encoding utf8 write UTF-16
    # or a BOM, and CRLF; the shell twin writes UTF-8 with LF. Two machines'
    # folders have to be comparable byte for byte, so every file this script
    # leaves goes through here.
    function Write-Utf8Lines([string]$path, [string[]]$lines) {
        $text = ''
        if ($lines.Count -gt 0) { $text = (($lines -replace "`r", '') -join "`n") + "`n" }
        [System.IO.File]::WriteAllText(
            $path, $text, (New-Object System.Text.UTF8Encoding($false)))
    }

    # Run the in-tree interpreter, show its output live, keep every line, and
    # return its own exit code.
    #
    # Two Windows PowerShell traps are handled here. Merging a native command's
    # stderr with 2>&1 wraps each line in an ErrorRecord, which Tee-Object would
    # spell out as six lines of "NativeCommandError" formatting inside the log,
    # so the records are unwrapped to their message first. And Tee-Object writes
    # the file itself, in UTF-16, so the lines are collected here and written by
    # Write-Utf8Lines instead. What is NOT done is piping into tail, head or
    # Select-Object: a pipeline reports the last command's status, so a
    # collection error would read as a pass.
    function Invoke-Logged([string[]]$argv, [string]$logPath) {
        $collected = New-Object System.Collections.Generic.List[string]
        & $py @argv 2>&1 | ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) {
                $line = [string]$_.Exception.Message
            }
            else {
                $line = [string]$_
            }
            $collected.Add($line)
            Write-Host $line
        }
        $code = $LASTEXITCODE
        Write-Utf8Lines $logPath $collected.ToArray()
        return $code
    }

    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("run_gpu_local-" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tmp | Out-Null
    $pytestLog = Join-Path $tmp 'pytest-gpu.txt'
    $gatesLog = Join-Path $tmp 'gpu-gates.log'
    $pytestCode = $null
    $gatesCode = $null

    $ver = & $py -c "import sys; print(sys.version.split()[0])"
    Write-Host "run_gpu_local: $ver at $py"
    Write-Host "run_gpu_local: report root $Out"

    if (-not $SkipPytest) {
        Write-Host ""
        Write-Host "=== pytest -m '$marker' ==="
        # -u because CPython block-buffers a stdout that is not a terminal, and
        # a ladder that prints nothing for minutes looks hung.
        $pytestCode = Invoke-Logged @('-u', '-m', 'pytest', '-m', $marker) $pytestLog
        Write-Host "pytest exit: $pytestCode"
    }

    if (-not $SkipGates) {
        Write-Host ""
        Write-Host "=== python -m caustica.validation gpu-gates ==="
        $gateArgs = @('-u', '-m', 'caustica.validation', 'gpu-gates', '--out', $Out)
        if ($Quick) { $gateArgs += @('--targets', $quickTargets) }
        $gatesCode = Invoke-Logged $gateArgs $gatesLog
        Write-Host "gpu-gates exit: $gatesCode"
    }

    # The suite names its own folder (<device slug>-<UTC stamp>); read it back
    # rather than recomputing the slug and risking a second, empty folder.
    $folder = $null
    if (Test-Path $gatesLog) {
        $line = Select-String -Path $gatesLog -Pattern '^report folder: ' | Select-Object -Last 1
        if ($line) { $folder = $line.Line -replace '^report folder: ', '' }
    }
    $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss')
    if (-not $folder) {
        # Either the suite was skipped, or it stopped before it named a folder
        # (no usable GPU is the ordinary case). Either way the run still leaves
        # its evidence somewhere, under a name that says which case it was.
        if ($SkipGates) { $folder = Join-Path $Out "tests-only-$stamp" }
        else { $folder = Join-Path $Out "no-gpu-$stamp" }
    }
    New-Item -ItemType Directory -Path $folder -Force | Out-Null

    function Get-LastLine([string]$path) {
        if (-not (Test-Path $path)) { return '' }
        $lines = @(Get-Content $path | Where-Object { $_.Trim() -ne '' })
        if ($lines.Count -eq 0) { return '' }
        return $lines[-1]
    }
    function Get-Match([string]$path, [string]$pattern) {
        if (-not (Test-Path $path)) { return '' }
        $m = Select-String -Path $path -Pattern $pattern | Select-Object -Last 1
        if ($m) { return $m.Line }
        return ''
    }

    $pytestLine = Get-LastLine $pytestLog
    $verdict = Get-Match $gatesLog '^overall: '
    $device = (Get-Match $gatesLog '^  gpu_name: ') -replace '^  gpu_name: ', ''
    if (-not $device) { $device = 'unknown' }

    if (Test-Path $pytestLog) { Copy-Item $pytestLog (Join-Path $folder 'pytest-gpu.txt') -Force }
    if (Test-Path $gatesLog) { Copy-Item $gatesLog (Join-Path $folder 'gpu-gates.log') -Force }

    if ($Quick) { $mode = 'quick' } else { $mode = 'full' }
    $now = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    $lines = @(
        '# Local GPU run',
        '',
        "Written by ``scripts/run_gpu_local.ps1`` on $now (UTC).",
        '',
        '| | |',
        '|---|---|',
        "| device | $device |",
        "| mode | $mode |",
        "| interpreter | $py |",
        '',
        '| leg | command | exit | summary |',
        '|---|---|---|---|'
    )
    if ($null -ne $pytestCode) {
        $lines += "| gpu tests | ``python -m pytest -m ""$marker""`` | $pytestCode | $pytestLine |"
    }
    else {
        $lines += '| gpu tests | skipped (`-SkipPytest`) | | |'
    }
    if ($null -ne $gatesCode) {
        $suffix = ''
        if ($Quick) { $suffix = " --targets $quickTargets" }
        $lines += "| gate suite | ``python -m caustica.validation gpu-gates$suffix`` | $gatesCode | $verdict |"
    }
    else {
        $lines += '| gate suite | skipped (`-SkipGates`) | | |'
    }
    if (Test-Path $gatesLog) {
        $verdicts = @(Select-String -Path $gatesLog -Pattern '^  (PASS|FAIL|SKIP|INCOMPLETE) ' |
            ForEach-Object { '- ' + $_.Line.Trim() })
        $lines += @('', '## Gates', '') + $verdicts + @(
            '',
            '`fullsize` needs a 512-cubed run at dx = 0.30 mm, 13.2 GiB predicted, so',
            'the ladder only reaches it on a device with about 15 GiB free. On a',
            'smaller card that gate stays INCOMPLETE and the suite exits 4 however',
            'green the rest is; the criterion closes on a hosted A100 or H100.',
            '',
            '`time` is the least reproducible of the five gates: it grades a plan',
            'built from a calibration probe taken seconds earlier, and on a laptop',
            'card the probe moves with the clocks. One `time` FAIL is a reason to',
            'run again, not a defect on its own.'
        )
    }
    $lines += @(
        '',
        'Full output: `pytest-gpu.txt`, `gpu-gates.log`. The suite''s own',
        'report is `REPORT.md` beside them, with `gpu_gates.json` as its',
        'machine half.'
    )
    $summaryPath = Join-Path (Resolve-Path $folder) 'SUMMARY.md'
    Write-Utf8Lines $summaryPath $lines

    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue

    Write-Host ""
    Write-Host "run_gpu_local: summary in $folder\SUMMARY.md"

    if (($null -ne $pytestCode) -and ($pytestCode -ne 0)) { exit 1 }
    if ($null -ne $gatesCode) { exit $gatesCode }
    exit 0
}
finally {
    Pop-Location
}
