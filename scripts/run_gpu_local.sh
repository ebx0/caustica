#!/usr/bin/env bash
# Run this machine's GPU evidence in one command: the gpu-marked tests, then
# the on-device gate suite, both landing in one stamped folder under
# benchmarks/reports/gpu_gates/<device>-<date>/.
#
# Usage:
#   bash scripts/run_gpu_local.sh [--quick] [--out DIR] [--skip-pytest] [--skip-gates]
#
#   --quick        small ladder rungs (targets 1 and 2 GiB) instead of the
#                  ones this device's free VRAM would pick. Every gate the
#                  device can close is still measured; only the rungs shrink.
#                  The time gate is the one it weakens, see the note beside
#                  the targets below.
#   --out DIR      report root (default: benchmarks/reports/gpu_gates)
#   --skip-pytest  gate suite only
#   --skip-gates   gpu-marked tests only
#
# Invoke it through `bash` as above. The file is committed without an
# executable bit, because this repository is developed on Windows where git
# records none, so `./scripts/run_gpu_local.sh` is a permission error on Linux.
#
# Exit code: 0 both legs green; 1 a gpu-marked test failed; 64 a bad option;
# 69 no in-tree interpreter; otherwise the gate suite's own code (2 no usable
# GPU, 4 a gate failed or is incomplete).
#
# The PowerShell twin is scripts/run_gpu_local.ps1; the two take the same
# decisions and write the same files, so a Windows and a Linux box produce
# comparable folders.

set -u

# The three values the twins must agree on, one assignment each so the test
# that compares them reads a value instead of searching the file for a phrase.
marker="gpu and not kwave and not network"
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
quick_targets="1.0,2.0"
default_out="benchmarks/reports/gpu_gates"

quick=0
out=""
run_pytest=1
run_gates=1

while [ $# -gt 0 ]; do
  case "$1" in
    --quick) quick=1 ;;
    --out)
      shift
      # Without this guard a trailing `--out` with nothing after it falls back
      # to the default root and writes a whole ladder into the repository.
      if [ $# -eq 0 ] || [ -z "$1" ]; then
        echo "run_gpu_local: --out needs a directory" >&2
        exit 64
      fi
      out="$1"
      ;;
    --skip-pytest) run_pytest=0 ;;
    --skip-gates) run_gates=0 ;;
    # The header comment IS the help text, so the two cannot fall out of step:
    # print from line 2 for as long as the lines are comments.
    -h|--help) sed -n '2,${/^#/!q;s/^# \{0,1\}//;p;}' "$0"; exit 0 ;;
    *) echo "run_gpu_local: unknown option $1 (try --help)" >&2; exit 64 ;;
  esac
  shift
done

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/.." && pwd)"
cd "$root" || exit 70

# caustica is installed in the in-tree .venv and in no other interpreter: a
# bare python collects dozens of ModuleNotFoundError and reads as a broken
# tree. Spell the interpreter out rather than trusting the active shell.
py="$root/.venv/Scripts/python.exe"
if [ ! -x "$py" ]; then py="$root/.venv/bin/python"; fi
if [ ! -x "$py" ]; then
  # 69 (EX_UNAVAILABLE), not 2: 2 is the gate suite's "no usable GPU", and a CI
  # step cannot tell a missing card from a missing environment if they share a
  # code.
  echo "run_gpu_local: no in-tree interpreter under $root/.venv" >&2
  echo "  fix: python -m venv .venv && .venv/bin/pip install -e '.[dev,gpu]'" >&2
  exit 69
fi

root_out="${out:-$default_out}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

pytest_log="$tmp/pytest-gpu.txt"
gates_log="$tmp/gpu-gates.log"
pytest_code=""
gates_code=""

echo "run_gpu_local: $("$py" -c 'import sys; print(sys.version.split()[0])') at $py"
echo "run_gpu_local: report root $root_out"

if [ "$run_pytest" -eq 1 ]; then
  echo ""
  echo "=== pytest -m '$marker' ==="
  # tee, never a pipe into tail or head: the shell reports the LAST command's
  # status, so a collection error would read as a pass. PIPESTATUS is what
  # keeps pytest's own code. -u because CPython block-buffers a stdout that is
  # not a terminal, and a ladder that prints nothing for minutes looks hung.
  "$py" -u -m pytest -m "$marker" 2>&1 | tee "$pytest_log"
  pytest_code="${PIPESTATUS[0]}"
  echo "pytest exit: $pytest_code"
fi

if [ "$run_gates" -eq 1 ]; then
  echo ""
  echo "=== python -m caustica.validation gpu-gates ==="
  gate_args=(--out "$root_out")
  if [ "$quick" -eq 1 ]; then gate_args+=(--targets "$quick_targets"); fi
  "$py" -u -m caustica.validation gpu-gates "${gate_args[@]}" 2>&1 | tee "$gates_log"
  gates_code="${PIPESTATUS[0]}"
  echo "gpu-gates exit: $gates_code"
fi

# The suite names its own folder (<device slug>-<UTC stamp>); read it back
# rather than recomputing the slug and risking a second, empty folder.
folder=""
if [ -f "$gates_log" ]; then
  folder="$(tr -d '\r' < "$gates_log" | grep '^report folder: ' | tail -1 | sed 's/^report folder: //' | tr '\\' '/')"
fi
if [ -z "$folder" ]; then
  # Either the suite was skipped, or it stopped before it named a folder (no
  # usable GPU is the ordinary case). Either way the run still leaves its
  # evidence somewhere, under a name that says which case it was.
  if [ "$run_gates" -eq 1 ]; then
    folder="$root_out/no-gpu-$(date -u +%Y%m%d-%H%M%S)"
  else
    folder="$root_out/tests-only-$(date -u +%Y%m%d-%H%M%S)"
  fi
fi
mkdir -p "$folder"

last_line() { [ -f "$1" ] && awk 'NF{last=$0} END{print last}' "$1" | tr -d '\r'; }

pytest_line="$(last_line "$pytest_log")"
verdict="$( [ -f "$gates_log" ] && tr -d '\r' < "$gates_log" | grep '^overall: ' | tail -1 )"
device="$( [ -f "$gates_log" ] && tr -d '\r' < "$gates_log" | grep '^  gpu_name: ' | tail -1 | sed 's/^  gpu_name: //' )"

# Strip the CR the child interpreter writes on Windows: the PowerShell twin's
# logs are LF, and the two shells' folders have to be comparable. On Linux this
# is a no-op.
[ -f "$pytest_log" ] && tr -d '\r' < "$pytest_log" > "$folder/pytest-gpu.txt"
[ -f "$gates_log" ] && tr -d '\r' < "$gates_log" > "$folder/gpu-gates.log"

{
  echo "# Local GPU run"
  echo ""
  echo "Written by \`scripts/run_gpu_local.sh\` on $(date -u +%Y-%m-%dT%H:%M:%SZ) (UTC)."
  echo ""
  echo "| | |"
  echo "|---|---|"
  echo "| device | ${device:-unknown} |"
  echo "| mode | $( [ "$quick" -eq 1 ] && echo quick || echo full ) |"
  echo "| interpreter | $py |"
  echo ""
  echo "| leg | command | exit | summary |"
  echo "|---|---|---|---|"
  if [ -n "$pytest_code" ]; then
    echo "| gpu tests | \`python -m pytest -m \"$marker\"\` | $pytest_code | ${pytest_line:-no output} |"
  else
    echo "| gpu tests | skipped (\`--skip-pytest\`) | | |"
  fi
  if [ -n "$gates_code" ]; then
    echo "| gate suite | \`python -m caustica.validation gpu-gates$( [ "$quick" -eq 1 ] && echo " --targets $quick_targets" )\` | $gates_code | ${verdict:-no verdict line} |"
  else
    echo "| gate suite | skipped (\`--skip-gates\`) | | |"
  fi
  if [ -f "$gates_log" ]; then
    echo ""
    echo "## Gates"
    echo ""
    tr -d '\r' < "$gates_log" | grep -E '^  (PASS|FAIL|SKIP|INCOMPLETE) ' | sed 's/^  /- /'
    echo ""
    echo "\`fullsize\` needs a 512-cubed run at dx = 0.30 mm, 13.2 GiB predicted, so"
    echo "the ladder only reaches it on a device with about 15 GiB free. On a"
    echo "smaller card that gate stays INCOMPLETE and the suite exits 4 however"
    echo "green the rest is; the criterion closes on a hosted A100 or H100."
    echo ""
    echo "\`time\` is the least reproducible of the five gates: it grades a plan"
    echo "built from a calibration probe taken seconds earlier, and on a laptop"
    echo "card the probe moves with the clocks. One \`time\` FAIL is a reason to"
    echo "run again, not a defect on its own."
  fi
  echo ""
  echo "Full output: \`pytest-gpu.txt\`, \`gpu-gates.log\`. The suite's own"
  echo "report is \`REPORT.md\` beside them, with \`gpu_gates.json\` as its"
  echo "machine half."
} > "$folder/SUMMARY.md"

echo ""
echo "run_gpu_local: summary in $folder/SUMMARY.md"

if [ -n "$pytest_code" ] && [ "$pytest_code" -ne 0 ]; then exit 1; fi
if [ -n "$gates_code" ]; then exit "$gates_code"; fi
exit 0
