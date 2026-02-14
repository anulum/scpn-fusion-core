#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Benchmark Result Collector
# Runs all Criterion benchmarks and saves results with hardware metadata.
# Usage: ./benchmarks/collect_results.sh
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTDIR="$REPO_ROOT/benchmarks/results/$TIMESTAMP"
mkdir -p "$OUTDIR"

# ── Collect hardware metadata ─────────────────────────────────────────
{
    echo "timestamp: $TIMESTAMP"
    echo "hostname: $(hostname)"
    echo "os: $(uname -srm)"
    if command -v lscpu &>/dev/null; then
        echo "cpu: $(lscpu | grep 'Model name' | sed 's/.*:\s*//')"
        echo "cores: $(nproc)"
    elif command -v sysctl &>/dev/null; then
        echo "cpu: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
        echo "cores: $(sysctl -n hw.ncpu 2>/dev/null || echo unknown)"
    else
        echo "cpu: unknown"
        echo "cores: unknown"
    fi
    if command -v free &>/dev/null; then
        echo "ram_mb: $(free -m | awk '/Mem:/{print $2}')"
    fi
    echo "rustc: $(rustc --version 2>/dev/null || echo not-installed)"
    echo "cargo: $(cargo --version 2>/dev/null || echo not-installed)"
    echo "python: $(python3 --version 2>/dev/null || python --version 2>/dev/null || echo not-installed)"
} > "$OUTDIR/hardware.txt"

echo "Hardware metadata saved to $OUTDIR/hardware.txt"

# ── Run Rust Criterion benchmarks ────────────────────────────────────
echo "Running Rust benchmarks (this may take a few minutes)..."
cd "$REPO_ROOT/scpn-fusion-rs"
cargo bench --message-format=json 2>/dev/null | tee "$OUTDIR/cargo_bench_raw.jsonl" || true

# Copy Criterion JSON results
if [ -d "target/criterion" ]; then
    cp -r target/criterion "$OUTDIR/criterion_data"
    echo "Criterion data copied to $OUTDIR/criterion_data/"
fi

# ── Run Python profiling ─────────────────────────────────────────────
cd "$REPO_ROOT"
echo "Running Python kernel profiling..."
python3 profiling/profile_kernel.py --top 50 --output-dir "$OUTDIR" 2>/dev/null || \
    python profiling/profile_kernel.py --top 50 --output-dir "$OUTDIR" 2>/dev/null || \
    echo "Python profiling skipped (install scpn-fusion first)"

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "Benchmark results collected in: $OUTDIR"
echo "Contents:"
ls -la "$OUTDIR"
echo ""
echo "To compare with previous runs, diff the criterion_data/ directories."
