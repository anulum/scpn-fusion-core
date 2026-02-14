# Benchmarks

This directory contains tools for reproducing and collecting benchmark results.

## Quick Start

```bash
# Run all benchmarks and save results with hardware metadata
./benchmarks/collect_results.sh
```

This will:
1. Record your hardware specs (CPU, RAM, OS, toolchain versions)
2. Run all Criterion benchmarks (`cargo bench`)
3. Copy raw Criterion JSON output
4. Run Python kernel profiling
5. Save everything to `benchmarks/results/<timestamp>/`

## What Gets Collected

| File | Contents |
|------|----------|
| `hardware.txt` | CPU model, core count, RAM, OS, Rust/Python versions |
| `cargo_bench_raw.jsonl` | Raw cargo bench output in JSON lines format |
| `criterion_data/` | Full Criterion statistical analysis (mean, median, CI, plots) |
| `kernel_solve.prof` | Python cProfile binary (loadable with `snakeviz`) |
| `kernel_solve.txt` | Top-N hotspots from cProfile |

## Criterion Benchmarks

Three benchmark suites are defined in the Rust workspace:

| Suite | Crate | What It Measures |
|-------|-------|------------------|
| `sor_bench` | fusion-math | SOR step at 65x65 and 128x128, Chebyshev vs fixed |
| `inverse_bench` | fusion-core | Full LM reconstruction (FD vs analytical Jacobian) |
| `neural_transport_bench` | fusion-ml | MLP inference single-point and 50-point batch |

## Comparing Results

Criterion stores historical results in `target/criterion/`. After running
benchmarks on different hardware or after code changes, Criterion automatically
reports percentage changes vs the previous run.

For cross-machine comparison, save the `benchmarks/results/<timestamp>/`
directories and compare the `hardware.txt` + Criterion estimates.

## Reference Numbers

See [`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md) for the reference timing
tables and hardware environment description.
