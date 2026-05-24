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

## Nonlinear GK Benchmark

`PYTHONPATH=src python benchmarks/gk_solver_comparison.py` runs the current
native nonlinear gyrokinetic APIs on a compact fixed grid and writes:

| File | Contents |
|------|----------|
| `validation/reports/gk_nonlinear_solver_comparison.json` | NumPy/JAX Krook and Sugama timing plus transport diagnostics |
| `validation/reports/gk_nonlinear_solver_comparison.md` | Human-readable benchmark table and Sugama moment residuals |

Adiabatic-electron benchmark cases keep the historical `chi_e = 0.5 * chi_i`
diagnostic convention. Kinetic-electron cases compute `chi_e` from the
electron distribution moment, so sign and magnitude are species-resolved rather
than inferred from the ion heat flux.

## Polyglot Grad-Shafranov Benchmark

`PYTHONPATH=src python benchmarks/polyglot_gs_solver_comparison.py` runs the
shared fixed-boundary Grad-Shafranov Picard/Jacobi case through the native
Python reference plus native Julia, Go, and Lean solver packages. Each
non-Python path implements the stencil, nonlinear source construction, Picard
loop, and boundary handling in its own language; these paths are not Python FFI
wrappers.

| File | Contents |
|------|----------|
| `validation/polyglot/gs_picard_reference.toml` | Shared cross-language case definition |
| `validation/reports/polyglot_gs_solver_comparison.json` | Hardware metadata, solver timings, and parity metrics |
| `validation/reports/polyglot_gs_solver_comparison.md` | Human-readable timing and parity report |

The local CLI benchmark includes process start-up and compilation-cache checks
for language CLI paths. Use long-lived processes or cloud GPU/CPU runners for
throughput comparisons.

## Criterion Benchmarks

Three benchmark suites are defined in the Rust workspace:

| Suite | Crate | What It Measures |
|-------|-------|------------------|
| `sor_bench` | fusion-math | SOR step at 65x65 and 128x128, Chebyshev vs fixed |
| `inverse_bench` | fusion-core | Full LM reconstruction (FD vs analytical Jacobian) |
| `picard_bench` | fusion-core | Full-order Grad-Shafranov Picard/SOR and Picard/multigrid solves |
| `vacuum_bench` | fusion-core | Full-order vacuum-field solve from ITER-like PF coil set |
| `neural_transport_bench` | fusion-ml | MLP inference single-point and 50-point batch |

## Latest Local Rust Equilibrium Numbers

Measured on 2026-05-24 with `cargo bench` on the local workstation:
Intel Core i5-11600K (6C/12T), 31.1 GB RAM, Linux 6.17.0-29-generic,
Rust 1.95.0, Python 3.12.3, NVIDIA GTX 1060 6GB present for JAX/CUDA
comparisons. Rust benches used the release profile with fat LTO.

| Benchmark | Scope | Criterion centre estimate |
|-----------|-------|---------------------------|
| `picard_gs_solve/sor_33x33` | Full-order Grad-Shafranov SOR, 33x33 grid, 10 Picard iterations | 412.83 us |
| `picard_multigrid_solve/multigrid_33x33` | Full-order Grad-Shafranov Picard multigrid, 33x33 grid, 10 Picard iterations | 844.59 us |
| `vacuum_field_33x33_6coils` | Vacuum field from six ITER-like coils, 33x33 grid | 139.55 us |
| `vacuum_field_65x65_6coils` | Vacuum field from six ITER-like coils, 65x65 grid | 488.86 us |

These are full-order Rust equilibrium component timings. They are separate
from the reduced-order Rust flight-simulator kernel measured by
`validation/verify_10khz_rust.py`.

## Comparing Results

Criterion stores historical results in `target/criterion/`. After running
benchmarks on different hardware or after code changes, Criterion automatically
reports percentage changes vs the previous run.

For cross-machine comparison, save the `benchmarks/results/<timestamp>/`
directories and compare the `hardware.txt` + Criterion estimates.

## Reference Numbers

See [`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md) for the reference timing
tables and hardware environment description.
