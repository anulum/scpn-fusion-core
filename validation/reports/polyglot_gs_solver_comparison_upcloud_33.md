<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core - Polyglot Grad-Shafranov Benchmark -->

# Polyglot Grad-Shafranov Solver Benchmark

Local workstation benchmark for native Python, Julia, Go, Rust, and Lean fixed-boundary Grad-Shafranov Picard/Jacobi solvers. Each non-Python path executes its own implementation rather than a Python FFI wrapper.

## Hardware

- CPU: AMD EPYC 9575F 64-Core Processor
- Machine: x86_64
- OS: Linux-6.8.0-117-generic-x86_64-with-glibc2.39
- Python: 3.12.3
- Julia: julia version 1.10.10
- Go: go version go1.22.2 linux/amd64
- Rust: rustc 1.95.0 (59807616e 2026-04-14)
- Lean: Lean (version 4.29.1, x86_64-unknown-linux-gnu, commit f72c35b3f637c8c6571d353742168ab66cc22c00, Release)

## Case

- Grid: 33x33
- Picard iterations: 8
- Jacobi sweeps per Picard step: 16
- Target plasma current: 1e+06 A

## Timing

| Language | Implementation | Wall time (s) |
|----------|----------------|---------------|
| Python | `gs_solve_np` | 0.002393 |
| Julia | `SCPNFusionSolvers.solve_grad_shafranov` | 0.990421 |
| Go | `gssolver.Solve` | 0.002761 |
| Rust | `fusion_polyglot::solve_grad_shafranov` | 0.001527 |
| Lean | `SCPNFusionSolvers.solveGradShafranov` | 0.645246 |

## Numerical Parity

| Language | Interior relative L2 vs Python | Interior max abs error vs Python | Axis flux abs error vs Python | Boundary absolute maximum |
|----------|--------------------------------|----------------------------------|-------------------------------|---------------------------|
| Julia | 1.055953e-16 | 6.938894e-18 | 0.000000e+00 | 0.000000e+00 |
| Go | 1.691122e-16 | 6.938894e-18 | 6.938894e-18 | 0.000000e+00 |
| Rust | 6.179584e-16 | 1.734723e-17 | 1.040834e-17 | 0.000000e+00 |
| Lean | 2.445130e-14 | 5.065393e-16 | 2.289835e-16 | 0.000000e+00 |

## Physics Invariants

| Language | Axis flux value | Vertical symmetry absolute maximum | Axis midplane offset (cells) | Axis radial-center offset (cells) | Axis boundary distance (cells) | Axis local dominance margin | Axis discrete Laplacian | GS residual absolute maximum | GS residual relative maximum | Midplane radial monotonicity violations | Axis-column vertical monotonicity violations | Negative flux absolute maximum |
|----------|-----------------|------------------------------------|------------------------------|------------------------------------|--------------------------------|------------------------------|--------------------------|------------------------------|------------------------------|-------------------------------------------|-----------------------------------------------|--------------------------------|
| Python | 1.811911e-02 | 0.000000e+00 | 0 | 0 | 16 | 4.265380e-08 | -2.600785e-04 | 8.572827e-01 | 9.277462e-01 | 0 | 0 | 0.000000e+00 |
| Julia | 1.811911e-02 | 0.000000e+00 | 0 | 0 | 16 | 4.265380e-08 | -2.600785e-04 | 8.572827e-01 | 9.277462e-01 | 0 | 0 | 0.000000e+00 |
| Go | 1.811911e-02 | 0.000000e+00 | 0 | 0 | 16 | 4.265380e-08 | -2.600785e-04 | 8.572827e-01 | 9.277462e-01 | 0 | 0 | 0.000000e+00 |
| Rust | 1.811911e-02 | 0.000000e+00 | 0 | 0 | 16 | 4.265380e-08 | -2.600785e-04 | 8.572827e-01 | 9.277462e-01 | 0 | 0 | 0.000000e+00 |
| Lean | 1.811911e-02 | 0.000000e+00 | 0 | 0 | 16 | 4.265380e-08 | -2.600785e-04 | 8.572827e-01 | 9.277462e-01 | 0 | 0 | 0.000000e+00 |

These local timings include process start-up for CLI paths. The Go and Rust rows build solver binaries before timing and exclude toolchain orchestration from the measured solver invocation. Use long-lived processes or cloud CPU/GPU runners for throughput comparisons.
