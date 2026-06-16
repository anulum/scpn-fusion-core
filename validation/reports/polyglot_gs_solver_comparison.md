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

- CPU: 11th Gen Intel(R) Core(TM) i5-11600K @ 3.90GHz
- Machine: x86_64
- OS: Linux-6.17.0-35-generic-x86_64-with-glibc2.39
- Python: 3.12.3
- Julia: julia version 1.12.6
- Go: go version go1.24.0 linux/amd64
- Rust: rustc 1.96.0 (ac68faa20 2026-05-25)
- Lean: Lean (version 4.31.0, x86_64-unknown-linux-gnu, commit 68218e876d2a38b1985b8590fff244a83c321783, Release)

## Case

- Grid: 17x17
- Picard iterations: 8
- Jacobi sweeps per Picard step: 16
- Target plasma current: 1e+06 A

## Timing

| Language | Implementation | Wall time (s) |
|----------|----------------|---------------|
| Python | `gs_solve_np` | 0.007884 |
| Julia | `SCPNFusionSolvers.solve_grad_shafranov` | 5.462643 |
| Go | `gssolver.Solve` | 0.003616 |
| Rust | `fusion_polyglot::solve_grad_shafranov` | 0.004898 |
| Lean | `SCPNFusionSolvers.solveGradShafranov` | 9.135552 |

## Numerical Parity

| Language | Interior relative L2 vs Python | Interior max abs error vs Python | Axis flux abs error vs Python | Boundary absolute maximum |
|----------|--------------------------------|----------------------------------|-------------------------------|---------------------------|
| Julia | 8.983125e-17 | 1.387779e-17 | 0.000000e+00 | 0.000000e+00 |
| Go | 1.790625e-16 | 1.387779e-17 | 0.000000e+00 | 0.000000e+00 |
| Rust | 5.906398e-16 | 4.163336e-17 | 3.469447e-17 | 0.000000e+00 |
| Lean | 1.123308e-14 | 4.857226e-16 | 2.844947e-16 | 0.000000e+00 |

## Physics Invariants

| Language | Axis flux value | Vertical symmetry absolute maximum | Axis midplane offset (cells) | Axis radial-center offset (cells) | Axis boundary distance (cells) | Axis local dominance margin | Axis discrete Laplacian | GS residual absolute maximum | GS residual relative maximum | Midplane radial monotonicity violations | Axis-column vertical monotonicity violations | Negative flux absolute maximum |
|----------|-----------------|------------------------------------|------------------------------|------------------------------------|--------------------------------|------------------------------|--------------------------|------------------------------|------------------------------|-------------------------------------------|-----------------------------------------------|--------------------------------|
| Python | 4.265183e-02 | 0.000000e+00 | 0 | 0 | 8 | 1.368324e-04 | -2.321758e-03 | 9.034946e-01 | 8.617210e-01 | 0 | 0 | 0.000000e+00 |
| Julia | 4.265183e-02 | 0.000000e+00 | 0 | 0 | 8 | 1.368324e-04 | -2.321758e-03 | 9.034946e-01 | 8.617210e-01 | 0 | 0 | 0.000000e+00 |
| Go | 4.265183e-02 | 0.000000e+00 | 0 | 0 | 8 | 1.368324e-04 | -2.321758e-03 | 9.034946e-01 | 8.617210e-01 | 0 | 0 | 0.000000e+00 |
| Rust | 4.265183e-02 | 0.000000e+00 | 0 | 0 | 8 | 1.368324e-04 | -2.321758e-03 | 9.034946e-01 | 8.617210e-01 | 0 | 0 | 0.000000e+00 |
| Lean | 4.265183e-02 | 0.000000e+00 | 0 | 0 | 8 | 1.368324e-04 | -2.321758e-03 | 9.034946e-01 | 8.617210e-01 | 0 | 0 | 0.000000e+00 |

These local timings include process start-up for CLI paths. The Go and Rust rows build solver binaries before timing and exclude toolchain orchestration from the measured solver invocation. Use long-lived processes or cloud CPU/GPU runners for throughput comparisons.
