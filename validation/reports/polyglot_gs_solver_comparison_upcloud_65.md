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

- Grid: 65x65
- Picard iterations: 8
- Jacobi sweeps per Picard step: 16
- Target plasma current: 1e+06 A

## Timing

| Language | Implementation | Wall time (s) |
|----------|----------------|---------------|
| Python | `gs_solve_np` | 0.004030 |
| Julia | `SCPNFusionSolvers.solve_grad_shafranov` | 0.997984 |
| Go | `gssolver.Solve` | 0.010574 |
| Rust | `fusion_polyglot::solve_grad_shafranov` | 0.003018 |
| Lean | `SCPNFusionSolvers.solveGradShafranov` | 0.670504 |

## Numerical Parity

| Language | Interior relative L2 vs Python | Interior max abs error vs Python | Axis flux abs error vs Python | Boundary absolute maximum |
|----------|--------------------------------|----------------------------------|-------------------------------|---------------------------|
| Julia | 7.890752e-17 | 5.204170e-18 | 1.734723e-18 | 0.000000e+00 |
| Go | 3.743443e-16 | 8.673617e-18 | 0.000000e+00 | 0.000000e+00 |
| Rust | 5.017542e-16 | 1.561251e-17 | 3.469447e-18 | 0.000000e+00 |
| Lean | 3.681780e-14 | 5.013351e-16 | 1.665335e-16 | 0.000000e+00 |

## Physics Invariants

| Language | Axis flux value | Vertical symmetry absolute maximum | Axis midplane offset (cells) | Axis radial-center offset (cells) | Axis boundary distance (cells) | Axis local dominance margin | Axis discrete Laplacian | GS residual absolute maximum | GS residual relative maximum | Midplane radial monotonicity violations | Axis-column vertical monotonicity violations | Negative flux absolute maximum |
|----------|-----------------|------------------------------------|------------------------------|------------------------------------|--------------------------------|------------------------------|--------------------------|------------------------------|------------------------------|-------------------------------------------|-----------------------------------------------|--------------------------------|
| Python | 1.199868e-02 | 0.000000e+00 | 0 | 0 | 32 | 6.539908e-16 | -4.593119e-05 | 3.032679e+00 | 3.466129e+00 | 0 | 0 | 0.000000e+00 |
| Julia | 1.199868e-02 | 0.000000e+00 | 0 | 0 | 32 | 6.522560e-16 | -4.593119e-05 | 3.032679e+00 | 3.466129e+00 | 0 | 0 | 0.000000e+00 |
| Go | 1.199868e-02 | 0.000000e+00 | 0 | 0 | 32 | 6.522560e-16 | -4.593119e-05 | 3.032679e+00 | 3.466129e+00 | 0 | 0 | 0.000000e+00 |
| Rust | 1.199868e-02 | 0.000000e+00 | 0 | 0 | 32 | 6.487866e-16 | -4.593119e-05 | 3.032679e+00 | 3.466129e+00 | 0 | 0 | 0.000000e+00 |
| Lean | 1.199868e-02 | 0.000000e+00 | 0 | 0 | 32 | 9.992007e-16 | -4.593119e-05 | 3.032679e+00 | 3.466129e+00 | 0 | 0 | 0.000000e+00 |

These local timings include process start-up for CLI paths. The Go and Rust rows build solver binaries before timing and exclude toolchain orchestration from the measured solver invocation. Use long-lived processes or cloud CPU/GPU runners for throughput comparisons.
