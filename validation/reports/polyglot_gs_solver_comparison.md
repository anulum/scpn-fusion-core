<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core - Polyglot Grad-Shafranov Benchmark -->

# Polyglot Grad-Shafranov Solver Benchmark

Local workstation benchmark for native Python, Julia, Go, and Lean fixed-boundary Grad-Shafranov Picard/Jacobi solvers. Each non-Python path executes its own implementation rather than a Python FFI wrapper.

## Hardware

- CPU: 11th Gen Intel(R) Core(TM) i5-11600K @ 3.90GHz
- Machine: x86_64
- OS: Linux-6.17.0-29-generic-x86_64-with-glibc2.39
- Python: 3.12.3
- Julia: julia version 1.12.6
- Go: go version go1.26.2 linux/amd64
- Lean: Lean (version 4.29.1, x86_64-unknown-linux-gnu, commit f72c35b3f637c8c6571d353742168ab66cc22c00, Release)

## Case

- Grid: 17x17
- Picard iterations: 8
- Jacobi sweeps per Picard step: 16
- Target plasma current: 1e+06 A

## Timing

| Language | Implementation | Wall time (s) |
|----------|----------------|---------------|
| Python | `gs_solve_np` | 0.003296 |
| Julia | `SCPNFusionSolvers.solve_grad_shafranov` | 5.871064 |
| Go | `gssolver.Solve` | 0.069161 |
| Lean | `SCPNFusionSolvers.solveGradShafranov` | 1.092461 |

## Numerical Parity

| Language | Interior relative L2 vs Python | Boundary absolute maximum |
|----------|--------------------------------|---------------------------|
| Julia | 8.983125e-17 | 0.000000e+00 |
| Go | 1.790625e-16 | 0.000000e+00 |
| Lean | 1.123308e-14 | 0.000000e+00 |

These local timings include process start-up and compilation-cache checks for CLI paths. Use long-lived processes or cloud CPU/GPU runners for throughput comparisons.
