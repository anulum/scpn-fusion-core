<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core - Polyglot Grad-Shafranov Benchmark -->

# Polyglot Grad-Shafranov Solver Benchmark

Local workstation benchmark for native Python and native Julia fixed-boundary Grad-Shafranov Picard/Jacobi solvers. The Julia path executes the `SCPNFusionSolvers` package directly; it is not a Python FFI wrapper.

## Hardware

- CPU: 11th Gen Intel(R) Core(TM) i5-11600K @ 3.90GHz
- Machine: x86_64
- OS: Linux-6.17.0-29-generic-x86_64-with-glibc2.39
- Python: 3.12.3
- Julia: julia version 1.12.6

## Case

- Grid: 17x17
- Picard iterations: 8
- Jacobi sweeps per Picard step: 16
- Target plasma current: 1e+06 A

## Timing

| Language | Implementation | Wall time (s) |
|----------|----------------|---------------|
| Python | `gs_solve_np` | 0.007014 |
| Julia | `SCPNFusionSolvers.solve_grad_shafranov` | 9.161843 |

## Numerical Parity

- Interior relative L2 error, Julia vs Python: 8.983125e-17
- Julia boundary absolute maximum: 0.000000e+00

These local timings include process start-up for the Julia CLI path. Use cloud or long-lived process benchmarks for throughput comparisons.
