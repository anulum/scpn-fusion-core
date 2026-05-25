<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core - Polyglot Warm Throughput Benchmark -->

# Polyglot Warm Throughput Benchmark

- Case: `65x65` Grad-Shafranov Picard/Jacobi
- Runs per language: `100` after `5` warm-up solves
- Timing excludes language/tool startup for Python, Go, Rust, Julia, and Lean.

| Language | Median ms | P95 ms |
|---|---:|---:|
| Python | 3.680793 | 4.109378 |
| Go | 4.022329 | 4.808413 |
| Rust | 1.302658 | 1.878885 |
| Julia | 1.663381 | 3.752034 |
| Lean | 1503.000000 | 1593.000000 |
