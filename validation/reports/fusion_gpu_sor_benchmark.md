<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core - Fusion GPU SOR Benchmark -->

# Fusion GPU SOR Benchmark

- Benchmark: `fusion_gpu_sor_apples_to_apples`
- Command: `cd scpn-fusion-rs && cargo bench -p fusion-gpu --bench gpu_sor_bench -- --sample-size 10`
- Iterations: `20`
- Omega: `1.3`
- Interpretation: small-grid GPU timings are launch/readback dominated; no speedup claim is supported at these grid sizes.

| Benchmark | Low | Median | High |
|---|---:|---:|---:|
| `cpu_sor_solve_33x33_20iter` | 43.505 µs | 45.215 µs | 47.272 µs |
| `gpu_sor_solve_full_33x33_20iter` | 964.26 µs | 965.68 µs | 967.51 µs |
| `cpu_sor_solve_65x65_20iter` | 175.61 µs | 177.11 µs | 179.19 µs |
| `gpu_sor_solve_full_65x65_20iter` | 961.68 µs | 965.96 µs | 969.68 µs |
| `cpu_sor_solve_129x129_20iter` | 701.17 µs | 709.97 µs | 716.33 µs |
| `gpu_sor_solve_full_129x129_20iter` | 981.46 µs | 984.41 µs | 986.54 µs |
