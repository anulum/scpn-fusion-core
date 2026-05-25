<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core - UpCloud L4 Native Solver Benchmark Bundle -->

# UpCloud L4 Native Solver Benchmark Bundle

- Date: 2026-05-25
- Host: UpCloud fi-hel2, NVIDIA L4 23034 MiB, driver 595.71.05
- CPU: 8x AMD EPYC 9575F 64-Core Processor
- RAM: 62 GiB
- OS: Linux 6.8.0-117-generic x86_64

## Tracked Rust CPU/GPU SOR benchmark

| Benchmark | Low | Median | High |
|---|---:|---:|---:|
| `cpu_sor_solve_33x33_20iter` | 43.505 µs | 45.215 µs | 47.272 µs |
| `gpu_sor_solve_full_33x33_20iter` | 964.26 µs | 965.68 µs | 967.51 µs |
| `cpu_sor_solve_65x65_20iter` | 175.61 µs | 177.11 µs | 179.19 µs |
| `gpu_sor_solve_full_65x65_20iter` | 961.68 µs | 965.96 µs | 969.68 µs |
| `cpu_sor_solve_129x129_20iter` | 701.17 µs | 709.97 µs | 716.33 µs |
| `gpu_sor_solve_full_129x129_20iter` | 981.46 µs | 984.41 µs | 986.54 µs |

## Polyglot Grad-Shafranov scaling

| Grid | Language | Wall time s |
|---|---|---:|
| 33x33 | Python | 0.002393 |
| 33x33 | Julia | 0.990421 |
| 33x33 | Go | 0.002761 |
| 33x33 | Rust | 0.001527 |
| 33x33 | Lean | 0.645246 |
| 65x65 | Python | 0.004030 |
| 65x65 | Julia | 0.997984 |
| 65x65 | Go | 0.010574 |
| 65x65 | Rust | 0.003018 |
| 65x65 | Lean | 0.670504 |

## Polyglot warm-throughput timing

| Language | Median ms | P95 ms |
|---|---:|---:|
| Python | 3.680793 | 4.109378 |
| Go | 4.022329 | 4.808413 |
| Rust | 1.302658 | 1.878885 |
| Julia | 1.663381 | 3.752034 |
| Lean | 1503.000000 | 1593.000000 |

Lean was executed with a single `lake env lean --run` process and `100` solves
inside that process.

## Persistent-buffer GPU SOR timing

| Grid | Runs | Persistent solve median ms | Persistent solve P95 ms | Download ms |
|---|---:|---:|---:|---:|
| `129x129` | 100 | 0.760128 | 2.940710 | 0.053754 |
| `257x257` | 100 | 0.764012 | 2.897592 | 0.165949 |
| `513x513` | 50 | 0.861687 | 3.009115 | 0.343303 |

## CUDA-JAX nonlinear GK

- JAX available: `True`

| Case | Backend row | Backend used | Elapsed s | Converged |
|---|---|---|---:|---:|
| `krook` | `numpy` | `numpy` | 0.020947 | True |
| `krook` | `jax` | `jax` | 4.051154 | True |
| `sugama` | `numpy` | `numpy` | 0.022762 | True |
| `sugama` | `jax` | `jax` | 1.648905 | True |
| `sugama_electromagnetic_kinetic` | `numpy` | `numpy` | 0.045141 | True |
| `sugama_electromagnetic_kinetic` | `jax` | `jax` | 1.662950 | True |

## Equilibrium reconstruction status

| Contract | Status | Evidence |
|---|---|---|
| Free-boundary coil/vacuum benchmark | PASS | `validation/reports/free_boundary_benchmark.json` |
| FreeGS strict backend comparison | FAIL | scalar derivative compatibility patched; FreeGS now fails with no O-points or Picard non-convergence in all five configured cases |
| EFIT/GEQDSK raw profile-source gate | FAIL | public rows `0/8` under `psi_N RMSE <= 0.05`, worst `jet/jet_lmode_2MA.geqdsk` |
| Operator-source gate | PASS | public rows `8/8` under `1e-6` |
| Adapted profile-source gate | PASS | accepted adapter rows `4/4` under `psi_N RMSE <= 0.05` |
| Native operator/current closure | PASS | radial order `2.000000` |

## Rust kernel Criterion bundle

| Log | Benchmark | Median |
|---|---|---:|
| `picard_bench.log` | `picard_gs_solve/sor_33x33` | 194.08 µs |
| `picard_bench.log` | `picard_multigrid_solve/multigrid_33x33` | 408.26 µs |
| `vacuum_bench.log` | `vacuum_field_33x33_6coils` | 73.153 µs |
| `vacuum_bench.log` | `vacuum_field_65x65_6coils` | 277.52 µs |
| `fusion-math_sor_bench.log` | `sor_step_33x33` | 1.8483 µs |
| `fusion-math_sor_bench.log` | `sor_solve_33x33_500iter` | 921.00 µs |
| `fusion-math_sor_bench.log` | `sor_solve_65x65_500iter` | 3.7896 ms |
| `fusion-math_sor_bench.log` | `sor_residual_65x65` | 8.0632 µs |
| `fusion-math_gmres_bench.log` | `gmres_33x33` | 203.79 µs |
| `fusion-math_gmres_bench.log` | `gmres_65x65` | 1.2071 ms |
| `fusion-math_multigrid_bench.log` | `multigrid_33x33` | 404.59 µs |
| `fusion-math_multigrid_bench.log` | `multigrid_65x65` | 1.5946 ms |
| `fusion-math_fft_bench.log` | `fft2_real_64x64` | 18.660 µs |
| `fusion-math_fft_bench.log` | `ifft2_real_64x64` | 21.421 µs |
| `fusion-math_fft_bench.log` | `cfft2_cifft2_complex_64x64` | 41.566 µs |
| `fusion-core_inverse_bench.log` | `inverse_reconstruct_analytic_60probes` | 42.133 µs |
| `fusion-core_inverse_bench.log` | `inverse_fd_vs_analytical/finite_difference_60probes` | 520.29 µs |
| `fusion-core_inverse_bench.log` | `inverse_fd_vs_analytical/analytical_60probes` | 338.89 µs |
| `fusion-core_transport_bench.log` | `transport_step/lmode_single_step` | 754.06 ns |
| `fusion-core_transport_bench.log` | `transport_step/hmode_single_step` | 866.43 ns |
| `fusion-core_transport_bench.log` | `transport_step/hmode_neoclassical_single_step` | 3.4128 µs |
| `fusion-core_transport_bench.log` | `chang_hinton_chi_50pts` | 1.3872 µs |
| `fusion-physics_hall_mhd_bench.log` | `bench_hall_mhd_step_64` | 864.60 µs |
| `fusion-physics_hall_mhd_bench.log` | `bench_hall_mhd_step_128` | 5.2128 ms |
| `fusion-physics_hall_mhd_bench.log` | `bench_hall_mhd_run_100_64` | 82.088 ms |
