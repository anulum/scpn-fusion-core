<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core - GPU Backend Alternatives Benchmark -->

# GPU Backend Alternatives

- Schema: `gpu-backend-alternatives.v1`
- Generated UTC: `2026-06-01T20:36:40.498244+00:00`
- Publishable GPU throughput present: `True`

| Lane | Status | Evidence |
| --- | --- | --- |
| `wgpu_physical` | `passed` | selected: NVIDIA GeForce GTX 1060 6GB (Vulkan, DiscreteGpu); inventory: Intel(R) Graphics (RKL GT1), NVIDIA GeForce GTX 1060 6GB, AMD Radeon RX 6600 XT (RADV NAVI23), AMD Radeon RX 6600 XT (RADV NAVI23), AMD Radeon RX 6600 XT (RADV NAVI23), AMD Radeon RX 6600 XT (RADV NAVI23), AMD Radeon RX 6600 XT (RADV NAVI23), llvmpipe (LLVM 20.1.2, 256 bits) |
| `cuda_jax` | `passed` | cuda:0 |

## Timing Details

### `wgpu_physical`

| Benchmark | Criterion time line |
| --- | --- |
| `cpu_sor_solve_129x129_20iter` | `time:   [2.2020 ms 2.3545 ms 2.5163 ms]` |
| `cpu_sor_solve_33x33_20iter` | `time:   [118.60 µs 123.24 µs 127.83 µs]` |
| `cpu_sor_solve_65x65_20iter` | `time:   [466.99 µs 490.65 µs 516.72 µs]` |
| `gpu_sor_solve_full_129x129_20iter` | `time:   [1.9710 ms 2.0289 ms 2.0877 ms]` |
| `gpu_sor_solve_full_33x33_20iter` | `time:   [2.0806 ms 2.1627 ms 2.2482 ms]` |
| `gpu_sor_solve_full_65x65_20iter` | `time:   [2.4200 ms 2.5852 ms 2.7623 ms]` |

### `cuda_jax`

- Median: `0.000215` s
- Result SHA-256: `981894544f3fdb73390ea6b23d4eaaf457e2fc74d702aab235e16625f9439718`
