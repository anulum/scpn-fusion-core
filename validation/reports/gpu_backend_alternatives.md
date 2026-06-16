<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core - GPU Backend Alternatives Benchmark -->

# GPU Backend Alternatives

- Schema: `gpu-backend-alternatives.v1`
- Generated UTC: `2026-06-16T23:47:57.821570+00:00`
- Publishable GPU throughput present: `True`

| Lane | Status | Evidence |
| --- | --- | --- |
| `wgpu_physical` | `passed` | selected: AMD Radeon RX 6600 XT (RADV NAVI23) (Vulkan, DiscreteGpu); inventory: AMD Radeon RX 6600 XT (RADV NAVI23), NVIDIA GeForce GTX 1060 6GB, AMD Radeon RX 6600 XT (RADV NAVI23), AMD Radeon RX 6600 XT (RADV NAVI23), AMD Radeon RX 6600 XT (RADV NAVI23), AMD Radeon RX 6600 XT (RADV NAVI23), Intel(R) Graphics (RKL GT1), llvmpipe (LLVM 20.1.2, 256 bits) |
| `cuda_jax` | `passed` | cuda:0 |

## Timing Details

### `wgpu_physical`

| Benchmark | Criterion time line |
| --- | --- |
| `cpu_sor_solve_33x33_20iter` | `time:   [109.50 µs 113.42 µs 117.84 µs]` |
| `gpu_sor_solve_full_33x33_20iter` | `time:   [104.01 ms 107.52 ms 111.09 ms]` |
| `cpu_sor_solve_65x65_20iter` | `time:   [360.26 µs 382.96 µs 410.02 µs]` |
| `gpu_sor_solve_full_65x65_20iter` | `time:   [103.32 ms 106.70 ms 110.18 ms]` |
| `cpu_sor_solve_129x129_20iter` | `time:   [1.3451 ms 1.3942 ms 1.4515 ms]` |
| `gpu_sor_solve_full_129x129_20iter` | `time:   [99.586 ms 103.33 ms 107.15 ms]` |

### `cuda_jax`

- Median: `0.000203` s
- Result SHA-256: `5572368b6af1edafdf98ec4eca1c58d4b918d617a35c95d1d3c77d3dba108ac8`
