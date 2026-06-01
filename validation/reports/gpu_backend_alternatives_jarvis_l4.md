<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core - GPU Backend Alternatives Benchmark -->

# GPU Backend Alternatives

- Schema: `gpu-backend-alternatives.v1`
- Generated UTC: `2026-06-01T20:50:28.867984+00:00`
- Publishable GPU throughput present: `True`

| Lane | Status | Evidence |
| --- | --- | --- |
| `wgpu_physical` | `blocked_cpu_adapter` | llvmpipe (LLVM 15.0.7, 256 bits) |
| `cuda_jax` | `passed` | cuda:0 |

## Timing Details

### `wgpu_physical`

- Blocker: `vulkan_wgpu_exposes_only_cpu_software_adapter`

### `cuda_jax`

- Median: `8.9e-05` s
- Result SHA-256: `c37f3690af105089f344784d0a6e43b7083f301301c368e08640cffa6086246b`
