<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core — SCPN inverse fit on FreeGSNKE's public inverse problems -->
# SCPN inverse coil-current fit on FreeGSNKE's public inverse problems

Status: **FAIL** (completeness gates; agreement is reported, not gated)

Synthetic public machine descriptions only. No facility, control, safety or speed claim; no comparison with measured discharges.

Loss columns are L2 norms over all targets. At FreeGSNKE's currents: FreeGSNKE's loss, SCPN's evaluator on FreeGSNKE's flux map, and SCPN's loss on its own equilibrium. At SCPN's fitted currents: SCPN's loss and FreeGSNKE's.

| Case | Start | Stop | FreeGSNKE own | SCPN evaluator, FreeGSNKE flux | SCPN at FreeGSNKE currents | SCPN fit | FreeGSNKE at SCPN currents | Current change (rel. L2) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| mastu_like | freegsnke | no_descent_step | 0.00154 | 0.00155 | 0.00224 | 1.92e-08 | 0.00211 | 0.014 |
| mastu_like | perturbed | jacobian_solve_not_converged | 0.00154 | 0.00155 | 0.00224 | 0.0196 | — | 0.642 |
| sparc_like | freegsnke | no_descent_step | 0.0349 | 0.0348 | 0.0311 | 4.92e-08 | 0.00778 | 0.00967 |
| sparc_like | perturbed | no_descent_step | 0.0349 | 0.0348 | 0.0311 | 1.21e-06 | 0.00775 | 0.127 |
| iter_like | freegsnke | max_outer_iterations | 1.71 | 1.71 | 1.81 | 1.39 | 1.39 | 0.0126 |
| iter_like | perturbed | max_outer_iterations | 1.71 | 1.71 | 1.81 | 1.41 | 1.41 | 0.0121 |

Method differences:

- FreeGSNKE linearises the targets with the plasma frozen; SCPN takes central differences through the full free-boundary solve.
- FreeGSNKE penalises current-limit violations through slack variables; SCPN projects the currents onto the limits.
- Both apply the Tikhonov weight to the current step.
- FreeGSNKE interpolates the plasma flux with a bicubic spline; SCPN with Keys cubic convolution.

Payload SHA-256: `2bf02dd351da4a711f862bdb4a1060f2b6795b80e276f3405fee57b52f53ebf9`
