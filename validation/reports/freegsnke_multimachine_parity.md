<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core — FreeGSNKE same-case parity on three public machines -->
# FreeGSNKE same-case parity on three public machine descriptions

Status: **PASS**

Synthetic public machine descriptions only. No facility, control, safety or speed claim; no comparison with measured discharges.

| Case | Grid | Vacuum max (Wb) | Source rel. L2 | Warm ψ_N RMSE | Frozen ψ_N RMSE | Cold start |
|---|---|---:|---:|---:|---:|---|
| mastu_like | 65×129 | 1.11e-16 | 3.51e-08 | 0.003543 | 0.0003777 | no_magnetic_axis_inside_limiter (600 iterations) |
| sparc_like | 129×129 | 1.78e-14 | 9.17e-08 | 0.0004591 | 0.0001518 | same_fixed_point_as_warm_start (153 iterations) |
| iter_like | 129×129 | 7.67e-13 | 2.99e-08 | 0.0006418 | 0.0001216 | same_fixed_point_as_warm_start (270 iterations) |

Payload SHA-256: `2365bc8bebf2c0840b8f630a9c7ce06dfb1e201c5d74dd7272cd8d26845df21b`
