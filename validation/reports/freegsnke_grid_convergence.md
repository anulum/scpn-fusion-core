<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core — grid convergence of the same-case FreeGSNKE comparison -->
# Grid convergence of the MAST-U-like same-case FreeGSNKE comparison

Status: **PASS** (completeness gates; differences are reported)

| Grid (nR × nZ) | Warm ψ_N RMSE | Frozen ψ_N RMSE | Axis error (m) | Worst X-point error (m) | Source rel. L2 | Vacuum max (Wb/rad) |
|---|---:|---:|---:|---:|---:|---:|
| 65 × 129 | 0.003543 | 0.0003777 | 0.00135 | 0.00211 | 3.51e-08 | 1.11e-16 |
| 129 × 257 | 0.002904 | 9.434e-05 | 0.00106 | 0.000659 | 5.27e-08 | 1.67e-16 |

Changes against the base grid:

- 129 × 257: FreeGSNKE currents change by 0.000486 (relative L2); warm RMSE ratio 0.82; frozen RMSE ratio 0.25.

Synthetic public machine description only. No facility, control, safety or speed claim; no comparison with measured discharges.

Payload SHA-256: `7867292cfc24f6594888ee8365c8f22f384a2ed7c97a6d7daa94a9646a302680`
