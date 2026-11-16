<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core — study of the two misses of the linear reduced-MHD acceptance -->
# Study of the two misses of the linear reduced-MHD acceptance

Status: **PASS** (completeness only; findings are reported)

## Cylinder 2/1 tearing (Δ′ = 9.695)

| η | Growth rate | Ratio to asymptotic | Local slope to next η |
|---|---:|---:|---:|
| 1e-06 | 0.0003049 | 0.7715 | 0.5366 |
| 1e-07 | 8.862e-05 | 0.8928 | 0.5724 |
| 1e-08 | 2.372e-05 | 0.9514 | 0.5878 |
| 1e-09 | 6.128e-06 | 0.9784 | 0.5947 |
| 1e-10 | 1.558e-06 | 0.9905 | — |

Doubling the elements at η = 1e-10 changes the growth rate by 0.000173.

## Harris sheet, ka = 1.2 (Δ′ < 0)

Every solve is a dense QZ search over all finite eigenvalues; 0 means no eigenvalue with
real part above 1e-12.

| η | Root, 200 elements | Root, 400 elements | ka = 0.5 tearing rate |
|---|---:|---:|---:|
| 1e-06 | 7.92e-09 | 1.04e-09 | 0.000248 |
| 1e-07 | 1.05e-08 | 0 | 6.28e-05 |
| 1e-08 | 1.36e-08 | 1.86e-07 | 1.58e-05 |

Source revision: `03fdaf1d158f43539656d4e2aea4b8c2272681ea`
