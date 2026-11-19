<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core — near-zero spectrum of the linear reduced-MHD pencil -->
# Near-zero spectrum of the linear reduced-MHD Harris pencil

Status: **PASS** (completeness only; findings are reported)

Case: ka = 1.2, η = 1e-06, 200 elements per side, 1600 unknowns.

Singular values of A: largest 1.324e+02, smallest 1.454e-17. Finite eigenvalues with |λ| < 1e-04: 158.

| Re λ | Im λ | Condition | Error bound | Residual |
|---:|---:|---:|---:|---:|
| -1.633e-11 | 0.000e+00 | 1.57e+10 | 4.63e-04 | 1.4e-14 |
| -2.613e-11 | 0.000e+00 | 4.56e+10 | 1.35e-03 | 1.4e-14 |
| -3.227e-10 | 0.000e+00 | 5.78e+10 | 1.71e-03 | 1.5e-14 |
| -5.398e-10 | 0.000e+00 | 2.29e+10 | 6.77e-04 | 1.4e-14 |
| 6.052e-10 | 0.000e+00 | 2.39e+11 | 7.07e-03 | 1.4e-14 |
| 1.014e-10 | -1.366e-09 | 1.90e+12 | 5.62e-02 | 1.4e-14 |
| -1.576e-09 | 0.000e+00 | 2.80e+10 | 8.26e-04 | 1.5e-14 |
| -1.916e-09 | 6.485e-10 | 3.34e+11 | 9.86e-03 | 1.5e-14 |
| -1.901e-09 | 1.759e-09 | 8.00e+11 | 2.36e-02 | 1.3e-14 |
| -2.944e-09 | 0.000e+00 | 1.16e+11 | 3.43e-03 | 1.4e-14 |
| -8.061e-11 | -2.997e-09 | 1.06e+12 | 3.12e-02 | 1.4e-14 |
| -3.029e-09 | 1.464e-10 | 1.39e+11 | 4.10e-03 | 1.5e-14 |

Largest real part: 6.052e-10. After relative perturbations of 1e-15 to A: -8.866e-12, 3.831e-11, 6.224e-10.

Source revision: `b5a7614169f1230d24a1ea04895509a9148a5707`
