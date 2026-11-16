<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core — split-conformal variants for the QLKNN-10D surrogate -->
# Split-conformal variants for the QLKNN-10D surrogate, per regime

Status: **PASS** (completeness gates). Level 0.90; holdout rows 47828.

Cells: coverage, then the median relative width 2h/y in brackets (positive targets).
Variants marked (post hoc) were added after the first comparison (940937b9, report 753c658a); chosen after seeing its results.

| Channel | Variant | Marginal | Zero flux | Near threshold | Unstable |
|---|---|---:|---:|---:|---:|
| chi_e | marginal_absolute | 0.901 | 0.997 | 0.999 (216) | 0.709 (2.74) |
| chi_e | mondrian_input_label | 0.901 | 0.995 | 0.998 (229) | 0.713 (2.9) |
| chi_e | normalised_gyro_bohm | 0.900 | 0.999 | 0.998 (55.7) | 0.701 (2.22) |
| chi_e | normalised_prediction | 0.899 | 0.995 | 0.998 (93.4) | 0.706 (1.75) |
| chi_e | mondrian_predicted_flux (post hoc) | 0.902 | 0.991 | 0.996 (66.3) | 0.724 (2.67) |
| chi_e | mondrian_predicted_flux_normalised_prediction (post hoc) | 0.901 | 0.992 | 0.997 (54.2) | 0.719 (2.52) |
| chi_i | marginal_absolute | 0.900 | 0.995 | 0.991 (1.22e+04) | 0.711 (3.17) |
| chi_i | mondrian_input_label | 0.901 | 0.994 | 0.987 (5.72e+03) | 0.716 (3.31) |
| chi_i | normalised_gyro_bohm | 0.901 | 0.998 | 0.991 (6.6e+03) | 0.709 (2.19) |
| chi_i | normalised_prediction | 0.902 | 0.992 | 0.988 (4.33e+03) | 0.723 (1.73) |
| chi_i | mondrian_predicted_flux (post hoc) | 0.903 | 0.989 | 0.989 (2.51e+03) | 0.733 (2.95) |
| chi_i | mondrian_predicted_flux_normalised_prediction (post hoc) | 0.904 | 0.988 | 0.988 (2.07e+03) | 0.735 (1.93) |
| D_e | marginal_absolute | 0.903 | 0.997 | 0.993 (296) | 0.665 (3.07) |
| D_e | mondrian_input_label | 0.903 | 0.996 | 0.996 (334) | 0.671 (3.34) |
| D_e | normalised_gyro_bohm | 0.902 | 0.999 | 0.988 (83.1) | 0.659 (2.51) |
| D_e | normalised_prediction | 0.902 | 0.997 | 0.993 (152) | 0.664 (2.24) |
| D_e | mondrian_predicted_flux (post hoc) | 0.903 | 0.996 | 0.986 (100) | 0.669 (3.01) |
| D_e | mondrian_predicted_flux_normalised_prediction (post hoc) | 0.903 | 0.997 | 0.990 (86.1) | 0.666 (2.93) |

Surrogate on the unstable rows:

- chi_e: 0.683 of 15975 rows predicted at most r0; median prediction/target 0.493.
- chi_i: 0.645 of 15960 rows predicted at most r0; median prediction/target 0.602.
- D_e: 0.738 of 13618 rows predicted at most r0; median prediction/target 0.305.

Source revision: `9bd8a7a762c14dd7a25cfae458cd308bc4c05980`
