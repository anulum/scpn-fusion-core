# GACODE TGLF runtime evidence

Status: **PASS**

This report records a PATH-resolved official GACODE TGLF activation run. It is
runtime and parser evidence, not a cross-solver accuracy claim.

| Gate | Result |
|---|---:|
| expected_gacode_revision | PASS |
| finite_public_output | PASS |
| nonempty_consistent_spectrum | PASS |
| official_regression_9_of_9 | PASS |
| signed_fluxes_preserved | PASS |

- GACODE revision: `b4933975 [2026-08-20]`
- Official regression: 9/9 cases passed
- Activation spectrum: 21 ky points
- Ion/electron heat flux: 18.802 / 6.6005 gyro-Bohm
- Electron/ion particle flux: -0.32596 / -0.32596 gyro-Bohm
- Activation runtime: 0.903475 s (orientation only)
- Regression runtime: 9.462037 s (orientation only)
- Machine CPU: 11th Gen Intel(R) Core(TM) i5-11600K @ 3.90GHz

## Limits

- Elapsed times are orientation-only measurements from the reported workstation.
- This activation benchmark does not establish surrogate accuracy or uncertainty calibration.
- This activation benchmark does not establish superiority over another transport solver.
