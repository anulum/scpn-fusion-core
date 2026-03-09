# SCPN RMSE Dashboard

- Generated: `2026-03-09T01:18:56.599113+00:00`
- Runtime: `3.02s`

## Auto-Flag Summary

| Metric | Value | Flag |
|--------|-------|------|
| tau_E MAE % | `50.09%` | **[FAIL]** |
| beta_N RMSE | `0.1731` | **[FAIL]** |
| SPARC axis RMSE (m) | `1.594950` | **[FAIL]** |

## Confinement RMSE (ITPA H-mode)

- Samples: `53`
- `tau_E` RMSE: `0.0969 s`
- `tau_E` mean absolute relative error: `50.09%` **[FAIL]**
- `H98(y,2)` RMSE: `0.2954`

![tau_E scatter](./tau_e_scatter.png)

## Confinement RMSE (ITER + SPARC references)

- Samples: `2`
- `tau_E` RMSE: `0.0299 s`

## Beta_N RMSE (ITER + SPARC references)

- Samples: `2`
- `beta_N` RMSE: `0.1731` **[FAIL]**

![beta_N scatter](./beta_n_scatter.png)

## SPARC GEQDSK Axis Error

- Files: `8`
- Axis RMSE: `1.594950 m` **[FAIL]**

## SPARC Point-wise psi(R,Z) RMSE

- Files: `8`
- Mean normalised psi RMSE: `0.928002`
- Mean relative L2: `0.803325`
- Mean GS residual (rel L2): `11.8310`
- Worst file: `lmode_vh.geqdsk` (psi_N RMSE = `1.478894`)

## Forward Diagnostics RMSE

- Interferometer channels: `3`
- Interferometer phase RMSE: `3.378620e-03 rad`
- Neutron-count relative error: `3.000%`
- Thomson channels: `3`
- Thomson voltage RMSE: `6.105161e-07 V`

## Notes

- `beta_N` estimates are derived from `DynamicBurnModel` steady-state thermal energy with a profile-peaking correction factor (1.446), calibrated against ITER and SPARC targets.
- Use this report for trend tracking; not as a replacement for full transport/MHD validation.
