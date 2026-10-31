# SCPN RMSE Dashboard

- Generated: `2026-10-31T11:54:34.529841+00:00`
- Runtime: `6.04s`

## Auto-Flag Summary

| Metric | Value | Flag |
|--------|-------|------|
| tau_E MAE % | `50.09%` | **[FAIL]** |
| beta_N RMSE | `0.0894` | **[WARN]** |
| SPARC axis RMSE (m) | `1.594950` | **[FAIL]** |

## Confinement RMSE (ITPA H-mode)

- Samples: `53`
- `tau_E` RMSE: `0.0969 s`
- `tau_E` mean absolute relative error: `50.09%` **[FAIL]**
- `H98(y,2)` RMSE: `0.2954`

![tau_E scatter](./tau_e_scatter.png)

## Confinement RMSE (ITER + SPARC references)

- Samples: `2`
- `tau_E` RMSE: `0.0254 s`

## Beta_N RMSE (ITER + SPARC references)

- Samples: `2`
- `beta_N` RMSE: `0.0894` **[WARN]**

![beta_N scatter](./beta_n_scatter.png)

## SPARC GEQDSK Axis Error

- Files: `8`
- Axis RMSE: `1.594950 m` **[FAIL]**

## SPARC Point-wise psi(R,Z) RMSE

- Files: `8`
- Mean normalised psi RMSE: `0.928047`
- Mean relative L2: `0.803365`
- Mean GS residual (rel L2): `11.8315`
- Worst file: `lmode_vh.geqdsk` (psi_N RMSE = `1.478971`)

## Forward Diagnostics RMSE

- Interferometer channels: `3`
- Interferometer phase RMSE: `3.378620e-03 rad`
- Neutron-count relative error: `3.000%`
- Thomson channels: `3`
- Thomson voltage RMSE: `6.105161e-07 V`

## Notes

- `beta_N` estimates are derived from `DynamicBurnModel` steady-state thermal energy with a profile/normalization correction factor (3.335), fitted to the ITER and SPARC targets as a reduced-order regression guard; the two-point beta_N RMSE is a fit residual, not a validation result.
- Use this report for trend tracking; not as a replacement for full transport/MHD validation.
