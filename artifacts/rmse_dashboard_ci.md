# SCPN RMSE Dashboard

- Generated: `2026-02-17T22:55:26.565818+00:00`
- Runtime: `1.13s`

## Auto-Flag Summary

| Metric | Value | Flag |
|--------|-------|------|
| tau_E MAE % | `32.53%` | **[FAIL]** |
| beta_N RMSE | `0.0417` | **[PASS]** |
| SPARC axis RMSE (m) | `1.594950` | **[FAIL]** |

## Confinement RMSE (ITPA H-mode)

- Samples: `20`
- `tau_E` RMSE: `0.1287 s`
- `tau_E` mean absolute relative error: `32.53%` **[FAIL]**
- `H98(y,2)` RMSE: `0.1983`

![tau_E scatter](./tau_e_scatter.png)

## Confinement RMSE (ITER + SPARC references)

- Samples: `2`
- `tau_E` RMSE: `0.0299 s`

## Beta_N RMSE (ITER + SPARC references)

- Samples: `2`
- `beta_N` RMSE: `0.0417` **[PASS]**

![beta_N scatter](./beta_n_scatter.png)

## SPARC GEQDSK Axis Error

- Files: `8`
- Axis RMSE: `1.594950 m` **[FAIL]**

## Forward Diagnostics RMSE

- Interferometer channels: `3`
- Interferometer phase RMSE: `3.378620e-03 rad`
- Neutron-count relative error: `3.000%`
- Thomson channels: `3`
- Thomson voltage RMSE: `6.105161e-07 V`

## Notes

- `beta_N` estimates are derived from `DynamicBurnModel` steady-state thermal energy with a profile-peaking correction factor (1.446), calibrated against ITER and SPARC targets.
- Use this report for trend tracking; not as a replacement for full transport/MHD validation.
