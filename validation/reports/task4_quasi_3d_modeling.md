# Task 4 Quasi-3D Modeling Report

- Generated: `2026-02-15T22:58:37.666154+00:00`
- Runtime: `0.964 s`
- Overall pass: `YES`

## Quasi-3D Force Balance

- NFP: `4`
- Force-balance RMSE: `2.777%` (threshold `<= 8.0%`)
- Force-residual P95: `3.105%` (threshold `<= 12.0%`)
- Asymmetry index: `0.1757`

## Hall-MHD + TEMHD Coupling

- Hall backend: `hall_mhd`
- Hall zonal ratio: `0.0825`
- Two-fluid index: `0.1593` (threshold `>= 0.10`)
- Two-fluid temperature split index: `0.4444`
- TEMHD cooling gain: `1.000%` (threshold `>= 1.0%`)

## JET / SOLPS-ITER Proxy Heat Flux

- JET reference files: `5`
- RMSE: `11.373%` (threshold `<= 15.0%`)
- Mean reference heat flux: `9.544e+05 W/m^2`
- Mean predicted heat flux: `9.449e+05 W/m^2`

## Erosion-Calibrated TBR Guard

- Raw TBR: `1.1360`
- Estimated erosion: `0.0265 mm/y`
- ASDEX reference erosion: `0.2500 mm/y`
- Erosion-curve RMSE: `33.992%` (threshold `<= 35.0%`)
- Calibrated TBR: `0.9815` (threshold `<= 1.10`)
