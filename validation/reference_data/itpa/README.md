# ITPA H-Mode Confinement Database (Subset)

Reference data for validating energy confinement time predictions against
the IPB98(y,2) empirical scaling law.

> **Disclaimer:** This directory contains a **20-row illustrative subset**
> manually derived from published tables in the cited paper. It is **not**
> the full ITPA global H-mode confinement database. The data is used
> solely for regression testing of the IPB98(y,2) scaling law
> implementation in SCPN Fusion Core. For the authoritative dataset,
> contact the ITPA Confinement Database Working Group or download from
> the Princeton DataSpace link below.

## Source

Verdoolaege et al., "The updated ITPA global H-mode confinement database:
description and analysis", Nuclear Fusion **61** (2021) 076006.

Full dataset: https://dataspace.princeton.edu/handle/88435/dsp01m900nx49h

## IPB98(y,2) Scaling Law

τ_E = 0.0562 × I_p^0.93 × B_T^0.15 × n_e19^0.41 × P_loss^{-0.69}
      × R^1.97 × κ^0.78 × ε^0.58 × M^0.19

where:
- I_p: plasma current (MA)
- B_T: toroidal field (T)
- n_e19: line-average density (10^19 m^-3)
- P_loss: loss power = P_heat - dW/dt (MW)
- R: major radius (m)
- κ: elongation
- ε: inverse aspect ratio (a/R)
- M: effective ion mass (AMU)

## Files

- `hmode_confinement.csv` — curated subset of multi-machine data
- `ipb98y2_coefficients.json` — scaling law exponents and uncertainties
