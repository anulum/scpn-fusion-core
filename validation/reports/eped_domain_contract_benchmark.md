# EPED Domain Contract Benchmark

- Cases: `5` (in-domain `2`, out-of-domain `3`)
- In-domain contract pass: `YES`
- Out-of-domain flag pass: `YES`
- Penalty bounds pass: `YES`
- Overall pass: `YES`

| Case | In domain | Score | Penalty | Violations | Delta_ped | T_ped [keV] |
|------|-----------|-------|---------|------------|-----------|-------------|
| in_ref | YES | 0.000 | 1.000 | — | 0.0100 | 0.1000 |
| in_compact | YES | 0.000 | 1.000 | — | 0.0100 | 0.1000 |
| out_density | NO | 0.538 | 0.812 | n_ped_1e19=22 outside [2, 15] 1e19 m^-3 | 0.0100 | 0.1000 |
| out_temperature | NO | 0.513 | 0.821 | T_ped_guess_keV=12 outside [0.2, 8] keV | 0.0100 | 0.1000 |
| out_shape | NO | 0.333 | 0.883 | kappa=2.8 outside [1.2, 2.4] - | 0.0100 | 0.1000 |