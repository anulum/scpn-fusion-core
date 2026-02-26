# Multi-Ion Transport Conservation Benchmark

- Config: `iter_config.json`
- Steps: `30` at dt=`0.100` s, P_aux=`30.0` MW
- Finite pass: `YES`
- Positivity pass: `YES`
- Quasi-neutrality pass: `YES`
- Late-energy pass: `YES`
- He-ash growth pass: `YES`
- Overall pass: `YES`

| Metric | Value | Threshold |
|--------|-------|-----------|
| quasineutral_residual | 0.000e+00 | <= 1.0e-10 |
| late_energy_error_p95 | 0.5746 | <= 2.00 |
| he_ash_peak_1e19m3 | 0.000883 | >= 1.0e-04 |
