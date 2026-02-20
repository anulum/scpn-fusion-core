# Transport Uncertainty Envelope Benchmark

- ITPA CSV: `validation/reference_data/itpa/hmode_confinement.csv`
- Shots: `20`
- Transport pass: `YES`
- Envelope fields pass: `YES`
- Coverage pass: `YES`
- |rel error| p95 pass: `YES`
- z-score p95 pass: `YES`
- Overall pass: `YES`

| Metric | Value | Threshold |
|--------|-------|-----------|
| within_2sigma_fraction | 0.95 | >= 0.80 |
| abs_relative_error_p95 | 0.8312 | <= 1.00 |
| zscore_p95 | 2.0736 | <= 2.50 |
| sigma_s_p95 | 0.5172 | > 0.00 |
