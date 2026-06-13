# SCPN End-to-End Latency Benchmark

- Generated: `2026-06-13T20:54:49.358157+00:00`
- Runtime: `0.572 s`
- Steps: `320`
- SNN runtime backend: `rust`
- Threshold profile: `accelerated`
- Overall pass: `YES`
- SNN full/surrogate p95 ratio: `0.598`

## Surrogate Physics Mode

| Controller | RMSE | p95 loop [ms] | p95 sensor [ms] | p95 controller [ms] | p95 actuator [ms] | p95 physics [ms] |
|------------|------|---------------|------------------|---------------------|-------------------|------------------|
| SNN | 0.074583 | 0.760918 | 0.015188 | 0.723525 | 0.023883 | 0.001942 |
| PID | 0.136634 | 0.034823 | 0.010704 | 0.012828 | 0.009215 | 0.000767 |
| MPC-lite | 0.051153 | 0.228498 | 0.014347 | 0.176755 | 0.038486 | 0.001507 |

## Full Physics Mode

| Controller | RMSE | p95 loop [ms] | p95 sensor [ms] | p95 controller [ms] | p95 actuator [ms] | p95 physics [ms] |
|------------|------|---------------|------------------|---------------------|-------------------|------------------|
| SNN | 0.074439 | 0.455023 | 0.011731 | 0.368325 | 0.013400 | 0.025048 |
| PID | 0.138171 | 0.046091 | 0.011864 | 0.010827 | 0.006859 | 0.018128 |
| MPC-lite | 0.052585 | 0.259869 | 0.015407 | 0.165361 | 0.031030 | 0.040290 |
