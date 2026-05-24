# SCPN End-to-End Latency Benchmark

- Generated: `2026-05-24T09:19:37.159402+00:00`
- Runtime: `0.235 s`
- Steps: `320`
- SNN runtime backend: `numpy`
- Threshold profile: `numpy_fallback`
- Overall pass: `YES`
- SNN full/surrogate p95 ratio: `1.037`

## Surrogate Physics Mode

| Controller | RMSE | p95 loop [ms] | p95 sensor [ms] | p95 controller [ms] | p95 actuator [ms] | p95 physics [ms] |
|------------|------|---------------|------------------|---------------------|-------------------|------------------|
| SNN | 0.074583 | 0.283756 | 0.006405 | 0.271609 | 0.008693 | 0.000628 |
| PID | 0.136634 | 0.012165 | 0.003780 | 0.004437 | 0.003704 | 0.000194 |
| MPC-lite | 0.051153 | 0.085373 | 0.004106 | 0.073007 | 0.009639 | 0.000374 |

## Full Physics Mode

| Controller | RMSE | p95 loop [ms] | p95 sensor [ms] | p95 controller [ms] | p95 actuator [ms] | p95 physics [ms] |
|------------|------|---------------|------------------|---------------------|-------------------|------------------|
| SNN | 0.074439 | 0.294143 | 0.007221 | 0.248414 | 0.007386 | 0.030486 |
| PID | 0.138171 | 0.046631 | 0.008111 | 0.007889 | 0.006502 | 0.023693 |
| MPC-lite | 0.052585 | 0.131835 | 0.009138 | 0.085636 | 0.014261 | 0.030579 |
