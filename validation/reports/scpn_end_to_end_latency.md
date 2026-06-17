# SCPN End-to-End Latency Benchmark

- Generated: `2026-06-17T17:15:09.150307+00:00`
- Runtime: `2.625 s`
- Steps: `320`
- SNN runtime backend: `numpy`
- Threshold profile: `numpy_fallback`
- Overall pass: `YES`
- SNN full/surrogate p95 ratio: `0.609`

## Digital-Twin Sensor-to-Control Path

Python CPU and Rust rows are local non-isolated wall-clock measurements. GPU and HIL rows remain blocked unless status is measured.

| Lane | Status | p50 loop [ms] | p95 loop [ms] | p99 loop [ms] | Boundary |
|------|--------|---------------|---------------|---------------|----------|
| Python CPU | measured | 0.033269 | 0.038555 | 0.046308 | local non-isolated wall-clock |
| Rust native | measured | 0.019566 | 0.019960 | 0.025563 | local release binary when measured |
| GPU | measured | 0.665243 | 0.847707 | 1.005122 | Local non-isolated CuPy/CUDA measurement of the same reduced-order sensor-to-control contract; includes host snapshot assembly and output serialisation. |

### CPU Pipeline Stages

| Stage | p50 [ms] | p95 [ms] | p99 [ms] |
|-------|----------|----------|----------|
| input_validation | 0.001116 | 0.001241 | 0.004204 |
| feature_assembly | 0.001112 | 0.001399 | 0.002282 |
| digital_twin_update | 0.007240 | 0.008066 | 0.012727 |
| controller_decision | 0.001394 | 0.001531 | 0.002049 |
| fallback_policy | 0.013026 | 0.014224 | 0.017563 |
| output_serialization | 0.007625 | 0.008765 | 0.010804 |

### Degraded Modes

| Case | Fallbacks | Safe output rate | Pass | Reasons |
|------|-----------|------------------|------|---------|
| nominal | 0 | 1.000 | YES | none |
| stale_input | 46 | 1.000 | YES | stale_snapshot_safe_mode:46 |
| out_of_distribution_input | 30 | 1.000 | YES | ood_snapshot_safe_mode:30 |
| missing_diagnostic | 64 | 1.000 | YES | missing_q95_defaulted:64 |
| non_finite_value | 25 | 1.000 | YES | non_finite_input_replaced:25 |
| actuator_saturation | 36 | 1.000 | YES | actuator_saturation_clamped:36 |
| controller_fallback | 32 | 1.000 | YES | controller_non_finite_or_bad_shape:32 |

## Surrogate Physics Mode

| Controller | RMSE | p95 loop [ms] | p95 sensor [ms] | p95 controller [ms] | p95 actuator [ms] | p95 physics [ms] |
|------------|------|---------------|------------------|---------------------|-------------------|------------------|
| SNN | 0.074583 | 0.343356 | 0.007033 | 0.322189 | 0.010369 | 0.000596 |
| PID | 0.136634 | 0.012219 | 0.003466 | 0.004163 | 0.003249 | 0.000213 |
| MPC-lite | 0.051153 | 0.080873 | 0.003917 | 0.069578 | 0.006252 | 0.000276 |

## Full Physics Mode

| Controller | RMSE | p95 loop [ms] | p95 sensor [ms] | p95 controller [ms] | p95 actuator [ms] | p95 physics [ms] |
|------------|------|---------------|------------------|---------------------|-------------------|------------------|
| SNN | 0.074439 | 0.209194 | 0.006500 | 0.185647 | 0.006814 | 0.009328 |
| PID | 0.138171 | 0.023085 | 0.005342 | 0.005630 | 0.004589 | 0.007589 |
| MPC-lite | 0.052585 | 0.092227 | 0.004156 | 0.071764 | 0.007452 | 0.007789 |
