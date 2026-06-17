# SCPN End-to-End Latency Benchmark

- Generated: `2026-06-17T17:28:24.538134+00:00`
- Runtime: `15.482 s`
- Steps: `320`
- SNN runtime backend: `numpy`
- Threshold profile: `numpy_fallback`
- Overall pass: `YES`
- SNN full/surrogate p95 ratio: `0.508`

## Digital-Twin Sensor-to-Control Path

Python CPU and Rust rows are local non-isolated wall-clock measurements. GPU and HIL rows remain blocked unless status is measured.

| Lane | Status | p50 loop [ms] | p95 loop [ms] | p99 loop [ms] | Boundary |
|------|--------|---------------|---------------|---------------|----------|
| Python CPU | measured | 0.034026 | 0.046184 | 0.058611 | local non-isolated wall-clock |
| Rust native | measured | 0.019957 | 0.022279 | 0.026322 | local release binary when measured |
| GPU | measured | 0.735895 | 1.019303 | 1.280550 | Local non-isolated CuPy/CUDA measurement of the same reduced-order sensor-to-control contract; includes host snapshot assembly and output serialisation. |

### CPU Pipeline Stages

| Stage | p50 [ms] | p95 [ms] | p99 [ms] |
|-------|----------|----------|----------|
| input_validation | 0.001069 | 0.001367 | 0.001799 |
| feature_assembly | 0.001171 | 0.001679 | 0.002207 |
| digital_twin_update | 0.007266 | 0.009281 | 0.013172 |
| controller_decision | 0.001424 | 0.001676 | 0.002582 |
| fallback_policy | 0.013247 | 0.019105 | 0.023702 |
| output_serialization | 0.007945 | 0.010855 | 0.013249 |

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

## Actuator-Count Scaling

Actuator-count scaling measures command fanout and output serialisation for the reduced-order local benchmark. It is not actuator hardware timing.

| Actuators | CPU p95 [ms] | Rust p95 [ms] | CUDA p95 [ms] | Measured lanes |
|-----------|--------------|---------------|---------------|----------------|
| 2 | 0.084140 | 0.029088 | 1.079834 | 3 |
| 16 | 0.131076 | 0.023659 | 1.136501 | 3 |
| 64 | 0.136059 | 0.044102 | 1.148275 | 3 |
| 128 | 0.126704 | 0.034328 | 1.214223 | 3 |
| 256 | 0.356137 | 0.052569 | 1.064266 | 3 |

## Predictive-Horizon Timing

Forecast timing is a reduced-order local digital-twin rollout. It is not a validated 50-100 ms plasma-instability prediction horizon.

| Horizon [ms] | p50 forecast [ms] | p95 forecast [ms] | p99 forecast [ms] | p95 real-time factor | Pass |
|--------------|-------------------|-------------------|-------------------|----------------------|------|
| 50 | 0.409457 | 0.672888 | 1.143809 | 74.307 | YES |
| 100 | 0.951235 | 1.628676 | 1.723556 | 61.400 | YES |

## Surrogate Physics Mode

| Controller | RMSE | p95 loop [ms] | p95 sensor [ms] | p95 controller [ms] | p95 actuator [ms] | p95 physics [ms] |
|------------|------|---------------|------------------|---------------------|-------------------|------------------|
| SNN | 0.074583 | 0.376301 | 0.007868 | 0.337136 | 0.010566 | 0.000718 |
| PID | 0.136634 | 0.011627 | 0.003026 | 0.003849 | 0.003024 | 0.000176 |
| MPC-lite | 0.051153 | 0.073739 | 0.004164 | 0.062514 | 0.006708 | 0.000277 |

## Full Physics Mode

| Controller | RMSE | p95 loop [ms] | p95 sensor [ms] | p95 controller [ms] | p95 actuator [ms] | p95 physics [ms] |
|------------|------|---------------|------------------|---------------------|-------------------|------------------|
| SNN | 0.074439 | 0.190975 | 0.005320 | 0.173100 | 0.005700 | 0.008574 |
| PID | 0.138171 | 0.017300 | 0.003453 | 0.003729 | 0.002889 | 0.004492 |
| MPC-lite | 0.052585 | 0.082936 | 0.004801 | 0.064268 | 0.006467 | 0.007803 |
