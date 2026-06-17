# SCPN End-to-End Latency Benchmark

- Generated: `2026-06-17T17:43:04.437895+00:00`
- Runtime: `15.515 s`
- Steps: `320`
- SNN runtime backend: `numpy`
- Threshold profile: `numpy_fallback`
- Overall pass: `YES`
- SNN full/surrogate p95 ratio: `0.938`

## Digital-Twin Sensor-to-Control Path

Python CPU and Rust rows are local non-isolated wall-clock measurements. GPU rows are accelerator measurements only when status is measured. HIL rows are simulated host ADC/DAC timing unless hardware_status names a physical rig.

| Lane | Status | p50 loop [ms] | p95 loop [ms] | p99 loop [ms] | Boundary |
|------|--------|---------------|---------------|---------------|----------|
| Python CPU | measured | 0.034983 | 0.041437 | 0.061708 | local non-isolated wall-clock |
| Rust native | measured | 0.020350 | 0.025929 | 0.029981 | local release binary when measured |
| GPU | measured | 0.784985 | 1.374682 | 3.878414 | Local non-isolated CuPy/CUDA measurement of the same reduced-order sensor-to-control contract; includes host snapshot assembly and output serialisation. |

### Simulated HIL Sensor-to-Actuator Scaffold

Measured host-side simulated ADC/DAC sensor-to-actuator scaffold; not a physical HIL rig, FPGA bitstream, plant CODAC, or actuator hardware timing claim.

| Status | Hardware status | Actuators | p50 [us] | p95 [us] | p99 [us] | Pass |
|--------|-----------------|-----------|----------|----------|----------|------|
| measured_simulated_hil | simulated_host_adc_dac_loop | 256 | 166.574500 | 203.927000 | 270.075290 | YES |

| Scenario | Fallbacks | Safe output rate | p95 [us] | Pass | Reasons |
|----------|-----------|------------------|----------|------|---------|
| nominal | 0 | 1.000 | 203.927000 | YES | none |
| sensor_dropout | 20 | 1.000 | 203.897850 | YES | invalid_sensor_sample_replaced:20 |
| noisy_sensor | 27 | 1.000 | 342.709550 | YES | noisy_sensor_envelope_safe_mode:27 |
| actuator_saturation | 18 | 1.000 | 347.775900 | YES | actuator_saturation_clamped:18 |
| controller_nonfinite | 16 | 1.000 | 353.215750 | YES | controller_nonfinite_fail_closed:16 |

### CPU Pipeline Stages

| Stage | p50 [ms] | p95 [ms] | p99 [ms] |
|-------|----------|----------|----------|
| input_validation | 0.001080 | 0.001219 | 0.001836 |
| feature_assembly | 0.001196 | 0.001573 | 0.002541 |
| digital_twin_update | 0.007627 | 0.009241 | 0.015000 |
| controller_decision | 0.001423 | 0.001631 | 0.002336 |
| fallback_policy | 0.013718 | 0.015631 | 0.023172 |
| output_serialization | 0.007859 | 0.009926 | 0.014942 |

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
| 2 | 0.082381 | 0.031709 | 0.945895 | 3 |
| 16 | 0.067500 | 0.022876 | 0.877984 | 3 |
| 64 | 0.104616 | 0.029172 | 0.918282 | 3 |
| 128 | 0.140701 | 0.050461 | 0.997855 | 3 |
| 256 | 0.194781 | 0.047781 | 1.078829 | 3 |

## Predictive-Horizon Timing

Forecast timing is a reduced-order local digital-twin rollout. It is not a validated 50-100 ms plasma-instability prediction horizon.

| Horizon [ms] | p50 forecast [ms] | p95 forecast [ms] | p99 forecast [ms] | p95 real-time factor | Pass |
|--------------|-------------------|-------------------|-------------------|----------------------|------|
| 50 | 0.410938 | 0.785309 | 0.868070 | 63.669 | YES |
| 100 | 0.865991 | 1.658421 | 1.796235 | 60.298 | YES |

## Surrogate Physics Mode

| Controller | RMSE | p95 loop [ms] | p95 sensor [ms] | p95 controller [ms] | p95 actuator [ms] | p95 physics [ms] |
|------------|------|---------------|------------------|---------------------|-------------------|------------------|
| SNN | 0.074583 | 0.309689 | 0.006728 | 0.292521 | 0.008781 | 0.000692 |
| PID | 0.136634 | 0.027094 | 0.007937 | 0.009062 | 0.007758 | 0.000458 |
| MPC-lite | 0.051153 | 0.091335 | 0.003671 | 0.073105 | 0.006674 | 0.000341 |

## Full Physics Mode

| Controller | RMSE | p95 loop [ms] | p95 sensor [ms] | p95 controller [ms] | p95 actuator [ms] | p95 physics [ms] |
|------------|------|---------------|------------------|---------------------|-------------------|------------------|
| SNN | 0.074439 | 0.290571 | 0.007209 | 0.264102 | 0.008163 | 0.011517 |
| PID | 0.138171 | 0.023113 | 0.005732 | 0.005857 | 0.004737 | 0.007926 |
| MPC-lite | 0.052585 | 0.116509 | 0.005181 | 0.093634 | 0.008993 | 0.009492 |
