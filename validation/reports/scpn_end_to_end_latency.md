# SCPN End-to-End Latency Benchmark

- Schema: `scpn-fusion-core.end_to_end_latency_report.v1`
- Status: `accepted_local_reduced_order_latency_report`
- Claim boundary: `Local reduced-order control latency evidence; not physical HIL, FPGA, plant CODAC, or actuator hardware timing.`
- Physical HIL ready: `NO`
- FPGA timing ready: `NO`
- CODAC timing ready: `NO`
- Actuator hardware timing ready: `NO`
- Generated: `2026-06-29T21:05:14.232207+00:00`
- Runtime: `63.127 s`
- Steps: `320`
- SNN runtime backend: `rust`
- Threshold profile: `accelerated`
- Overall pass: `YES`
- SNN full/surrogate p95 ratio: `0.813`

## Digital-Twin Sensor-to-Control Path

Reduced-order sensor-to-control benchmark with a host-side simulated HIL scaffold. This section does not certify physical HIL, FPGA, plant CODAC, or actuator hardware timing.

Python CPU and Rust rows are local non-isolated wall-clock measurements. GPU rows are accelerator measurements only when status is measured. HIL rows are simulated host ADC/DAC timing unless hardware_status names a physical rig.

| Lane | Status | p50 loop [ms] | p95 loop [ms] | p99 loop [ms] | Boundary |
|------|--------|---------------|---------------|---------------|----------|
| Python CPU | measured | 0.049953 | 0.134760 | 0.154150 | non_isolated_local_workstation |
| Rust native | measured | 0.052549 | 0.115893 | 1.811680 | local release binary when measured |
| GPU | measured | 2.481406 | 8.575209 | 16.886498 | Local non-isolated CuPy/CUDA measurement of the same reduced-order sensor-to-control contract; includes host snapshot assembly and output serialisation. |

### Measurement Metadata

- Command: `not_recorded`
- CPU isolation method: `non_isolated_local_workstation`
- Reserved CPU set: `not_recorded`
- Host load before: `[27.81103515625, 26.12939453125, 25.36279296875]`
- Host load after: `[27.81103515625, 26.12939453125, 25.36279296875]`
- CPU governors: `['powersave']`
- Concurrent heavy jobs: `not_recorded`

### Simulated HIL Sensor-to-Actuator Scaffold

Measured host-side simulated ADC/DAC sensor-to-actuator scaffold; not a physical HIL rig, FPGA bitstream, plant CODAC, or actuator hardware timing claim.

| Status | Hardware status | Actuators | p50 [us] | p95 [us] | p99 [us] | Pass |
|--------|-----------------|-----------|----------|----------|----------|------|
| measured_simulated_hil | simulated_host_adc_dac_loop | 256 | 216.513500 | 395.166000 | 500.434190 | YES |

| Scenario | Fallbacks | Safe output rate | p95 [us] | Pass | Reasons |
|----------|-----------|------------------|----------|------|---------|
| nominal | 0 | 1.000 | 395.166000 | YES | none |
| sensor_dropout | 20 | 1.000 | 440.465450 | YES | invalid_sensor_sample_replaced:20 |
| noisy_sensor | 27 | 1.000 | 456.901600 | YES | noisy_sensor_envelope_safe_mode:27 |
| actuator_saturation | 18 | 1.000 | 393.388550 | YES | actuator_saturation_clamped:18 |
| controller_nonfinite | 16 | 1.000 | 398.435850 | YES | controller_nonfinite_fail_closed:16 |

### CPU Pipeline Stages

| Stage | p50 [ms] | p95 [ms] | p99 [ms] |
|-------|----------|----------|----------|
| input_validation | 0.001496 | 0.003270 | 0.004041 |
| feature_assembly | 0.001778 | 0.005226 | 0.007960 |
| digital_twin_update | 0.010416 | 0.028516 | 0.035042 |
| controller_decision | 0.001886 | 0.005237 | 0.006340 |
| fallback_policy | 0.019286 | 0.055997 | 0.062685 |
| output_serialization | 0.011623 | 0.030747 | 0.037170 |

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
| 2 | 0.193261 | 0.143219 | 1.313268 | 3 |
| 16 | 0.104858 | 0.034980 | 5.021439 | 3 |
| 64 | 0.246943 | 0.118186 | 3.753956 | 3 |
| 128 | 1.292813 | 0.474949 | 1.563021 | 3 |
| 256 | 0.493068 | 0.052667 | 15.660111 | 3 |

## Predictive-Horizon Timing

Forecast timing is a reduced-order local digital-twin rollout. It is not a validated 50-100 ms plasma-instability prediction horizon.

| Horizon [ms] | p50 forecast [ms] | p95 forecast [ms] | p99 forecast [ms] | p95 real-time factor | Pass |
|--------------|-------------------|-------------------|-------------------|----------------------|------|
| 50 | 1.192322 | 2.871309 | 9.860543 | 17.414 | YES |
| 100 | 2.405479 | 5.350235 | 6.316376 | 18.691 | YES |

## Surrogate Physics Mode

| Controller | RMSE | p95 loop [ms] | p95 sensor [ms] | p95 controller [ms] | p95 actuator [ms] | p95 physics [ms] |
|------------|------|---------------|------------------|---------------------|-------------------|------------------|
| SNN | 0.074583 | 0.402123 | 0.011659 | 0.364727 | 0.013818 | 0.001063 |
| PID | 0.136634 | 0.038902 | 0.012666 | 0.013785 | 0.010935 | 0.001041 |
| MPC-lite | 0.051153 | 0.206647 | 0.015067 | 0.159759 | 0.030990 | 0.001196 |

## Full Physics Mode

| Controller | RMSE | p95 loop [ms] | p95 sensor [ms] | p95 controller [ms] | p95 actuator [ms] | p95 physics [ms] |
|------------|------|---------------|------------------|---------------------|-------------------|------------------|
| SNN | 0.074439 | 0.327030 | 0.012684 | 0.283431 | 0.013579 | 0.022949 |
| PID | 0.138171 | 0.035547 | 0.008219 | 0.008824 | 0.006871 | 0.011098 |
| MPC-lite | 0.052585 | 0.159016 | 0.008497 | 0.122391 | 0.014298 | 0.013899 |
