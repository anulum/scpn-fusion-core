# SCPN End-to-End Latency Benchmark

- Generated: `2026-06-17T17:52:57.038434+00:00`
- Runtime: `13.794 s`
- Steps: `320`
- SNN runtime backend: `numpy`
- Threshold profile: `numpy_fallback`
- Overall pass: `YES`
- SNN full/surrogate p95 ratio: `0.738`

## Digital-Twin Sensor-to-Control Path

Taskset-affinity benchmark evidence on operator-reserved logical CPUs 10,11; not shielded cpuset or dedicated-runner evidence. GPU and HIL boundaries remain separately scoped.

| Lane | Status | p50 loop [ms] | p95 loop [ms] | p99 loop [ms] | Boundary |
|------|--------|---------------|---------------|---------------|----------|
| Python CPU | measured | 0.034281 | 0.053408 | 0.062749 | taskset_affinity_operator_reserved_cores |
| Rust native | measured | 0.019581 | 0.020284 | 0.024611 | local release binary when measured |
| GPU | measured | 0.710571 | 0.840560 | 1.011992 | Local non-isolated CuPy/CUDA measurement of the same reduced-order sensor-to-control contract; includes host snapshot assembly and output serialisation. |

### Measurement Metadata

- Command: `taskset -c 10,11 env SCPN_BENCHMARK_ISOLATION_METHOD=taskset_affinity_operator_reserved_cores SCPN_BENCHMARK_CPUSET=10,11 SCPN_BENCHMARK_CONCURRENT_HEAVY_JOBS=none_intentionally_started_by_this_task SCPN_BENCHMARK_CLAIM_BOUNDARY='Taskset-affinity benchmark evidence on operator-reserved logical CPUs 10,11; not shielded cpuset or dedicated-runner evidence. GPU and HIL boundaries remain separately scoped.' .venv/bin/python validation/scpn_end_to_end_latency.py --steps 320 --strict`
- CPU isolation method: `taskset_affinity_operator_reserved_cores`
- Reserved CPU set: `10,11`
- Host load before: `[7.2265625, 5.3642578125, 4.83251953125]`
- Host load after: `[7.2265625, 5.3642578125, 4.83251953125]`
- CPU governors: `['powersave']`
- Concurrent heavy jobs: `none_intentionally_started_by_this_task`

### Simulated HIL Sensor-to-Actuator Scaffold

Measured host-side simulated ADC/DAC sensor-to-actuator scaffold; not a physical HIL rig, FPGA bitstream, plant CODAC, or actuator hardware timing claim.

| Status | Hardware status | Actuators | p50 [us] | p95 [us] | p99 [us] | Pass |
|--------|-----------------|-----------|----------|----------|----------|------|
| measured_simulated_hil | simulated_host_adc_dac_loop | 256 | 167.140500 | 232.522000 | 294.771870 | YES |

| Scenario | Fallbacks | Safe output rate | p95 [us] | Pass | Reasons |
|----------|-----------|------------------|----------|------|---------|
| nominal | 0 | 1.000 | 232.522000 | YES | none |
| sensor_dropout | 20 | 1.000 | 266.350700 | YES | invalid_sensor_sample_replaced:20 |
| noisy_sensor | 27 | 1.000 | 226.725950 | YES | noisy_sensor_envelope_safe_mode:27 |
| actuator_saturation | 18 | 1.000 | 198.920100 | YES | actuator_saturation_clamped:18 |
| controller_nonfinite | 16 | 1.000 | 223.609450 | YES | controller_nonfinite_fail_closed:16 |

### CPU Pipeline Stages

| Stage | p50 [ms] | p95 [ms] | p99 [ms] |
|-------|----------|----------|----------|
| input_validation | 0.001066 | 0.001517 | 0.002063 |
| feature_assembly | 0.001138 | 0.001939 | 0.002616 |
| digital_twin_update | 0.007461 | 0.012753 | 0.015234 |
| controller_decision | 0.001438 | 0.002172 | 0.002712 |
| fallback_policy | 0.013327 | 0.021719 | 0.024084 |
| output_serialization | 0.007922 | 0.013405 | 0.016184 |

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
| 2 | 0.070917 | 0.021200 | 0.859985 | 3 |
| 16 | 0.083223 | 0.023589 | 0.848174 | 3 |
| 64 | 0.114598 | 0.038520 | 1.046033 | 3 |
| 128 | 0.133026 | 0.052389 | 0.873599 | 3 |
| 256 | 0.219528 | 0.059727 | 0.885944 | 3 |

## Predictive-Horizon Timing

Forecast timing is a reduced-order local digital-twin rollout. It is not a validated 50-100 ms plasma-instability prediction horizon.

| Horizon [ms] | p50 forecast [ms] | p95 forecast [ms] | p99 forecast [ms] | p95 real-time factor | Pass |
|--------------|-------------------|-------------------|-------------------|----------------------|------|
| 50 | 0.404858 | 0.538453 | 0.632840 | 92.859 | YES |
| 100 | 0.794599 | 0.925584 | 1.022272 | 108.040 | YES |

## Surrogate Physics Mode

| Controller | RMSE | p95 loop [ms] | p95 sensor [ms] | p95 controller [ms] | p95 actuator [ms] | p95 physics [ms] |
|------------|------|---------------|------------------|---------------------|-------------------|------------------|
| SNN | 0.074583 | 0.254913 | 0.005889 | 0.235810 | 0.007110 | 0.000484 |
| PID | 0.136634 | 0.009763 | 0.002848 | 0.003428 | 0.002717 | 0.000165 |
| MPC-lite | 0.051153 | 0.067794 | 0.003191 | 0.057310 | 0.005488 | 0.000266 |

## Full Physics Mode

| Controller | RMSE | p95 loop [ms] | p95 sensor [ms] | p95 controller [ms] | p95 actuator [ms] | p95 physics [ms] |
|------------|------|---------------|------------------|---------------------|-------------------|------------------|
| SNN | 0.074439 | 0.188081 | 0.004273 | 0.169782 | 0.004821 | 0.007095 |
| PID | 0.138171 | 0.015783 | 0.003223 | 0.003439 | 0.002925 | 0.004200 |
| MPC-lite | 0.052585 | 0.070495 | 0.003404 | 0.056565 | 0.004726 | 0.005281 |
