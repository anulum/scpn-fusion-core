# SCPN End-to-End Latency Benchmark

- Schema: `scpn-fusion-core.end_to_end_latency_report.v1`
- Status: `accepted_local_reduced_order_latency_report`
- Claim boundary: `Local reduced-order control latency evidence; not physical HIL, FPGA, plant CODAC, or actuator hardware timing.`
- Physical HIL ready: `NO`
- FPGA timing ready: `NO`
- CODAC timing ready: `NO`
- Actuator hardware timing ready: `NO`
- Generated: `2026-11-15T17:05:21.036813+00:00`
- Runtime: `18.659 s`
- Steps: `320`
- SNN runtime backend: `rust`
- Threshold profile: `accelerated`
- Overall pass: `YES`
- SNN full/surrogate p95 ratio: `0.934`

## Digital-Twin Sensor-to-Control Path

Reduced-order sensor-to-control benchmark with a host-side simulated HIL scaffold. This section does not certify physical HIL, FPGA, plant CODAC, or actuator hardware timing.

Taskset-affinity benchmark evidence on logical CPUs 10,11 of a shared workstation; the cores were not reserved and other processes could run on them. Not shielded cpuset or dedicated-runner evidence. GPU and HIL boundaries remain separately scoped.

| Lane | Status | p50 loop [ms] | p95 loop [ms] | p99 loop [ms] | Boundary |
|------|--------|---------------|---------------|---------------|----------|
| Python CPU | measured | 0.041568 | 0.076077 | 0.100267 | taskset_affinity_shared_workstation_cores |
| Rust native | measured | 0.023582 | 0.030349 | 0.034881 | local release binary when measured |
| GPU | measured | 1.562348 | 1.911913 | 2.103508 | Local non-isolated CuPy/CUDA measurement of the same reduced-order sensor-to-control contract; includes host snapshot assembly and output serialisation. |

### Measurement Metadata

- Command: `taskset -c 10,11 env SCPN_BENCHMARK_ISOLATION_METHOD=taskset_affinity_shared_workstation_cores SCPN_BENCHMARK_CPUSET=10,11 SCPN_BENCHMARK_CONCURRENT_HEAVY_JOBS=none_started_by_this_task_other_seats_share_the_host SCPN_BENCHMARK_ACCELERATOR_ISOLATION=shared_local_device_not_reserved SCPN_BENCHMARK_CLAIM_BOUNDARY='Taskset-affinity benchmark evidence on logical CPUs 10,11 of a shared workstation; the cores were not reserved and other processes could run on them. Not shielded cpuset or dedicated-runner evidence. GPU and HIL boundaries remain separately scoped.' .venv/bin/python validation/scpn_end_to_end_latency.py --steps 320 --strict --commit-evidence`
- CPU isolation method: `taskset_affinity_shared_workstation_cores`
- Pinned CPU set: `10,11`
- Accelerator isolation: `shared_local_device_not_reserved`
- Host load before: `[7.8154296875, 11.00244140625, 11.7314453125]`
- Host load after: `[7.8154296875, 11.00244140625, 11.7314453125]`
- CPU governors: `['powersave']`
- Concurrent heavy jobs: `none_started_by_this_task_other_seats_share_the_host`

### Simulated HIL Sensor-to-Actuator Scaffold

Measured host-side simulated ADC/DAC sensor-to-actuator scaffold; not a physical HIL rig, FPGA bitstream, plant CODAC, or actuator hardware timing claim.

| Status | Hardware status | Actuators | p50 [us] | p95 [us] | p99 [us] | Pass |
|--------|-----------------|-----------|----------|----------|----------|------|
| measured_simulated_hil | simulated_host_adc_dac_loop | 256 | 209.979000 | 273.154650 | 336.719570 | YES |

| Scenario | Fallbacks | Safe output rate | p95 [us] | Pass | Reasons |
|----------|-----------|------------------|----------|------|---------|
| nominal | 0 | 1.000 | 273.154650 | YES | none |
| sensor_dropout | 20 | 1.000 | 345.344850 | YES | invalid_sensor_sample_replaced:20 |
| noisy_sensor | 27 | 1.000 | 318.414200 | YES | noisy_sensor_envelope_safe_mode:27 |
| actuator_saturation | 18 | 1.000 | 376.830350 | YES | actuator_saturation_clamped:18 |
| controller_nonfinite | 16 | 1.000 | 380.811950 | YES | controller_nonfinite_fail_closed:16 |

### CPU Pipeline Stages

| Stage | p50 [ms] | p95 [ms] | p99 [ms] |
|-------|----------|----------|----------|
| input_validation | 0.001412 | 0.002254 | 0.002847 |
| feature_assembly | 0.001407 | 0.002685 | 0.003930 |
| digital_twin_update | 0.010135 | 0.018350 | 0.024478 |
| controller_decision | 0.001712 | 0.002943 | 0.003580 |
| fallback_policy | 0.013683 | 0.024273 | 0.045578 |
| output_serialization | 0.009600 | 0.017629 | 0.024760 |

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
| 2 | 0.138533 | 0.032018 | 1.692250 | 3 |
| 16 | 0.112619 | 0.040907 | 2.004491 | 3 |
| 64 | 0.228114 | 0.053792 | 1.632200 | 3 |
| 128 | 0.204335 | 0.049056 | 1.968878 | 3 |
| 256 | 0.298261 | 0.088392 | 1.985865 | 3 |

## Predictive-Horizon Timing

Forecast timing is a reduced-order local digital-twin rollout. It is not a validated 50-100 ms plasma-instability prediction horizon.

| Horizon [ms] | p50 forecast [ms] | p95 forecast [ms] | p99 forecast [ms] | p95 real-time factor | Pass |
|--------------|-------------------|-------------------|-------------------|----------------------|------|
| 50 | 0.590227 | 0.967393 | 1.159521 | 51.685 | YES |
| 100 | 1.132572 | 1.862350 | 2.095646 | 53.696 | YES |

## Surrogate Physics Mode

| Controller | RMSE | p95 loop [ms] | p95 sensor [ms] | p95 controller [ms] | p95 actuator [ms] | p95 physics [ms] |
|------------|------|---------------|------------------|---------------------|-------------------|------------------|
| SNN | 0.074583 | 0.233691 | 0.008367 | 0.212739 | 0.010296 | 0.000887 |
| PID | 0.136634 | 0.023420 | 0.006730 | 0.008048 | 0.006493 | 0.000406 |
| MPC-lite | 0.051153 | 0.147931 | 0.007366 | 0.126487 | 0.014663 | 0.000656 |

## Full Physics Mode

| Controller | RMSE | p95 loop [ms] | p95 sensor [ms] | p95 controller [ms] | p95 actuator [ms] | p95 physics [ms] |
|------------|------|---------------|------------------|---------------------|-------------------|------------------|
| SNN | 0.074439 | 0.218182 | 0.008934 | 0.187097 | 0.009140 | 0.011676 |
| PID | 0.138171 | 0.033363 | 0.007653 | 0.008233 | 0.006567 | 0.007821 |
| MPC-lite | 0.052585 | 0.145195 | 0.007268 | 0.116628 | 0.011503 | 0.009989 |
