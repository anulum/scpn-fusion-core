# Task 2 Surrogate + Benchmark Report

- Generated: `2026-02-15T23:29:50.741355+00:00`
- Runtime: `0.694 s`
- Overall pass: `YES`

## Disruption Predictor AUC

- TM1 proxy AUC: `0.9919` (threshold `>= 0.95`)
- TokamakNET proxy AUC: `0.9772` (threshold `>= 0.95`)
- Published standard: `TM1/TokamakNET proxy` (published=`True`)

## Pretrained Surrogates

- MLP RMSE: `13.489%` (threshold `<= 20.0%`)
- FNO eval relative L2: `0.7925` (threshold `<= 0.80`)
- Pretrained coverage: `28.6%` of listed surrogate lanes
- Surrogates requiring user training: `5`

## Equilibrium Latency (10x Fault Runs)

- Backend: `torch_fallback`
- P95 estimate: `0.0029 ms` (threshold `<= 1.0 ms`)
- Fault P95 estimate: `0.0031 ms` (threshold `<= 1.0 ms`)
- P95 wall latency: `0.3226 ms` (threshold `<= 10.0 ms`)
- Fault P95 wall latency: `0.5104 ms` (threshold `<= 10.0 ms`)

## Consumer Hardware Latency Profiles

- `host_measured_runtime` backend=`torch_fallback` p95_est=`0.0029 ms` p95_wall=`0.3226 ms`
- `cpu_reference_model` projected p95_est=`0.0587 ms` source=`Deterministic reference throughput used by GPURuntimeBridge CPU estimate.`
- `gpu_sim_reference_model` projected p95_est=`0.0047 ms` source=`Deterministic reference throughput used by GPURuntimeBridge GPU-sim estimate.`
- `consumer_rtx_3060_projected` projected p95_est=`0.0021 ms` source=`Projected throughput for consumer RTX 3060-class hardware.`
- `consumer_rtx_4090_projected` projected p95_est=`0.0007 ms` source=`Projected throughput for consumer RTX 4090-class hardware.`
