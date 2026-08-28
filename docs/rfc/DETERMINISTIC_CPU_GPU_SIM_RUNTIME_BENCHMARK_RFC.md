<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Fusion Core — deterministic CPU/GPU-sim runtime benchmark RFC
-->

# Deterministic CPU/GPU-Sim Runtime Benchmark

Status: Implemented local validation surface

## Scope and decision context

This RFC defines the deterministic comparison between the shipped scalar CPU
and vectorised GPU-simulation lanes for multigrid smoothing and SNN inference.
The acceptance metrics are operation-count estimates. Wall times remain
observable diagnostics and are not hardware-neutral performance evidence.

The benchmark does not claim CUDA, ROCm, wgpu, multi-GPU, production-scale,
realtime-control, or accelerator-hardware readiness. Separate hardware-bound
benchmark and provenance gates own those claims.

## Owning surfaces

- Runtime bridge: `src/scpn_fusion/core/gpu_runtime.py`
- Validation CLI: `validation/deterministic_cpu_gpu_sim_runtime_benchmark.py`
- Default local reports:
  - `validation/reports/deterministic_cpu_gpu_sim_runtime_benchmark.json`
  - `validation/reports/deterministic_cpu_gpu_sim_runtime_benchmark.md`
- Tests: `tests/test_deterministic_cpu_gpu_sim_runtime_benchmark.py`

## Data and dependency boundary

- Inputs are deterministic arrays generated locally from fixed numerical
  ranges and seeds.
- The baseline benchmark requires NumPy only.
- Optional JAX and PyTorch compatibility backends are exposed by the runtime
  bridge but do not convert the GPU-sim acceptance lane into hardware evidence.
- No external dataset, device service, network access, or untracked artefact is
  required.

## Metrics and acceptance criteria

- GPU-sim multigrid P95 operation-count estimate: at most `2.0 ms`.
- GPU-sim SNN P95 operation-count estimate: at most `1.0 ms`.
- Estimated multigrid speedup over the scalar CPU lane: at least `4.0x`.
- Estimated SNN speedup over the scalar CPU lane: at least `4.0x`.
- The JSON report uses schema version 2 and the descriptive
  `deterministic_cpu_gpu_sim_runtime_benchmark` identity.

The estimate divides declared operation counts by fixed lane-throughput
surrogates. It is a deterministic regression contract, not a measured device
latency or throughput result. Shared-host wall times are reported separately
and do not determine acceptance.

## Regression and safety contract

- CPU and GPU-sim lanes execute both multigrid and SNN kernels through the
  public runtime bridge.
- Invalid backend names, trial counts, grid sizes, fault parameters and report
  thresholds fail explicitly.
- The equilibrium-latency API records nominal and injected-fault observations.
- Strict CLI mode returns a non-zero status when any configured threshold
  fails.
- The current report validator rejects the obsolete unversioned coded payload.

## Delivery state

- [x] Deterministic CPU and GPU-sim multigrid execution
- [x] Deterministic CPU and GPU-sim SNN execution
- [x] Estimated latency and speedup acceptance contract
- [x] Optional JAX/PyTorch equilibrium compatibility observations
- [x] Versioned JSON and Markdown report contract
- [x] Focused real-surface tests and strict local gates
