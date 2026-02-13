<!--
SCPN Fusion Core — RFC GDEP-02
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# RFC GDEP-02 — GPU Runtime Integration Bridge

Status: Draft for review

## 1. Task Identity

- Task ID: `GDEP-02`
- Title: Add deterministic GPU-sim runtime bridge for multigrid + SNN lanes
- Owner: Unassigned
- Proposed sprint: Phase 2, Sprint S2

## 2. Problem Statement

The deployment lane needs a reproducible interface for comparing CPU and GPU-like runtimes across key workloads (multigrid solve and SNN inference). This RFC introduces a deterministic runtime bridge with estimated latency/speedup contracts and wall-time observability.

## 3. Target Paths

- Python module(s):
  - `src/scpn_fusion/core/gpu_runtime.py`
  - `src/scpn_fusion/core/__init__.py`
- Validation/reporting path:
  - `validation/gdep_02_gpu_integration.py`
  - `validation/reports/gdep_02_gpu_integration.json`
  - `validation/reports/gdep_02_gpu_integration.md`
  - `tests/test_gdep_02_gpu_integration.py`

## 4. Data and Dependency Readiness

- External dataset(s): none required.
- External dependencies: none beyond NumPy.
- Offline fallback path: deterministic CPU and `gpu_sim` lanes run locally.

## 5. Metrics and Acceptance Criteria

- Primary metrics:
  - GPU-sim P95 estimate for multigrid lane.
  - GPU-sim P95 estimate for SNN inference lane.
  - Estimated speedups vs CPU baseline.
- Minimum acceptable threshold:
  - Multigrid P95 estimate <= `2.0 ms`.
  - SNN P95 estimate <= `1.0 ms`.
  - Both speedups >= `4x`.
- Runtime budget:
  - Strict validation <= `30 s`.

## 6. Regression and Safety Plan

- CI jobs to update:
  - Add targeted pytest and strict GDEP-02 validation invocation.
- Property/invariant tests:
  - GPU-sim estimates are consistently faster than CPU estimates.
  - Validation report includes both estimated and wall metrics.
- Failure rollback strategy:
  - Keep runtime bridge additive; no default solver path replacement.

## 7. Delivery Plan

- Milestone 1:
  - Add CPU and GPU-sim execution kernels for multigrid and SNN.
- Milestone 2:
  - Add deterministic latency/speedup estimation and benchmark API.
- Milestone 3:
  - Add validation report and tests.

## 8. Go/No-Go Checklist

- [x] Path mapping reviewed
- [x] Metrics protocol approved
- [x] CI cost estimated
- [ ] Owner committed
