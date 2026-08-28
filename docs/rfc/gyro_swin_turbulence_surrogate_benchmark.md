<!--
SCPN Fusion Core — GyroSwin-Like Turbulence Surrogate Benchmark RFC
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# RFC — GyroSwin-Like Turbulence Surrogate Benchmark

Status: Draft for review

## Scope and decision context

This RFC is a planning contract for an explicit implementation lane in SCPN-FUSION-CORE. It records the intended scope, target modules, metrics, and acceptance path before work is promoted to default behavior or external release claims.

## 1. Benchmark identity

- Responsibility: deterministic GyroSwin-like turbulence surrogate benchmark
- Owner: Unassigned
- Proposed sprint: Phase 2, Sprint S2

## 2. Problem Statement

Current transport stack includes analytic and neural surrogate paths, but there is no deterministic CI lane that tests the specific advanced-track goal: high-speed surrogate inference with bounded turbulence error. This RFC defines a synthetic, reproducible benchmark lane that can run offline now while external dataset and licensing gates are finalized.

## 3. Target Paths

- Python module(s):
  - `src/scpn_fusion/core/gyro_swin_surrogate.py`
  - `src/scpn_fusion/core/neural_transport.py` (consumer context only; no hard dependency changes required)
- Validation/reporting path:
  - `validation/gyro_swin_turbulence_surrogate_benchmark.py`
  - `validation/reports/gyro_swin_turbulence_surrogate_benchmark.json`
  - `validation/reports/gyro_swin_turbulence_surrogate_benchmark.md`
  - `tests/test_gyro_swin_turbulence_surrogate_benchmark.py`
- Rust path mapping for later phase:
  - `scpn-fusion-rs/crates/fusion-core/src/transport.rs`
  - `scpn-fusion-rs/crates/fusion-physics/src/turbulence.rs`

## 4. Data and Dependency Readiness

- External dataset(s): not required for v1 (synthetic benchmark only).
- License and usage confirmation: synthetic generator only (no external data license obligations).
- External repository dependencies: none required for v1.
- Offline fallback path: benchmark remains fully offline (`numpy`, existing project modules).

## 5. Metrics and Acceptance Criteria

- Primary metrics:
  - RMSE (% of mean target turbulence coefficient) for surrogate vs synthetic reference.
  - Per-sample speedup of surrogate vs deterministic "GENE-like proxy" baseline.
- Minimum acceptable threshold:
  - RMSE <= `10%`.
  - Speedup >= `1000x`.
- Runtime budget:
  - Validation smoke run <= `30 s` on GitHub-hosted runner.

## 6. Regression and Safety Plan

- CI jobs to update:
  - Add a GyroSwin turbulence benchmark smoke invocation in the existing Python
    test lane (targeted pytest plus strict validation CLI).
- Property/invariant tests:
  - Deterministic dataset generation for fixed seed.
  - Finite positive surrogate outputs.
  - Stable threshold pass on smoke configuration.
- Failure rollback strategy:
  - Keep additive-only implementation.
  - Do not modify default transport solver behavior.

## 7. Delivery Plan

- Milestone 1:
  - Implement synthetic target law + deterministic dataset generator.
- Milestone 2:
  - Implement lightweight GyroSwin-like surrogate fit/predict path and speed benchmark harness.
- Milestone 3:
  - Add validation report script, tests, and tracker updates.

## 8. Go/No-Go Criteria

- [x] Path mapping reviewed
- [x] Data/license approved (synthetic-only v1 scope)
- [x] Metrics protocol approved (RMSE + speedup thresholds)
- [x] CI cost estimated
- [ ] Owner committed
