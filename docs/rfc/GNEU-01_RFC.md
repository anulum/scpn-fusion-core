<!--
SCPN Fusion Core — RFC GNEU-01
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# RFC GNEU-01 — SNN vs RL Tearing-Mode Baseline

Status: Draft for review

## 1. Task Identity

- Task ID: `GNEU-01`
- Title: Benchmark SNN compiler against DIII-D-style RL baseline for tearing-mode avoidance
- Owner: Unassigned
- Proposed sprint: Phase 2, Sprint S2

## 2. Problem Statement

Current control/disruption paths include ML and SNN-oriented components but do not provide a direct, reproducible benchmark against an RL baseline under synthetic fault/noise stress. This RFC defines a constrained benchmark lane that can run in CI without external heavy dependencies and establishes acceptance criteria before adding larger data integrations.

## 3. Target Paths

- Python module(s):
  - `src/scpn_fusion/scpn/compiler.py`
  - `src/scpn_fusion/scpn/controller.py`
  - `src/scpn_fusion/control/disruption_predictor.py`
- Rust crate/module(s):
  - `scpn-fusion-rs/crates/fusion-ml/src/disruption.rs` (optional parity metric path)
- Validation/reporting path:
  - `validation/control_resilience_campaign.py`
  - `validation/reports/gneu_01_benchmark.json`
  - `validation/reports/gneu_01_benchmark.md`
  - `tests/test_gneu_01_benchmark.py`

## 4. Data and Dependency Readiness

- External dataset(s): none required for v1 benchmark; use deterministic synthetic tearing-mode traces.
- License and usage confirmation: synthetic generator only (no external license constraints).
- External repository dependencies: none for v1.
- Offline fallback path: benchmark remains fully offline and deterministic (`numpy` + existing repo modules).

## 5. Metrics and Acceptance Criteria

- Primary metrics:
  - SNN-vs-RL label agreement on disruption risk decision boundary.
  - Absolute risk-score difference distribution.
  - Fault-recovery behavior under injected bit flips and additive noise.
- Baseline comparison:
  - RL baseline is a lightweight deterministic policy head over the same feature vector.
  - SNN path is the compiled/control path from existing `scpn` modules.
- Minimum acceptable threshold:
  - Agreement >= `0.95` on campaign samples.
  - Mean absolute risk delta <= `0.08`.
  - P95 recovery steps <= configured recovery window.
- Runtime budget:
  - CI benchmark job <= 60 seconds on GitHub-hosted runner.

## 6. Regression and Safety Plan

- CI jobs to update:
  - Extend CI with `GNEU-01 benchmark smoke` (single deterministic seed).
- Property/invariant tests:
  - Reproducibility under fixed seed.
  - Risk score bounded in `[0, 1]`.
  - Fault injection never produces NaN/inf risk output.
- Numerical stability checks:
  - Validate campaign metrics remain finite.
  - Clamp/guard feature extraction for extreme inputs.
- Failure rollback strategy:
  - Keep benchmark additive; do not alter existing default runtime/control behavior.
  - Feature-flag new benchmark path if instability appears.

## 7. Delivery Plan

- Milestone 1:
  - Add deterministic RL baseline scorer and shared feature extraction bridge.
- Milestone 2:
  - Add SNN-vs-RL benchmark runner + JSON/Markdown report output.
- Milestone 3:
  - Add CI smoke + regression tests and document expected thresholds.

## 8. Go/No-Go Checklist

- [x] Path mapping reviewed
- [x] Data/license approved (synthetic-only v1 scope)
- [x] Metrics protocol approved (defined in this RFC)
- [x] CI cost estimated
- [ ] Owner committed
