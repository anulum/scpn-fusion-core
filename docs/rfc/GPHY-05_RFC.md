<!--
SCPN Fusion Core — RFC GPHY-05
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# RFC GPHY-05 — Latency-Aware Control and Stochastic Noise Layer

Status: Draft for review

## 1. Task Identity

- Task ID: `GPHY-05`
- Title: Extend control lane with vector OU noise, actuator delay-line, and lag-aware MPC rollout
- Owner: Unassigned
- Proposed sprint: Phase 2, Sprint S2

## 2. Problem Statement

Controller performance degrades when latency and stochastic perturbations are ignored. This task adds deterministic noise and delay abstractions in the Rust control stack, including a reduced DDE-like lagged-delay MPC rollout path.

## 3. Target Paths

- Rust module(s):
  - `scpn-fusion-rs/crates/fusion-control/src/digital_twin.rs`
  - `scpn-fusion-rs/crates/fusion-control/src/mpc.rs`
- Validation path:
  - Unit tests in `digital_twin.rs` and `mpc.rs`

## 4. Data and Dependency Readiness

- External dataset(s): none.
- License and usage confirmation: synthetic deterministic test scenarios only.
- External dependencies: none beyond existing crate stack.
- Offline fallback path: additive APIs; existing controller paths remain available.

## 5. Metrics and Acceptance Criteria

- Primary metrics:
  - Deterministic vector OU noise under fixed seed.
  - Explicit delay-line behavior with controllable lag dynamics.
  - Delay-aware MPC action generation with clipping and directional correction.
- Minimum acceptable threshold:
  - Seeded vector OU channels produce reproducible sequences.
  - Delay-line tests verify delayed and lagged action application.
  - Delay-aware MPC tests maintain bounded output and target-directed actuation.
- Runtime budget:
  - `cargo test -p fusion-control` within standard CI budget.

## 6. Regression and Safety Plan

- CI jobs to update:
  - Ensure `fusion-control` tests include new delay/noise paths.
- Property/invariant tests:
  - Delay-line shape guard and reset behavior.
  - OU noise channel determinism.
  - MPC output clipping under delay-aware rollout.
- Failure rollback strategy:
  - Keep new methods additive with no default behavior replacement.

## 7. Delivery Plan

- Milestone 1:
  - Add vector OU noise abstraction and actuator delay-line with lag.
- Milestone 2:
  - Add delay-dynamics MPC rollout method.
- Milestone 3:
  - Add deterministic regression tests for new paths.

## 8. Go/No-Go Checklist

- [x] Path mapping reviewed
- [x] Data/license approved (synthetic deterministic scope)
- [x] Metrics protocol approved
- [x] CI cost estimated
- [ ] Owner committed
