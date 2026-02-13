<!--
SCPN Fusion Core — RFC GPHY-01
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# RFC GPHY-01 — Reduced Particle Tracker Feedback (Boris + J_phi Overlay)

Status: Draft for review

## 1. Task Identity

- Task ID: `GPHY-01`
- Title: Add reduced hybrid particle-current feedback path to Grad-Shafranov kernel
- Owner: Unassigned
- Proposed sprint: Phase 2, Sprint S2

## 2. Problem Statement

Current equilibrium source updates are fluid-only. This task introduces a deterministic particle overlay lane with a Boris pusher and toroidal-current deposition so particle-derived `J_phi` can be blended into the Grad-Shafranov source term each Picard step.

## 3. Target Paths

- Rust module(s):
  - `scpn-fusion-rs/crates/fusion-core/src/particles.rs`
  - `scpn-fusion-rs/crates/fusion-core/src/kernel.rs`
  - `scpn-fusion-rs/crates/fusion-core/src/lib.rs`
- Validation path:
  - Rust unit tests in `particles.rs` and `kernel.rs`

## 4. Data and Dependency Readiness

- External dataset(s): none.
- License and usage confirmation: synthetic deterministic particle states only.
- External dependencies: none beyond existing crate stack.
- Offline fallback path: particle overlay remains optional and disabled by default.

## 5. Metrics and Acceptance Criteria

- Primary metrics:
  - Boris no-electric-field speed invariance check.
  - Non-zero toroidal current deposition from moving particles.
  - Feedback blend preserves target total plasma current.
- Minimum acceptable threshold:
  - Relative Boris speed drift <= `1e-9` (test target set to `1e-10`).
  - Deposited toroidal current map has finite non-zero support.
  - Blended `J_phi` integral matches `I_target` within numerical tolerance.
- Runtime budget:
  - `cargo test -p fusion-core` under standard CI budget.

## 6. Regression and Safety Plan

- CI jobs to update:
  - Ensure `fusion-core` unit tests include particle-overlay checks.
- Property/invariant tests:
  - Shape guard for particle current map mismatch.
  - Coupling clamp to `[0, 1]`.
  - Optional feedback path does not alter default behavior when disabled.
- Failure rollback strategy:
  - Keep overlay additive with explicit setter/clear APIs.

## 7. Delivery Plan

- Milestone 1:
  - Add charged-particle state and Boris pusher primitives.
- Milestone 2:
  - Add toroidal current deposition and current-blend helper.
- Milestone 3:
  - Wire optional particle feedback into kernel Picard loop with tests.

## 8. Go/No-Go Checklist

- [x] Path mapping reviewed
- [x] Data/license approved (synthetic-only v1 scope)
- [x] Metrics protocol approved
- [x] CI cost estimated
- [ ] Owner committed
