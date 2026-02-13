<!--
SCPN Fusion Core — RFC GPHY-02
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# RFC GPHY-02 — Symplectic Integrator Baseline in fusion-math

Status: Draft for review

## 1. Task Identity

- Task ID: `GPHY-02`
- Title: Add velocity-Verlet symplectic trajectory lane for canonical Hamiltonian systems
- Owner: Unassigned
- Proposed sprint: Phase 2, Sprint S2

## 2. Problem Statement

Long-horizon plasma/control trajectories can accumulate integration drift when using non-symplectic steppers. This task introduces a deterministic symplectic baseline in `fusion-math` with explicit energy-drift regression checks and RK4 reference comparison on coarse-step long-horizon runs.

## 3. Target Paths

- Rust module(s):
  - `scpn-fusion-rs/crates/fusion-math/src/symplectic.rs`
  - `scpn-fusion-rs/crates/fusion-math/src/lib.rs`
- Validation path:
  - Unit tests in `symplectic.rs`

## 4. Data and Dependency Readiness

- External dataset(s): none.
- License and usage confirmation: analytic synthetic Hamiltonian trajectories only.
- External dependencies: none beyond existing crate stack.
- Offline fallback path: deterministic local trajectory integration and drift checks.

## 5. Metrics and Acceptance Criteria

- Primary metrics:
  - Long-horizon Hamiltonian drift under velocity-Verlet.
  - Symplectic drift comparison against RK4 at coarse step.
- Minimum acceptable threshold:
  - Bounded drift on long horizon (`dt=0.3`, `N=5000`) under configured tolerance.
  - `max_drift(verlet) < max_drift(rk4)` for coarse-step regression profile.
- Runtime budget:
  - `cargo test -p fusion-math` within standard CI budget.

## 6. Regression and Safety Plan

- CI jobs to update:
  - Ensure `fusion-math` tests execute symplectic module checks.
- Property/invariant tests:
  - `dt=0` identity.
  - Energy drift bounded on long horizon.
  - Symplectic drift lower than RK4 drift under coarse-step stress profile.
- Failure rollback strategy:
  - Keep module additive; no replacement of existing solver paths by default.

## 7. Delivery Plan

- Milestone 1:
  - Add canonical state and Hamiltonian-system trait.
- Milestone 2:
  - Add velocity-Verlet and RK4 steppers plus trajectory helpers.
- Milestone 3:
  - Add drift metrics and long-horizon regression tests.

## 8. Go/No-Go Checklist

- [x] Path mapping reviewed
- [x] Data/license approved (analytic synthetic scope)
- [x] Metrics protocol approved
- [x] CI cost estimated
- [ ] Owner committed
