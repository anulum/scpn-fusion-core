<!--
SCPN Fusion Core — RFC GPHY-04
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# RFC GPHY-04 — Reduced IGA/NURBS Boundary Lane

Status: Draft for review

## 1. Task Identity

- Task ID: `GPHY-04`
- Title: Add NURBS-based boundary sampling primitives for smooth reactor-wall geometry
- Owner: Unassigned
- Proposed sprint: Phase 2, Sprint S2

## 2. Problem Statement

Polygonal boundary approximations can introduce artificial geometric artifacts. This task adds a reduced isogeometric lane with NURBS curve evaluation/sampling and integrates it into first-wall geometry generation to provide smoother boundary contours for physics pipelines.

## 3. Target Paths

- Rust module(s):
  - `scpn-fusion-rs/crates/fusion-math/src/iga.rs`
  - `scpn-fusion-rs/crates/fusion-math/src/lib.rs`
  - `scpn-fusion-rs/crates/fusion-nuclear/src/wall_interaction.rs`
- Validation path:
  - Unit tests in `iga.rs` and `wall_interaction.rs`

## 4. Data and Dependency Readiness

- External dataset(s): none.
- License and usage confirmation: deterministic synthetic control points only.
- External dependencies: none beyond existing crate stack.
- Offline fallback path: existing analytic wall generator remains available.

## 5. Metrics and Acceptance Criteria

- Primary metrics:
  - Valid NURBS endpoint evaluation and uniform sampling.
  - Closed NURBS first-wall contour generation.
  - NURBS wall envelope remains near analytic D-shape envelope.
- Minimum acceptable threshold:
  - NURBS contour closure error below tolerance.
  - Finite sampled geometry values.
  - Envelope deviation bounded by configured regression tolerance.
- Runtime budget:
  - `cargo test -p fusion-math` and `cargo test -p fusion-nuclear` within CI budget.

## 6. Regression and Safety Plan

- CI jobs to update:
  - Ensure `fusion-math` and `fusion-nuclear` tests include NURBS/IGA checks.
- Property/invariant tests:
  - Knot monotonicity and expected knot-vector length.
  - Endpoint and sample-size invariants.
  - Closed-loop and envelope sanity checks for NURBS wall contour.
- Failure rollback strategy:
  - Keep NURBS generator additive; legacy analytic path unchanged.

## 7. Delivery Plan

- Milestone 1:
  - Add NURBS curve and open-uniform knot helpers.
- Milestone 2:
  - Add sampling/evaluation tests in `fusion-math`.
- Milestone 3:
  - Add NURBS first-wall generator and geometric regression tests in `fusion-nuclear`.

## 8. Go/No-Go Checklist

- [x] Path mapping reviewed
- [x] Data/license approved (synthetic control-point scope)
- [x] Metrics protocol approved
- [x] CI cost estimated
- [ ] Owner committed
