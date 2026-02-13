<!--
SCPN Fusion Core — RFC GPHY-03
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# RFC GPHY-03 — Reduced Non-LTE Collisional-Radiative Lookup (ADAS-style)

Status: Draft for review

## 1. Task Identity

- Task ID: `GPHY-03`
- Title: Add charge-state-resolved collisional-radiative emissivity lookup for impurity cooling
- Owner: Unassigned
- Proposed sprint: Phase 2, Sprint S2

## 2. Problem Statement

Existing wall/impurity handling lacks explicit charge-state radiative differentiation. This task introduces a deterministic ADAS-style reduced lookup path for PEC values and derived radiative losses, enabling non-LTE-like impurity cooling estimates without external runtime dependencies.

## 3. Target Paths

- Rust module(s):
  - `scpn-fusion-rs/crates/fusion-nuclear/src/wall_interaction.rs`
- Validation path:
  - Unit tests in `wall_interaction.rs`

## 4. Data and Dependency Readiness

- External dataset(s): none for v1.
- License and usage confirmation: synthetic reduced lookup tables bundled in source.
- External dependencies: none beyond existing crate stack.
- Offline fallback path: deterministic in-repo lookup/interpolation only.

## 5. Metrics and Acceptance Criteria

- Primary metrics:
  - Distinct PEC values by impurity charge state (e.g., W20+ vs W40+).
  - Finite positive collisional-radiative power density outputs.
  - Stable behavior for unsupported charge states (`None` contract).
- Minimum acceptable threshold:
  - `PEC(W40+) > PEC(W20+)` at same reference plasma point in regression test.
  - Radiative power/loss outputs finite and positive for supported states.
  - Unsupported states do not panic and return `None`.
- Runtime budget:
  - `cargo test -p fusion-nuclear` within standard CI budget.

## 6. Regression and Safety Plan

- CI jobs to update:
  - Ensure `fusion-nuclear` tests include CR lookup checks.
- Property/invariant tests:
  - Bilinear interpolation bounds via clamped axis bracketing.
  - Unsupported lookup states return `None`.
  - Output remains finite for positive physical inputs.
- Failure rollback strategy:
  - Keep CR model additive; existing wall-interaction behavior preserved.

## 7. Delivery Plan

- Milestone 1:
  - Add reduced PEC table definitions for selected impurity/charge states.
- Milestone 2:
  - Add lookup and bilinear interpolation helpers.
- Milestone 3:
  - Add radiative power/loss helpers and regression tests.

## 8. Go/No-Go Checklist

- [x] Path mapping reviewed
- [x] Data/license approved (synthetic reduced table scope)
- [x] Metrics protocol approved
- [x] CI cost estimated
- [ ] Owner committed
