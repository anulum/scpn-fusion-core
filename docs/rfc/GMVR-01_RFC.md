<!--
SCPN Fusion Core — RFC GMVR-01
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# RFC GMVR-01 — Compact MVR Constraint Normalization

Status: Draft for review

## 1. Task Identity

- Task ID: `GMVR-01`
- Title: Update compact-reactor scanner with 2025/2026-style engineering caps (divertor, Zeff, HTS peak field)
- Owner: Unassigned
- Proposed sprint: Phase 2, Sprint S2

## 2. Problem Statement

The current MVR scanner does not enforce explicit compact-device caps for divertor heat flux, impurity level proxy, and HTS peak magnetic field. This RFC adds these constraints directly to scanner outputs and introduces a compact-mode scan to verify that `R ~ 1.2..1.5 m` designs can still achieve `Q > 5`.

## 3. Target Paths

- Python module(s):
  - `src/scpn_fusion/core/global_design_scanner.py`
  - `src/scpn_fusion/core/heat_ml_shadow_surrogate.py` (consumer integration)
- Validation/reporting path:
  - `validation/gmvr_01_compact_constraints.py`
  - `validation/reports/gmvr_01_compact_constraints.json`
  - `validation/reports/gmvr_01_compact_constraints.md`
  - `tests/test_gmvr_01_compact_constraints.py`

## 4. Data and Dependency Readiness

- External dataset(s): not required for v1 (synthetic scan lanes only).
- License and usage confirmation: synthetic-only scanner constraints.
- External repository dependencies: none.
- Offline fallback path: deterministic NumPy/Pandas scan remains available.

## 5. Metrics and Acceptance Criteria

- Primary metrics:
  - Feasible design count in compact radius window (`1.2..1.5 m`).
  - Best feasible design satisfies `Q > 5`.
  - Constraint satisfaction: `Div_Load <= 45 MW/m2`, `Zeff <= 0.4`, `B_peak_HTS <= 21 T`.
- Minimum acceptable threshold:
  - Feasible design count >= `1`.
  - Best feasible design satisfies all caps.
- Runtime budget:
  - Strict validation <= `30 s`.

## 6. Regression and Safety Plan

- CI jobs to update:
  - Add targeted pytest and strict GMVR-01 validation invocation.
- Property/invariant tests:
  - Compact scan deterministic for fixed seed.
  - Constraint fields are present and bounded.
- Failure rollback strategy:
  - Keep additive fields (`Div_Load_Baseline`, `Div_Load_Optimized`, `Constraint_OK`) for audit.

## 7. Delivery Plan

- Milestone 1:
  - Add scanner-level caps and compact-mode scan ranges.
- Milestone 2:
  - Add HTS peak and Zeff proxy fields to design outputs.
- Milestone 3:
  - Add strict compact feasibility validation report + tests.

## 8. Go/No-Go Checklist

- [x] Path mapping reviewed
- [x] Data/license approved (synthetic-only v1 scope)
- [x] Metrics protocol approved
- [x] CI cost estimated
- [ ] Owner committed
