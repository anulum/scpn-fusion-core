<!--
SCPN Fusion Core — RFC GMVR-02
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# RFC GMVR-02 — TEMHD Divertor MHD and Velocity-Dependent Evaporation

Status: Draft for review

## 1. Task Identity

- Task ID: `GMVR-02`
- Title: Add reduced TEMHD MHD pressure-loss + velocity-dependent evaporation model
- Owner: Unassigned
- Proposed sprint: Phase 2, Sprint S2

## 2. Problem Statement

Divertor modeling currently includes thermal and vapor shielding behavior but lacks explicit MHD flow-loss terms and velocity-sensitive evaporation behavior needed for liquid-metal channel tradeoffs. This RFC adds a deterministic reduced model and validates both slow and fast flow regimes, including toroidal stability checks.

## 3. Target Paths

- Python module(s):
  - `src/scpn_fusion/core/divertor_thermal_sim.py`
- Validation/reporting path:
  - `validation/gmvr_02_temhd_divertor.py`
  - `validation/reports/gmvr_02_temhd_divertor.json`
  - `validation/reports/gmvr_02_temhd_divertor.md`
  - `tests/test_gmvr_02_temhd_divertor.py`

## 4. Data and Dependency Readiness

- External dataset(s): none for v1 (synthetic reduced model only).
- License and usage confirmation: synthetic path.
- External dependencies: none (NumPy only).
- Offline fallback path: deterministic model and toroidal sweep remain in-repo.

## 5. Metrics and Acceptance Criteria

- Primary metrics:
  - Pressure-loss ratio (fast flow vs slow flow).
  - Evaporation-rate ratio (fast flow vs slow flow).
  - 3D toroidal stability rate in non-axisymmetric sweep.
- Minimum acceptable threshold:
  - Pressure ratio >= `1000`.
  - Evaporation ratio < `1.0`.
  - Toroidal stability rate >= `0.95`.
- Runtime budget:
  - Strict validation <= `30 s`.

## 6. Regression and Safety Plan

- CI jobs to update:
  - Add targeted pytest and strict GMVR-02 validation invocation.
- Property/invariant tests:
  - Fast flow increases pressure loss.
  - Fast flow reduces evaporation in this reduced model.
  - Stability index remains bounded and finite across toroidal sweep.
- Failure rollback strategy:
  - Keep additive method API on `DivertorLab`; do not remove existing methods.

## 7. Delivery Plan

- Milestone 1:
  - Add reduced MHD pressure-loss model and velocity-dependent evaporation estimator.
- Milestone 2:
  - Add integrated TEMHD simulation method and stability index.
- Milestone 3:
  - Add toroidal sweep validation + tests and report generation.

## 8. Go/No-Go Checklist

- [x] Path mapping reviewed
- [x] Data/license approved (synthetic-only v1 scope)
- [x] Metrics protocol approved
- [x] CI cost estimated
- [ ] Owner committed
