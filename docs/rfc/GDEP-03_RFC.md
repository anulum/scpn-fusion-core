<!--
SCPN Fusion Core — RFC GDEP-03
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# RFC GDEP-03 — Blind Validation Dashboard (EU-DEMO/K-DEMO Synthetic Holdout)

Status: Draft for review

## 1. Task Identity

- Task ID: `GDEP-03`
- Title: Add deterministic blind-validation dashboard for EU-DEMO/K-DEMO-like unseen scenarios
- Owner: Unassigned
- Proposed sprint: Phase 2, Sprint S2

## 2. Problem Statement

Current validation reports cover ITER/SPARC and in-sprint deterministic campaigns, but there is no explicit blind holdout lane that emulates unseen reactor-class scenarios. This task adds a bounded offline dashboard for EU-DEMO/K-DEMO-like synthetic references to measure parity and regression drift before external-data integrations.

## 3. Target Paths

- Python module(s):
  - `validation/gdep_03_blind_validation.py`
- Validation/reference data:
  - `validation/reference_data/blind/eu_demo_reference.json`
  - `validation/reference_data/blind/k_demo_reference.json`
- Validation/reporting path:
  - `validation/reports/gdep_03_blind_validation.json`
  - `validation/reports/gdep_03_blind_validation.md`
  - `tests/test_gdep_03_blind_validation.py`

## 4. Data and Dependency Readiness

- External dataset(s): none for v1.
- License and usage confirmation: synthetic holdout references only.
- External dependencies: none beyond current stdlib/NumPy stack.
- Offline fallback path: deterministic local report generation using bundled synthetic references.

## 5. Metrics and Acceptance Criteria

- Primary metrics:
  - Confinement `tau_E` RMSE on blind set.
  - `beta_N` RMSE on blind set.
  - Core-edge match RMSE on blind set.
  - Aggregate parity score (higher is better).
- Minimum acceptable threshold:
  - `tau_E` RMSE <= `0.35 s`.
  - `beta_N` RMSE <= `0.15`.
  - Core-edge match RMSE <= `0.020`.
  - Aggregate parity >= `95%`.
- Runtime budget:
  - Strict validation <= `30 s`.

## 6. Regression and Safety Plan

- CI jobs to update:
  - Add targeted pytest and strict GDEP-03 validation invocation.
- Property/invariant tests:
  - Blind reference loader includes both EU-DEMO and K-DEMO lanes.
  - Aggregate and per-machine metrics satisfy threshold contract.
  - Markdown report includes threshold and aggregate sections.
- Failure rollback strategy:
  - Keep blind-validation lane additive and isolated from default solver execution.

## 7. Delivery Plan

- Milestone 1:
  - Add synthetic blind reference files for EU-DEMO and K-DEMO.
- Milestone 2:
  - Add metric pipeline and parity scoring in GDEP-03 validation module.
- Milestone 3:
  - Add report rendering and regression tests.

## 8. Go/No-Go Checklist

- [x] Path mapping reviewed
- [x] Data/license approved (synthetic-only v1 scope)
- [x] Metrics protocol approved
- [x] CI cost estimated
- [ ] Owner committed
