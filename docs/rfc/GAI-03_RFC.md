<!--
SCPN Fusion Core — RFC GAI-03
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# RFC GAI-03 — HEAT-ML Magnetic-Shadow Surrogate for MVR Scanner

Status: Draft for review

## 1. Task Identity

- Task ID: `GAI-03`
- Title: Add HEAT-ML magnetic-shadow surrogate and integrate into global MVR scanner
- Owner: Unassigned
- Proposed sprint: Phase 2, Sprint S2

## 2. Problem Statement

Current divertor load estimation in the global design scanner uses a static scaling path and does not model magnetic-shadow attenuation from advanced divertor geometry. This RFC introduces a deterministic HEAT-ML surrogate lane to reduce heat-flux estimation bias and expose optimization leverage in compact-reactor scans.

## 3. Target Paths

- Python module(s):
  - `src/scpn_fusion/core/heat_ml_shadow_surrogate.py`
  - `src/scpn_fusion/core/global_design_scanner.py`
- Validation/reporting path:
  - `validation/gai_03_heat_ml_shadow.py`
  - `validation/reports/gai_03_heat_ml_shadow.json`
  - `validation/reports/gai_03_heat_ml_shadow.md`
  - `tests/test_gai_03_heat_ml_shadow.py`

## 4. Data and Dependency Readiness

- External dataset(s): not required for v1 (synthetic shadow references only).
- License and usage confirmation: synthetic path only.
- External repository dependencies: none.
- Offline fallback path: full benchmark and scanner integration remain NumPy/Pandas only.

## 5. Metrics and Acceptance Criteria

- Primary metrics:
  - Shadow prediction RMSE (% of mean target).
  - Inference runtime for high-volume query batch.
  - Mean divertor-load reduction when applied inside scanner path.
- Minimum acceptable threshold:
  - RMSE <= `10%`.
  - Inference runtime for 200k samples <= `1.0 s`.
  - Mean divertor-load reduction >= `8%`.
- Runtime budget:
  - Strict validation <= `30 s` on GitHub-hosted runner.

## 6. Regression and Safety Plan

- CI jobs to update:
  - Add targeted pytest and strict GAI-03 validation CLI invocation.
- Property/invariant tests:
  - Deterministic synthetic datasets for fixed seed.
  - Shadow fraction remains bounded.
  - Optimized divertor load does not exceed baseline load.
- Failure rollback strategy:
  - Keep additive integration in scanner; preserve baseline fields for auditability.

## 7. Delivery Plan

- Milestone 1:
  - Implement synthetic shadow dataset and HEAT-ML surrogate fit/predict.
- Milestone 2:
  - Integrate surrogate attenuation into global design scanner outputs.
- Milestone 3:
  - Add strict validation report and unit tests.

## 8. Go/No-Go Checklist

- [x] Path mapping reviewed
- [x] Data/license approved (synthetic-only v1 scope)
- [x] Metrics protocol approved
- [x] CI cost estimated
- [ ] Owner committed
