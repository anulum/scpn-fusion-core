<!--
SCPN Fusion Core — RFC GMVR-03
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# RFC GMVR-03 — Stellarator Geometry Extension + SNN Stability Lane

Status: Draft for review

## 1. Task Identity

- Task ID: `GMVR-03`
- Title: Extend geometry path with W7-X-like non-axisymmetric synthesis and SNN stability-control benchmark
- Owner: Unassigned
- Proposed sprint: Phase 2, Sprint S2

## 2. Problem Statement

The geometry stack already supports generic Fourier non-axisymmetry but lacks an explicit stellarator-oriented extension workflow and benchmark contract. This RFC adds a W7-X-like helper in `geometry_3d`, couples it with an SNN stability-control lane, and validates parity against a VMEC++-proxy reference.

## 3. Target Paths

- Python module(s):
  - `src/scpn_fusion/core/geometry_3d.py`
- Validation/reporting path:
  - `validation/gmvr_03_stellarator_extension.py`
  - `validation/reports/gmvr_03_stellarator_extension.json`
  - `validation/reports/gmvr_03_stellarator_extension.md`
  - `tests/test_gmvr_03_stellarator_extension.py`
  - `tests/test_geometry_3d.py` (helper coverage)

## 4. Data and Dependency Readiness

- External dataset(s): not required for v1 (synthetic W7-X-like mode set).
- License and usage confirmation: synthetic reduced model path only.
- External dependencies: none beyond in-repo modules.
- Offline fallback path: deterministic field-line and Poincare diagnostics remain local.

## 5. Metrics and Acceptance Criteria

- Primary metrics:
  - Final instability metric after SNN control loop.
  - Instability improvement vs baseline coupling.
  - VMEC++-proxy parity score.
- Minimum acceptable threshold:
  - Final instability metric <= `0.09`.
  - Improvement >= `10%`.
  - VMEC++-proxy parity >= `95%`.
- Runtime budget:
  - Strict validation <= `30 s`.

## 6. Regression and Safety Plan

- CI jobs to update:
  - Add targeted pytest and strict GMVR-03 validation invocation.
- Property/invariant tests:
  - Stellarator helper returns non-axisymmetric geometry (`phi`-dependent radius).
  - SNN loop remains deterministic and bounded.
  - Parity metric remains finite and within [0, 100].
- Failure rollback strategy:
  - Keep helper additive and preserve existing axisymmetric defaults.

## 7. Delivery Plan

- Milestone 1:
  - Add `build_stellarator_w7x_like_equilibrium` helper in geometry builder.
- Milestone 2:
  - Add SNN stability-control benchmark lane over non-axisymmetric field-line tracing.
- Milestone 3:
  - Add VMEC++-proxy parity metric and report/test harness.

## 8. Go/No-Go Checklist

- [x] Path mapping reviewed
- [x] Data/license approved (synthetic-only v1 scope)
- [x] Metrics protocol approved
- [x] CI cost estimated
- [ ] Owner committed
