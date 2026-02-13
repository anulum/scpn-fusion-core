<!--
SCPN Fusion Core — RFC GDEP-01
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# RFC GDEP-01 — NSTX-U/SPARC Digital-Twin Ingestion Hook

Status: Draft for review

## 1. Task Identity

- Task ID: `GDEP-01`
- Title: Add realtime data-ingest hook with SNN scenario planning for NSTX-U/SPARC emulation
- Owner: Unassigned
- Proposed sprint: Phase 2, Sprint S2

## 2. Problem Statement

The codebase has control and simulation lanes but no deterministic integration point that emulates realtime digital-twin telemetry ingest and immediate scenario planning. This RFC adds a lightweight hook that ingests machine telemetry packets and runs SNN-assisted look-ahead risk planning.

## 3. Target Paths

- Python module(s):
  - `src/scpn_fusion/control/digital_twin_ingest.py`
  - `src/scpn_fusion/control/__init__.py`
- Validation/reporting path:
  - `validation/gdep_01_digital_twin_hook.py`
  - `validation/reports/gdep_01_digital_twin_hook.json`
  - `validation/reports/gdep_01_digital_twin_hook.md`
  - `tests/test_gdep_01_digital_twin_hook.py`

## 4. Data and Dependency Readiness

- External dataset(s): none for v1 (deterministic NSTX-U/SPARC emulation).
- License and usage confirmation: synthetic telemetry stream only.
- External dependencies: none beyond current stdlib/NumPy stack.
- Offline fallback path: campaign remains fully deterministic and local.

## 5. Metrics and Acceptance Criteria

- Primary metrics:
  - Scenario-planning success rate per machine.
  - Mean predicted disruption risk per machine.
  - P95 planning latency per machine.
- Minimum acceptable threshold:
  - Success rate >= `0.90`.
  - Mean risk <= `0.75`.
  - P95 latency <= `1.5 ms`.
- Runtime budget:
  - Strict validation <= `30 s`.

## 6. Regression and Safety Plan

- CI jobs to update:
  - Add targeted pytest and strict GDEP-01 validation invocation.
- Property/invariant tests:
  - Stream generation deterministic under fixed seed.
  - Hook planning remains bounded with finite outputs.
  - Validation report includes per-machine and aggregate pass/fail status.
- Failure rollback strategy:
  - Keep hook additive; no change to default simulation pipelines.

## 7. Delivery Plan

- Milestone 1:
  - Add deterministic telemetry stream generator and ingest hook.
- Milestone 2:
  - Add SNN scenario planner with disruption-risk scoring.
- Milestone 3:
  - Add validation report and tests for NSTX-U/SPARC smoke profiles.

## 8. Go/No-Go Checklist

- [x] Path mapping reviewed
- [x] Data/license approved (synthetic-only v1 scope)
- [x] Metrics protocol approved
- [x] CI cost estimated
- [ ] Owner committed
