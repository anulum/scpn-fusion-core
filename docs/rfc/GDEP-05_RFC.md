<!--
SCPN Fusion Core — RFC GDEP-05
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# RFC GDEP-05 — v2.0-Cutting-Edge Release Staging Gate

Status: Draft for review

## 1. Task Identity

- Task ID: `GDEP-05`
- Title: Add deterministic release-readiness gate and changelog contract for `v2.0-cutting-edge`
- Owner: Unassigned
- Proposed sprint: Phase 2, Sprint S2

## 2. Problem Statement

Advanced lanes were delivered incrementally, but there is no deterministic gate ensuring that tracker state and release messaging remain synchronized before creating the `v2.0-cutting-edge` tag. This task adds a bounded validator that checks required advanced-task completion and changelog release note presence.

## 3. Target Paths

- Documentation:
  - `CHANGELOG.md`
  - `docs/rfc/GDEP-05_RFC.md`
- Validation/reporting path:
  - `validation/gdep_05_release_readiness.py`
  - `validation/reports/gdep_05_release_readiness.json`
  - `validation/reports/gdep_05_release_readiness.md`
  - `tests/test_gdep_05_release_readiness.py`
- Tracker path:
  - `docs/PHASE2_ADVANCED_RFC_TRACKER.md`

## 4. Data and Dependency Readiness

- External dataset(s): none.
- License and usage confirmation: repository-local metadata/doc validation only.
- External dependencies: stdlib only.
- Offline fallback path: deterministic local parsing of tracker/changelog files.

## 5. Metrics and Acceptance Criteria

- Primary metrics:
  - Required advanced tasks marked `Done` in tracker.
  - Required `v2.0-cutting-edge` changelog phrase present.
  - Aggregate release-readiness pass/fail status.
- Minimum acceptable threshold:
  - Done coverage = `100%` for required task list.
  - Changelog phrase presence = `True`.
  - Aggregate pass = `True`.
- Runtime budget:
  - Strict validation <= `10 s`.

## 6. Regression and Safety Plan

- CI jobs to update:
  - Add targeted pytest and strict GDEP-05 validation invocation.
- Property/invariant tests:
  - Tracker parsing is deterministic and task-ID based.
  - Validator fails if any required task is not `Done`.
  - Markdown report includes checklist and aggregate pass state.
- Failure rollback strategy:
  - Keep validation additive; no runtime solver-path behavior changes.

## 7. Delivery Plan

- Milestone 1:
  - Add release note clause for `v2.0-cutting-edge` in changelog.
- Milestone 2:
  - Add release-readiness validator and report renderer.
- Milestone 3:
  - Add tests and tracker updates.

## 8. Go/No-Go Checklist

- [x] Path mapping reviewed
- [x] Data/license approved (local metadata only)
- [x] Metrics protocol approved
- [x] CI cost estimated
- [ ] Owner committed
