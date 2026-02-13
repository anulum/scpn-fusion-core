<!--
SCPN Fusion Core — RFC GAI-02
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# RFC GAI-02 — TORAX-Hybrid Realtime Loop (Synthetic v1)

Status: Draft for review

## 1. Task Identity

- Task ID: `GAI-02`
- Title: Integrate TORAX-like surrogate with SNN control loop for realtime simulation path
- Owner: Unassigned
- Proposed sprint: Phase 2, Sprint S2

## 2. Problem Statement

The realtime simulation path lacks a deterministic integration lane that combines a TORAX-style transport surrogate with the existing neuro-symbolic controller. This RFC defines a bounded, offline implementation that exercises the realtime coupling contract before external TORAX repository synchronization and machine-data ingest.

## 3. Target Paths

- Python module(s):
  - `src/scpn_fusion/control/torax_hybrid_loop.py`
  - `run_realtime_simulation.py`
- Validation/reporting path:
  - `validation/gai_02_torax_hybrid.py`
  - `validation/reports/gai_02_torax_hybrid.json`
  - `validation/reports/gai_02_torax_hybrid.md`
  - `tests/test_gai_02_torax_hybrid.py`
- Future external coupling path (deferred):
  - TORAX upstream sync + adapter bridge in a dedicated integration branch.

## 4. Data and Dependency Readiness

- External dataset(s): none for v1; use deterministic NSTX-U-like synthetic scenarios.
- License and usage confirmation: synthetic-only path.
- External repository dependencies: none for v1.
- Offline fallback path: full campaign runs with stdlib + NumPy + in-repo modules.

## 5. Metrics and Acceptance Criteria

- Primary metrics:
  - Disruption-avoidance rate in NSTX-U-like disturbance scenarios.
  - TORAX parity metric (hybrid branch tracking vs TORAX-only branch).
  - Realtime loop latency budget (P95).
- Minimum acceptable threshold:
  - Disruption avoidance rate >= `0.90`.
  - TORAX parity >= `95%`.
  - P95 loop latency <= `1.0 ms`.
- Runtime budget:
  - Strict smoke validation <= `30 s`.

## 6. Regression and Safety Plan

- CI jobs to update:
  - Add targeted pytest and strict GAI-02 validation invocation in Python lane.
- Property/invariant tests:
  - Deterministic campaign under fixed seed.
  - Finite bounded state evolution.
  - Thresholded pass/fail contract in validation output.
- Failure rollback strategy:
  - Keep integration additive and behind explicit invocation path (`--torax-hybrid`).
  - Preserve legacy realtime simulation default behavior.

## 7. Delivery Plan

- Milestone 1:
  - Add reduced TORAX-like state update and policy head.
- Milestone 2:
  - Fuse with SNN correction lane and disruption-risk scoring.
- Milestone 3:
  - Add validation/report/test harness and wire smoke invocation into realtime script.

## 8. Go/No-Go Checklist

- [x] Path mapping reviewed
- [x] Data/license approved (synthetic-only v1 scope)
- [x] Metrics protocol approved
- [x] CI cost estimated
- [ ] Owner committed
