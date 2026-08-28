<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Fusion Core — NSTX-U/SPARC digital-twin ingestion RFC
-->

# NSTX-U/SPARC Digital-Twin Ingestion and Scenario Planning

Status: Implemented synthetic validation surface

## Scope and decision context

This RFC defines the deterministic NSTX-U/SPARC telemetry-ingestion campaign
and its immediate neuro-symbolic scenario-planning boundary. The campaign is a
synthetic runtime validation; it does not claim connection to either facility,
experimental-shot validation, control-system deployment, or realtime hardware
readiness.

## Problem statement

The control package needs a deterministic integration point that emulates
timestamped machine telemetry, bounds an in-memory stream, and evaluates
near-term mitigation plans through the shipped neuro-symbolic controller and
disruption-risk predictor.

## Owning surfaces

- Runtime: `src/scpn_fusion/control/digital_twin_ingest.py`
- Validation CLI: `validation/nstx_u_sparc_digital_twin_ingestion.py`
- Default local reports:
  - `validation/reports/nstx_u_sparc_digital_twin_ingestion.json`
  - `validation/reports/nstx_u_sparc_digital_twin_ingestion.md`
- Tests: `tests/test_nstx_u_sparc_digital_twin_ingestion.py`

## Data and dependency boundary

- The telemetry is generated locally from deterministic synthetic profiles.
- No external dataset, machine archive, network service, or new dependency is
  required.
- Machine labels select distinct emulated parameter ranges; they do not assert
  facility fidelity.

## Metrics and acceptance criteria

- Scenario-planning success rate per machine: at least `0.90`.
- Mean predicted disruption risk per machine: at most `0.75`.
- Deterministic planning-latency estimate: P95 at most `6.0 ms`.
- The JSON report uses schema version 2 and the descriptive
  `nstx_u_sparc_digital_twin_ingestion` report identity.
- Chaos inputs record channel dropout and Gaussian-noise injection rates.

Wall-clock latency remains diagnostic because shared-runner load is not an
isolated hardware benchmark. The deterministic estimate is the validation gate.

## Regression and safety contract

- Stream generation is repeatable for a fixed seed.
- Unsupported machines and invalid runtime bounds fail explicitly.
- The ring buffer never exceeds its configured capacity.
- Scenario planning fails closed without telemetry and reports bounded finite
  outputs after ingestion.
- The current report validator rejects the obsolete unversioned coded payload.
- Strict CLI mode returns a non-zero status when thresholds fail.

## Delivery state

- [x] Deterministic telemetry generator and bounded ingest runtime
- [x] Neuro-symbolic scenario planning with disruption-risk scoring
- [x] NSTX-U/SPARC synthetic campaign and chaos accounting
- [x] Versioned JSON and Markdown report contract
- [x] Focused real-surface tests and strict local gates
