<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Fusion Core — EU-DEMO/K-DEMO synthetic holdout validation RFC
-->

# EU-DEMO/K-DEMO Synthetic Holdout Validation

Status: Implemented synthetic validation surface

## Scope and decision context

This RFC defines a deterministic offline comparison of confinement and
core-edge proxies against bundled EU-DEMO-like and K-DEMO-like synthetic rows.
It is a regression holdout for locally generated reactor-class scenarios, not
an experimental or facility-supplied validation dataset.

The campaign does not claim EU-DEMO or K-DEMO design-team endorsement,
machine-shot validation, predictive design accuracy, external-solver parity, or
readiness for reactor design decisions.

## Owning surfaces

- Validation CLI: `validation/eu_demo_k_demo_synthetic_holdout_validation.py`
- Bundled synthetic inputs:
  - `validation/reference_data/blind/eu_demo_reference.json`
  - `validation/reference_data/blind/k_demo_reference.json`
- Default local reports:
  - `validation/reports/eu_demo_k_demo_synthetic_holdout_validation.json`
  - `validation/reports/eu_demo_k_demo_synthetic_holdout_validation.md`
- Tests: `tests/test_eu_demo_k_demo_synthetic_holdout_validation.py`

## Data and dependency boundary

- Both input files are generated synthetic references with repository-tracked
  provenance and licence metadata.
- The labels select two locally defined parameter regimes; they do not identify
  public experimental shots.
- The campaign uses the repository's IPB98 confinement helper plus explicit
  normalised-beta and core-edge proxy equations.
- No external dataset, network service, accelerator, or new dependency is
  required.

## Metrics and acceptance criteria

- Confinement-time RMSE: at most `0.35 s`.
- Normalised-beta RMSE: at most `0.15`.
- Core-edge proxy RMSE: at most `0.020`.
- Aggregate parity score: at least `95%`.
- Every machine regime and the aggregate must pass.
- The JSON report uses schema version 2 and the descriptive
  `eu_demo_k_demo_synthetic_holdout_validation` identity.

The parity score is a bounded composite of confinement, normalised-beta and
core-edge absolute relative errors. It is a local regression metric and must
not be presented as physical fidelity to either named design programme.

## Regression and safety contract

- Both required reference files must exist and contribute at least one row.
- Per-machine and aggregate metrics are emitted separately.
- Non-finite or out-of-range acceptance thresholds fail explicitly.
- Strict CLI mode returns a non-zero status when any configured gate fails.
- The current report validator rejects the obsolete unversioned coded payload.

## Delivery state

- [x] Bundled EU-DEMO-like and K-DEMO-like synthetic reference rows
- [x] Deterministic confinement, normalised-beta and core-edge proxies
- [x] Per-regime and aggregate error/parity metrics
- [x] Versioned JSON and Markdown report contract
- [x] Focused real-file and CLI tests with strict local gates
