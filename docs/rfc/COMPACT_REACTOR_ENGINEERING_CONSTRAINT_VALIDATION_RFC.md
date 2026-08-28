<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Compact-Reactor Engineering-Constraint Validation RFC

Status: Implemented synthetic validation contract

## Scope

This validation exercises the repository's reduced compact-reactor design
scanner against explicit major-radius, fusion-gain, divertor-flux, impurity and
HTS peak-field proxy thresholds. It uses deterministic synthetic sampling and
the public `GlobalDesignExplorer.run_compact_scan` surface.

The contract is a regression and integration gate for the repository model. It
is not experimental evidence, an engineering design, a licensed facility
assessment or an optimisation certificate.

## Public surfaces

- Scanner: `src/scpn_fusion/core/global_design_scanner.py`.
- Validation CLI:
  `validation/compact_reactor_engineering_constraint_validation.py`.
- Tests:
  `tests/test_compact_reactor_engineering_constraint_validation.py` and the
  scanner's existing public-surface cohort.
- Default reports:
  `validation/reports/compact_reactor_engineering_constraint_validation.json`
  and `.md`.

## Deterministic protocol

The CLI declares the random seed and synthetic sample count. The scanner uses
its compact envelope, evaluates each accepted design with the repository's
physics-scaling and HEAT-ML-shadow surrogates, and selects rows satisfying all
of the following default gates:

- major radius in `1.2..1.5 m`;
- reduced engineering-gain proxy greater than `5.0`;
- shadowed divertor-flux proxy at or below `45 MW/m2`;
- reduced `Zeff` proxy at or below `0.4`;
- reduced HTS peak-field proxy at or below `21 T`;
- at least one feasible synthetic design.

The best feasible row minimises the scanner's declared cost proxy. Seed,
sample count, thresholds, evaluated and feasible counts, best-row metrics and
runtime are serialized. Runtime is diagnostic shared-host context, not a
performance acceptance claim.

## Report contract

The JSON report uses schema version 2 and report kind
`compact_reactor_engineering_constraint_validation`. The payload key has the
same descriptive identity. Validation requires the exact current top-level key
set and rejects the obsolete unversioned coded payload rather than accepting a
compatibility alias.

The command returns status 2 under `--strict` when the configured minimum
feasible count is not met. Thresholds are public command-line and Python inputs
with explicit finite/range validation, so pass and fail behavior can be tested
through the real interface.

## Evidence boundary

All sampled designs and model responses are synthetic. The result does not
establish:

- feasibility or safety of a physical compact reactor;
- component lifetime, manufacturability or economic cost;
- plasma-scenario reachability or control-system performance;
- agreement with a named systems code, transport solver, equilibrium solver or
  experimental campaign;
- global optimality, uncertainty calibration or out-of-distribution validity;
- readiness for design, procurement, licensing, construction or operation.

Those claims require independently sourced data, declared uncertainty,
cross-code and experimental comparison, materials and structural analysis,
safety cases, and domain-expert review outside this synthetic regression lane.

## Acceptance and regression gates

- Deterministic replay produces identical feasible counts and best designs for
  a fixed seed and configuration.
- Default thresholds admit at least one synthetic design.
- A stricter public fusion-gain threshold produces an empty feasible set and a
  strict CLI failure.
- Every threshold rejects invalid finite/range inputs.
- Schema-2 JSON and Markdown expose only descriptive identities.
- The validation owner reaches complete statement and branch coverage through
  real scanner, report and CLI surfaces.
- Ruff, NumPy docstrings, strict typing, documentation and repository preflight
  remain green.
