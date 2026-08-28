<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Stellarator SNN Stability-Control Validation RFC

Status: Implemented reduced synthetic validation contract

## Scope

This validation builds the repository's synthetic W7-X-like Fourier geometry,
traces a reduced field line, applies an in-repository stochastic-neural-network
controller to its angular-step variability metric, and compares the geometry
with a deterministically perturbed in-repository reference.

It is a regression and integration gate for geometry, field-line and controller
surfaces. It is not a comparison with W7-X measurements, a VMEC or VMEC++ run,
or an externally validated stellarator stability result.

## Public surfaces

- Geometry/equilibrium: `src/scpn_fusion/core/geometry_3d.py` and
  `src/scpn_fusion/core/equilibrium_3d.py`.
- SNN compilation/control: the public `scpn_fusion.scpn` compiler, structure,
  contracts and controller modules.
- Validation CLI: `validation/stellarator_snn_stability_control_validation.py`.
- Tests: `tests/test_stellarator_snn_stability_control_validation.py` plus the
  existing geometry, equilibrium and controller cohorts.
- Reports: `validation/reports/stellarator_snn_stability_control_validation.json`
  and `.md`.

## Deterministic protocol

The default campaign performs six controller iterations and evaluates
synthetic-reference parity on 720 deterministically seeded flux coordinates.
It requires a final angular-step variability metric no greater than `0.025`, at
least `30%` improvement over the first iteration, and at least `95%` parity
with the perturbed in-repository reference.

Iteration count, parity sample count and all thresholds are public Python and
CLI inputs with explicit finite/range validation. The strict CLI returns status
2 whenever a declared gate fails. Runtime is diagnostic shared-host context,
not a performance claim.

## Report contract

The JSON report uses schema version 2 and report kind
`stellarator_snn_stability_control_validation`. The payload key has the same
descriptive identity. Validation requires the exact current top-level key set
and rejects the obsolete unversioned coded payload without a compatibility
alias.

The parity field is named `synthetic_reference_parity_pct`. The previous
VMEC++-proxy label was inaccurate because no VMEC++ executable, output or
independently generated equilibrium participates in this campaign.

## Evidence boundary

All geometry, controller inputs and reference perturbations are synthetic. The
result does not establish:

- agreement with W7-X, another stellarator experiment, VMEC or VMEC++;
- MHD, neoclassical, turbulence, coil-error or island-divertor fidelity;
- plasma controllability under facility sensors and actuators;
- global stability, confinement, engineering feasibility or safety;
- design, licensing, construction or operational readiness.

Those claims require independent solver and experimental data, uncertainty and
domain coverage, device-specific geometry and diagnostics, and expert review
outside this reduced synthetic lane.

## Acceptance and regression gates

- Fixed inputs reproduce identical metrics apart from runtime.
- Each public metric threshold can produce strict failure.
- Invalid iterations, samples and thresholds are refused.
- Schema-2 JSON and Markdown expose only descriptive identities and the honest
  in-repository synthetic-reference boundary.
- The validation owner reaches complete statement and branch coverage through
  real geometry, field-line, SNN, report and CLI surfaces.
- Ruff, NumPy docstrings, strict typing, documentation and repository preflight
  remain green.
