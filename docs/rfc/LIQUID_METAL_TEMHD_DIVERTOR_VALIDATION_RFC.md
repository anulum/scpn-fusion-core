<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Liquid-Metal TEMHD Divertor Validation RFC

Status: Implemented reduced synthetic validation contract

## Scope

This validation exercises the repository's reduced liquid-metal divertor model
at slow and fast flow velocities, then projects the fast-flow state across a
synthetic non-axisymmetric toroidal sweep. It uses the public `DivertorLab`
TEMHD surface and `VMECStyleEquilibrium3D` coordinate mapping.

The contract is a deterministic regression and integration gate. It is not a
validated divertor design, a materials-lifetime model or evidence from a
liquid-metal experiment.

## Public surfaces

- Reduced divertor model: `src/scpn_fusion/core/divertor_thermal_sim.py`.
- Synthetic 3D mapping: `src/scpn_fusion/core/equilibrium_3d.py`.
- Validation CLI: `validation/liquid_metal_temhd_divertor_validation.py`.
- Tests: `tests/test_liquid_metal_temhd_divertor_validation.py` plus the
  existing divertor and equilibrium public-surface cohorts.
- Default reports: `validation/reports/liquid_metal_temhd_divertor_validation.json`
  and `.md`.

## Deterministic protocol

The default campaign compares `0.001 m/s` and `10 m/s` reduced flow states at a
flux-expansion proxy of `40`. It computes fast-to-slow MHD pressure-loss and
evaporation ratios. A 36-point toroidal sweep modulates the fast-flow surface
heat-flux proxy with one declared non-axisymmetric Fourier mode and evaluates a
combined reduced-model stability index.

Default acceptance requires:

- both reduced flow states to report stable;
- fast-to-slow pressure-loss ratio at least `1000`;
- fast-to-slow evaporation ratio below `1.0`;
- combined toroidal stability index at or below `1.0`;
- at least `95%` of toroidal samples within that bound.

Velocities, expansion, sample count and all numeric acceptance thresholds are
public Python and CLI inputs with explicit finite/range validation. This lets
the real strict pass and failure surfaces be exercised without replacing
private functions.

## Report contract

The JSON report uses schema version 2 and report kind
`liquid_metal_temhd_divertor_validation`. The payload key has the same
descriptive identity. Validation requires the exact current top-level key set
and rejects the obsolete unversioned coded payload rather than accepting a
compatibility alias.

The CLI returns status 2 under `--strict` when any declared metric gate fails.
Runtime is serialized only as shared-host diagnostic context and is not a
performance acceptance claim.

## Evidence boundary

The pressure-loss, evaporation and stability quantities are reduced synthetic
proxies. This result does not establish:

- agreement with a named TEMHD solver or liquid-metal loop experiment;
- lithium or alloy compatibility, erosion, corrosion or tritium retention;
- free-surface, wetting, turbulence, electromagnetic coupling or failure-mode
  fidelity outside the implemented equations;
- structural, thermal-hydraulic, neutron, maintenance or safety feasibility;
- facility, licensing, construction, procurement or operational readiness.

Those claims require declared material properties, uncertainty, independently
sourced experimental and cross-code comparisons, coupled engineering analysis
and domain-expert review outside this reduced synthetic lane.

## Acceptance and regression gates

- Public slow/fast flow calls preserve the expected pressure and evaporation
  ordering.
- Fixed inputs reproduce identical metrics apart from runtime.
- Each public metric gate can produce a strict failure.
- Invalid velocities, sample counts and thresholds are refused.
- Schema-2 JSON and Markdown expose only descriptive identities.
- The validation owner reaches complete statement and branch coverage through
  real divertor, equilibrium, report and CLI surfaces.
- Ruff, NumPy docstrings, strict typing, documentation and repository preflight
  remain green.
