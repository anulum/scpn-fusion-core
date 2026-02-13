<!--
SCPN Fusion Core — 3D Gap Audit and Phase 2 Reset
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# 3D Gap Audit and Phase 2 Trajectory Reset

Date of restructuring: 2026-02-13

This document replaces the flat backlog with a gated plan so delivery remains progressive and non-regressive.

## Trajectory Assessment

Current trajectory is positive but at risk of scope collision.

- Positive: core Phase 1 3D baseline landed (`G3D-01`, `G3D-02`, `G3D-03`, `G3D-04`, `G3D-06`, `G3D-08`).
- Positive: CI includes dedicated 3D OBJ smoke and RMSE artifact checks.
- Risk: backlog mixed executable tasks with research-grade tasks in one lane.
- Risk: several requested target paths do not exist in the repository, which breaks execution sequencing.
- Risk: external-data-heavy tasks (TORAX, large gyrokinetic datasets, ADAS tables) were not gated by data/licensing readiness.

Conclusion: direction is correct, but work must be split into gated tracks with readiness criteria.

## Baseline 3D Gap Status

| Gap ID | Status | Current State | Required Extension |
|---|---|---|---|
| G3D-01 | Done | Reduced VMEC-like flux-coordinate interface in `src/scpn_fusion/core/equilibrium_3d.py` | Evolve toward full 3D force-balance closure in later phase |
| G3D-02 | Done | Reduced 3D field-line and Poincare diagnostics in `src/scpn_fusion/core/fieldline_3d.py` | Evolve toward self-consistent 3D field solve |
| G3D-03 | Done | Reduced low-order toroidal closure in `scpn-fusion-rs/crates/fusion-core/src/transport.rs` | Evolve to full `(rho, theta, phi)` resolved transport |
| G3D-04 | Done | Reduced toroidal-harmonic closure in `scpn-fusion-rs/crates/fusion-physics/src/fno.rs` | Evolve to native 3D spectral tensor evolution |
| G3D-05 | Done | Reduced 3D volumetric blanket surrogate added in `src/scpn_fusion/nuclear/blanket_neutronics.py` and `scpn-fusion-rs/crates/fusion-nuclear/src/neutronics.rs` | Evolve toward higher-fidelity mesh-driven neutronics in later phase |
| G3D-06 | Done | Reduced 3D strike-point asymmetry projection in `scpn-fusion-rs/crates/fusion-nuclear/src/divertor.rs` | Evolve toward full 3D SOL/target coupling |
| G3D-07 | Done | Toroidal asymmetry observables added in `src/scpn_fusion/core/fieldline_3d.py`, consumed in disruption paths `src/scpn_fusion/control/disruption_predictor.py` and `scpn-fusion-rs/crates/fusion-ml/src/disruption.rs` | Evolve toward richer mode-resolved instability signatures in later phase |
| G3D-08 | Done | CI includes 3D smoke and RMSE artifact checks | Keep green as a hard regression gate |

## Conflict Normalization (Path and Scope)

The following requested tasks are retained, but normalized to actual repository paths before execution.

| Requested Path (non-existent) | Normalized Target Path(s) |
|---|---|
| `scpn/neuro_symbolic_compiler.py` | `src/scpn_fusion/scpn/compiler.py`, `src/scpn_fusion/scpn/controller.py` |
| `control/disruption_predictor.rs` | `scpn-fusion-rs/crates/fusion-ml/src/disruption.rs`, `src/scpn_fusion/control/disruption_predictor.py` |
| `transport/toroidal_closure.rs` | `scpn-fusion-rs/crates/fusion-core/src/transport.rs`, `scpn-fusion-rs/crates/fusion-physics/src/turbulence.rs` |
| `engineering/divertor.py` | `src/scpn_fusion/core/divertor_thermal_sim.py`, `scpn-fusion-rs/crates/fusion-nuclear/src/divertor.rs` |
| `modes/fueling.py` | `src/scpn_fusion/control/spi_mitigation.py` (extend with fueling mode) or new module under `src/scpn_fusion/control/` |

## Restructured Phase 2 Tracks

### Track A: Stability and Regression Guard (mandatory)

- Keep `cargo clippy --all-targets --all-features -- -D warnings` green.
- Keep 3D smoke (`examples/run_3d_flux_quickstart.py`) green in CI.
- No Track B/C work merges without passing Track A gates.

### Track B: Close Remaining Baseline 3D Gaps

- B1: Deliver `G3D-05` reduced 3D volumetric blanket surrogate.
- B2: Deliver `G3D-07` toroidal asymmetry observables and control feature wiring.

### Track C: Control Realism and Resilience (incremental)

- C1: Add bounded fault/noise campaign harness on existing control/disruption path.
- C2: Add delay/noise primitives in `digital_twin.rs` and latency-aware interfaces in `mpc.rs`.

### Track D: Advanced Research (incubation only until gates are met)

- Includes `GNEU-*`, `GAI-*`, `GMVR-*`, `GDEP-*`, `GPHY-*`.
- Each item requires an RFC with:
  - data source and license confirmation,
  - benchmark protocol,
  - target path mapping to existing modules,
  - fallback if external dependency is unavailable.

## Reclassified Backlog State

- Ready now (execution lane): `GDEP-04` (scoped to existing modules only).
- Needs RFC before coding: `GNEU-01..03`, `GAI-01..03`, `GMVR-01..03`, `GDEP-01..03`, `GDEP-05`, `GPHY-01..06`.
- Blocked by external dependency readiness: tasks requiring TORAX fork sync, large external gyrokinetic datasets, or ADAS data ingestion.

## Next Sprint Queue

Execution queue is defined in `docs/NEXT_SPRINT_EXECUTION_QUEUE.md`.

Only items in that queue are considered in-sprint active work.

## Phase 2 Exit Criteria

- Baseline reduced 3D gaps `G3D-01..G3D-08` complete with tests.
- CI remains green for Python + Rust + 3D smoke.
- At least one resilience campaign report merged (fault/noise sensitivity summary).
