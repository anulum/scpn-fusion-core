<!--
SCPN Fusion Core — Next Sprint Execution Queue
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# Next Sprint Execution Queue (Phase 2, Sprint S1)

Date: 2026-02-13

Sprint theme: close baseline 3D physics gaps without destabilizing CI.

## Sprint Guardrails

- Maximum active work-in-progress: 2 tasks.
- Merge order is strict: S1-001 -> S1-002 -> S1-003 -> S1-004.
- No advanced `GNEU-*`, `GAI-*`, `GMVR-*`, `GPHY-*` implementation in this sprint.
- Every task must map to existing repository modules.
- Every task must pass Rust and Python gates before merge.

## Execution Queue

| ID | Priority | Task | Target Files | Definition of Done | Validation |
|---|---|---|---|---|---|
| S1-001 | P0 | Implement reduced 3D volumetric blanket surrogate (`G3D-05`) | `src/scpn_fusion/nuclear/blanket_neutronics.py`, `scpn-fusion-rs/crates/fusion-nuclear/src/neutronics.rs`, `tests/` | Volumetric (not wall-only) blanket estimator with deterministic outputs and documented assumptions | `python -m pytest -v`, `cargo test -p fusion-nuclear` |
| S1-002 | P0 | Implement toroidal asymmetry instability observables (`G3D-07`) | `src/scpn_fusion/core/fieldline_3d.py`, `src/scpn_fusion/control/disruption_predictor.py`, `scpn-fusion-rs/crates/fusion-ml/src/disruption.rs` | Observable vector includes toroidal asymmetry terms and is consumed by disruption path in both Python and Rust lanes | `python -m pytest -v`, `cargo test -p fusion-ml` |
| S1-003 | P1 | Harden 3D geometry extraction regressions | `src/scpn_fusion/core/geometry_3d.py`, `tests/test_geometry_3d.py`, `.github/workflows/ci.yml` | LCFS extraction robustness tests include low-point edge cases; CI validates both axisymmetric and VMEC-like smoke paths | `python -m pytest tests/test_geometry_3d.py -v`, `python examples/run_3d_flux_quickstart.py --toroidal 12 --poloidal 12 --output artifacts/ci_3d_smoke.obj` |
| S1-004 | P1 | Add control fault/noise resilience campaign baseline (`GDEP-04`, scoped) | `src/scpn_fusion/control/disruption_predictor.py`, `scpn-fusion-rs/crates/fusion-control/src/digital_twin.rs`, `scpn-fusion-rs/crates/fusion-control/src/mpc.rs`, `validation/` | Deterministic synthetic fault/noise campaign with summary metrics and regression thresholds | `python -m pytest -v`, `cargo test -p fusion-control` |
| S1-005 | P2 | Backlog normalization RFC for advanced tracks | `docs/3d_gaps.md`, `docs/PHASE2_ADVANCED_RFC_TEMPLATE.md` | RFC template added; each deferred item has required data/licensing/benchmark checklist | Docs review + CI docs job |

## Progress Update (2026-02-13)

- Completed: `S1-001` (reduced 3D volumetric blanket surrogate with Python + Rust tests).
- Completed: `S1-002` (toroidal asymmetry observables + disruption-path integration in Python and Rust).
- Completed: `S1-003` (added low-point LCFS fallback regression test and VMEC-like geometry CI smoke coverage).
- Completed: `S1-004` (deterministic control fault/noise resilience baseline in Python + Rust with campaign reporting and thresholds).
- Completed: `S1-005` (RFC template + tracker: `docs/PHASE2_ADVANCED_RFC_TEMPLATE.md`, `docs/PHASE2_ADVANCED_RFC_TRACKER.md`).
- Completed: `GAI-01` (synthetic GyroSwin-like turbulence surrogate benchmark lane + strict RMSE/speedup validation).
- Completed: `GAI-02` (synthetic TORAX-hybrid realtime loop lane + NSTX-U-like disturbance campaign thresholds).
- Completed: `GAI-03` (HEAT-ML magnetic-shadow surrogate integrated into MVR scanner with strict runtime/accuracy gates).
- Completed: `GMVR-01` (compact-constraint scanner update with divertor/Zeff/HTS caps in R=1.2..1.5m window).
- Completed: `GMVR-02` (TEMHD divertor MHD pressure-loss + velocity-dependent evaporation model with 3D toroidal stability sweep).
- Completed: `GMVR-03` (stellarator geometry extension with SNN stability loop and VMEC++-proxy parity benchmark).
- Completed: `GDEP-01` (NSTX-U/SPARC digital-twin ingest hook with SNN scenario planning and thresholded validation).
- Completed: `GDEP-02` (GPU runtime integration bridge for multigrid + SNN lanes with deterministic latency/speedup gates).
- Completed: `GDEP-03` (blind EU-DEMO/K-DEMO synthetic holdout dashboard with strict RMSE/parity thresholds).
- Completed: `GDEP-05` (v2.0-cutting-edge release-readiness gate and changelog contract validation lane).
- Completed: `GPHY-01` (reduced Boris-pusher particle tracker with toroidal current feedback blending into Grad-Shafranov source updates).
- Completed: `GPHY-02` (velocity-Verlet symplectic integration baseline in `fusion-math` with long-horizon drift checks).
- Completed: `GPHY-03` (reduced non-LTE collisional-radiative lookup for impurity charge-state PEC and radiative-loss estimation).
- Completed: `GPHY-04` (reduced IGA/NURBS boundary lane with smooth first-wall contour generation and regression checks).
- Completed: `GPHY-05` (latency-aware control lane with vector OU noise, actuator delay-line, and lagged MPC rollout).
- Completed: `GPHY-06` (reduced runtime regime-specialized kernel cache/hot-swap lane in `fusion-core`).
- Next active task: moved to `docs/PHASE3_EXECUTION_REGISTRY.md` (S4 baseline queue closed).

## Explicitly Deferred (Not In Sprint S1)

- `GNEU-01`, `GNEU-02`, `GNEU-03`
- `GAI-01`, `GAI-02`, `GAI-03`
- `GMVR-01`, `GMVR-02`, `GMVR-03`
- `GDEP-01`, `GDEP-02`

## Merge Gate Checklist (Per PR)

1. `cargo clippy --all-targets --all-features -- -D warnings`
2. `cargo test --all-features`
3. `python -m pytest -v`
4. 3D smoke artifacts generated and cleaned in CI path
5. `docs/3d_gaps.md` status line updated for any completed queue item

## Exit Criteria for Sprint S1

- `S1-001` and `S1-002` merged.
- At least one of `S1-003` or `S1-004` merged.
- Main branch CI remains fully green.
