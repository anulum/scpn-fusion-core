<!--
SCPN Fusion Core — 3D Gap Audit
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# 3D Gap Audit (Phase 1 Baseline)

Current 3D support exists primarily in `geometry_3d.py` and visualization-oriented workflows.  
The following gaps block full 3D physics fidelity.

## Identified Extension Points

| Gap ID | Current State | Limitation | Required Extension |
|---|---|---|---|
| G3D-01 | LCFS extracted from 2D axisymmetric equilibrium and revolved, plus reduced VMEC-like flux-coordinate interface in `src/scpn_fusion/core/equilibrium_3d.py` | Reduced closure only (not a full 3D force-balance VMEC solve) | Extend reduced VMEC-like interface to a full 3D equilibrium solve with force-balance residual closure |
| G3D-02 | Ray-based LCFS tracing over 2D grid with reduced 3D field-line and Poincare diagnostics in `src/scpn_fusion/core/fieldline_3d.py` | Reduced closure only (does not yet solve full 3D field-line integration from self-consistent 3D MHD fields) | Extend reduced tracer to full field solve on self-consistent 3D magnetic geometry |
| G3D-03 | Transport is radial (1.5D) with low-order `n!=0` closure in `fusion-core/src/transport.rs` | Closure model only (no resolved `(rho, phi)` PDE transport) | Extend from reduced spectral closure to native toroidal transport solve |
| G3D-04 | FNO turbulence is 2D with reduced toroidal-harmonic spectral closure in `fusion-physics/src/fno.rs` | No native `(rho, theta, phi)` 3D spectral tensor evolution | Extend to full 3D FFT/FNO transport with explicit toroidal coordinate state |
| G3D-05 | Neutronics simplified wall flux methods | No volumetric 3D blanket transport | Add 3D mesh-based neutronics surrogate |
| G3D-06 | Divertor model includes reduced 3D strike-point asymmetry projection in `fusion-nuclear/src/divertor.rs` | Projection closure only (not a full 3D SOL/target transport solve) | Extend to self-consistent 3D SOL transport and wall-coupled strike-point evolution |
| G3D-07 | Control logic built on 2D state summaries | No 3D instability observables | Add toroidal asymmetry features (ELM/RMP indicators) |
| G3D-08 | CI validates 2D-heavy paths | No dedicated 3D smoke/regression pipeline | Add CI smoke test generating and validating OBJ mesh artifacts |

## Priority Order

1. G3D-08 (CI safety for 3D path) - completed
2. G3D-03 (reduced low-order toroidal transport coupling) - completed
3. G3D-04 (reduced toroidal-harmonic turbulence coupling) - completed
4. G3D-06 (reduced 3D strike-point asymmetry mapping) - completed
5. G3D-01 (reduced VMEC-like native 3D flux coordinates) - completed
6. G3D-02 (reduced field-line tracing + Poincare diagnostics) - completed
7. G3D-05 (3D blanket neutronics surrogate)
8. G3D-07 (3D instability observables for control)

## Additional Requested Tasks (2026-02-13)

| Task ID | Scope | Target Metric | Planned Extension |
|---|---|---|---|
| GNEU-01 | Benchmark SNN compiler vs SOTA RL tearing-mode avoidance baseline (DIII-D-style, 2025 references) | `>95%` match vs DeepMind TORAX baseline; bit-flip resilience recovery `<1 ms` on synthetic disruptions | Extend `scpn/neuro_symbolic_compiler.py` with RL baseline harness + fault-injection resilience tests |
| GNEU-02 | Neuromorphic anomaly detection aligned with Startorus supervised/unsupervised early-alarm style | Robust early MHD/liquid-flow instability alarms under random perturbations | Extend `control/disruption_predictor.rs` with supervised+unsupervised alarm path and `proptest` perturbation coverage |
| GNEU-03 | Ice pellet fueling control mode via Petri-to-SNN controller path | Plasma density tracking error `<=1e-3` on ITER config scenarios | Add fueling mode implementation in `modes/fueling.py` with ITER-oriented simulation loop + regression tests |
| GAI-01 | GyroSwin-like turbulence surrogate for 5D transport closure | `~1000x` speedup vs reference GENE-style runs; RMSE `<10%` on core turbulence benchmarks | Extend `transport/toroidal_closure.rs` with Torch-trained surrogate pipeline over public gyrokinetic datasets (JET/ITPA) |
| GAI-02 | TORAX hybrid real-time control loop integration | Stable hybrid loop execution on NSTX-U digital-twin scenarios with closed-loop control hooks | Integrate/fork DeepMind TORAX workflow into `run_realtime_simulation.py` and couple with existing SNN controller path |
| GAI-03 | HEAT-ML magnetic shadow surrogate for divertor optimization | Magnetic shadow prediction runtime `<1 s` per scenario with optimization-ready outputs | Add HEAT-ML-style surrogate and integrate into MVR scanner heat-flux optimization workflow |

## Additional Requested Tasks (2026-02-13, Batch B: Compact/Liquid-Metal + Deployment)

| Task ID | Scope | Target Metric | Planned Extension |
|---|---|---|---|
| GMVR-01 | Reground MVR scanner constraints using compact/liquid-metal + HTS assumptions | Feasible `R ~1.2-1.5 m`, `Q > 5`, divertor flux `<=45 MW/m^2`, `Zeff <=0.4`, REBCO-like peak field up to `21 T` | Update MVR constraint envelope and rerun scan with DOE liquid-Li and compact-HTS bounded assumptions |
| GMVR-02 | TEMHD divertor MHD pressure-loss + velocity-coupled evaporation | Stable operating windows in reduced 3D toroidal runs across slow (`~1 mm/s`) to fast (`~10 m/s`) flow regimes | Prototype in code execution path and integrate reduced model into `engineering/divertor.py` |
| GMVR-03 | Stellarator extension and non-axisymmetric control compatibility | Benchmarkable non-axisymmetric geometry/control path against VMEC++-style references | Extend `geometry_3d.py` for stellarator-like non-axisymmetric fields and couple with SNN stability hooks |
| GDEP-01 | NSTX-U/SPARC digital-twin data hook | Real-time ingest + scenario-planning loop over synthetic/streamed diagnostics | Add ingestion and planning scaffolding under `validation/` for STELLAR-AI-style emulation |
| GDEP-02 | GPU integration completion (multigrid + neuromorphic inference) | `~2 ms` multigrid solves with end-to-end GPU path and deployable edge-inference hooks | Complete `wgpu` multigrid pipeline and add SNN inference acceleration path (including Loihi-facing API abstractions) |
| GDEP-03 | Blind validation on unseen EU-DEMO/K-DEMO-style scenarios | Internal RMSE dashboards at SOTA-parity envelopes for core-edge metrics | Add unseen-data validation harness and reporting in `validation/` dashboards |
| GDEP-04 | Resilience proof suite for noisy/faulty control loops | Demonstrated robustness advantage of SNN over baseline NN/RL under fault/noise campaigns | Add fault/noise campaign runner (bit-flips + stochastic noise) and comparative control metrics |
| GDEP-05 | v2.0-cutting-edge release tag and narrative | Versioned release package with changelog traceability for new hybrid stack | Prepare `v2.0-cutting-edge` release notes and version tags once required tasks are complete |

## Additional Requested Tasks (2026-02-13, Batch C: Grey-Zone Physics + Numerical Structure)

| Task ID | Scope | Target Metric | Planned Extension |
|---|---|---|---|
| GPHY-01 | PIC/MHD hybrid overlay for runaway-electron and SOL non-Maxwellian behavior | Particle tracker coupled to grid current with bounded runtime overhead and disruption-wall impact observables | Add `fusion-core/src/particles.rs` with Boris pusher and feed particle-derived current back into Grad-Shafranov/current closures |
| GPHY-02 | Symplectic/geometric integration integrity for long-pulse runs | Reduced numerical energy drift over long-pulse scenarios relative to RK baselines | Introduce symplectic integrators in `fusion-math` and preserve field-line/Poincare invariants in integration pipelines |
| GPHY-03 | Hamiltonian-constrained ML surrogate path | Improved long-horizon conservation behavior vs unconstrained operator baselines | Add Hamiltonian-informed surrogate track (HNN-oriented) alongside FNO/Torch paths for conservation-sensitive modules |
| GPHY-04 | Non-LTE wall/impurity radiative model integration | Improved divertor/wall radiation fidelity via species-state-aware emission coefficients | Integrate reduced collisional-radiative/PEC lookup workflow (ADAS-derived tables) into wall/divertor interaction models |
| GPHY-05 | Isogeometric (NURBS/B-spline) boundary fidelity path | Reduced boundary-shape discretization artifacts in magnetic/field calculations | Add spline/NURBS-ready geometry representation and coupling hooks for Grad-Shafranov boundary handling |
| GPHY-06 | Causality-aware control and runtime specialization | Delay/noise-aware MPC stability under realistic sensor/actuator latency plus runtime-specialized kernels for regime switching | Extend `digital_twin.rs` with stochastic noise layers, `mpc.rs` with delay-aware formulation, and investigate LLVM/InkWell JIT hooks for runtime solver specialization |

## Backlog Summary

- Remaining 3D gaps: `G3D-05`, `G3D-07` (2 tasks)
- Added neuromorphic/control tasks: `GNEU-01`, `GNEU-02`, `GNEU-03` (3 tasks)
- Added AI-surrogate integration tasks: `GAI-01`, `GAI-02`, `GAI-03` (3 tasks)
- Added compact/liquid-metal + deployment tasks: `GMVR-01..03`, `GDEP-01..05` (8 tasks)
- Added grey-zone physics + numerical-structure tasks: `GPHY-01..06` (6 tasks)
- Total remaining tracked tasks: 22

## Phase 2 Entry Criteria

- G3D-08 complete and stable in CI.
- At least one physics module upgraded from 2D/1.5D to 3D-aware coupling.
- Benchmark suite includes explicit 3D accuracy and runtime metrics.
