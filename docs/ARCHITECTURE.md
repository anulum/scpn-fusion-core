<!-- SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Project: SCPN Fusion Core -->
<!-- Description: Architecture map — capabilities, wiring, inputs/outputs, processing models, backends. -->

# SCPN Fusion Core — Architecture

This document is the structural map of `scpn-fusion-core`: what the package contains, how
the pieces are wired, what it consumes (inputs) and produces (outputs), the processing
models behind each subsystem, and the multi-backend acceleration layer. It is written for
**sibling repositories** (`scpn-control`, `scpn-quantum-control`, `scpn-mif-core`) and for
**SCPN STUDIO** verb/contract design that consume this repository's surfaces. For the
reasoning behind the durable design choices, see
[`docs/ARCHITECTURE_DECISIONS.md`](ARCHITECTURE_DECISIONS.md).

Every claim here is grounded in the source tree. Capability *maturity* (production-grade
positioning, fidelity, what is reduced-order vs full-fidelity, and what is fail-closed) is
governed by the dedicated evidence artifacts — see [Evidence and validation](#evidence-and-validation)
— not by this map. Where a model is reduced-order or a path is dormant, this document says so.

---

## 1. Ecosystem role

`scpn-fusion-core` is one repository in a multi-repository SCPN ecosystem. Each repository owns
a distinct surface; the boundaries are explicit contracts rather than copied code.

| Repository | Role | Owns |
|---|---|---|
| **`scpn-fusion-core`** (this repo) | **Physics-solver laboratory** | Broad physics kernels, numerical formulation, external-code validation campaigns, facility-data integration, Rust + GPU solver paths, 3D / stellarator / VMEC / free-boundary internals, neural-physics training, and the Petri→SNN control-compiler internals. |
| `scpn-control` | Control-grade integration + facade | Bounded public control APIs, NMPC / controller-loop integration, replay/campaign metadata, HIL / CODAC / EPICS / WebSocket safety boundaries, traceable claim surfaces. Ports/wraps the control-loop subset of FUSION's physics. |
| `scpn-quantum-control` | Quantum + phase-dynamics research | Quantum disruption classifiers, Qiskit/PennyLane execution, quantum Kuramoto/UPDE variants. |
| `scpn-mif-core` | FRC / magneto-inertial fusion | Consumes FUSION's FRC equilibrium / Faraday / compression seams (see [§11](#11-cross-repository-contracts)). |
| `scpn-studio` | Ecosystem hub / verb registry | Consumes per-repo "verbs" + evidence; FUSION is a physics-verb provider. |

A physics model matures **in `scpn-fusion-core` first**; downstream repos wrap the subset that
has a clear control-loop or research contract. FUSION is the canonical owner of the solver-level
mathematics, so the same model is not re-implemented elsewhere.

---

## 2. Architecture at a glance

The package is `src/scpn_fusion/` plus a Rust workspace `scpn-fusion-rs/`
and a validation/evidence tree. It is organised as **physics/control lanes over a
multi-backend dispatch floor**, with a self-auditing validation layer. The
generated capability manifest is the count source of truth for package, Rust,
test, documentation, and workflow surfaces.

```
                            ┌──────────────────────────── ENTRY POINTS ───────────────────────────┐
                            │  CLI (scpn-fusion, 27 modes)   ·   scpn-dashboard (Streamlit)        │
                            └──────────────────────────────────┬──────────────────────────────────┘
                                                               │
   INPUTS                          ┌────────────────────────── LANES ──────────────────────────┐         OUTPUTS
 ┌──────────┐                      │  core/      physics: equilibrium, transport, GK, MHD,      │      ┌──────────────┐
 │ GEQDSK   │                      │             disruption, FRC, 3D, edge/pedestal, surrogates │      │ IMAS IDS JSON│
 │ IMAS IDS │ ── io/ ingest ───►   │  control/   controllers, disruption, free-boundary, flight │ ──►  │ GEQDSK       │
 │ NPZ      │                      │             sim, RL, neuro-symbolic                        │      │ NPZ weights  │
 │ ITPA CSV │                      │  scpn/      Petri-net → SNN verifiable control compiler     │      │ .scpnctl.json│
 │ MDSplus  │                      │  phase/     Kuramoto / UPDE phase-synchronisation engine    │      │ PNG figures  │
 │ Zarr/S3  │                      │  diagnostics/ synthetic sensors + forward models           │      │ JSON/MD      │
 │ JSON cfg │                      │  nuclear/ engineering/ hpc/ ui/                             │      │  evidence    │
 └──────────┘                      └────────────────────────────┬───────────────────────────────┘      └──────────────┘
                                                                │
                            ┌──────────────────── MULTI-BACKEND DISPATCH FLOOR ────────────────────┐
                            │  core/_multi_compat.py  ·  fastest-first: RUST → … → JAX → NUMPY      │
                            │  Rust tier: scpn-fusion-rs (PyO3)   ·   NumPy: always-available floor  │
                            └──────────────────────────────────┬──────────────────────────────────┘
                                                               │
                            ┌──────────────────── EVIDENCE / VALIDATION LAYER ─────────────────────┐
                            │  validation/*.py benchmarks  ·  validation/reports/ evidence          │
                            │  release preflight gate  ·  claims/evidence guards  ·  Lean proofs    │
                            └─────────────────────────────────────────────────────────────────────┘
```

The committed generated inventory lives in `docs/_generated/capability_manifest.json`
and `docs/_generated/capability_snapshot.md`; this document keeps only the qualitative
subsystem map.

---

## 3. Backends and wiring

The package separates **physics contracts** (Python reference implementations) from
**acceleration tiers**. The NumPy reference path is always available; faster tiers are used
when present.

### 3.1 The fastest-first dispatcher (`core/_multi_compat.py`)

A `BackendTier` priority enum orders accelerators fastest→slowest:

```
RUST (0) → GPU (1) → MOJO (2) → JULIA (3) → GO (4) → JAX (5) → NUMPY (6)
```

`register_kernel(name, tier, provider)` registers a provider for a kernel at a tier;
`dispatch(name)` returns the fastest **available** provider, probing once and caching the
winner, and records a `fallback_telemetry` event when a non-fastest tier is selected.
`register_kernel_class` / `dispatch_kernel_class` do the same for stateful backends (the
equilibrium kernel), using lazy loader thunks to break import cycles.
Rust-only extension symbols without a NumPy floor are resolved through
`dispatch_rust_symbol(name)`, which is the single production import boundary for optional
`scpn_fusion_rs` callables such as SCPN runtime kernels and `PyRustFlightSim`.

**Currently registered tiers (verified):**

| Kernel | Rust tier | NumPy tier | Notes |
|---|---|---|---|
| `equilibrium_kernel` (class) | `RustAcceleratedKernel` | `FusionKernel` | drop-in swap; Grad-Shafranov solve |
| `hall_mhd_discovery` (class) | `PyHallMHD` | `HallMHD` | reconciled reduced Hall-MHD; statistically equivalent seeded trajectories |
| `shafranov_bv` | ✓ | ✓ | bit-exact parity |
| `solve_coil_currents` | ✓ | ✓ | tolerance-aware (ridge) |
| `measure_magnetics` | ✓ | ✓ | tolerance-aware (bilinear + probe θ) |
| `multigrid_solve` | ✓ | ✓ | machine-precision parity |
| `simulate_tearing_mode` | ✓ | ✓ | deterministic-step bit-exact + statistical trajectory |
| `kuramoto_step` | ✓ (`fusion-phase`) | ✓ | deterministic; agreement bounded by fp summation order (measured rel L2 ~1e-16) |
| `upde_tick` / `upde_run` | ✓ (`fusion-phase`) | ✓ | flat multi-layer contract (non-uniform N); `upde_run` batches the whole loop behind one boundary crossing |
| `gs_rb_sor_smooth` | GPU tier (`PyGpuSolver`, wgpu f32) | ✓ (`mg_smooth`, f64) | fixed-sweep Red-Black SOR of the toroidal GS* operator; f32-bounded agreement (rel L2 ~5e-6); GPU tier exists only when the extension is built with `--features gpu` AND a physical adapter passes the runtime probe |

All dispatched kernels have a **NumPy floor** — the package runs with no Rust extension
present. In addition, `diagnostics.PlasmaTomography.reconstruct` prefers the Rust
Tikhonov-NNLS backend (`PyTomography`, resolved through `dispatch_rust_symbol`) for its
`auto` method and degrades to SciPy `lsq_linear` then SART; the backends share the same
endpoint-inclusive geometry matrix and convex objective. The `MOJO`/`JULIA`/`GO` tiers
exist in the enum as a forward-looking chain but have no registered providers today.

The dispatcher proof surface is tracked by
`validation/reports/dispatcher_kernel_tiers_benchmark.json`, generated with
`PYTHONPATH=src python benchmarks/bench_dispatcher_kernel_tiers.py`. The report records the
selected tier and timing row for each A2 function kernel, then forces the Rust tier unavailable
and requires `fallback_telemetry_validation` to emit one `multi_backend` fallback event per
kernel while selecting the NumPy floor. The GPU smoother tier is benchmarked by
`validation/reports/gpu_gs_solver_benchmark.json` (`benchmarks/bench_gpu_gs_solver.py`):
on a discrete adapter the wgpu tier runs the identical 200-sweep workload 11.8x
(129x129), 38.8x (257x257) and 28.4x (513x513) faster than the NumPy floor.

### 3.2 Rust acceleration layer (`scpn-fusion-rs/`, 13 crates)

| Crate | Responsibility | Backs (Python) |
|---|---|---|
| `fusion-types` | config / error / state / constants | (error conversion) |
| `fusion-math` | FFT, multigrid, GMRES, SOR, Chebyshev, tridiagonal, interpolation, AMR, IGA, symplectic | `multigrid_vcycle` |
| `fusion-core` | Grad-Shafranov kernel, equilibrium, current diffusion, transport, X-point, inversion, particles | `PyFusionKernel`, `PyTransportSolver`, `PyInverseSolver`, particle/Boris fns |
| `fusion-polyglot` | standalone native GS reference solver | (Rust-only) |
| `fusion-phase` | Kuramoto-Sakaguchi step + multi-layer UPDE tick/run (SCPN phase lane) | `py_kuramoto_step`, `py_upde_tick`, `py_upde_run` |
| `fusion-physics` | Fokker-Planck, FNO, nonlinear GK, Hall-MHD, drift-wave, reduced-MHD sawtooth, FRC, MRTI, Faraday, compression | `PyFokkerPlanckSolver`, `PyFnoController`, `PyHallMHD`, `PyDriftWave`, `PyReducedMHD`, `PyNonlinearGKSolver`, FRC fns |
| `fusion-nuclear` | neutronics, TEMHD, sputtering, wall interaction, divertor, BOP | `PyBreedingBlanket` |
| `fusion-engineering` | plant design, magnets, tritium, blanket lifetime, economics | `PyPlantModel` |
| `fusion-control` | PID, optimal, MPC, SNN, digital twin, flight sim, SPI, analytic | `PyRustFlightSim`, `PySnnController`, `PySnnPool`, `PyMpcController`, `PyPlasma2D`, `PySpiAblationSolver`, `shafranov_bv`, `solve_coil_currents` |
| `fusion-diagnostics` | synthetic probes, bolometer, soft X-ray tomography | `PyTomography` |
| `fusion-ml` | neural equilibrium (PCA+MLP), neural transport, disruption (Transformer), PCE | `PyNeuralTransport`, `simulate_tearing_mode`, `rutherford_island_growth` |
| `fusion-python` | **PyO3 bindings** — this is the Rust→Python API | ~31 classes + ~31 functions |
| `fusion-gpu` | wgpu compute: Red-Black SOR + multigrid V-cycle GS solver | `PyGpuSolver` (feature-gated `gpu`) |

The PyO3 surface (`fusion-python/src/lib.rs`) is the complete Rust→Python API. Most exports are
consumed either through the dispatcher (the function kernels above, the equilibrium, Hall-MHD,
Fokker-Planck runaway-electron (`fokker_planck_re`), FNO-turbulence (`fno_turbulence`),
canonical-configuration surrogate-MPC (`neural_surrogate_mpc`), and reactor-design evaluator
(`global_design_scan`) class kernels, and the tomography solver path) or by direct import in a small number of performance-critical sites (the SCPN
runtime kernels, the Rust flight simulator). The remaining exports are an explicit
**library-only capability surface** — built and tested but deliberately not on a production
dispatch path, each for a stated reason: `py_advance_boris`/`PyParticle` (uniform-field
full-orbit pusher; the production `orbit_following` lane is a guiding-centre integrator in
spatially varying fields, so the two are not interchangeable), `PyDriftWave` and `PyReducedMHD`
(no Python twin implements the same reduced model — the Python `DriftWavePhysics` and
`ReducedMHD` classes are different formulations), `PyPlasma2D` (the Rust scalar
`(temperature, position)` plant is not the Python `Plasma2D` 2D poloidal-field twin), and
`PyInverseSolver`, `PyPlantModel` (Rust-only reduced capabilities with no Python twin
implementing the same model — the Rust pedestal-parameter inverse fit and 0D
systems-engineering formulas have no direct Python analogue; the Python `RealtimeEFIT`/
`KineticEFIT` full-grid flux reconstruction and `NuclearEngineeringLab` 2D wall-loading are
different, higher-fidelity formulations, so there is no parity contract to wire). The
single-point reactor-design evaluator `py_evaluate_design` is now a numerical twin of the
Python `GlobalDesignExplorer.evaluate_design` (Troyon/H-mode `beta_N` shaping, Eich divertor
scaling, and the HEAT-ML magnetic-shadow ridge attenuation with the same frozen weights and
engineering-constraint caps), agreeing to ~1e-15 relative across the design envelope with
identical `Constraint_OK`, and is dispatched Rust → NumPy through the `global_design_scan`
class kernel; the Monte Carlo `py_run_design_scan` driver stays a Rust-only convenience (its
rejection sampler uses a language-native RNG stream, so it is not point-for-point identical
to the Python scan — both are valid Monte Carlo draws). The GPU solver (`PyGpuSolver`) is feature-gated (`gpu`) and not compiled into
the default extension; when built with `--features gpu` on a machine with a physical
adapter, it backs the `gs_rb_sor_smooth` dispatcher kernel (GPU tier, W-2) with the
NumPy `mg_smooth` floor. Reconciliation of these pairs is tracked internally and follows the
same pattern used for the Hall-MHD and tomography lanes.

### 3.3 JAX research tier

The `jax_*` modules (`core/jax_gs_solver`, `jax_equilibrium_solver`, `jax_transport_solver`,
`jax_solvers`, `jax_gk_solver`, `jax_gk_nonlinear`, `jax_neural_equilibrium`,
`control/jax_traceable_runtime`) are **differentiable reference implementations** — JIT-compilable
and GPU-capable, enabling `jax.grad` through full solves for sensitivity/optimisation. They are
re-exported from `core/__init__` but are **not on the production dispatch path**; they are
research/benchmark backends, not the canonical CPU floor (NumPy is).

This is a deliberate disposition (master-plan W-3), not an omission: (a) JAX is an OPTIONAL
pinned extra, and a dispatch tier must not silently change production numerics based on which
optional dependency happens to be installed; (b) none of the JAX solvers is contract-reconciled
against its NumPy twin (each pair would need a Hall-MHD-style reconciliation before tier
registration); (c) their value is differentiability, not throughput — the consumers are research
scripts and the traceable-runtime validation lane (`validation/traceable_runtime_parity.py`),
which exercises them explicitly. The `BackendTier.JAX` enum slot stays reserved for a future
reconciled kernel; nothing registers it today.

### 3.4 Native C++ bridge (`hpc/hpc_bridge.py`)

An optional ctypes FFI to a native Grad-Shafranov solver, with HMAC+SHA256 source-trust
checking and a compiler whitelist. Used as an optional fallback under `FusionKernel`; the
Python/Rust paths are the default.

---

## 4. Inputs (ingest seam — `io/` + `core/eqdsk`)

The repository can ingest the following external formats:

| Format | Module(s) | Content |
|---|---|---|
| **GEQDSK / EQDSK** (EFIT) | `core/eqdsk.py`, `io/tokamak_archive_profiles` | poloidal flux ψ(R,Z), 1-D profiles (fpol, pres, q, pprime, ffprime), boundary/limiter |
| **IMAS IDS (JSON)** | `io/imas_connector*` (facade + common/equilibrium/transport/digital_twin/storage) | equilibrium, core_profiles, core_transport, summary IDS structures |
| **OMAS** (optional) | `io/imas_connector_omas` | bridge to ITER OMAS equilibrium and core-profile data structures (gated, soft import) |
| **NPZ** | `io/tokamak_disruption_archive`, `tokamak_synthetic_archive`, weights loaders | shot time-series, disruption flags, neural weights (loaded via DoS-bounded `safe_loaders`, no pickle) |
| **ITPA CSV** | `io/tokamak_archive` | H-mode confinement reference (Ip, BT, H98y2, κ, δ, τ_E) |
| **MDSplus (live)** | `io/tokamak_archive.fetch_mdsplus_profiles` | live facility data (β_N, q95, τ_E, contours, toroidal modes) — optional dependency |
| **Zarr / S3 (FAIR MAST)** | `io/mast_ingestor` | MAST (UKAEA) public shot summaries + magnetic probes — optional dependency |
| **JSON config** | `core/fusion_kernel`, CLI modes | FusionKernel config (grid, coils, boundary, plasma profiles); bounded depth/size |

`io/safe_loaders.py` enforces size bounds and disables pickle on all JSON/NPZ loads.

---

## 5. Outputs (emit seam)

| Format | Producer | Content |
|---|---|---|
| **IMAS IDS (JSON)** | `io/imas_connector_storage.write_ids`, `validation/torax_imas_interchange.py` | serialized equilibrium/core_profiles/summary/core_transport IDS, including the tracked TORAX basic-config `core_profiles` fixture |
| **GEQDSK** | `core/eqdsk.write_geqdsk` | EFIT-format equilibrium |
| **`.scpnctl.json`** | `scpn/artifact.save_artifact` | compiled SNN controller artifact (topology + weights + readout + packed bitstreams) |
| **NPZ** | surrogate/training modules | neural weights, surrogate datasets |
| **PNG figures** | `diagnostics/`, `ui/dashboard_generator` | tomography reconstructions, sensor geometry, Poincaré maps |
| **JSON + Markdown evidence** | `validation/*.py` | 137 benchmark/validation reports (see [§9](#9-evidence-and-validation)) |
| **Structured logs** | `io/logging_config` | JSON log records with physics context |

---

## 6. Subsystem map (`core/`, `control/`, `diagnostics/`, `nuclear/`, `engineering/`)

### 6.1 Equilibrium
- **`fusion_kernel.py` — `FusionKernel`**: non-linear **free-boundary Grad-Shafranov** solver,
  Picard iteration with under-relaxation; inner elliptic solves via Jacobi / SOR / **geometric
  multigrid V-cycle** (`fusion_kernel_iterative_solver` + `multigrid_solve`, Rust-parity).
  Newton-Kantorovich path (`fusion_kernel_newton_solver`, GMRES + Anderson) and free-boundary
  mixin. Rust tier via `RustAcceleratedKernel`; vacuum field via Green's function (elliptic
  integrals). **Input:** grid, coil set, pressure/current profiles. **Output:** ψ(R,Z), magnetic
  axis, X-point, B-field.
- **JAX equilibrium** (`jax_gs_solver`, `jax_equilibrium_solver`): differentiable fixed-boundary
  GS for sensitivity/coil optimisation.
- **`kinetic_efit.py`**: kinetic reconstruction with anisotropic fast-ion pressure + synthetic MSE.
- **`force_balance.py`**: public 2D free-boundary post-solve utility that adjusts paired PF3/PF4
  currents with a Newton radial-force residual loop and writes a balanced equilibrium config. It is
  consumed by validation workflows and documented as a config-balancing utility, not as the quasi-3D
  reduced-order force residual model in `quasi_3d_contracts.py`.
- **`amr_patch.py`**: validation/offline two-level patch AMR utility for X-point and pedestal
  refinement studies. It provides finite-input validation, gradient-threshold patch detection,
  prolongation/restriction, and cylindrical Jacobi patch smoothing; it is not wired into the
  production Grad-Shafranov solve or Rust dispatcher until a parity-backed AMR production lane
  exists. **`eqdsk.py`** handles I/O.
- **3D:** `geometry_3d` (LCFS→toroidal mesh), `equilibrium_3d` (VMEC-style Fourier),
  `fieldline_3d` (tracing/Poincaré), `vmec_lite` (reduced-order spectral), `stellarator_geometry`.

### 6.2 Transport
- **`integrated_transport_solver.py` — `IntegratedTransportSolver`**: 1.5-D transport orchestration;
  Rust tier (`PyTransportSolver`) with Python mixin fallback. Closures: TGLF (external),
  **QLKNN / neural** (`neural_transport`, MLP surrogate) with **out-of-distribution escalation**
  to TGLF, and `neoclassical` (Chang-Hinton χ, Sauter bootstrap). **Input:** Te/Ti/ne profiles +
  geometry. **Output:** evolved profiles, χ_e/χ_i/D, bootstrap current.
- **Gyrokinetics** (`gk_*`): linear eigenvalue solver (Miller geometry, Sugama collisions,
  response matrix), quasilinear saturation, TGLF subprocess interface, nonlinear δf (Rust/JAX),
  domain decomposition, explicit online-learner library API, and spot-check scheduler. The six `GKSolverBase`
  implementations (TGLF external, TGLF native, CGYRO, GS2, GENE, QuaLiKiz) are discoverable
  through the string-keyed **`gk_registry`** (`create_gk_solver` / `available_gk_solvers` /
  `resolve_tglf_solver`, exported from `core`); the canonical TGLF path resolves to the GACODE
  binary when on PATH and to the native quasilinear model otherwise. Three deliberately
  distinct TGLF roles (not duplicates): `gk_tglf` wraps the external binary (GKSolverBase
  deck contract), `gk_tglf_native` is the always-available quasilinear model, and
  `tglf_interface` is the TransportSolver comparison framework (JSON references, benchmark
  tables) used by the `tglf_live` pipeline backend. `gk_nonlinear.NonlinearGKSolver` is
  config-driven 5D δf and intentionally outside the deck-file registry. `gk_online_learner.py` is
  validation/offline workflow support for supervised surrogate refreshes from accepted GK samples; no
  live transport pipeline mutates deployed weights through it.
- **Supporting:** `impurity_transport` (ADAS charge-states), `momentum_transport` (E×B shear),
  `current_diffusion`, `current_drive` (ECCD/NBI), `blob_transport`.

### 6.3 Stability, disruption, FRC, edge
- **MHD stability** (`stability_mhd` + `stability_mhd_extended`, `ballooning_solver`,
  `alfven_eigenmodes`, `ntm_dynamics`, `locked_mode`): seven criteria — Mercier, ballooning
  (ODE shooting), Kruskal-Shafranov, Troyon, NTM (Modified Rutherford), RWM, peeling-ballooning —
  with cited references; TAE/Alfvén continuum; NTM island MRE.
- **Disruption physics** (`disruption_sequence`, `runaway_electrons`): reduced TQ/CQ/RE/halo
  sequence; DREAM-style runaway (Dreicer/Connor-Hastie, Rosenbluth-Putvinski avalanche, hot-tail).
- **FRC / magneto-inertial** (`frc_rigid_rotor`, `tilt_mode_frc`, `pulsed_compression`,
  `faraday_recovery`, `hall_mhd_pulsed`): rigid-rotor equilibrium (Steinhauer no-rotation limit; the
  **rotating BVP is implemented** against the source-verified Rostoker & Qerushi (2002)
  one-ion rigid-rotor closure with Python/Rust parity — verbatim Steinhauer Fig. 3 digitised
  parity remains a separate external gate), tilt-mode
  diagnostics, pulsed-compression dynamics, Faraday back-EMF recovery.
- **Edge/pedestal** (`sol_model` two-point + Eich, `eped_pedestal` reduced-order, `elm_model`,
  `divertor_thermal_sim`, `marfe`, `lh_transition`, `scaling_laws` IPB98(y,2)).

### 6.4 Control (`control/`)
- **Classical / optimal**: H-infinity (`h_infinity_controller`), μ-synthesis (`mu_synthesis`),
  constrained NMPC (`nmpc_controller`, `fusion_nmpc_jax`), super-twisting sliding-mode
  (`sliding_mode_vertical`), gain-scheduled PID (`gain_scheduled_controller`), fault-tolerant
  reconfiguration (`fault_tolerant_control`), SVD/MPC trajectory control (`fusion_optimal_control`,
  `neural_surrogate_mpc`), RWM feedback, shape control, density control (with EKF estimator),
  burn control (Bosch-Hale alpha heating). `controller_tuning.py` is an offline/library tuning
  helper for PID gains and reduced H-infinity search parameters; it validates Gym/Gymnasium-style
  PID environments and H-infinity target dictionaries, then returns bounded parameter dictionaries.
  It is not a live controller-loop mutator or deployment path.
- **Disruption**: Transformer predictor (`disruption_predictor`, torch, with checkpoint-trust
  policy + analytical fallback), deterministic linear-risk + tearing physics
  (`disruption_risk_runtime`), contracts, federated learning, Fokker-Planck RE, SPI ablation/mitigation.
- **Free-boundary tracking** (`_free_boundary_*` + `free_boundary_tracking`): kernel-in-the-loop
  isoflux/gap/X-point control with a supervisory safety policy; modular mixin architecture.
- **Flight sim / digital twin**: `tokamak_flight_sim` (actuator lag + delay + noise),
  `rust_flight_sim_wrapper` (Rust 10 kHz), `tokamak_digital_twin`, GDEP digital-twin ingest with
  SNN scenario planning, HIL harness, TORAX hybrid loop.
- **RL / neuro-symbolic**: Gymnasium env (`gym_tokamak_env`), constrained RL with Lagrangian
  multipliers (`safe_rl_controller`), spiking neuro-cybernetic controller (`neuro_cybernetic_controller`,
  embedded LIF), CODAC/EPICS interface (`codac_interface`), state estimator (EKF), RZIP vertical model.

### 6.5 Diagnostics, nuclear, engineering
- **`diagnostics/`**: forward models (interferometer, neutron, Thomson, ECE, soft X-ray,
  bolometer, CXRS), `synthetic_sensors` (magnetics/bolometer/interferometer), `tomography`
  (lsq_linear / SART / ridge inversion).
- **`nuclear/`**: 1-D breeding-blanket neutronics (TBR), neutron wall loading + ash poisoning,
  sputtering/erosion (PWI), TEMHD divertor stabiliser — reduced-order, deterministic, tested.
- **`engineering/`**: balance-of-plant (Rankine + parasitics), CAD ray-trace surface loading,
  thermal-hydraulics (coolant pumping power).

---

## 7. The SCPN verifiable-control compiler (`scpn/` + `phase/`)

This is the repository's distinguishing control surface: a **Petri-net → spiking-neural-network
compiler with formal safety contracts**, paired with a **Kuramoto/UPDE phase-synchronisation
engine**.

### 7.1 Compilation pipeline (`scpn/`)

```
control spec
  → StochasticPetriNet  (structure.py)        places + transitions + arcs → sparse (W_in, W_out)
                                              + boundedness / liveness / topology verification
  → FusionCompiler      (compiler.py)         → CompiledNet: float matrices + uint64 bitstreams
                                              + one stochastic LIF neuron per transition
  → Artifact            (artifact*.py)         → .scpnctl.json  (topology + weights + readout +
                                              packed weights via u64-zlib-base64 codec; schema-
                                              and range-validated)
  → NeuroSymbolicController (controller.py)    runtime: dual ORACLE (float) + SC (stochastic) paths,
                                              feature extraction (obs→[0,1]), place injection,
                                              marking update, slew-limited action decode
  → contracts.py + safety_interlocks.py        physics invariants (q_min, β_N, Greenwald, T_i,
                                              energy-conservation) + inhibitor-arc hard stops
```

The runtime activation/marking/sampling kernels have a Rust fast-path
(`scpn_dense_activations` / `scpn_marking_update` / `scpn_sample_firing`, resolved through
`core/_multi_compat.py`) with a NumPy floor.

### 7.2 Phase-synchronisation engine (`phase/`)
- **`kuramoto.py`**: Kuramoto-Sakaguchi mean-field step with an exogenous global driver Ψ;
  order parameter, Lyapunov V and exponent.
- **`upde.py` + `knm.py` + `plasma_knm.py`**: multi-layer Unified Phase Dynamics with an
  inter-layer coupling matrix K_nm (Paper-27 16-layer + plasma-native 8/16-layer hierarchies
  mapping tokamak timescales: micro-turbulence → zonal flow → MHD → sawtooth → transport barrier
  → current profile → equilibrium → plasma-wall).
- **`adaptive_knm.py`**: online, diagnostic-driven adaptation of K_nm (β_N / disruption-risk /
  coherence-PI channels, rate-limited, guard-vetoed).
- **`lyapunov_guard.py`**, **`realtime_monitor.py`**, **`ws_phase_stream.py`**: stability
  guardrail, tick-by-tick dashboard hook, and a WebSocket streaming server.

> Wiring note: the phase engine (`phase/kuramoto.py`, `phase/upde.py`) runs on a **NumPy reference
> implementation — the canonical and only execution path**. There is no Rust acceleration tier for
> the Kuramoto/UPDE kernels: the Rust workspace does not implement `kuramoto_step` / `upde_tick`, and
> the earlier guarded import + dead dispatch branches that implied one have been removed (a Rust phase
> tier is a tracked forward optimisation, not a current path).

### 7.3 Machine-checked compiler contract (`scpn-fusion-lean/`)

The Petri→SNN compilation is backed by Lean 4 proofs (`scpn-fusion-lean/`, toolchain pinned in
`lean-toolchain`, built by the `lean-safety-proofs` CI lane with a no-`sorry` gate). The contract has
two layers:

- **Static graph contract** (`SNNReachabilityPreservation.lean`): every Petri adjacency edge survives
  compilation, so reachability is preserved *and* reflected (`compile_reachability_equivalent`), edge
  well-formedness is preserved, and the compiled net has no spurious reachable path
  (`compile_has_no_spurious_reachable_path`).
- **Dynamic interlock + replay contract** (`InterlockReplayInvariance.lean`): over token markings, a
  raised interlock (a marked guard place) disables its transition and firing is a safe no-op
  (`interlock_raised_noop`); compilation preserves enabledness and one-step behaviour
  (`compile_preserves_enabled`, `compile_step_commutes`) and cannot weaken a raised interlock
  (`compile_preserves_interlock_block`); replay is a pure total fold, so it is deterministic and
  machine-independent — `compile_replay_commutes` proves compiling then replaying equals replaying then
  compiling, `replay_append` gives the prefix/checkpoint law behind a replay certificate, and
  `replay_keeps_guard_clear` is a safety invariant across a whole episode. Every theorem depends only on
  the standard Lean axioms (`propext`, `Quot.sound`) — no `sorryAx`.

The proof sources are hashed into a stable, source-based checksum by
`scpn/proof_manifest.py` (`compute_proof_checksum`, `proof_contract_manifest`); `scpn/artifact.py`
carries it into the emitted `.scpnctl.json` under `meta.compiler.{proof_system, proof_checksum}`
(`stamp_proof_contract`), so a controller artifact records exactly which machine-checked contract
certifies its compilation. The fields are optional and backwards compatible.

---

## 8. Entry points and public API

- **CLI** — `scpn-fusion` (`cli.py`): a registry of **27 modes** with three maturity tiers:
  - *public* (no token): `kernel`, `flight`, `optimal`, `learning`, `digital-twin`, `control-room`,
    `sandpile`, `nuclear`, `breeding`, `safety`, `optimizer`, `divertor`, `diagnostics`, `sawtooth`,
    `geometry`, `stellarator-control-replay-benchmark`, `spi`, `scanner`, `heating`, `wdm`,
    `neuro-control`, `rust-flight`.
  - *surrogate* (`--surrogate`): `neural`, `fno-training`.
  - *experimental* (`--experimental` + ack token): `quantum`, `q-control`, `director`.
  - Supports `--mode all`, `--dry-run`, per-mode timeout, `--list-modes`, startup health check.
- **Dashboard** — `scpn-dashboard` (`ui/dashboard_launcher`): launches the Streamlit app
  (`ui/app.py`, 5 tabs: physics, ignition/Q, nuclear, power-plant, shot-replay) with hardened config.
- **Public Python API** — `scpn_fusion.__all__ = ["setup_fusion_logging", "__version__"]`. The
  surface is deliberately thin; consumers import domain classes from subpackages directly
  (e.g. `from scpn_fusion.core.fusion_kernel import FusionKernel`).
- **Optional extras** (12): `ui`, `ml` (jax), `rl` (gymnasium), `dev`, `fuzz` (atheris),
  `rust` (maturin), `snn`, `benchmark` (freegs), `full-physics` (freegs+omas), `gpu`, `mpi`, `full`.

Cross-cutting infrastructure: `_data_paths.data_root()` (install-layout-independent data
resolution), `_data_paths.default_iter_config_path()` (bundled `validation/iter_config.json`
for CLI, dashboard, benchmark, and wheel defaults), `fallback_telemetry` (budget-gated
fallback accounting), `neurocore_compat` (embedded LIF neuron + entropy, no external
neuromorphic dependency).

---

## 9. Evidence and validation

Self-auditing is a first-class architectural layer, not an afterthought:
- `validation/` holds ~30 `benchmark_*.py` scripts and **152 generated reports** (`validation/reports/`,
  JSON + Markdown pairs) covering equilibrium parity (FreeGS), transport conservation, disruption
  replay/transfer, EPED domain contracts, FRC acceptance gates, end-to-end latency, and more.
- The **release preflight gate** (`tools/preflight.py --gate release`, delegating to
  `tools/run_python_preflight.py` in the CI Python-3.12 lane) runs
  metadata/claims/packaging/manifest/typing/docstring guards plus the strict benchmarks as a
  single sequential gate.
- **Lean** proofs provide machine-checkable safety lemmas for the control compiler.
- Claims are bounded: stale-number, claim-range, and evidence-map guards keep the public surface
  honest, and several physics lanes are **fail-closed** (they refuse to emit a parity claim until a
  redistributable reference exists).

This document maps *what exists and how it is wired*. For *how well-validated and how mature relative to the published field*
each component is, consult the evidence artifacts: `docs/BENCHMARKS.md`,
`docs/PHYSICS_VALIDATION_STATUS.md`, `docs/PHYSICS_METHODS_COMPLETE.md`, `docs/formal_verification.md`,
`docs/competitive_analysis.md`, and the per-report files under `validation/reports/`.

---

## 10. Symmetric ecosystem architecture

The SCPN repositories are intentionally asymmetric in implementation depth but symmetric in
contract responsibility: each repository owns one public boundary and consumes the others through
documented seams. `scpn-fusion-core` owns physics-solver development and evidence generation; it
does not own downstream deployment posture, facility interlock policy, quantum execution, MIF
system integration, or Studio routing. Those boundaries keep a solver improvement from becoming an
unreviewed deployment claim in another repository.

### 10.1 Responsibility matrix

| Boundary | FUSION owns | Sibling owns | Shared contract |
|---|---|---|---|
| Solver-to-control | Physics kernels, solver configuration, accepted/blocked validation reports, backend dispatch telemetry | Control facade, replay campaign shape, controller safety envelope, HIL/CODAC/EPICS/WebSocket integration | Imported solver subset, deterministic replay metadata, report status, and claim boundary |
| Solver-to-MIF | FRC equilibrium, Faraday recovery, pulsed-compression solver seams, and their evidence gates | Magneto-inertial workflow orchestration, pulsed-system kinematics, RTL/hardware-adjacent paths | FRC public symbols and validation status without duplicated solver mathematics |
| Classical-to-quantum | Classical phase/disruption feature contracts and local evidence | Quantum circuit/model execution, quantum-specific benchmarks, hardware-provider metadata | Feature schemas, phase/replay inputs, and bounded comparison reports |
| Solver-to-Studio | Physics verbs, capability manifest, evidence schemas, architecture-map metadata | Hub registry, user-facing routing, Studio workbench UX, cross-repo aggregation | Generated Studio manifest, schema-A verbs, content digests, and evidence links |

The symmetric rule is operational: if a change alters a shared contract, this repository updates
the FUSION side of the contract, evidence references, and public docs in the same commit. The
sibling repository remains responsible for its facade, runtime adapter, or user workflow. Where a
contract has not been reconciled, public docs must say that the row is blocked or diagnostic rather
than implying downstream readiness.

### 10.2 Integration surface hierarchy

Consumers should prefer the narrowest stable surface that satisfies the use case:

1. **Generated manifests and reports** for discovery, claim status, evidence schemas, and Studio
   ingestion.
2. **Public CLI modes** for reproducible runs whose command line, inputs, and outputs can be
   recorded.
3. **Domain subpackages** for Python integrations that need solver objects or adapters directly.
4. **Dispatcher kernels and IO seams** for native/runtime integrations that need backend selection,
   fallback telemetry, GEQDSK, IMAS, NPZ, or JSON contracts.

Internal `_` modules, gitignored TODO/planning files, and un-reconciled native exports are not
sibling integration contracts. A sibling may inspect them while planning, but a public dependency
needs a documented seam, tests, and evidence references before it becomes an ecosystem contract.

## 11. Cross-repository contracts

What `scpn-fusion-core` provides to siblings:

- **To `scpn-control`**: the equilibrium/transport/free-boundary solver subset and the Petri→SNN
  compiler internals; CONTROL wraps the control-loop subset behind bounded public APIs and the
  Studio control vertical. (The GS multigrid defect-sign correction and the Solovev exact-solution
  validation suite are an active two-way contract between the repos.)
- **To `scpn-mif-core`**: FRC rigid-rotor equilibrium, Faraday induction recovery, and pulsed-compression
  seams; the canonical FRC physics lives here.
- **To `scpn-quantum-control`**: the classical Kuramoto/UPDE phase engine and disruption-feature
  contracts; quantum variants live in QUANTUM.
- **To `scpn-studio`**: physics "verbs" (solve / diagnose / validate) and their evidence, consumed
  via the studio platform package.

Consumers should treat the **dispatched kernels** ([§3.1](#31-the-fastest-first-dispatcher-core_multi_compatpy))
and the **IO seam** ([§4](#4-inputs-ingest-seam--io--coreeqdsk)–[§5](#5-outputs-emit-seam)) as the stable
integration surface, and the **public CLI modes** ([§8](#8-entry-points-and-public-api)) as the stable
execution surface. Internal `_`-prefixed modules and the un-wired forward-capability Rust/JAX/GPU
surfaces are not integration contracts.
