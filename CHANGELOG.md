# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Changelog
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

# Changelog

## [2.0.0] — 2026-02-17

### Added — v2.0.0: Publication-Grade Physics Validation

#### Equilibrium Solver
- Pure-Python geometric multigrid V-cycle (full-weighting restriction, bilinear prolongation, Red-Black SOR smoother) with toroidal GS* stencil
- Python multigrid wired as default solver (replaces SOR) — 129x129 converges to <1e-6 in <500 V-cycles
- PyO3 binding for Rust multigrid `multigrid_vcycle()` with Python fallback
- Real DIII-D EFIT GEQDSK equilibria replacing synthetic Solov'ev files
- Psi/q-profile overlay validation against EFIT ground truth
- Neural surrogate retrained on expanded 18+ GEQDSK dataset

#### Transport Physics
- EPED-like pedestal model (Snyder 2009 simplified scaling) for H-mode boundary conditions
- Gyro-Bohm transport calibration against ITPA H-mode confinement database (20 shots)
- IPB98(y,2) uncertainty quantification with log-linear error propagation (Verdoolaege 2021)
- Transport validation against ITPA CSV with RMSE dashboard integration

#### Control & Disruption
- Proper H-infinity synthesis via Riccati equations (Doyle-Glover-Khargonekar) replacing fixed-gain PID
- Real DIII-D disruption shot data (5 disruptions + 5 safe controls)
- Disruption predictor retrained on mixed real+synthetic data (>70% recall target)
- Closed-loop disruption mitigation with real shot replay
- Realistic actuator model (sensor noise, coil rate limits, measurement delay)
- SNN vs MPC vs PID disturbance rejection benchmark
- Petri net formal verification documentation

#### Integration & IO
- IMAS IDS conformance audit with JSON Schema validation
- MDSplus automated download with NPZ caching
- GitHub issue templates for shot replay requests
- GitHub issue template for shot replay requests (`shot_replay_request.md`); Streamlit Shot Replay tab planned for v2.1

#### Validation
- `validate_real_shots.py` capstone validation (5 shots, equilibrium + transport + disruption)
- Rust/Python parity test suite (relative tolerance < 1e-3)
- RMSE regression gate in CI (tau_E RMSE, psi NRMSE thresholds)
- HIL demo documentation with FPGA register map

### Changed
- Default equilibrium solver changed from SOR to multigrid
- Transport model uses gyro-Bohm scaling instead of constant chi_base=0.5
- Version bumped to 2.0.0

---

## [2.0.0-pre] — 2026-02-15 (Pre-release hardening)

### Added

#### 3D MHD and Exascale Readiness
- VMEC fixed-boundary 3D equilibrium solver (Hirshman & Whitson 1983 variational approach)
  in `fusion-core/src/vmec_interface.rs`
- BOUT++ coupling interface with Newton-iteration flux surface tracing, field-aligned metric
  tensors (`g^ij`, Jacobian, `|B|`), and stability result parsing
- 2D Cartesian MPI domain decomposition with Additive Schwarz distributed GS solver
  (Rayon thread-parallel) in `fusion-core/src/mpi_domain.rs`
- Optimal process-grid factorisation (surface-to-volume minimisation)
- 4-face halo exchange (serial reference, rsmpi-ready interface)
- GS residual L2 norm for convergence monitoring

#### Physics and Transport
- Coulomb collision operator (Fokker-Planck Monte Carlo with NRL Coulomb logarithm,
  Spitzer slowing-down time, critical velocity, pitch-angle Langevin scattering,
  deterministic xorshift64 PRNG) in `fusion-core/src/particles.rs`
- GPU equilibrium solver via wgpu compute shaders (WGSL Red-Black SOR on
  DX12/Vulkan/Metal with R-singularity protection) in new `fusion-gpu` crate
- Neoclassical transport closure (Chang-Hinton 1982 ion thermal diffusivity,
  banana/plateau/PS regimes, bootstrap current) in Rust and Python

#### Validation and Experimental Data
- DIII-D shot 187070 and JET shot 92436 synthetic Solov'ev GEQDSK validation profiles
  (10 files: L-mode, H-mode, negative triangularity, snowflake, DT, hybrid, high-Ip)
  on 129x129 grids with generation script and validation runner
- DIII-D/JET validation runner with strict GS-operator and GEQDSK payload integrity contracts

#### Documentation and Community
- Sphinx documentation site (26 files): Furo-themed conf.py with autodoc, napoleon,
  mathjax, intersphinx; installation, quickstart, and 8 user guide chapters with
  physics equations; API reference for all 59+ Python modules across 8 subpackages
- Paper A LaTeX manuscript: GS equilibrium solver (840 lines, target Nuclear Fusion / CPC)
- Paper B LaTeX manuscript: SNN vertical stability controller (939 lines, target FED / IEEE TPS)
- Shared bibliography (55 BibTeX entries, 643 lines)
- DOE ARPA-E convergence pitch document (785 lines): GAMOW, BETHE, FES, ASCR program
  alignment matrices with 36-month milestone roadmap
- GitHub PR template with testing/quality/physics checklists
- Issue template config (security contact, blank issues disabled)
- CONTRIBUTING.md comprehensive upgrade (426 lines: dev setup, style guide, PR process)
- Security policy (SECURITY.md) expanded with version table, hardening summary, known limitations
- Coverage badge, Docker quickstart, pure-Python install, and data licensing in README

#### Reactor Engineering Elevation (H6 wave, 9 tasks)
- VMEC IO bridge for non-axisymmetric equilibrium interoperability (`H6-001`)
- Hybrid kinetic-fluid alpha/runaway test-particle tracker with orbit/energy statistics (`H6-002`)
- Synthetic forward diagnostics (interferometer phase, neutron count channels) (`H6-003`)
- IMAS/IDS adapter pattern for Digital Twin state interchange (`H6-004`)
- CAD geometry integration lane (STEP/STL mesh ingestion for heat/neutron load tracing) (`H6-005`)
- Explicit actuator transfer-function dynamics in flight simulation/control path (`H6-006`)
- Domain-randomization chaos monkey for deterministic sensor dropout/noise injection (`H6-007`)
- Multi-node MPI domain-decomposition scaffolding for core grid solves (`H6-008`)
- Traceable control-loop runtime lane to reduce Python interpreter overhead (`H6-009`)

### Changed

#### SCPN Controller Optimisation (H5 wave)
- Vectorized injection, action decode, and array-native stepping in SCPN controller
- Internal marking state kept in NumPy arrays with compatibility accessors
- Compiled feature-evaluation path removes per-tick dict churn
- Preallocated dense/update work buffers for tick-loop allocation reduction
- Binomial sampling replaces high-allocation mirrored-draw stochastic path
- Rust stochastic-firing kernel bridge for optional backend offload
- Default runtime profile changed to adaptive (binary probabilistic margin)
- Expanded Rust offload eligibility and chunked antithetic sampling for large nets
- Precomputed transition-delay indices remove per-tick timing mask churn

#### README Repositioning
- Repositioned project identity as a **neuro-symbolic control framework** rather than a
  physics simulation suite competing with TRANSP/GENE/JINTRAC
- Expanded Design Philosophy, Neuro-Symbolic Compiler section, simulation mode maturity
  tiers, Code Health & Hardening section, and Known Limitations & Roadmap
- Honest benchmark claims: separated validated from projected performance numbers

#### Runtime and Configuration
- Experimental modes gated behind `--experimental` flag
- Removed `sys.path` hacks from 20+ core/validation/test modules (replaced with package imports)
- Realtime simulation entrypoint resolves config relative to script root with `--config` support
- Docker image hardened with prod-by-default dependency install and non-root user
- HPC bridge restricts native solver library loading to trusted package paths with
  explicit `SCPN_SOLVER_LIB` env override
- Stabilized GAI-02 latency metric with deterministic hardware-normalized proxy
- PID/MPC and GNEU benchmark controllers pinned to adaptive non-zero binary margin
- NumPy 2.4 compatibility restored for blanket TBR integration (trapezoid fallback)
- GitHub Pages deployment gated behind explicit `DEPLOY_GH_PAGES` repo variable

#### Licensing and Legal
- AGPL-3.0-or-later added to all package manifests
- SPARC MIT license notice and ITPA provenance disclaimer added
- TestPyPI publish made best-effort in CI

### Fixed

#### Hardening Waves (266 tasks completed across 8 waves)

**S2 (8 tasks):** HPC bridge in-place solve paths, C++ SOR convergence guards,
SCPN benchmark stochastic-vs-float equivalence gate, disruption predictor fallback,
PWI angle-energy invariants and redeposition bounds, TEMHD solver edge-case regression,
path mapping normalisation, release gate queue health.

**S3 (6 tasks):** SCPN topology diagnostics for dead nodes and unseeded place cycles,
inhibitor-arc support with explicit opt-in, compact artifact serialization mode,
control simulation fallback entry points, HPC bridge edge-path validation,
S3 queue health in release-readiness report.

**S4 (4 tasks):** Compiler topology/inhibitor compile controls, deterministic compact
artifact codec smoke checks, deterministic flight-sim CI path, S4 release gate lane.

**H5 (37 tasks):** SCPN controller optimisation (vectorized injection, array-native
stepping, Rust offload, binomial sampling, compiled feature evaluation, reusable scratch
buffers); HPC library loading security (trusted-path restriction); artifact decompression
bounds (zip-bomb prevention); solver fail-fast on divergence; Docker hardening (non-root);
CI `sc-neurocore` pinning; strict py312 mypy typing; adaptive margin regression locks;
`sys.path` removal across 20+ modules.

**H6 (9 tasks):** Reactor engineering elevation -- VMEC IO, particle tracker, forward
diagnostics, IMAS/IDS adapter, CAD raytrace, actuator dynamics, chaos monkey, MPI
scaffolding, traceable control loop.

**H7 (90 tasks):** Comprehensive Python runtime hardening -- deterministic non-interactive
summary APIs for all control/nuclear/diagnostics modules; scoped RNG (elimination of
global `np.random` mutation) across disruption predictor, digital twin, flight sim,
optimal control, SOC learning, FNO turbulence, design scanner, diagnostics, and
benchmark runners; strict input guards (finite/range/type validation) across all
control, nuclear, diagnostics, validation, SCPN contract, HPC, and IO modules;
NumPy LIF fallback for neuro-cybernetic controller; director interface fallback
oversight; tomography non-SciPy solve path.

**H8 (112 tasks):** Rust runtime hardening -- finite-value guards, shape validation,
and domain bounds enforcement across all `fusion-core` crates (kernel, inverse,
particles, transport, AMR, stability, RF heating, vacuum, B-field, X-point, memory
transport, plasma source, VMEC interface, MPI domain, JIT, pedestal, ignition) and
all `fusion-control` crates (MPC, PID, IsoFlux, SNN, optimal, SPI, analytic, SOC
learning, digital twin with OU noise/chaos monkey/SimpleMLP/Plasma2D/actuator);
explicit `FusionResult` error propagation replacing silent coercions/clamps/fallbacks;
rustfmt and clippy CI fixes; Python-side IDS validation tightening, forward diagnostics
grid-axis monotonicity enforcement, psi RMSE convergence/encoding/input contracts,
solver-method bridge regression tests.

---

## [2.0.0-pre] — Pre-Hardening

### Point-Wise ψ RMSE Validation (All 8 SPARC GEQDSKs)

- Added `validation/psi_pointwise_rmse.py` — self-contained module for point-wise
  ψ(R,Z) reconstruction error metrics on real SPARC equilibrium data:
  - `gs_operator()`: finite-difference Grad-Shafranov operator Δ*ψ
  - `gs_residual()`: relative L2 and max-abs GS residual
  - `manufactured_solve_vectorised()`: Red-Black SOR with reference boundary conditions
  - `compute_psi_rmse()`: normalised ψ RMSE, plasma-region RMSE, max pointwise error
  - `validate_file()` / `validate_all_sparc()`: per-file and aggregate runners
  - `sparc_psi_rmse()`: drop-in for rmse_dashboard integration
- Added `tests/test_psi_pointwise_rmse.py` — 17 tests covering GS operator
  analytics, SPARC integration, manufactured solve, RMSE metrics
- Integrated psi RMSE into `validation/rmse_dashboard.py` report output

### Neural Equilibrium Rewrite & Mode Deprecation

- Rewrote `src/scpn_fusion/core/neural_equilibrium.py` from 157 → 530 lines:
  - Removed `sklearn` dependency → `MinimalPCA` (pure NumPy SVD)
  - Removed `pickle` persistence → `.npz` save/load (`allow_pickle=False`)
  - Added `SimpleMLP` with He initialisation, ReLU hidden layers, full backprop
  - Added `train_from_geqdsk()`: trains on real SPARC GEQDSK data with profile
    perturbations (8-dim input: I_p, B_t, R_axis, Z_axis, pprime_scale,
    ffprime_scale, simag, sibry)
  - Added `train_on_sparc()` convenience function for all 8 files
  - Added `benchmark()` timing method
- Added `tests/test_neural_equilibrium.py` — 19 tests (PCA round-trip, MLP shapes,
  save/load, SPARC training integration, benchmark timing)
- Moved "neural" mode from `PUBLIC_MODES` to `SURROGATE_MODES` in
  `run_fusion_suite.py` — now behind `--surrogate` / `SCPN_SURROGATE=1` flag

### Multi-Regime SPARC-Parameterized FNO Training

- Added multi-regime turbulence data generator to `fno_training.py`:
  - `SPARC_REGIMES` dict: ITG / TEM / ETG parameter ranges (adiabaticity α,
    gradient drive κ, viscosity ν, nonlinear damping, spectral cutoff k_c)
  - `_generate_multi_regime_pairs()`: modified Hasegawa-Wakatani with regime-
    dependent dispersion ω = α·k_y/(α+k²), growth γ = κ·k_y·k²/(α+k²)² − ν·k⁴,
    and spectral filtering
  - `train_fno_multi_regime()`: full training loop with per-regime validation
    breakdown and regime distribution logging
- Added `tests/test_fno_multi_regime.py` — 18 tests (regime sampling, spectral
  character verification, training smoke, weights round-trip)
- Weights saved to `weights/fno_turbulence_sparc.npz` (multi-regime); legacy
  single-regime weights at `weights/fno_turbulence.npz` preserved

### Documentation — README Repositioning

- Repositioned project identity in README.md as a **neuro-symbolic control framework**
  rather than a physics simulation suite competing with TRANSP/GENE/JINTRAC.
- Added Design Philosophy table (control-first, graceful degradation, explicit errors,
  real data validation, reduced-order by design).
- Expanded Neuro-Symbolic Compiler section from 7 lines to ~40 lines with pipeline
  diagram, 5-stage description, and "Why This Matters" rationale.
- Reorganised simulation modes into 4 maturity tiers (Production / Validated /
  Reduced-order / Experimental) with hardening task counts.
- Added Code Health & Hardening section documenting all 8 hardening waves (248 tasks).
- Added Known Limitations & Roadmap section with honest gap table, strengths table,
  and DOE Fusion S&T Roadmap alignment.

### Multigrid Wiring and Experimental Validation

- Wired geometric multigrid V-cycle solver into `FusionKernel` Picard loop:
  - Added `SolverMethod` enum (`PicardSor`, `PicardMultigrid`) in `fusion-core/src/kernel.rs`
  - Added `set_solver_method()` / `solver_method()` accessors on `FusionKernel`
  - Exported `multigrid` module from `fusion-math/src/lib.rs`
  - Inner Picard solve now dispatches to Red-Black SOR (default) or multigrid V-cycle
- Exposed solver method control in PyO3 bindings:
  - `PyFusionKernel.set_solver_method("sor" | "multigrid")` in `fusion-python/src/lib.rs`
  - `RustAcceleratedKernel.set_solver_method()` / `.solver_method()` in `_rust_compat.py`
- Added `validation/benchmark_solvers.py`: 3-way SOR vs Multigrid vs Python timing comparison
- Added `validation/run_experimental_validation.py`: unified runner for all 8 SPARC equilibrium files
  with topology checks (axis, q-profile, GS sign) and solver validation modes

### Cutting-Edge Release Staging

- Prepared `v2.0-cutting-edge` release contract:
  - Novum elevated - SNN + GyroSwin hybrids for resilient, 1000x fast control; MVR grounded in 2025-2026 liquid metal/HTS.
- Added deterministic release-readiness validator (`validation/gdep_05_release_readiness.py`) with tracker/changelog gate checks.

### Stability and CI

- Hardened 3D LCFS extraction for coarse smoke meshes:
  - per-ray nearest-boundary fallback when strict crossing is absent
  - conservative ellipse fallback for sparse contours
- Added geometry regression coverage for sparse/non-crossing LCFS rays.
- Fixed CI formatting regression in inverse Jacobian tests (`cargo fmt --check` pass restored).

### Solver Validation and Numerics

- Added kernel-level analytical-vs-finite-difference Jacobian consistency test for inverse reconstruction.
- Added reduced particle feedback lane in `fusion-core`:
  - new `particles.rs` with Boris pusher and toroidal current deposition
  - optional kernel hook `set_particle_current_feedback(...)` to blend particle `J_phi` into GS source updates
  - regression tests for Boris invariance, deposition non-zero support, blend renormalization, and shape guard
- Added symplectic integration baseline in `fusion-math`:
  - new `symplectic.rs` with canonical velocity-Verlet stepper and RK4 reference
  - long-horizon Hamiltonian drift checks for harmonic-oscillator stress profiles
  - regression coverage for bounded symplectic drift and coarse-step drift superiority vs RK4
- Added reduced non-LTE impurity-radiation lookup in `fusion-nuclear/wall_interaction.rs`:
  - ADAS-style synthetic PEC grids with charge-state distinction (including W20+ and W40+)
  - bilinear PEC interpolation in `(log ne, log Te)` space
  - collisional-radiative power-density and volume-integrated radiative-loss helpers
- Added reduced IGA/NURBS geometry lane:
  - new `fusion-math/src/iga.rs` for NURBS curve evaluation and open-uniform knot generation
  - NURBS-based first-wall contour generator (`generate_first_wall_nurbs`) in `fusion-nuclear`
  - regression checks for endpoint consistency, contour closure, and analytic-envelope proximity
- Added latency-aware control realism extensions in `fusion-control`:
  - vector Ornstein-Uhlenbeck noise layer for multi-channel perturbations
  - actuator delay-line with configurable lag blending
  - DDE-like delayed rollout path in MPC (`plan_with_delay_dynamics`)
- Added reduced runtime regime-specialized kernel lane in `fusion-core`:
  - new `fusion-core/src/jit.rs` with deterministic regime routing and specialization specs
  - compile-cache and hot-swap manager (`RuntimeKernelJit`) for regime-triggered kernel activation
  - regression tests for cache reuse, hot-swap response divergence, and identity-safe fallback
- Upgraded AMR hierarchy and kernel integration to level-aware scaling (`2^level`) for multi-level patch colocation/prolongation.
- Added AMR multilevel tests for hierarchy generation and level-scaled interpolation.
- Added low-order toroidal mode coupling closure in `fusion-core` transport:
  - configurable `n=1..N` mode spectrum via `set_toroidal_mode_spectrum(...)`
  - edge-weighted diffusivity coupling factor for reduced `n!=0` transport effects
  - regression tests for baseline parity, edge-dominant coupling, and clamp safety
- Added reduced 3D-aware toroidal-harmonic coupling in `fusion-physics` FNO turbulence:
  - configurable toroidal harmonic spectrum via `set_toroidal_harmonics(...)`
  - non-zonal low-k spectral amplification closure with bounded coupling gain
  - regression tests for default parity, non-zonal energy increase, and clamp safety
- Added reduced 3D divertor strike-point asymmetry projection in `fusion-nuclear`:
  - toroidal/poloidal heat-flux map projection via `project_heat_flux_3d(...)`
  - low-order toroidal mode shaping through `ToroidalMode { n, amplitude, phase }`
  - map normalization preserving Eich-derived mean target load
  - regression tests for strike localization, toroidal asymmetry increase, and mean-load conservation
- Added reduced VMEC-like native 3D equilibrium interface:
  - new `VMECStyleEquilibrium3D` + `FourierMode3D` in `src/scpn_fusion/core/equilibrium_3d.py`
  - native flux-coordinate mapping `(rho, theta, phi) -> (R, Z) -> (x, y, z)`
  - integration with `Reactor3DBuilder` for direct 3D surface generation without 2D revolve-only path
  - builder utility `build_vmec_like_equilibrium(...)` to infer baseline shaping from traced 2D LCFS
  - regression coverage in `tests/test_equilibrium_3d.py` and quickstart support for VMEC-like mode
- Added reduced 3D field-line and Poincare diagnostics:
  - new `FieldLineTracer3D` in `src/scpn_fusion/core/fieldline_3d.py`
  - reduced helical field-line tracing over VMEC-like flux coordinates
  - Poincare section generation for arbitrary toroidal cut planes
  - `Reactor3DBuilder` helpers: `create_fieldline_tracer(...)` and `generate_poincare_map(...)`
  - new quickstart workflow in `examples/run_3d_fieldline_quickstart.py`
  - regression coverage in `tests/test_fieldline_3d.py`
- Added PCE checked APIs:
  - `PCEModel::try_fit(...) -> FusionResult<PCEModel>`
  - `PCEModel::try_predict(...) -> FusionResult<Array1<f64>>`
- Added PCE edge-case tests for non-finite inputs and prediction dimension mismatch.
- Extended pedestal model tests for ELM cooldown gating and minimum profile floor guarantees.

### Release and Metadata

- Added cross-file release metadata consistency test (`tests/test_version_metadata.py`) covering:
  - `pyproject.toml`
  - `setup.py`
  - `CITATION.cff`
  - `src/scpn_fusion/__init__.py`
  - `docs/sphinx/conf.py`
- Removed hardcoded Sphinx release fallback; now resolves dynamically from package source metadata.

### SCPN Controller Artifact

- Centralized artifact schema version as `ARTIFACT_SCHEMA_VERSION` in `scpn/artifact.py`.
- Updated compiler artifact export to use package version for `meta.compiler.version`.
- Added git SHA resolution for artifacts:
  - prefers `SCPN_GIT_SHA`, `GITHUB_SHA`, `CI_COMMIT_SHA`
  - falls back to `git rev-parse --short HEAD`
  - final fallback `0000000`
- Added controller/compiler tests for schema version, package version, and env-driven git SHA stamping.

### Documentation

- Added Sphinx GPU roadmap page (`docs/sphinx/gpu_roadmap.rst`) and linked it in docs navigation.

## 1.0.2 - 2026-02-12

### Solvers and Numerics

- Added analytical Jacobian pathway for inverse reconstruction with selectable `JacobianMode`.
- Added Chebyshev-accelerated SOR and benchmark comparison against fixed SOR.
- Added phase-space memory kernel transport solver.
- Added patch-based AMR baseline:
  - `fusion-math/src/amr.rs` hierarchy and refinement criterion (`|∇²J_phi|`)
  - `fusion-core/src/amr_kernel.rs` AMR-assisted solve wrapper

### Physics and Transport

- Added EPED-like pedestal model (`fusion-core/src/pedestal.rs`):
  - pedestal width scaling: `Δ_ped ~ sqrt(beta_p,ped) * (rho_s / R)`
  - ELM trigger and crash application on edge profiles
- Integrated pedestal model into the transport solver H-mode path.

### Machine Learning and UQ

- Added multi-layer FNO training pipeline and Rust weight loading path.
- Added neural transport surrogate in Rust with `.npz` loading.
- Added neural transport training and validation scripts/docs.
- Added Polynomial Chaos Expansion UQ module in Rust:
  - multivariate Hermite basis generation
  - Latin Hypercube sampling utility
  - fit/predict and first-order Sobol indices

### Engineering and Bindings

- Implemented `fusion-engineering` modules for tritium, blanket, magnets, and layout.
- Expanded PyO3 bindings to expose inverse, transport, and plant APIs.
- Added benchmark suites for inverse Jacobians and neural transport throughput.

### Validation and Documentation

- Added ITER/SPARC regression reference data and validation tests.
- Added GPU acceleration implementation roadmap (`docs/GPU_ACCELERATION_ROADMAP.md`).
- Added this changelog and CFF citation metadata.

### Versioning

- Python package version bumped to `1.0.2` (`pyproject.toml`, `setup.py`, `src/scpn_fusion/__init__.py`).
- `fusion-engineering` crate version bumped to `0.2.0`.
