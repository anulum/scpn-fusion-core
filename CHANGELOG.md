# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Changelog
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

# Changelog

## Unreleased

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
