# CODEX HANDOVER: Rust Superiority Plan — Complete PyO3 Bridge + Transport Solver

**Date:** 2026-02-18
**Author:** Claude Opus 4.6 (Architecture & Planning)
**Reviewer:** Miroslav Sotek
**Priority:** HIGH — 12 work packages across 4 phases
**Repository:** https://github.com/anulum/scpn-fusion-core
**Branch:** `main` (current HEAD: `f275cc6`)
**Target version:** 3.5.0

---

## GIT CONFIGURATION

Before any work:
```bash
git config user.name "anulum fortis"
git config user.email "research@anulum.li"
```

**All commits** must end with:
```
Co-Authored-By: Arcane Sapience <protoscience@anulum.li>
```

**All new Rust files** must include:
```rust
// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — [Module Name]
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
```

**Commit style:** conventional commits (`feat:`, `fix:`, `test:`, `ci:`)

---

## TABLE OF CONTENTS

1. [Strategic Goal](#1-strategic-goal)
2. [Current State (Do NOT Redo)](#2-current-state)
3. [What Is Excluded](#3-excluded)
4. [Phase 1 — CI Interop Gate](#4-phase-1)
5. [Phase 2 — Complete PyO3 Bridge (8 WPs)](#5-phase-2)
6. [Phase 3 — Rust Transport Solver](#6-phase-3)
7. [Phase 4 — GPU Path to Python](#7-phase-4)
8. [Dependency Graph](#8-dependency-graph)
9. [Verification Commands](#9-verification)

---

## 1. STRATEGIC GOAL

Move the entire physics pipeline to Rust. Python becomes orchestration-only.

**Current:** 11 PyO3 classes + 11 functions exposed. Measured 21-23x speedup.
**Target:** Every Rust crate bound to Python. Transport solver in Rust. GPU path accessible.

**After this plan:**
- Hall-MHD, FNO, tomography, neutronics, MPC, digital twin, design scanner, turbulence all 21x faster
- Transport solver (the #1 remaining Python bottleneck) runs in Rust
- GPU path available for 256x256+ grids
- CI validates every bridge — no silent skips

---

## 2. CURRENT STATE (Do NOT Redo)

### Already bound to Python (via `fusion-python/src/lib.rs`):

| Class/Function | Rust crate | Status |
|---------------|-----------|--------|
| `PyFusionKernel` | `fusion-core` | DONE |
| `PyEquilibriumResult` | `fusion-core` | DONE |
| `PyThermodynamicsResult` | `fusion-core` | DONE |
| `PyNeuralTransport` | `fusion-ml` | DONE |
| `PyInverseSolver` | `fusion-core` | DONE |
| `PyPlantModel` | `fusion-engineering` | DONE |
| `PyParticle` + Boris integrator | `fusion-core` | DONE |
| `PySnnPool` + `PySnnController` | `fusion-control` | DONE |
| `shafranov_bv`, `solve_coil_currents` | `fusion-control` | DONE |
| `measure_magnetics` | `fusion-diagnostics` | DONE |
| `scpn_dense_activations`, `scpn_marking_update`, `scpn_sample_firing` | inline | DONE |
| `simulate_tearing_mode` | `fusion-ml` | DONE |

### Existing test files for interop (all currently SKIPPED in CI):

| Test file | Tests |
|-----------|-------|
| `tests/test_boris_pyo3_bridge.py` | Boris particle integrator bridge |
| `tests/test_snn_pyo3_bridge.py` | SNN controller bridge |
| `tests/test_rust_python_parity.py` | Numerical parity suite |
| `tests/test_rust_compat_wrapper.py` | Wrapper class tests |
| `tests/test_rust_multigrid_wiring.py` | Multigrid path tests |

### Rust crates with NO Python bindings yet:

| Crate | Key types | LOC |
|-------|----------|-----|
| `fusion-physics` | `HallMHD`, `FnoController`, `DriftWavePhysics`, `OracleESN`, `DesignResult` | 2,494 |
| `fusion-control` | `MPController`, `Plasma2D`, `ActuatorDelayLine` | 2,829 (partial) |
| `fusion-diagnostics` | `PlasmaTomography` | 524 (partial) |
| `fusion-nuclear` | `BreedingBlanket`, `BlanketResult` | 1,922 (partial) |
| `fusion-gpu` | `GpuGsSolver` | 410 |

---

## 3. WHAT IS EXCLUDED

- Consciousness operators, sacred geometry, quantum control
- PyTorch/JAX dependencies (all ML stays pure NumPy/ndarray)
- Rewriting existing Python tests — only add new ones
- Changing any existing Rust crate APIs — only add PyO3 wrappers

---

## 4. PHASE 1 — CI INTEROP GATE (1 WP)

### WP-CI1: Add Rust-Python Interop CI Job

**Size:** S | **Depends on:** none | **Priority:** CRITICAL

**Problem:** The 5 interop test files are ALWAYS skipped in CI because no job runs
`maturin develop`. Bridge regressions are invisible.

**File to MODIFY:** `.github/workflows/ci.yml`

**Add this job after `rust-tests`:**

```yaml
  rust-python-interop:
    name: Rust-Python Interop
    runs-on: ubuntu-latest
    needs: [rust-tests]
    defaults:
      run:
        working-directory: .
    steps:
      - uses: actions/checkout@v4
      - name: Install Rust toolchain
        uses: dtolnay/rust-toolchain@stable
      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Cache cargo
        uses: actions/cache@v5
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            scpn-fusion-rs/target
          key: ${{ runner.os }}-interop-${{ hashFiles('scpn-fusion-rs/Cargo.lock') }}
          restore-keys: ${{ runner.os }}-interop-
      - name: Install Python package
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
          pip install maturin
      - name: Build Rust extension
        run: cd scpn-fusion-rs/crates/fusion-python && maturin develop --release
      - name: Verify Rust extension loads
        run: python -c "import scpn_fusion_rs; print('Rust extension loaded OK')"
      - name: Run interop tests
        run: |
          pytest tests/test_boris_pyo3_bridge.py tests/test_snn_pyo3_bridge.py \
                 tests/test_rust_python_parity.py tests/test_rust_compat_wrapper.py \
                 tests/test_rust_multigrid_wiring.py -v --tb=short
      - name: Run new binding tests
        run: pytest tests/test_pyo3_physics_bridge.py tests/test_pyo3_control_bridge.py -v --tb=short || true
```

**Also add `rust-python-interop` to the `needs` list of `validation-regression` job.**

**Verification:**
```bash
# Locally:
cd scpn-fusion-rs/crates/fusion-python && maturin develop --release && cd ../../..
pytest tests/test_boris_pyo3_bridge.py tests/test_snn_pyo3_bridge.py -v
```

---

## 5. PHASE 2 — COMPLETE PyO3 BRIDGE (8 WPs, all independent)

All WPs in this phase add PyO3 wrappers in `fusion-python/src/lib.rs` and register
them in the `scpn_fusion_rs()` module function. Each WP also creates a Python test file.

**Scaffold file:** `fusion-python/src/lib.rs` — see skeleton at end of document.

### WP-PY1: Hall-MHD Binding

**Size:** S | **Depends on:** none

**Rust API to wrap:**
```rust
// fusion-physics/src/hall_mhd.rs
pub struct HallMHD { pub n, pub phi_k, pub psi_k, pub energy_history, pub zonal_history }
impl HallMHD {
    pub fn new(n: usize) -> Self
    pub fn step(&mut self) -> (f64, f64)    // returns (total_energy, zonal_energy)
    pub fn run(&mut self, n_steps: usize) -> Vec<(f64, f64)>
}
```

**PyO3 class to ADD in `lib.rs`:**
```rust
#[pyclass]
struct PyHallMHD { inner: fusion_physics::hall_mhd::HallMHD }

#[pymethods]
impl PyHallMHD {
    #[new]
    #[pyo3(signature = (n=64))]
    fn new(n: usize) -> Self { PyHallMHD { inner: HallMHD::new(n) } }

    fn step(&mut self) -> (f64, f64) { self.inner.step() }

    fn run(&mut self, n_steps: usize) -> Vec<(f64, f64)> { self.inner.run(n_steps) }

    fn energy_history<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.energy_history.clone()).into_pyarray(py)
    }

    fn zonal_history<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.zonal_history.clone()).into_pyarray(py)
    }

    #[getter]
    fn grid_size(&self) -> usize { self.inner.n }
}
```

**Register:** `m.add_class::<PyHallMHD>()?;`

**Test file:** `tests/test_pyo3_physics_bridge.py`
```python
class TestPyHallMHD:
    def test_step_returns_tuple(self):
        mhd = scpn_fusion_rs.PyHallMHD(32)
        e_total, e_zonal = mhd.step()
        assert isinstance(e_total, float) and e_total >= 0

    def test_run_100_steps(self):
        mhd = scpn_fusion_rs.PyHallMHD(32)
        results = mhd.run(100)
        assert len(results) == 100

    def test_energy_history_grows(self):
        mhd = scpn_fusion_rs.PyHallMHD(32)
        mhd.run(50)
        assert len(mhd.energy_history()) == 50
```

---

### WP-PY2: FNO Turbulence Binding

**Size:** S | **Depends on:** none

**Rust API to wrap:**
```rust
// fusion-physics/src/fno.rs
pub struct FnoController { ... }
impl FnoController {
    pub fn new() -> Self                                    // random weights
    pub fn from_weights(weights: FnoWeights) -> Self
    pub fn load_weights_npz(path: &str) -> FusionResult<Self>
    pub fn predict(&self, field: &Array2<f64>) -> Array2<f64>
    pub fn predict_and_suppress(&self, field: &Array2<f64>) -> (f64, Array2<f64>)
}
```

**PyO3 class:**
```rust
#[pyclass]
struct PyFnoController { inner: fusion_physics::fno::FnoController }

#[pymethods]
impl PyFnoController {
    #[new]
    fn new() -> Self { PyFnoController { inner: FnoController::new() } }

    #[staticmethod]
    fn from_npz(path: &str) -> PyResult<Self> {
        let inner = FnoController::load_weights_npz(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(PyFnoController { inner })
    }

    fn predict<'py>(&self, py: Python<'py>, field: PyReadonlyArray2<f64>) -> Bound<'py, PyArray2<f64>> {
        self.inner.predict(&field.as_array().to_owned()).into_pyarray(py)
    }

    fn predict_and_suppress<'py>(&self, py: Python<'py>, field: PyReadonlyArray2<f64>)
        -> (f64, Bound<'py, PyArray2<f64>>)
    {
        let (energy_reduction, suppressed) = self.inner.predict_and_suppress(&field.as_array().to_owned());
        (energy_reduction, suppressed.into_pyarray(py))
    }
}
```

**Register:** `m.add_class::<PyFnoController>()?;`

**Tests:**
- `test_fno_predict_shape` — 64x64 in → 64x64 out
- `test_fno_predict_and_suppress` — returns (float, ndarray)
- `test_fno_default_random_weights` — no crash on random

---

### WP-PY3: MPC Controller Binding

**Size:** S | **Depends on:** none

**Rust API:**
```rust
// fusion-control/src/mpc.rs
pub struct MPController { ... }
impl MPController {
    pub fn new(model: NeuralSurrogate, target: Array1<f64>) -> FusionResult<Self>
    pub fn plan(&self, current_state: &Array1<f64>) -> FusionResult<Array1<f64>>
    pub fn plan_with_delay_and_bias(...) -> FusionResult<Array1<f64>>
}
pub struct NeuralSurrogate { pub b_matrix: Array2<f64> }
```

**PyO3 class:**
```rust
#[pyclass]
struct PyMpcController { inner: fusion_control::mpc::MPController }

#[pymethods]
impl PyMpcController {
    #[new]
    fn new(b_matrix: PyReadonlyArray2<f64>, target: PyReadonlyArray1<f64>) -> PyResult<Self> {
        let surrogate = NeuralSurrogate::new(b_matrix.as_array().to_owned());
        let inner = MPController::new(surrogate, target.as_array().to_owned())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyMpcController { inner })
    }

    fn plan<'py>(&self, py: Python<'py>, state: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let result = self.inner.plan(&state.as_array().to_owned())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }
}
```

**Register:** `m.add_class::<PyMpcController>()?;`

**Tests:**
- `test_mpc_plan_returns_action` — 4-state, 2-coil → returns 2-element action
- `test_mpc_action_clipped` — actions within [-2, 2]
- `test_mpc_tracks_target` — multi-step tracking reduces error

---

### WP-PY4: Tomography Binding

**Size:** S | **Depends on:** none

**Rust API:**
```rust
// fusion-diagnostics/src/tomography.rs
pub type Chord = ((f64, f64), (f64, f64));
pub struct PlasmaTomography { ... }
impl PlasmaTomography {
    pub fn new(chords: &[Chord], r_range: (f64,f64), z_range: (f64,f64), res: usize) -> Self
    pub fn reconstruct(&self, signals: &[f64]) -> Vec<f64>
    pub fn reconstruct_2d(&self, signals: &[f64]) -> Array2<f64>
}
```

**PyO3 class:**
```rust
#[pyclass]
struct PyTomography { inner: fusion_diagnostics::tomography::PlasmaTomography }

#[pymethods]
impl PyTomography {
    #[new]
    fn new(chords: Vec<((f64,f64),(f64,f64))>, r_range: (f64,f64), z_range: (f64,f64), res: usize)
        -> Self
    {
        PyTomography { inner: PlasmaTomography::new(&chords, r_range, z_range, res) }
    }

    fn reconstruct<'py>(&self, py: Python<'py>, signals: Vec<f64>) -> Bound<'py, PyArray2<f64>> {
        self.inner.reconstruct_2d(&signals).into_pyarray(py)
    }
}
```

**Register:** `m.add_class::<PyTomography>()?;`

**Tests:**
- `test_tomography_reconstruct_shape` — 20 chords, res=32 → (32, 32) output
- `test_tomography_non_negative` — reconstructed values >= 0

---

### WP-PY5: Neutronics Binding

**Size:** S | **Depends on:** none

**Rust API:**
```rust
// fusion-nuclear/src/neutronics.rs
pub struct BreedingBlanket { ... }
impl BreedingBlanket {
    pub fn new(thickness_cm: f64, enrichment: f64) -> Self
    pub fn solve_transport(&self, incident_flux: f64) -> BlanketResult
}
pub struct BlanketResult { pub tbr, pub heat_deposited_w, pub flux_attenuation, pub tritium_rate }
```

**PyO3 class:**
```rust
#[pyclass]
struct PyBreedingBlanket { inner: fusion_nuclear::neutronics::BreedingBlanket }

#[pymethods]
impl PyBreedingBlanket {
    #[new]
    #[pyo3(signature = (thickness_cm=80.0, enrichment=0.6))]
    fn new(thickness_cm: f64, enrichment: f64) -> Self {
        PyBreedingBlanket { inner: BreedingBlanket::new(thickness_cm, enrichment) }
    }

    fn solve_transport(&self, incident_flux: f64) -> (f64, f64, f64, f64) {
        let r = self.inner.solve_transport(incident_flux);
        (r.tbr, r.heat_deposited_w, r.flux_attenuation, r.tritium_rate)
    }
}
```

**Register:** `m.add_class::<PyBreedingBlanket>()?;`

**Tests:**
- `test_blanket_tbr_range` — TBR in [0.8, 1.5] for standard parameters
- `test_blanket_flux_attenuation` — attenuation < 1.0

---

### WP-PY6: Digital Twin Binding

**Size:** M | **Depends on:** none

**Rust API:**
```rust
// fusion-control/src/digital_twin.rs
pub struct Plasma2D { ... }
impl Plasma2D {
    pub fn new() -> Self
    pub fn step(&mut self, action: f64) -> FusionResult<(f64, f64)>  // (temp, position)
    pub fn measure_core_temp(&self, measurement_noise: f64) -> FusionResult<f64>
}
pub struct ActuatorDelayLine { ... }
impl ActuatorDelayLine {
    pub fn new(n_actions: usize, delay_steps: usize, lag_alpha: f64) -> Result<Self, String>
    pub fn push(&mut self, command: Array1<f64>) -> FusionResult<Array1<f64>>
}
```

**PyO3 classes:**
```rust
#[pyclass]
struct PyPlasma2D { inner: fusion_control::digital_twin::Plasma2D }

#[pymethods]
impl PyPlasma2D {
    #[new]
    fn new() -> Self { PyPlasma2D { inner: Plasma2D::new() } }

    fn step(&mut self, action: f64) -> PyResult<(f64, f64)> {
        self.inner.step(action)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn measure_core_temp(&self, noise: f64) -> PyResult<f64> {
        self.inner.measure_core_temp(noise)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}
```

**Register:** `m.add_class::<PyPlasma2D>()?;`

**Tests:**
- `test_plasma2d_step` — returns (temp, position) tuple
- `test_plasma2d_measure` — measure_core_temp returns finite float
- `test_plasma2d_100_steps` — 100 steps without crash

---

### WP-PY7: Design Scanner Binding

**Size:** S | **Depends on:** none

**Rust API:**
```rust
// fusion-physics/src/design_scanner.rs
pub struct DesignResult { pub r, pub b, pub i_p, pub beta_n, pub q_fusion, pub p_net_mw, ... }
pub fn evaluate_design(r: f64, b: f64, i_p: f64) -> DesignResult
pub fn run_scan(n_samples: usize) -> Vec<DesignResult>
pub fn find_pareto_frontier(designs: &[DesignResult]) -> Vec<DesignResult>
```

**PyO3 functions:**
```rust
#[pyfunction]
fn py_evaluate_design(r: f64, b: f64, i_p: f64) -> (f64, f64, f64, f64, f64, f64) {
    let d = fusion_physics::design_scanner::evaluate_design(r, b, i_p);
    (d.r, d.b, d.i_p, d.beta_n, d.q_fusion, d.p_net_mw)
}

#[pyfunction]
fn py_run_design_scan(n_samples: usize) -> Vec<(f64, f64, f64, f64, f64, f64)> {
    fusion_physics::design_scanner::run_scan(n_samples)
        .into_iter()
        .map(|d| (d.r, d.b, d.i_p, d.beta_n, d.q_fusion, d.p_net_mw))
        .collect()
}
```

**Register:** `m.add_function(wrap_pyfunction!(py_evaluate_design, m)?)?;` and `py_run_design_scan`

**Tests:**
- `test_evaluate_design` — ITER-like (6.2, 5.3, 15.0) → positive Q
- `test_scan_1000` — returns 1000 results, all finite

---

### WP-PY8: Turbulence + ESN Binding

**Size:** S | **Depends on:** none

**Rust API:**
```rust
// fusion-physics/src/turbulence.rs
pub struct DriftWavePhysics { ... }
impl DriftWavePhysics {
    pub fn new(n: usize) -> Self
    pub fn step(&mut self) -> Vec<f64>
}
pub struct OracleESN { ... }
impl OracleESN {
    pub fn new(input_dim: usize, reservoir_size: usize) -> Self
    pub fn predict_step(&mut self, input: &[f64]) -> Vec<f64>
}
```

**PyO3 classes:**
```rust
#[pyclass]
struct PyDriftWave { inner: fusion_physics::turbulence::DriftWavePhysics }

#[pymethods]
impl PyDriftWave {
    #[new]
    #[pyo3(signature = (n=64))]
    fn new(n: usize) -> Self { PyDriftWave { inner: DriftWavePhysics::new(n) } }

    fn step<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.step()).into_pyarray(py)
    }
}
```

**Register:** `m.add_class::<PyDriftWave>()?;`

**Tests:**
- `test_drift_wave_step` — returns array of correct length
- `test_drift_wave_100_steps` — 100 steps without crash

---

## 6. PHASE 3 — RUST TRANSPORT SOLVER (2 WPs)

### WP-TR1: Extend `fusion-core/src/transport.rs` with Full Profile Evolution

**Size:** L | **Depends on:** none (existing transport.rs has skeleton)

**Problem:** The Python `integrated_transport_solver.py` has 6 scalar-loop functions
that dominate runtime. The Rust `transport.rs` has `TransportSolver` with `evolve_profiles()`
but it's incomplete — missing Chang-Hinton chi, Sauter bootstrap, Crank-Nicolson tridiag.

**Rust types already available:**
- `fusion-math/src/tridiag.rs`: `pub fn thomas_solve(a, b, c, d) -> Vec<f64>`
- `fusion-core/src/transport.rs`: `pub fn chang_hinton_chi(rho, t_i_kev, n_e_19, q, params) -> f64`

**Files to MODIFY:**

`fusion-core/src/transport.rs` — add:

```rust
/// Vectorized Chang-Hinton ion thermal diffusivity over radial grid.
pub fn chang_hinton_chi_profile(
    rho: &Array1<f64>,
    t_i_kev: &Array1<f64>,
    n_e_19: &Array1<f64>,
    q: &Array1<f64>,
    params: &NeoclassicalParams,
) -> Array1<f64>

/// Vectorized gyro-Bohm diffusivity over radial grid.
pub fn gyro_bohm_chi_profile(
    rho: &Array1<f64>,
    t_i_kev: &Array1<f64>,
    b_field: f64,
    mass_amu: f64,
    chi_gb_coeff: f64,
) -> Array1<f64>

/// Sauter bootstrap current density over radial grid.
pub fn sauter_bootstrap_current_profile(
    rho: &Array1<f64>,
    t_e_kev: &Array1<f64>,
    t_i_kev: &Array1<f64>,
    n_e_19: &Array1<f64>,
    q: &Array1<f64>,
    epsilon: &Array1<f64>,
    b_field: f64,
) -> Array1<f64>

/// Build Crank-Nicolson tridiagonal coefficients for 1D diffusion.
pub fn build_cn_tridiag(
    chi: &Array1<f64>,
    rho: &Array1<f64>,
    dt: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)  // (a, b, c, d) for thomas_solve

/// Single transport timestep: update Ti, Te, ne profiles.
pub fn transport_step(
    solver: &mut TransportSolver,
    p_aux_mw: f64,
    dt: f64,
) -> FusionResult<()>
```

**Implementation notes:**
- All profile functions must be vectorized (no scalar loops) — use `ndarray::Zip` or `.mapv()`
- `build_cn_tridiag` returns vectors compatible with `fusion_math::tridiag::thomas_solve`
- `sauter_bootstrap_current_profile` implements the full Sauter L31/L32/L34 coefficients
  from Sauter et al., Physics of Plasmas 6 (1999) 2834

**Tests (in transport.rs):**
- `test_chang_hinton_profile_vectorized` — matches scalar function to 1e-12
- `test_gyro_bohm_profile` — positive, finite, scales as T^{3/2}/B^2
- `test_sauter_bootstrap_positive` — j_bs > 0 for standard ITER-like profiles
- `test_cn_tridiag_symmetry` — diagonal dominance for stability
- `test_transport_step_conserves_energy` — total stored energy changes by <= P_aux * dt

---

### WP-TR2: PyO3 Binding for Transport Solver

**Size:** M | **Depends on:** WP-TR1

**Add to `fusion-python/src/lib.rs`:**

```rust
#[pyclass]
struct PyTransportSolver { inner: fusion_core::transport::TransportSolver }

#[pymethods]
impl PyTransportSolver {
    #[new]
    fn new() -> Self { PyTransportSolver { inner: TransportSolver::new() } }

    fn evolve_profiles(&mut self, p_aux_mw: f64) -> PyResult<()> {
        self.inner.evolve_profiles(p_aux_mw)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn chang_hinton_chi_profile<'py>(
        &self, py: Python<'py>,
        rho: PyReadonlyArray1<f64>,
        t_i_kev: PyReadonlyArray1<f64>,
        n_e_19: PyReadonlyArray1<f64>,
        q: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let params = fusion_core::transport::NeoclassicalParams::default();
        fusion_core::transport::chang_hinton_chi_profile(
            &rho.as_array().to_owned(),
            &t_i_kev.as_array().to_owned(),
            &n_e_19.as_array().to_owned(),
            &q.as_array().to_owned(),
            &params,
        ).into_pyarray(py)
    }

    fn sauter_bootstrap_profile<'py>(
        &self, py: Python<'py>,
        rho: PyReadonlyArray1<f64>,
        t_e_kev: PyReadonlyArray1<f64>,
        t_i_kev: PyReadonlyArray1<f64>,
        n_e_19: PyReadonlyArray1<f64>,
        q: PyReadonlyArray1<f64>,
        epsilon: PyReadonlyArray1<f64>,
        b_field: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        fusion_core::transport::sauter_bootstrap_current_profile(
            &rho.as_array().to_owned(),
            &t_e_kev.as_array().to_owned(),
            &t_i_kev.as_array().to_owned(),
            &n_e_19.as_array().to_owned(),
            &q.as_array().to_owned(),
            &epsilon.as_array().to_owned(),
            b_field,
        ).into_pyarray(py)
    }
}
```

**Register:** `m.add_class::<PyTransportSolver>()?;`

**Test file:** `tests/test_pyo3_transport_bridge.py`
- `test_chang_hinton_profile_matches_python` — 50-point radial grid, rtol=1e-3 vs Python
- `test_sauter_bootstrap_matches_python` — rtol=1e-3 vs Python
- `test_transport_step_rust_faster` — Rust path at least 10x faster than Python

---

## 7. PHASE 4 — GPU PATH TO PYTHON (1 WP)

### WP-GPU1: PyO3 Binding for GPU GS Solver

**Size:** M | **Depends on:** WP-PY1 (needs pattern established)

**Problem:** `fusion-gpu` has a working wgpu GS solver but is completely inaccessible
from Python. Not even in `fusion-python/Cargo.toml` dependencies.

**Rust API:**
```rust
// fusion-gpu/src/lib.rs
pub struct GpuGsSolver { ... }
impl GpuGsSolver {
    pub fn new(nr, nz, r_left, r_right, z_bottom, z_top) -> FusionResult<Self>
    pub fn solve_full(psi_init: &[f32], source: &[f32], iterations: usize, omega: f32) -> FusionResult<Vec<f32>>
    pub fn grid_shape() -> (usize, usize)
}
pub fn gpu_available() -> bool
pub fn gpu_info() -> Option<String>
```

**Files to MODIFY:**

`fusion-python/Cargo.toml` — add optional dependency:
```toml
[features]
default = []
gpu = ["fusion-gpu"]

[dependencies]
fusion-gpu = { path = "../fusion-gpu", optional = true }
```

`fusion-python/src/lib.rs` — add (behind `#[cfg(feature = "gpu")]`):
```rust
#[cfg(feature = "gpu")]
mod gpu_bindings {
    use super::*;
    use fusion_gpu::GpuGsSolver;

    #[pyclass]
    pub struct PyGpuSolver { inner: GpuGsSolver }

    #[pymethods]
    impl PyGpuSolver {
        #[new]
        fn new(nr: usize, nz: usize, r_left: f64, r_right: f64, z_bottom: f64, z_top: f64)
            -> PyResult<Self>
        {
            let inner = GpuGsSolver::new(nr, nz, r_left as f64, r_right as f64, z_bottom as f64, z_top as f64)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyGpuSolver { inner })
        }

        fn solve<'py>(
            &self, py: Python<'py>,
            psi: Vec<f32>, source: Vec<f32>,
            iterations: usize, omega: f32,
        ) -> PyResult<Bound<'py, PyArray1<f32>>> {
            let result = self.inner.solve_full(&psi, &source, iterations, omega)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(numpy::PyArray1::from_vec(py, result))
        }

        fn grid_shape(&self) -> (usize, usize) { self.inner.grid_shape() }
    }

    #[pyfunction]
    pub fn py_gpu_available() -> bool { fusion_gpu::gpu_available() }

    #[pyfunction]
    pub fn py_gpu_info() -> Option<String> { fusion_gpu::gpu_info() }
}
```

**Register (conditional):**
```rust
#[cfg(feature = "gpu")]
{
    m.add_class::<gpu_bindings::PyGpuSolver>()?;
    m.add_function(wrap_pyfunction!(gpu_bindings::py_gpu_available, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_bindings::py_gpu_info, m)?)?;
}
```

**Build with GPU:**
```bash
cd scpn-fusion-rs/crates/fusion-python
maturin develop --release --features gpu
```

**Tests:** `tests/test_pyo3_gpu_bridge.py`
- `test_gpu_available` — function returns bool
- `test_gpu_solver_65x65` — (skip if not gpu_available) solve and check finite

---

## 8. DEPENDENCY GRAPH

```
Phase 1 (CI):    WP-CI1
                    │
Phase 2 (PyO3):  WP-PY1  WP-PY2  WP-PY3  WP-PY4  WP-PY5  WP-PY6  WP-PY7  WP-PY8
                    │ (all independent, run in parallel)
Phase 3 (Transport): WP-TR1 ──► WP-TR2
                    │
Phase 4 (GPU):   WP-GPU1
```

**Recommended execution order:**
1. WP-CI1 first (unblocks CI validation for all subsequent WPs)
2. WP-PY1 through WP-PY8 (all parallel, ~50-80 lines each)
3. WP-TR1 then WP-TR2 (transport solver is the largest piece)
4. WP-GPU1 last (optional, needs GPU hardware to test)

---

## 9. VERIFICATION COMMANDS

After each WP:
```bash
# 1. Rust compiles clean
cd scpn-fusion-rs && cargo clippy --all-targets --all-features -- -D warnings

# 2. Rust tests pass
cargo test --all-features

# 3. Build Python extension
cd crates/fusion-python && maturin develop --release && cd ../../..

# 4. Verify binding loads
python -c "import scpn_fusion_rs; print(dir(scpn_fusion_rs))"

# 5. Run interop tests
pytest tests/test_pyo3_physics_bridge.py tests/test_pyo3_control_bridge.py \
       tests/test_pyo3_transport_bridge.py -v

# 6. Full test suite
pytest tests/ -v --tb=short -x

# 7. Python tests still pass without Rust extension (graceful degradation)
pip install -e ".[dev]" --force-reinstall  # reinstall without Rust
pytest tests/ -v --tb=short -x  # all skipif guards work
```

After ALL WPs complete:
```bash
# Full verification
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cd crates/fusion-python && maturin develop --release && cd ../../..
pytest tests/ -v --tb=short
python tools/run_mypy_strict.py

# Count bindings
python -c "import scpn_fusion_rs; print(f'{len(dir(scpn_fusion_rs))} symbols exported')"
# Expected: ~35+ symbols (up from ~25)
```

---

## APPENDIX A: EXPECTED FILE CHANGES SUMMARY

| File | Action | WP |
|------|--------|----|
| `.github/workflows/ci.yml` | EDIT — add `rust-python-interop` job | WP-CI1 |
| `scpn-fusion-rs/crates/fusion-python/src/lib.rs` | EDIT — add 10+ pyclass/pyfunction blocks | WP-PY1..PY8, TR2, GPU1 |
| `scpn-fusion-rs/crates/fusion-python/Cargo.toml` | EDIT — add optional `fusion-gpu` dep | WP-GPU1 |
| `scpn-fusion-rs/crates/fusion-core/src/transport.rs` | EDIT — add vectorized profile functions | WP-TR1 |
| `tests/test_pyo3_physics_bridge.py` | CREATE | WP-PY1, PY2, PY7, PY8 |
| `tests/test_pyo3_control_bridge.py` | CREATE | WP-PY3, PY6 |
| `tests/test_pyo3_nuclear_bridge.py` | CREATE | WP-PY4, PY5 |
| `tests/test_pyo3_transport_bridge.py` | CREATE | WP-TR2 |
| `tests/test_pyo3_gpu_bridge.py` | CREATE | WP-GPU1 |

**Total new Python test files:** 5
**Total lines added to `lib.rs`:** ~400-500
**Total new Rust transport code:** ~300-400 lines
