# Architecture

SCPN Fusion Core is a three-tier plasma physics framework: a Rust GPU/WGPU
compute engine, a Python equilibrium/transport/control layer, and a
validation pipeline backed by real tokamak data.

**Totals:** 234 Python source files, 62,570 lines, 11 Rust crates, 2859 tests.

## Python Subsystems

### `core/` — 118 files

Equilibrium, transport, neural surrogates, gyrokinetic, MHD, disruptions, and
edge physics.

| Subarea | Key Modules | Count |
|---------|-------------|-------|
| GS equilibrium | `fusion_kernel`, `force_balance`, `eqdsk`, `amr_patch`, `neural_equilibrium` | 10 |
| JAX solvers | `jax_gs_solver`, `jax_neural_equilibrium`, `jax_solvers` (Thomas + Crank-Nicolson) | 3 |
| Transport | `integrated_transport_solver`, `jax_transport_solver`, `neural_transport`, `scaling_laws`, `neoclassical_transport` | 8 |
| GK three-path | Native linear eigenvalue, quasilinear flux, 5 external interfaces (TGLF, GENE, GS2, CGYRO, QuaLiKiz), OOD detection, correction, scheduling, online learning, verification | 18 |
| MHD stability | `stability_mhd`, `mhd_sawtooth`, `hall_mhd_discovery`, `stability_analyzer` | 6 |
| Disruptions | `disruption_sequence`, `locked_mode`, `tearing_mode_coupling` | 3 |
| Edge physics | `eped_pedestal`, `divertor_thermal_sim`, `blob_transport`, `marfe`, `lh_transition` | 7 |
| Neural surrogates | `fno_training`, `fno_jax_training`, `fno_turbulence_suppressor`, `gyro_swin_surrogate`, `pretrained_surrogates`, `neural_turbulence` | 8 |
| Scenario | `scenario_simulator`, `compact_reactor_optimizer`, `global_design_scanner`, `plasma_startup` | 5 |
| Geometry & 3D | `equilibrium_3d`, `fieldline_3d`, `geometry_3d`, `vmec_lite` | 4 |
| UQ & config | `uncertainty`, `config_schema`, `state_space`, `gpu_runtime`, `quantum_bridge` | 8 |
| Remaining | turbulence oracle, WDM engine, heating/neutronics contracts, RF heating, sandpile reactor, Lazarus/Vibrana bridges, etc. | 38 |

### `control/` — 54 files

Real-time plasma control: classical, robust, optimal, neural, and RL.

| Subarea | Key Modules | Count |
|---------|-------------|-------|
| Classical | PID, LQR, fusion control room | 4 |
| Robust | H-infinity, mu-synthesis, RWM feedback | 3 |
| Optimal | `fusion_optimal_control`, `fusion_sota_mpc`, `fusion_nmpc_jax` | 3 |
| Neural / SNN | `neuro_cybernetic_controller`, `nengo_snn_wrapper`, SNN compiler target | 3 |
| RL | `gym_tokamak_env`, advanced SOC fusion learning | 2 |
| Free-boundary | free-boundary tracking, state estimator (EKF), volt-second manager | 3 |
| Disruption | `disruption_predictor`, `disruption_contracts`, `spi_mitigation`, `spi_ablation` | 4 |
| Burn / scenario | burn controller, detachment controller, density controller, fueling mode | 4 |
| Runtime | `jax_traceable_runtime`, `rust_flight_sim_wrapper`, `tokamak_flight_sim`, `tokamak_digital_twin` | 4 |
| Infrastructure | `hil_harness`, `replay_pipeline`, `digital_twin_ingest`, `director_interface`, `torax_hybrid_loop`, `analytic_solver` | 6 |
| Edge physics | `fokker_planck_re`, `halo_re_physics`, runaway electrons | 3 |
| Remaining | orbit following, pellet injection, plasma-wall interaction, etc. | 15 |

### `phase/` — 10 files

Phase dynamics subpackage bridging SCPN oscillator theory to plasma physics.

| Module | Purpose |
|--------|---------|
| `kuramoto_upde` | Unified Phase Dynamics Equation solver |
| `adaptive_knm` | Adaptive coupling matrix K_nm |
| `gk_upde_bridge` | Gyrokinetic-to-UPDE coupling |
| `plasma_knm` | Plasma-physics-derived coupling matrix |
| `lyapunov_guard` | Lyapunov stability monitor |
| + 5 supporting modules | Config, types, utilities |

### `scpn/` — 12 files

Neuro-symbolic Petri net compiler and runtime.

| Module | Purpose |
|--------|---------|
| `compiler` | SCPN Petri net → SNN compilation |
| `contracts` | Formal verification contracts |
| `controller` | SCPN-based control dispatch |
| `safety_interlocks` | Safety interlock logic |
| `artifact` | Build artifact management |
| `structure` | Network structure definitions |
| + 6 supporting modules | Init, types, utilities |

### `io/` — 15 files

Data I/O and archival.

| Module | Purpose |
|--------|---------|
| `imas_connector` | IMAS IDS read/write |
| `imas_adapter` | IMAS data adaptation layer |
| `tokamak_archive` | Multi-machine shot archive (DIII-D, JET, KSTAR) |
| `eqdsk` | GEQDSK equilibrium file read/write |
| `logging_config` | Structured logging |
| + 10 supporting modules | Format converters, utilities |

### Other Python Subsystems

| Subsystem | Files | Key Modules |
|-----------|-------|-------------|
| `diagnostics/` | 5 | Forward models, synthetic sensors, tomography |
| `engineering/` | 4 | Balance of plant, CAD raytrace, thermal hydraulics |
| `nuclear/` | 5 | Blanket neutronics, PWI erosion, tritium breeding |
| `hpc/` | 2 | C++ solver bridge, HPC types |
| `ui/` | 4 | Streamlit dashboard, dashboard generator/launcher |

## Rust Workspace — 11 Crates

```
scpn-fusion-rs/crates/
├── fusion-core/        Grad-Shafranov solver, vacuum field
├── fusion-control/     PID controller, flight-sim kernel
├── fusion-diagnostics/ Synthetic diagnostic forward models
├── fusion-engineering/  Engineering subsystems
├── fusion-gpu/         WGPU compute shaders (gs_solver.wgsl)
├── fusion-math/        Linear algebra, interpolation
├── fusion-ml/          Surrogate model inference
├── fusion-nuclear/     Blanket neutronics, tritium breeding
├── fusion-physics/     MHD stability, scaling laws
├── fusion-python/      PyO3 Python ↔ Rust bridge (maturin)
└── fusion-types/       Shared types and configs
```

All Rust functionality has pure-Python fallbacks. The PyO3 bridge
(`fusion-python`) exposes the Rust kernels to Python with zero-copy
NumPy array interop.

## Data Flow

```
Python API  ──►  Rust Engine (PyO3)  ──►  WGPU compute shaders
    │                  │
    │                  ├──►  PID controller kernel (sub-µs latency)
    │                  │
    ▼                  ▼
Neural surrogates    Validation pipeline
(JAX/NumPy)         (FreeGS, TORAX, SPARC parity)
    │
    ├──►  GK three-path (native / TGLF / GENE / GS2 / CGYRO / QuaLiKiz)
    │
    ├──►  Phase dynamics (Kuramoto UPDE ↔ plasma K_nm)
    │
    ▼
IMAS connector  ──►  Tokamak data archive (DIII-D, JET, KSTAR)
```

## Build Targets

| Target | Command |
|--------|---------|
| Python package | `pip install -e ".[dev]"` |
| Rust workspace | `cd scpn-fusion-rs && cargo build --release` |
| Python ↔ Rust bridge | `cd scpn-fusion-rs/crates/fusion-python && maturin develop --release` |
| Tests (Python) | `pytest tests/ -v` |
| Tests (Rust) | `cd scpn-fusion-rs && cargo test` |
| Preflight gate | `python tools/run_python_preflight.py` |
| Docs | `mkdocs serve` |
| Validation suite | `python validation/full_validation_pipeline.py` |
