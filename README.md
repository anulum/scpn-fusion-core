# SCPN Fusion Core

[![CI](https://github.com/anulum/scpn-fusion-core/actions/workflows/ci.yml/badge.svg)](https://github.com/anulum/scpn-fusion-core/actions/workflows/ci.yml)
[![Docs](https://github.com/anulum/scpn-fusion-core/actions/workflows/docs.yml/badge.svg)](https://anulum.github.io/scpn-fusion-core/)
[![PyPI](https://img.shields.io/pypi/v/scpn-fusion.svg)](https://pypi.org/project/scpn-fusion/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/scpn-fusion.svg)](https://pypi.org/project/scpn-fusion/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org/)
[![DOI](https://img.shields.io/badge/DOI-Zenodo-blue.svg)](https://zenodo.org/)

A comprehensive tokamak plasma physics simulation and control suite with neuro-symbolic compilation. SCPN Fusion Core models the full lifecycle of a fusion reactor — from Grad-Shafranov equilibrium and MHD stability through transport, heating, neutronics, and real-time disruption prediction — with optional Rust acceleration via PyO3 and an optional bridge to [SC-NeuroCore](https://github.com/anulum/sc-neurocore) spiking neural networks.

## Architecture

```
scpn-fusion-core/
├── src/scpn_fusion/           # Python package (46 modules)
│   ├── core/                  # Plasma physics engines
│   │   ├── fusion_kernel.py           Grad-Shafranov + transport solver
│   │   ├── compact_reactor_optimizer  MVR-0.96 compact reactor search
│   │   ├── mhd_sawtooth.py           MHD sawtooth crash simulator
│   │   ├── rf_heating.py             ICRH/ECRH/LHCD heating models
│   │   ├── divertor_thermal_sim.py   Divertor heat-flux solver
│   │   ├── hall_mhd_discovery.py     Hall-MHD two-fluid effects
│   │   ├── sandpile_fusion_reactor   SOC criticality model
│   │   ├── neural_equilibrium.py     Neural-network equilibrium solver
│   │   ├── fno_turbulence_suppressor Fourier Neural Operator turbulence model
│   │   ├── turbulence_oracle.py      ITG/TEM turbulence predictor
│   │   ├── wdm_engine.py             Warm dense matter EOS
│   │   ├── geometry_3d.py            3D flux-surface geometry
│   │   ├── global_design_scanner.py  Multi-objective design space explorer
│   │   └── integrated_transport      Coupled transport solver
│   ├── control/               # Reactor control & AI
│   │   ├── tokamak_flight_sim.py     Real-time flight simulator
│   │   ├── tokamak_digital_twin.py   Digital twin with live telemetry
│   │   ├── fusion_optimal_control    Model-predictive controller
│   │   ├── fusion_sota_mpc.py        State-of-the-art MPC
│   │   ├── disruption_predictor.py   ML disruption early-warning
│   │   ├── spi_mitigation.py         Shattered pellet injection
│   │   ├── fusion_control_room.py    Integrated control room sim
│   │   ├── neuro_cybernetic_controller  SNN-based feedback controller
│   │   └── advanced_soc_fusion_learning Self-organized criticality RL
│   ├── nuclear/               # Nuclear engineering
│   │   ├── blanket_neutronics.py     Tritium breeding ratio solver
│   │   ├── nuclear_wall_interaction  PMI / first-wall damage
│   │   ├── pwi_erosion.py            Plasma-wall erosion model
│   │   └── temhd_peltier.py          Thermoelectric MHD effects
│   ├── diagnostics/           # Synthetic diagnostics
│   │   ├── synthetic_sensors.py      Virtual instrument suite
│   │   └── tomography.py             Soft X-ray tomographic inversion
│   ├── engineering/           # Balance of plant
│   │   └── balance_of_plant.py       Thermal cycle, turbine, cryo
│   ├── scpn/                  # Neuro-symbolic compiler
│   │   ├── compiler.py               Petri nets → stochastic neurons
│   │   ├── controller.py             SNN-driven plasma control
│   │   ├── structure.py              Petri net data structures
│   │   ├── contracts.py              Formal verification contracts
│   │   └── artifact.py               Compilation artifact storage
│   ├── hpc/                   # High-performance computing
│   │   └── hpc_bridge.py             C++/Rust FFI bridge
│   └── ui/                    # Dashboard
│       └── app.py                    Streamlit real-time dashboard
├── scpn-fusion-rs/            # Rust workspace (10 crates)
│   ├── crates/
│   │   ├── fusion-types/      # Shared data types
│   │   ├── fusion-math/       # Linear algebra, FFT, interpolation
│   │   ├── fusion-core/       # Grad-Shafranov, transport in Rust
│   │   ├── fusion-physics/    # MHD, heating, turbulence
│   │   ├── fusion-nuclear/    # Neutronics, wall erosion
│   │   ├── fusion-engineering/ # Balance of plant
│   │   ├── fusion-control/    # PID, MPC, disruption predictor
│   │   ├── fusion-diagnostics/ # Sensor models
│   │   ├── fusion-ml/         # Inference engine
│   │   └── fusion-python/     # PyO3 bindings → scpn_fusion_rs.pyd
│   └── Cargo.toml             # Workspace manifest
├── tests/                     # Python test suite
├── docs/                      # Technical documentation
├── validation/                # ITER validation configurations
├── calibration/               # Optimization tools
└── schemas/                   # JSON schemas
```

## Quick Start

```bash
# Clone
git clone https://github.com/anulum/scpn-fusion-core.git
cd scpn-fusion-core

# Install (Python)
pip install -e .

# Run a simulation
python run_fusion_suite.py kernel       # Grad-Shafranov equilibrium
python run_fusion_suite.py optimizer    # Compact reactor search (MVR-0.96)
python run_fusion_suite.py flight       # Tokamak flight simulator
python run_fusion_suite.py neural       # Neural equilibrium solver
python examples/run_3d_flux_quickstart.py --toroidal 24 --poloidal 24
python examples/run_3d_flux_quickstart.py --toroidal 24 --poloidal 24 --preview-png artifacts/SCPN_Plasma_3D_quickstart.png

# Run tests
pytest tests/ -v

# Generate validation RMSE dashboard
python validation/rmse_dashboard.py
```

The 3D quickstart writes an OBJ mesh to `artifacts/SCPN_Plasma_3D_quickstart.obj` and can optionally render a PNG preview.

### Rust Acceleration (Optional)

```bash
cd scpn-fusion-rs
cargo build --release
cargo test

# Build Python bindings (requires maturin)
pip install maturin
cd crates/fusion-python
maturin develop --release
```

The Python package auto-detects the Rust extension and falls back to NumPy if unavailable.

### Testing

```bash
# Python unit + property-based tests
pip install -e ".[dev]"
pytest tests/ -v

# Rust unit + property-based tests
cd scpn-fusion-rs
cargo test --all-features

# Rust benchmarks
cargo bench
```

The test suites include property-based tests powered by [Hypothesis](https://hypothesis.readthedocs.io/) (Python) and [proptest](https://crates.io/crates/proptest) (Rust), covering numerical invariants, topology preservation, and solver convergence properties.

## Tutorial Notebooks

| Notebook | Description |
|----------|-------------|
| `01_compact_reactor_search` | MVR-0.96 compact reactor optimizer walkthrough |
| `02_neuro_symbolic_compiler` | Petri net → stochastic neuron compilation pipeline |
| `03_grad_shafranov_equilibrium` | Free-boundary equilibrium solver tutorial |
| `04_divertor_and_neutronics` | Divertor heat flux & tritium breeding ratio |
| `05_validation_against_experiments` | Cross-validation vs SPARC GEQDSK & ITPA scaling |
| `06_inverse_and_transport_benchmarks` | Inverse solver & neural transport surrogate benchmarks |

## Validation Against Experimental Data

The `validation/` directory contains reference data from real tokamaks for cross-checking simulation outputs:

| Dataset | Source | Contents |
|---------|--------|----------|
| **SPARC GEQDSK** | [SPARCPublic](https://github.com/cfs-energy/SPARCPublic) | 8 equilibrium files (B=12.2 T, I_p up to 8.7 MA) |
| **ITPA H-mode** | Verdoolaege et al., NF 61 (2021) | Confinement data from 10 tokamaks (JET, DIII-D, C-Mod, ...) |
| **IPB98(y,2)** | ITER Physics Basis | Scaling law coefficients + uncertainties |
| **ITER configs** | Internal | 4 coil-optimised ITER configurations |
| **SPARC config** | Creely et al., JPP 2020 | Machine parameters for compact high-field design |
| **DIII-D config** | Luxon, NF 42 (2002) | Medium-size US tokamak parameters |
| **JET config** | Pamela et al. (2007) | Largest tokamak, DT fusion data |

```bash
# Run validation script
python validation/validate_against_sparc.py

# Read a GEQDSK equilibrium
python -c "from scpn_fusion.core.eqdsk import read_geqdsk; eq = read_geqdsk('validation/reference_data/sparc/lmode_vv.geqdsk'); print(f'B={eq.bcentr:.1f}T, Ip={eq.current/1e6:.1f}MA')"
```

## 26 Simulation Modes

| Mode | Description |
|------|-------------|
| `kernel` | Grad-Shafranov equilibrium + coupled transport |
| `flight` | Real-time tokamak flight simulator |
| `optimal` | Model-predictive optimal control |
| `learning` | Self-organized criticality reinforcement learning |
| `digital-twin` | Live digital twin with telemetry |
| `control-room` | Integrated control room simulation |
| `sandpile` | SOC sandpile criticality model |
| `nuclear` | Plasma-wall interaction & first-wall damage |
| `breeding` | Tritium breeding blanket neutronics |
| `safety` | ML disruption predictor + early warning |
| `optimizer` | Compact reactor design search (MVR-0.96) |
| `divertor` | Divertor thermal load simulation |
| `diagnostics` | Synthetic diagnostic instrument suite |
| `sawtooth` | MHD sawtooth crash dynamics |
| `neural` | Neural-network equilibrium solver |
| `geometry` | 3D flux-surface geometry |
| `spi` | Shattered pellet injection mitigation |
| `scanner` | Multi-objective global design scanner |
| `heating` | RF heating (ICRH / ECRH / LHCD) |
| `wdm` | Warm dense matter equation of state |
| `quantum` | Quantum computing bridge |
| `q-control` | Quantum control agent |
| `neuro-control` | SNN-based cybernetic controller |
| `neuro-quantum` | Neuro-quantum hybrid controller |
| `lazarus` | Lazarus protocol bridge |
| `director` | Director AI interface |
| `vibrana` | Vibrana resonance bridge |

## Minimum Viable Reactor (MVR-0.96)

The compact reactor optimizer (`python run_fusion_suite.py optimizer`) performs multi-objective design-space exploration to find the smallest tokamak configuration that achieves Q >= 10 ignition. The "0.96" refers to the normalized minor radius target. Key parameters explored:

- Major/minor radius, elongation, triangularity
- Magnetic field strength, plasma current
- Heating power allocation (NBI, ICRH, ECRH)
- Tritium breeding ratio constraints
- Divertor heat-flux limits

## Neuro-Symbolic Compiler

The `scpn/` subpackage implements a **Petri net → stochastic neuron** compiler:

1. **Petri Net Definition** — plasma control logic expressed as place/transition nets with formal contracts
2. **Compilation** — Petri net transitions mapped to stochastic LIF neurons (using [SC-NeuroCore](https://github.com/anulum/sc-neurocore) when available, NumPy fallback otherwise)
3. **Execution** — SNN-driven real-time plasma control with sub-millisecond latency
4. **Verification** — formal contract checking on compiled artifacts

## SC-NeuroCore Integration

SCPN Fusion Core has an **optional** dependency on [sc-neurocore](https://github.com/anulum/sc-neurocore). When installed, the neuro-symbolic compiler uses hardware-accurate stochastic LIF neurons and Bernoulli bitstream encoding. Without it, all paths fall back to NumPy float computation:

```python
try:
    from sc_neurocore import StochasticLIFNeuron, generate_bernoulli_bitstream
    _HAS_SC_NEUROCORE = True
except ImportError:
    _HAS_SC_NEUROCORE = False  # NumPy float-path fallback
```

## Rust Workspace

The `scpn-fusion-rs/` directory contains a 10-crate Rust workspace that mirrors the Python package structure. Key features:

- **Performance**: `opt-level = 3`, fat LTO, single codegen unit for maximum optimization
- **FFI**: `fusion-python` crate provides PyO3 bindings producing `scpn_fusion_rs.so/.pyd`
- **Dependencies**: `ndarray`, `nalgebra`, `rayon` (parallelism), `rustfft`, `serde`
- **No external runtime**: pure Rust with no C/Fortran dependencies

## Benchmarks

### Headline Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Equilibrium solve** | **15 ms** @ 65×65 | Rust multigrid V-cycle, **50× faster** than Python |
| **Inverse reconstruction** | **~4 s** full (5 LM iters) | Competitive with EFIT (Fortran) |
| **Neural transport** | **5 µs/point** MLP inference | **200,000× faster** than gyrokinetic |
| **Memory** | **0.7 MB** full equilibrium | 5 KFLOP for MLP inference |
| **GPU (projected)** | **~2 ms** equilibrium | wgpu roadmap, RTX 4090-class |

### Community Comparison

Representative single-shot runtimes on contemporary hardware (2024-2025 publications):

| Code | Category | Typical Runtime | Language |
|------|----------|-----------------|----------|
| **GENE** | 5D gyrokinetic | ~10⁶ CPU-h | Fortran/MPI |
| **JINTRAC** | Integrated modelling | ~10 min/shot | Fortran/Python |
| **CHEASE** | Fixed-boundary equilibrium | ~5 s | Fortran |
| **EFIT** | Current-filament reconstruction | ~2 s | Fortran |
| **TORAX** | Integrated (JAX) | ~30 s (GPU) | Python/JAX |
| **DREAM** | Disruption / runaway electrons | ~1 s | C++ |
| **P-EFIT** | GPU-accelerated reconstruction | <1 ms | Fortran+OpenACC |
| **SCPN (Rust)** | **Full-stack** | **~4 s recon, 15 ms equil.** | **Rust+Python** |

Struggling with convergence? See the [Solver Tuning Guide](docs/SOLVER_TUNING_GUIDE.md) + benchmarks notebook Part F.

### Resources

- **Full comparison tables:** [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)
- **Static figures for PDF/arXiv:** [`docs/BENCHMARK_FIGURES.md`](docs/BENCHMARK_FIGURES.md) (includes LaTeX table snippets)
- **Interactive notebook:** [`examples/06_inverse_and_transport_benchmarks.ipynb`](examples/06_inverse_and_transport_benchmarks.ipynb)
- **Pre-built HTML notebooks:** [`docs/notebooks/`](docs/notebooks/) (also served via [GitHub Pages](https://anulum.github.io/scpn-fusion-core/notebooks/))

## Documentation

- [Solver Tuning Guide](docs/SOLVER_TUNING_GUIDE.md) (relaxation, Tikhonov, Huber, grid sizing, common pitfalls)
- [Benchmarks & Comparisons](docs/BENCHMARKS.md)
- [Benchmark Figures (static export)](docs/BENCHMARK_FIGURES.md)
- [Compact Reactor Findings](docs/COMPACT_REACTOR_FINDINGS.md)
- [Physics Methods](docs/PHYSICS_METHODS_COMPLETE.md)
- [ITER Validation](docs/VALIDATION_AGAINST_ITER.md)
- [Neuro-Symbolic Compiler Architecture](docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md)
- [Packet C Control API](docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md)
- [Future Applications](docs/FUTURE_APPLICATIONS.md)
- [Phase 1 3D Execution Plan](docs/PHASE1_3D_EXECUTION_PLAN.md)
- [3D Gap Audit](docs/3d_gaps.md)
- [Next Sprint Execution Queue](docs/NEXT_SPRINT_EXECUTION_QUEUE.md)
- [Profiling Quickstart](profiling/README.md)
- [Comprehensive Technical Study](SCPN_FUSION_CORE_COMPREHENSIVE_STUDY.md) (30,000+ words)

## Citation

If you use SCPN Fusion Core in your research, please cite using the [CITATION.cff](CITATION.cff) file or:

```bibtex
@software{scpn_fusion_core,
  title   = {SCPN Fusion Core: Tokamak Plasma Physics Simulation and Neuro-Symbolic Control Suite},
  author  = {Sotek, Miroslav and Reiprich, Michal},
  year    = {2026},
  url     = {https://github.com/anulum/scpn-fusion-core},
  version = {1.0.0}
}
```

This software is archived on **Zenodo** (DOI pending first release deposit) and published on **Academia.edu**.

## Authors

- **Miroslav Sotek** — ANULUM CH & LI — [ORCID](https://orcid.org/0009-0009-3560-0851)
- **Michal Reiprich** — ANULUM CH & LI

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE).

For commercial licensing inquiries, contact: protoscience@anulum.li
