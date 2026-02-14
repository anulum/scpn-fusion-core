# SCPN Fusion Core

[![CI](https://github.com/anulum/scpn-fusion-core/actions/workflows/ci.yml/badge.svg)](https://github.com/anulum/scpn-fusion-core/actions/workflows/ci.yml)
[![Docs](https://github.com/anulum/scpn-fusion-core/actions/workflows/docs.yml/badge.svg)](https://anulum.github.io/scpn-fusion-core/)
[![PyPI](https://img.shields.io/pypi/v/scpn-fusion.svg)](https://pypi.org/project/scpn-fusion/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/scpn-fusion.svg)](https://pypi.org/project/scpn-fusion/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org/)
[![codecov](https://codecov.io/gh/anulum/scpn-fusion-core/graph/badge.svg)](https://codecov.io/gh/anulum/scpn-fusion-core)
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

### Docker (One-Click Run)

```bash
# Run the Streamlit dashboard
docker compose up

# Or build and run manually
docker build -t scpn-fusion-core .
docker run -p 8501:8501 scpn-fusion-core

# With dev dependencies (for running tests inside the container)
docker build --build-arg INSTALL_DEV=1 -t scpn-fusion-core:dev .
docker run scpn-fusion-core:dev pytest tests/ -v
```

### Pure Python (No Rust Toolchain Required)

The entire simulation suite works without Rust. Every module auto-detects the
Rust extension and falls back to NumPy/SciPy:

```bash
pip install scpn-fusion          # from PyPI (pre-built wheels include Rust)
# OR
pip install -e .                 # from source (pure Python, no cargo needed)
```

If the Rust extension is not available, you'll see a one-time info message at
import and all computations run on NumPy. The only difference is speed
(Rust kernels are ~10-50x faster for equilibrium solves).

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

## 21 Simulation Modes

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
| `neuro-control` | SNN-based cybernetic controller |

Additional experimental modes (quantum, vibrana, lazarus, director) are
available for SCPN framework integration work. These require external
components not shipped in this repo:

```bash
python run_fusion_suite.py --experimental quantum
# or
SCPN_EXPERIMENTAL=1 python run_fusion_suite.py vibrana
```

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

### What's Validated

Validation results from 50 synthetic shots (2026-02-14). Raw data in
`validation/results/`. We encourage independent reproduction.

| Component | Status | Key Result | Evidence |
|-----------|--------|------------|----------|
| **Forward GS solver** | 50/50 converge | Circular RMSE 0.028, shaped 0.13-0.56 | `validation/results/forward/summary.csv` |
| **Inverse reconstruction** | 50/50 converge | 2489x mean RMSE improvement, 0.26 s/shot | `validation/results/inverse/summary.csv` |
| **SNN vs PID control** | 6 scenarios | SNN 10x better under plant uncertainty | `validation/results/control_benchmark.json` |
| **Controller latency** | Measured | PID 5 us, SNN 20 us (median) | `validation/results/latency_benchmark.json` |
| **Formal properties** | SNN proved | Boundedness, liveness, mutual exclusion | Contract checker on compiled Petri net |
| **IPB98(y,2) scaling** | Matches published law | 3-10% error vs ITPA dataset | `tests/test_uncertainty.py` |
| **SPARC GEQDSK** | Topology checks pass | Axis position, q-profile monotonicity | `validation/validate_against_sparc.py` |

**Honest limitations:**
- Forward solver uses flat Laplacian stencil (no 1/R correction) in Python --
  this is the primary cause of elevated RMSE on shaped plasmas (0.13-0.61)
- Inverse results suffer from **inverse crime** (same model generates and
  reconstructs data) -- real-world accuracy will be lower
- SNN controller has higher latency than PID (20 us vs 5 us) and slower
  settling on nominal plant (100 ms vs 4.6 ms)
- Both controllers are disrupted by 50ms sensor dropout

### Measured Performance (50 Synthetic Shots)

These numbers are real measurements, not projections. Reproduce them with
`python validation/run_forward_validation.py` and
`python validation/run_inverse_validation.py`.

| Metric | Measured Value | Notes |
|--------|---------------|-------|
| **Forward solve** (65x65, Python) | 0.45 s mean (0.25-1.44 s range) | 1200-1800 Picard iterations |
| **Forward RMSE** (circular) | 0.028 mean | Best category; flat stencil sufficient |
| **Forward RMSE** (shaped) | 0.13-0.56 | Flat stencil limitation; multigrid path improves this |
| **Inverse reconstruction** | 0.26 s mean, 3.6 iterations | 50/50 converge, 2489x improvement |
| **PID controller latency** | 5.1 us mean, 4.0 us median | Pure controller step, no plant sim |
| **SNN controller latency** | 20.3 us mean, 15.8 us median | NumPy backend; Rust/HW path faster |
| **PID steady-state error** (nominal) | 0.026 mm | Best-case tuned scenario |
| **SNN steady-state error** (plant uncertainty) | 2.6 mm | 10x better than PID (26.4 mm) |
| **Memory** | ~0.7 MB (65x65 equil.) | Estimated from array sizes |

> **Note on comparisons:** Earlier versions of this README cited "50x faster
> than Python" and "200,000x faster than gyrokinetic." These comparisons mixed
> different algorithms (multigrid vs SOR) and compared a microsecond-latency
> MLP surrogate against first-principles gyrokinetic solvers -- an apples-to-
> oranges comparison. We've removed these headlines pending proper A/B
> benchmarks and trained model validation.

### Community Context

For context, here are representative runtimes from published fusion codes
(2024–2025 literature). These are not direct comparisons with SCPN.

| Code | Category | Typical Runtime | Language | Reference |
|------|----------|-----------------|----------|-----------|
| GENE | 5D gyrokinetic | ~10⁶ CPU-h | Fortran/MPI | Jenko 2000 |
| JINTRAC | Integrated modelling | ~10 min/shot | Fortran/Python | Romanelli 2014 |
| CHEASE | Fixed-boundary equilibrium | ~5 s | Fortran | Lütjens 1996 |
| EFIT | Current-filament reconstruction | ~2 s | Fortran | Lao 1985 |
| TORAX | Integrated (JAX) | ~30 s (GPU) | Python/JAX | — |
| DREAM | Disruption / runaway electrons | ~1 s | C++ | Hoppe 2021 |

Struggling with convergence? See the [Solver Tuning Guide](docs/SOLVER_TUNING_GUIDE.md) + benchmarks notebook Part F.

### Resources

- **Full comparison tables:** [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)
- **Repro tooling:** [`benchmarks/`](benchmarks/) (Criterion collection + hardware metadata)
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

## Validation Data Licensing

The `validation/reference_data/` directory contains third-party data used
exclusively for regression testing. Each dataset has its own licensing terms:

| Dataset | License / Source | Redistribution |
|---------|-----------------|----------------|
| **SPARC GEQDSK** | MIT ([cfs-energy/SPARCPublic](https://github.com/cfs-energy/SPARCPublic)) | See `validation/reference_data/sparc/LICENSE` |
| **ITPA H-mode** | 20-row illustrative subset from Verdoolaege et al., NF 61 (2021) | See `validation/reference_data/itpa/README.md` |
| **ITER configs** | Internally generated from published parameters | No restrictions |
| **JET / DIII-D** | Manually constructed from published literature | No restrictions |
| **EU-DEMO / K-DEMO** | Synthetic reference configurations | No restrictions |

The SPARC data carries an MIT license from Commonwealth Fusion Systems. The
ITPA subset is a small illustrative extract from a published paper and is not
the full ITPA global confinement database. For the authoritative ITPA dataset,
contact the ITPA Confinement Database Working Group.

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
