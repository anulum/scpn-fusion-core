# SCPN Fusion Core

**Petri-net to spiking-neural-network compiler that turns fusion control policies into 0.5 us controllers with real QLKNN-10D physics and zero-disruption stress-test results.**

<p align="center">
  <img src="docs/assets/repo_header.png" alt="SCPN Fusion Core -- Neuro-Symbolic Tokamak Control">
</p>

[![CI](https://github.com/anulum/scpn-fusion-core/actions/workflows/ci.yml/badge.svg)](https://github.com/anulum/scpn-fusion-core/actions/workflows/ci.yml) [![Docs](https://github.com/anulum/scpn-fusion-core/actions/workflows/docs.yml/badge.svg)](https://github.com/anulum/scpn-fusion-core/actions/workflows/docs.yml) [![Coverage](https://codecov.io/gh/anulum/scpn-fusion-core/branch/main/graph/badge.svg)](https://codecov.io/gh/anulum/scpn-fusion-core) [![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://anulum.github.io/scpn-fusion-core/) [![PyPI](https://img.shields.io/pypi/v/scpn-fusion)](https://pypi.org/project/scpn-fusion/) [![License](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE) ![Version](https://img.shields.io/badge/Version-3.9.2-brightgreen.svg) ![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg) ![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange.svg) ![Tests](https://img.shields.io/badge/Tests-1899_Python_%7C_200%2B_Rust-green.svg)

Most fusion codes are physics-first -- solve equations, then bolt on control.
SCPN Fusion Core inverts this: **control-first**. Express plasma control logic
as stochastic Petri nets, compile to spiking neural networks, execute at
**10 kHz+** against physics-informed plant models. Pure Python with optional
Rust acceleration (6,600x speedup).

> **Minimal control-only package:** [`scpn-control`](https://github.com/anulum/scpn-control) (41 files, pip-installable).
> This repo is the full physics + experimental suite.

## Try in 60 Seconds

```bash
pip install -e .
scpn-fusion kernel          # Grad-Shafranov equilibrium
scpn-fusion flight           # Tokamak flight simulator
pytest tests/ -x -q          # 1,899 tests
```

Or run the **Golden Base** hero notebook -- formal proofs, closed-loop control,
shot replay, all in one:
[`examples/neuro_symbolic_control_demo_v2.ipynb`](examples/neuro_symbolic_control_demo_v2.ipynb)

```bash
# Docker one-click
docker compose up --build    # Streamlit dashboard at localhost:8501
```

## Key Results

| Metric | Value | Reproducibility |
|--------|-------|-----------------|
| Rust control-loop latency | **0.52 us P50** | `validation/verify_10khz_rust.py` |
| QLKNN-10D transport surrogate | test rel_L2 = 0.094 | `weights/neural_transport_qlknn.npz` |
| Disruption rate (1,000-shot campaign) | **0%** (Rust-PID) | `validation/stress_test_campaign.py` |
| ITPA H-mode confinement | 53 shots / 24 machines | `validation/reference_data/itpa/` |
| SPARC GEQDSK validation | 8 EFIT equilibria (MIT, CFS) | `validation/reference_data/sparc/` |
| HIL control loop | 27 us P50 | `python validation/collect_results.py` |
| Q >= 10 operating point | Q = 15 (ITER-like) | `RESULTS.md` |
| TBR | 1.14 (3-group blanket) | `RESULTS.md` |

Full numbers: [`RESULTS.md`](RESULTS.md) -- re-run `python validation/collect_results.py` to reproduce.

## Honest Scope

This is **not** a replacement for TRANSP, JINTRAC, or GENE. It does not solve
5D gyrokinetics or full 3D MHD. It is a **control-algorithm development
framework** with reduced-order physics models fast enough for real-time loop
closure. See [`docs/HONEST_SCOPE.md`](docs/HONEST_SCOPE.md) for the full
limitations assessment, and [`docs/CLAIMS_EVIDENCE_MAP.md`](docs/CLAIMS_EVIDENCE_MAP.md)
for every claim mapped to its evidence artifact.

## Core Innovation: Neuro-Symbolic Compiler

```
Petri Net (places + transitions + contracts)
    |
    v  compiler.py -- structure-preserving mapping
Stochastic LIF Network (neurons + synapses + thresholds)
    |
    v  controller.py -- closed-loop execution
Real-Time Plasma Control (sub-ms latency, deterministic replay)
    |
    v  artifact.py -- versioned, signed compilation artifact
Deployment Package (JSON + schema version + git SHA)
```

Control logic is the primary artifact -- expressed in a formally verifiable
Petri net formalism, compiled to spiking neural networks, executed at
hardware-compatible latencies. The physics modules provide a realistic plant
model for the controller to operate against.

| Property | How |
|----------|-----|
| Formal verification | Contract checking preserves Petri net invariants (boundedness, liveness, reachability) |
| Hardware targeting | Same Petri net compiles to NumPy, SC-NeuroCore (FPGA), or neuromorphic silicon |
| Graceful degradation | Every path has a pure-Python fallback |
| Deterministic replay | Identical inputs produce identical outputs across platforms |

## Controller Stress-Test Campaign (1,000 shots)

| Controller | P50 Latency | P95 Latency | Disruption Rate |
|-----------|------------|------------|-----------------|
| **Rust-PID** | **0.52 us** | 0.67 us | **0%** |
| PID (Python) | 3,431 us | 3,624 us | 0% |
| H-infinity | 3,227 us | 3,607 us | 100% |
| NMPC-JAX | 45,450 us | 49,773 us | 0% |
| Nengo-SNN | 23,573 us | 24,736 us | 0% |

---

<details>
<summary><strong>Architecture</strong></summary>

```
scpn-fusion-core/
+-- src/scpn_fusion/           # Python package (46 modules)
|   +-- core/                  # Plasma physics engines
|   +-- control/               # Reactor control & AI
|   +-- nuclear/               # Nuclear engineering
|   +-- diagnostics/           # Synthetic diagnostics
|   +-- engineering/           # Balance of plant
|   +-- scpn/                  # Neuro-symbolic compiler
|   +-- hpc/                   # High-performance computing
|   +-- ui/                    # Streamlit dashboard
+-- scpn-fusion-rs/            # Rust workspace (11 crates)
|   +-- crates/
|       +-- fusion-types/      # Shared data types
|       +-- fusion-math/       # Linear algebra, FFT
|       +-- fusion-core/       # Grad-Shafranov, transport
|       +-- fusion-physics/    # MHD, heating, turbulence
|       +-- fusion-control/    # PID, MPC, disruption
|       +-- fusion-ml/         # Inference engine
|       +-- fusion-python/     # PyO3 bindings
+-- tests/                     # 1,899 Python tests
+-- validation/                # Benchmark pipeline + reference data
+-- examples/                  # 10 Jupyter notebooks
```

</details>

<details>
<summary><strong>Installation</strong></summary>

### Python (no Rust required)

```bash
pip install -e .               # from source
pip install "scpn-fusion[full]"  # from PyPI with optional deps
```

Every module auto-detects Rust and falls back to NumPy/SciPy.

### Rust acceleration (optional)

```bash
cd scpn-fusion-rs
cargo build --release && cargo test
pip install maturin
cd crates/fusion-python && maturin develop --release
```

### Docker

```bash
docker compose up --build                              # dashboard
docker build --build-arg INSTALL_DEV=1 -t dev .        # with tests
docker run dev pytest tests/ -v
```

### Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v                 # Python (Hypothesis property tests)
cd scpn-fusion-rs && cargo test  # Rust (proptest)
cargo bench                      # Criterion benchmarks
```

</details>

<details>
<summary><strong>Tutorial Notebooks</strong></summary>

| Notebook | Description |
|----------|-------------|
| `01_compact_reactor_search` | MVR-0.96 compact reactor optimizer |
| `02_neuro_symbolic_compiler` | Petri net -> SNN compilation pipeline |
| **`neuro_symbolic_control_demo_v2`** | **Golden Base v2** -- formal proofs + closed-loop + replay |
| `03_grad_shafranov_equilibrium` | Free-boundary equilibrium solver |
| `04_divertor_and_neutronics` | Divertor heat flux & TBR |
| `05_validation_against_experiments` | Cross-validation vs SPARC & ITPA |
| `06_inverse_and_transport_benchmarks` | Inverse solver & neural transport |
| `07_multi_ion_transport` | Multi-species transport evolution |
| `08_mhd_stability` | Ballooning & Mercier criteria |
| `09_coil_optimization` | Coil current optimization (Tikhonov) |
| `10_uncertainty_quantification` | Monte Carlo UQ chain |

All notebooks include `%%timeit` benchmark cells.

</details>

<details>
<summary><strong>Validation Data</strong></summary>

| Dataset | Source | Contents |
|---------|--------|----------|
| **SPARC GEQDSK** | [SPARCPublic](https://github.com/cfs-energy/SPARCPublic) (MIT) | 8 EFIT equilibria (B=12.2 T, Ip up to 8.7 MA) |
| **ITPA H-mode** | Verdoolaege et al., NF 61 (2021) | 53 shots from 24 machines |
| **DIII-D disruptions** | Reference profiles (16 shots) | Locked mode, VDE, tearing, density, beta |
| **Multi-machine GEQDSK** | Synthetic Solov'ev | 100 equilibria (DIII-D, JET, EAST, KSTAR, ASDEX-U) |

```bash
python validation/validate_real_shots.py        # real-shot gate
python validation/collect_results.py            # full 14-lane benchmark
python validation/benchmark_disturbance_rejection.py
```

</details>

<details>
<summary><strong>Simulation Modes (tiered by maturity)</strong></summary>

### Production (hardened, CI-gated)

| Mode | Description |
|------|-------------|
| `kernel` | Grad-Shafranov + 1.5D transport |
| `neuro-control` | SNN cybernetic controller |
| `optimal` | MPC with gradient trajectory optimization |
| `flight` | Real-time flight simulator |
| `digital-twin` | Live twin with RL + chaos monkey |
| `safety` | ML disruption predictor |
| `rust-flight` | Rust-native 10kHz flight sim |

### Validated (tested, not yet hardened)

`optimizer` | `breeding` | `nuclear` | `diagnostics` | `spi` | `divertor` | `heating` | `sawtooth` | `scanner`

### Surrogate / Reduced-order

`neural` (PCA+MLP equilibrium) | `fno` (JAX turbulence) | `geometry` (3D Fourier) | `wdm` (dense matter EOS)

### Experimental

`quantum` | `vibrana` | `lazarus` | `director` -- integration bridges to external SCPN components.

</details>

<details>
<summary><strong>Benchmarks & Solver Performance</strong></summary>

All numbers are internal measurements. Reproduce with `cargo bench` and
`python validation/collect_results.py`.

| Metric | Value | Source |
|--------|-------|--------|
| SOR step @ 65x65 | us-range | `sor_bench.rs` |
| GMRES(30) @ 65x65 | ~45 iters | `gmres_bench.rs` |
| Multigrid V(3,3) @ 129x129 | ~10 cycles | `multigrid_bench.rs` |
| Rust flight sim | 0.3 us/step | `verify_10khz_rust.py` |
| Full equilibrium (Python) | ~5 s | `profile_kernel.py` |
| Neural transport MLP | ~5 us/point | `neural_transport_bench.rs` |

### Community context (not direct comparisons)

| Code | Category | Typical Runtime |
|------|----------|-----------------|
| GENE | 5D gyrokinetic | ~10^6 CPU-h |
| JINTRAC | Integrated modelling | ~10 min/shot |
| TORAX | Integrated (JAX) | ~30 s (GPU) |
| DREAM | Disruption / runaway | ~1 s |

</details>

<details>
<summary><strong>Rust Workspace (11 crates)</strong></summary>

`opt-level = 3`, fat LTO, single codegen unit. Pure Rust, no C/Fortran deps.

- `fusion-math`: Linear algebra, FFT, interpolation
- `fusion-core`: Grad-Shafranov, transport
- `fusion-physics`: MHD, heating, turbulence
- `fusion-control`: PID, MPC, disruption predictor, flight sim, digital twin
- `fusion-ml`: Neural inference engine
- `fusion-python`: PyO3 bindings (`scpn_fusion_rs.pyd`)

Features: 2D MPI domain decomposition (Rayon-parallel Additive Schwarz),
VMEC 3D interface, BOUT++ coupling.

</details>

<details>
<summary><strong>Documentation</strong></summary>

Full docs: **[GitHub Pages](https://anulum.github.io/scpn-fusion-core/)**

| Resource | Link |
|----------|------|
| Python API | [Sphinx docs](https://anulum.github.io/scpn-fusion-core/python/) |
| Rust API | [Rustdoc](https://anulum.github.io/scpn-fusion-core/rust/fusion_core/) |
| Notebooks | [HTML renders](https://anulum.github.io/scpn-fusion-core/notebooks/) |

Key technical docs:
[Solver Tuning](docs/SOLVER_TUNING_GUIDE.md) |
[Benchmarks](docs/BENCHMARKS.md) |
[HIL Demo](docs/hil_demo.md) |
[Physics Methods](docs/PHYSICS_METHODS_COMPLETE.md) |
[SCPN Compiler](docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md)

Two companion papers in preparation:
1. Equilibrium Solver (GS + multigrid + inverse, validated on 8 SPARC GEQDSKs)
2. SNN Controller (Petri net compilation + formal verification + deterministic replay)

</details>

<details>
<summary><strong>Code Health & Hardening</strong></summary>

263 hardening tasks across 8 waves (S2-S4, H5-H8). Every production-path
module returns structured errors. Full registry:
[`docs/PHASE3_EXECUTION_REGISTRY.md`](docs/PHASE3_EXECUTION_REGISTRY.md).

Audit artifacts:
- [Underdeveloped register](UNDERDEVELOPED_REGISTER.md) (312 flags, auto-generated)
- [Claims evidence map](docs/CLAIMS_EVIDENCE_MAP.md) (every claim mapped to evidence)
- [Validation gate matrix](docs/VALIDATION_GATE_MATRIX.md)
- [Release acceptance checklist](docs/RELEASE_ACCEPTANCE_CHECKLIST.md)

</details>

## Community

- [Discussions](https://github.com/anulum/scpn-fusion-core/discussions) — Q&A, ideas, show and tell
- [Roadmap](ROADMAP.md) — v4.0 targets and beyond
- [Contributing](CONTRIBUTING.md) — how to get started

## Citation

```bibtex
@software{scpn_fusion_core,
  title   = {SCPN Fusion Core: Neuro-Symbolic Tokamak Control Suite},
  author  = {Sotek, Miroslav and Reiprich, Michal},
  year    = {2026},
  url     = {https://github.com/anulum/scpn-fusion-core},
  version = {3.9.2}
}
```

## Authors

- **Miroslav Sotek** -- ANULUM CH & LI -- [ORCID](https://orcid.org/0009-0009-3560-0851)
- **Michal Reiprich** -- ANULUM CH & LI

## License

GNU Affero General Public License v3.0 -- see [LICENSE](LICENSE).
Commercial licensing: protoscience@anulum.li
