# Architecture

SCPN Fusion Core is a three-tier plasma physics framework: a Rust GPU/WGPU
compute engine, a Python equilibrium/transport/control layer, and a
validation pipeline backed by real tokamak data.

## Directory Map

```
scpn-fusion-core/
├── scpn-fusion-rs/             Rust workspace (PyO3 bindings via fusion-python)
│   └── crates/
│       ├── fusion-core/        Grad-Shafranov solver, equilibrium routines
│       ├── fusion-control/     PID controller, flight-sim kernel
│       ├── fusion-diagnostics/ Synthetic diagnostic forward models
│       ├── fusion-engineering/  Engineering subsystems
│       ├── fusion-gpu/         WGPU compute shaders (gs_solver.wgsl)
│       ├── fusion-math/        Linear algebra, interpolation
│       ├── fusion-ml/          Surrogate model inference
│       ├── fusion-nuclear/     Blanket neutronics, tritium breeding
│       ├── fusion-physics/     MHD stability, scaling laws
│       ├── fusion-python/      PyO3 Python ↔ Rust bridge (maturin build)
│       └── fusion-types/       Shared types and configs
│
├── src/scpn_fusion/            Python package (pip install -e ".[dev]")
│   ├── core/                   Equilibrium solvers, transport, neural surrogates
│   │   ├── fusion_kernel.py    Newton GS solver with free/fixed boundary
│   │   ├── neural_equilibrium.py  Neural equilibrium surrogate (MLP)
│   │   ├── neural_transport.py    QLKNN-10D transport surrogate
│   │   ├── integrated_transport_solver.py  Multi-ion coupled transport
│   │   ├── scaling_laws.py     ITER/SPARC confinement scaling
│   │   ├── stability_mhd.py   MHD stability analysis
│   │   └── uncertainty.py      Bayesian uncertainty quantification
│   ├── control/                Real-time plasma control
│   │   ├── disruption_predictor.py  ML disruption prediction
│   │   ├── hil_harness.py      Hardware-in-the-loop test harness
│   │   ├── jax_traceable_runtime.py  JAX-traceable control runtime
│   │   ├── neuro_cybernetic_controller.py  SNN-based controller
│   │   └── spi_mitigation.py   Shattered pellet injection
│   ├── diagnostics/            Synthetic diagnostic forward models
│   ├── io/                     IMAS connector, tokamak data archive
│   ├── nuclear/                Blanket neutronics, tritium breeding
│   ├── scpn/                   SCPN Petri-net compiler and controller
│   └── ui/                     Streamlit dashboard
│
├── tests/                      Python test suite (1899 tests)
├── validation/                 Physics validation benchmarks
│   ├── benchmark_vs_freegs.py  Manufactured-source equilibrium parity
│   ├── benchmark_vs_torax.py   Transport parity against TORAX
│   ├── benchmark_sparc_geqdsk_rmse.py  SPARC GEQDSK RMSE gate
│   ├── stress_test_campaign.py 1000-shot Rust vs Python campaign
│   └── validate_real_shots.py  DIII-D/JET/KSTAR real-shot validation
│
├── tools/                      CI gates, preflight, metadata sync
├── weights/                    Pretrained surrogate weights (.npz, Git LFS)
├── artifacts/                  CI-generated benchmark results (.json)
├── calibration/                Calibration data and configs
├── schemas/                    JSON schemas for configs and artifacts
└── docs/                       Sphinx + MkDocs site source
```

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
