# SCPN Fusion Core v1.0.0 — Announcement Drafts

## Development History

This GitHub repository was initialized in February 2026 when the codebase was
open-sourced. The underlying physics modules, validation suites, and
neuro-symbolic compiler were developed privately as part of the SCPN
(Self-Consistent Phenomenological Network) research framework starting in 1998,
with the fusion-specific components written between 2024-2026. The commit
history reflects the public release preparation, not the full development
timeline.

---

## X / Twitter (280 chars)

### Option A
Announcing SCPN Fusion Core v1.0.0 — open-source tokamak plasma physics simulation with 54 Python modules, 10 Rust crates, 26 simulation modes, and a neuro-symbolic Petri net → stochastic neuron compiler. AGPL v3.

https://github.com/anulum/scpn-fusion-core

#fusion #plasma #tokamak #Rust #Python #OpenScience

### Option B
We just open-sourced our full tokamak simulation suite: Grad-Shafranov equilibrium, MHD stability, disruption prediction, RF heating, neutronics, and a neuro-symbolic compiler — all with optional Rust acceleration via PyO3.

https://github.com/anulum/scpn-fusion-core

#FusionEnergy #PlasmaPhysics

---

## Reddit r/fusion

**Title:** SCPN Fusion Core v1.0.0 — Open-source tokamak simulation with 26 modes and Rust acceleration

**Body:**

We're releasing SCPN Fusion Core, a comprehensive tokamak plasma physics simulation and control suite. It's been developed as part of the SCPN (Self-Consistent Phenomenological Network) research framework.

**What it does:**

- Grad-Shafranov free-boundary equilibrium solver
- Coupled transport (electron/ion temperature, density, impurities)
- MHD sawtooth crash dynamics
- RF heating (ICRH, ECRH, LHCD)
- ML disruption predictor with shattered pellet injection mitigation
- Divertor thermal simulation
- Blanket neutronics (tritium breeding ratio)
- Compact reactor optimizer (MVR-0.96 — minimum viable reactor search)
- Neuro-symbolic compiler: Petri nets → stochastic LIF neurons for real-time control
- Tokamak flight simulator and digital twin
- 26 simulation modes total

**Tech stack:**

- 54 Python modules (NumPy/SciPy/Matplotlib)
- 10 Rust crates with PyO3 bindings (optional — auto-detected, graceful fallback)
- Criterion benchmarks for the SOR solver
- Property-based testing with Hypothesis (Python) and proptest (Rust)

**What makes it different from other fusion codes:**

1. Full lifecycle coverage — from equilibrium through control to neutronics
2. Neuro-symbolic compiler for real-time plasma control (sub-ms latency)
3. Optional SC-NeuroCore integration for hardware-accurate stochastic neural networks
4. Dual Python + Rust implementation with the same API

License: GNU AGPL v3 (commercial licensing available)

GitHub: https://github.com/anulum/scpn-fusion-core

We'd love feedback from the fusion community — especially on validation against experimental data (JET, DIII-D) and suggestions for improving the physics models.

---

## Reddit r/rust

**Title:** SCPN Fusion Core — 10-crate Rust workspace for tokamak plasma physics (PyO3 bindings)

**Body:**

We just published a Rust workspace for tokamak plasma physics simulation. The Rust side provides performance-critical kernels that are ~10-100× faster than the pure Python equivalents.

**Crate structure:**

- `fusion-types` — shared data types (Grid2D, PlasmaState, ReactorConfig) with serde
- `fusion-math` — linear algebra (SVD, eigendecomposition), SOR solver, Thomas algorithm, elliptic integrals, FFT, bilinear interpolation
- `fusion-core` — Grad-Shafranov equilibrium solver (Picard iteration with Red-Black SOR)
- `fusion-physics` — MHD, heating, turbulence
- `fusion-nuclear` — neutronics, wall erosion
- `fusion-engineering` — balance of plant
- `fusion-control` — PID, MPC, disruption predictor
- `fusion-diagnostics` — sensor models
- `fusion-ml` — inference engine
- `fusion-python` — PyO3 bindings → `scpn_fusion_rs.so/.pyd`

**Build:**

```
cargo build --release  # opt-level 3, fat LTO, single codegen unit
cargo test --all-features
cargo bench            # Criterion benchmarks
```

**Dependencies:** ndarray, nalgebra, rayon, rustfft, serde, pyo3, criterion, proptest

**Testing:** Unit tests + property-based tests with proptest (elliptic integral relations, SOR convergence, SVD reconstruction, grid invariants)

The Python package auto-detects the Rust extension and falls back to NumPy if unavailable.

GitHub: https://github.com/anulum/scpn-fusion-core/tree/main/scpn-fusion-rs

License: AGPL-3.0-or-later

---

## Reddit r/PlasmaPhysics

**Title:** Open-source Grad-Shafranov solver with 26 simulation modes (Python + Rust)

**Body:**

We've open-sourced our tokamak simulation suite, SCPN Fusion Core. It covers the full reactor physics lifecycle:

**Equilibrium & Stability:**
- Free-boundary Grad-Shafranov solver (SOR with toroidal corrections)
- MHD sawtooth crash dynamics
- Stability analysis (decay index, force balance)

**Transport & Heating:**
- Coupled electron/ion transport with impurities
- ICRH, ECRH, LHCD heating models
- ITG/TEM turbulence prediction

**Control:**
- Model-predictive optimal control
- ML disruption predictor
- Shattered pellet injection mitigation
- Real-time flight simulator

**Nuclear Engineering:**
- Blanket neutronics (tritium breeding ratio)
- Divertor thermal simulation
- Plasma-wall interaction / first-wall erosion

**Reactor Design:**
- Compact reactor optimizer (MVR-0.96) — multi-objective design space search
- Global design scanner (R, B₀, κ, δ sweeps)
- Balance of plant (thermal cycle, turbine, cryogenics)

The code has been validated against ITER reference configurations. We're looking for collaborators to help with experimental data validation (JET, DIII-D, KSTAR).

GitHub: https://github.com/anulum/scpn-fusion-core
License: AGPL v3

---

## Academia.edu Abstract

**Title:** SCPN Fusion Core: A Comprehensive Tokamak Plasma Physics Simulation and Neuro-Symbolic Control Suite

**Authors:** Miroslav Sotek (ORCID: 0009-0009-3560-0851), Michal Reiprich — ANULUM CH & LI

**Abstract:**

We present SCPN Fusion Core, an open-source software suite for tokamak plasma physics simulation, control, and reactor design. The package implements 54 Python modules across 7 subpackages (core physics, control, nuclear engineering, diagnostics, engineering, HPC, and neuro-symbolic compilation), with optional Rust acceleration through 10 crates and PyO3 bindings. Key capabilities include a free-boundary Grad-Shafranov equilibrium solver, coupled transport, MHD sawtooth dynamics, RF heating (ICRH/ECRH/LHCD), blanket neutronics, disruption prediction, and a compact reactor optimiser (MVR-0.96). A novel feature is the neuro-symbolic compiler that maps Petri net control logic to stochastic leaky integrate-and-fire (LIF) neuron networks, enabling sub-millisecond real-time plasma control with formally verifiable logic. The software integrates with the SC-NeuroCore spiking neural network framework for hardware-accurate stochastic computation. The code is released under GNU AGPL v3 and archived on Zenodo.

**Keywords:** tokamak, fusion, plasma physics, Grad-Shafranov, MHD, neuro-symbolic, spiking neural networks, reactor design, digital twin
