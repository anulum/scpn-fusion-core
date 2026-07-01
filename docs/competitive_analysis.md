# ──────────────────────────────────────────────────────────────────────

## Purpose

This comparison note summarizes how SCPN-FUSION-CORE is positioned relative to peer tooling across control, throughput, and validation posture, with explicit emphasis on traceable evidence and claim boundaries.

## Reader context

This document is used to separate practical engineering claims from marketing shorthand. It is not a ranking list of every fusion code; it is a structured comparison of how SCPN-FUSION-CORE currently demonstrates performance and validation behavior versus peers.

The intended use is to help contributors, collaborators, and evaluators decide whether a claim is supported by local reproducibility data, shared benchmark artifacts, and externally verifiable acceptance rules. Entries should be read together with `docs/BENCHMARKS.md` and the relevant files under `validation/reports/` before using any result in proposals or planning documents.

# SCPN Fusion Core — Competitive Analysis
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────

**Version:** 3.9.8 | **Last updated:** 2026-05-24

Community code timings are from published literature (references at end).
SCPN timings are CI-verified on GitHub Actions ubuntu-latest unless noted.
Latency IDs are defined in `docs/PERFORMANCE_METRIC_TAXONOMY.md`; this table
reports `control.pid_kernel_step_us` (Rust kernel) and
`control.closed_loop_step_us` (controller loop with explicit plant-update
mode). Reduced-order surrogate and full-physics paths are not treated as
same-work speedup pairs.

---

## 1. Real-Time Control Loop

| Code | Control Freq | Step Latency | Language | Source |
|------|-------------|-------------|----------|--------|
| **SCPN v3.10.0 (Rust reduced-order kernel)** | **~2 MHz** | **0.52 us P50 / 0.70 us P99** | Rust | Committed Rust-extension campaign artifact |
| **SCPN v3.10.0 (controller loop, surrogate plant)** | **>10 kHz** | **PID p95 0.012 ms; SNN p95 0.284 ms** | Python | `validation/reports/scpn_end_to_end_latency.md` |
| **SCPN v3.10.0 (controller loop, full-mode plant)** | **>10 kHz** | **PID p95 0.047 ms; SNN p95 0.294 ms** | Python | `validation/reports/scpn_end_to_end_latency.md` |
| DIII-D PCS (production) | 4--10 kHz | 100--250 us per physics cycle | C / Fortran | Penaflor 2024; Barr 2024 |
| ITER CODAC (spec) | ~1 kHz | ~1 ms processing budget | TBD | ITER RTF design docs |
| P-EFIT (GPU) | N/A (reconstruction) | 300--375 us per iter (129x129) | Fortran + CUDA | Sabbagh 2023 |
| TORAX | N/A (offline sim) | ~ms per timestep | Python / JAX | Citrin 2024 |
| FUSE | N/A (design code) | N/A | Julia | Meneghini 2024 |

The Rust PID kernel (0.52 us P50) is a reduced-order control-kernel artifact,
not a same-work comparison against Python Grad-Shafranov solving. The
end-to-end latency benchmark separately reports surrogate and full-mode plant
updates in one Python harness; use those rows for controller-loop comparison.

Full-order Rust equilibrium component timings are now also tracked: 33x33
Grad-Shafranov SOR is 413 us, 33x33 Picard multigrid is 845 us, and six-coil
vacuum-field assembly is 140 us at 33x33 / 489 us at 65x65 on an Intel
i5-11600K. These values are competitive for low-resolution control-support
physics, but they are intentionally not compared against the reduced-order
Rust flight-simulator kernel.

---

## 2. Transport and Gyrokinetic Simulation Speed

SCPN ships three fidelity tiers for transport, selectable per-surface and per-timestep.

### 2.1 Three-Path Gyrokinetic Architecture

| Path | Fidelity | Speed | Modules |
|------|----------|-------|---------|
| **A: Critical-gradient + QLKNN-10D** | Surrogate | ~24 ns/point (MLP), ~1.5--5.5 us/step (1.5D) | `neural_transport`, `integrated_transport_solver` |
| **B: Native linear GK eigenvalue** | High | ~0.3 s/surface | `gk_eigenvalue`, `gk_quasilinear` (Miller geometry, Sugama collisions) |
| **C: External full GK** | Production GK when installed | Minutes to CPU-hours (code-dependent) | `gk_gene`, `gk_cgyro`, `gk_gs2`, `gk_tglf`, `gk_qualikiz` |

Path selection is managed by `gk_scheduler` and `gk_ood_detector`: out-of-distribution
inputs detected by the OOD monitor trigger automatic fallback from Path A to B or C.
The Path C GENE/GS2/CGYRO adapters carry explicit model-fidelity metadata and
can generate nonlinear electrostatic or electromagnetic 5D input decks for
site-installed production solvers. The repository does not bundle those solvers
or claim native parity with their nonlinear turbulence engines.
Online learning via `gk_online_learner` retrains the surrogate from newly computed GK
points, and `gk_corrector` applies delta corrections when the surrogate drifts.

### 2.2 Competitive Comparison

| Code | Type | Runtime | Physics | Source |
|------|------|---------|---------|--------|
| GENE / CGYRO | Nonlinear GK | 10^5--10^6 CPU-hours | 5D Vlasov | Jenko 2000; Belli 2008 |
| JINTRAC + QuaLiKiz | Full integrated | ~217 hours (16 cores) | First-principles turbulence | TU/e 2021 |
| JINTRAC + QLKNN | NN surrogate | ~2 hours (1 core) | ML surrogate | van de Plassche 2020 |
| TORAX | 1D JAX | ~30 s (GPU) / minutes (CPU) | QLKNN10D | Citrin 2024 |
| FUSE | 1D Julia | ~25 ms per step (TJLF) | TJLF surrogate | Meneghini 2024 |
| **SCPN v3.10.0 (1.5D step)** | **1.5D coupled** | **1.5--5.5 us per step** | Crit-gradient + neoclassical | CI Criterion |
| **SCPN v3.10.0 (MLP)** | Neural surrogate | **~24 ns single-point** | Trained QLKNN-10D surrogate | CI Criterion |
| **SCPN v3.10.0 (native GK)** | Linear eigenvalue | **~0.3 s/surface** | Ballooning, Sugama collisions | Measured |
| QLKNN (TensorFlow) | NN inference | ~100 us (25 outputs) | Surrogate | van de Plassche 2020 |

The MLP surrogate is trained on 500K QLKNN-10D gyrokinetic data points
(Zenodo DOI 10.5281/zenodo.3497066) with test_rel_L2 = 0.094. The 10--25%
relative error range is typical for QLKNN surrogates. The native GK eigenvalue
solver uses the response-matrix formulation with Miller geometry
(Kotschenreuther 1995) and Sugama collision operator.

---

## 3. Equilibrium Reconstruction

| Code | Grid | Method | Runtime | Source |
|------|------|--------|---------|--------|
| EFIT (Fortran) | 65x65 | Current-filament Picard | ~2 s full recon | Lao 1985 |
| P-EFIT (GPU) | 65x65 | GPU-accelerated Picard | <1 ms per iter | Sabbagh 2023 |
| CHEASE (Fortran) | 257x257 | Fixed-boundary cubic Hermite | ~5 s | Lutjens 1996 |
| HELENA | 201 flux | Isoparametric | ~10 s | Huysmans 1991 |
| FreeGS | Variable | Picard + multigrid | ~seconds | FreeGS GitHub |
| FreeGSNKE | Variable | Newton-Krylov | Faster than FreeGS | FreeGSNKE 2024 |
| **SCPN v3.10.0 (Rust)** | 65x65 | Picard + SOR | **~100 ms** | Measured |
| **SCPN v3.10.0 (Neural)** | 129x129 | PCA + MLP surrogate | Latency not public-claimed in this release | Tracked surrogate artifact |
| **SCPN v3.10.0 (Multigrid)** | 65x65 | V-cycle | **~15 ms** | Projected |

The neural-equilibrium surface is shipped as a scenario-scanning surrogate
artifact, but this comparison does not claim P-EFIT-class CPU latency until a
tracked inference-latency report is generated. High-fidelity retraining,
cross-machine validation, and public latency evidence remain separate release
gates.

---

## 4. Feature Breadth Matrix

Y = implemented and tested. N = not present. P = partial.

| Capability | SCPN | TORAX | FUSE | FreeGS | DREAM | PROCESS |
|------------|------|-------|------|--------|-------|---------|
| **Equilibrium** | | | | | | |
| Fixed-boundary GS | Y | Y (spectral) | Y | Y (Picard) | N | N |
| Free-boundary GS | Y (Green's + coil opt) | P | Y | Y | N | N |
| Neural equilibrium | Y (standard surrogate; no public latency claim) | N | N | N | N | N |
| JAX-differentiable GS | Y | Y | Y (Julia AD) | N | N | N |
| GEQDSK I/O | Y (read + validate) | N | Y | Y (read + write) | N | N |
| **Transport** | | | | | | |
| 1.5D coupled transport | Y | Y (1D flux-driven) | Y (1D) | N | P (0--1D) | P (0D) |
| Neural surrogate (QLKNN) | Y (test_rel_L2=0.094) | Y (QLKNN10D) | N (TJLF) | N | N | N |
| FNO turbulence surrogate | Y (JAX, val_rel_L2=0.055) | N | N | N | N | N |
| Impurity transport | Y | P | Y | N | N | Y |
| Momentum transport | Y | N | P | N | N | N |
| Current diffusion | Y | Y | Y | N | P | N |
| **Gyrokinetics** | | | | | | |
| Native linear GK eigenvalue | **Y** | N | N | N | N | N |
| External GK (GENE) | Y | N | N | N | N | N |
| External GK (CGYRO) | Y | N | N | N | N | N |
| External GK (GS2) | Y | N | N | N | N | N |
| External GK (TGLF) | Y | N | Y (TJLF variant) | N | N | N |
| External GK (QuaLiKiz) | Y | Y (via QLKNN) | N | N | N | N |
| GK OOD detection + scheduler | **Y** | N | N | N | N | N |
| GK online learner + corrector | **Y** | N | N | N | N | N |
| **Control** | | | | | | |
| Real-time PID (<1 us) | **Y** (0.52 us Rust) | N | N | N | N | N |
| H-infinity controller | **Y** (Riccati synthesis) | N | N | N | N | N |
| NMPC (JAX-differentiable) | Y | N | N | N | N | N |
| Free-boundary tracking | **Y** (kernel + supervisor + EKF) | N | N | N | N | N |
| Burn control (alpha heating) | Y | P | Y | N | N | Y |
| RZIP rigid plasma response | Y | N | Y | N | N | N |
| State estimation (EKF) | Y | N | N | N | N | N |
| SNN compiler (SPN to SNN) | **Y** | N | N | N | N | N |
| Gymnasium RL environment | Y | Y (Gym-TORAX) | N | N | N | N |
| **Disruption & Mitigation** | | | | | | |
| Disruption prediction (ML) | Y | N | N | N | N | N |
| Disruption chain (contract) | Y | N | N | N | N | N |
| SPI mitigation | Y | N | N | N | Y | N |
| ELM model + RMP suppression | Y | N | N | N | N | N |
| Runaway electrons (Fokker-Planck) | Y | N | N | N | **Y** (comprehensive) | N |
| Halo current + RE physics | Y | N | N | N | Y | N |
| Pellet injection | Y | N | Y | N | Y | N |
| **MHD Stability (7 criteria)** | | | | | | |
| Mercier interchange | Y | N | N | N | N | N |
| Ballooning (Connor-Hastie-Taylor) | Y | N | N | N | N | N |
| Kruskal-Shafranov (external kink) | Y | N | N | N | N | N |
| Troyon beta limit | Y | N | Y | N | N | Y |
| NTM seeding threshold | Y | N | N | N | N | N |
| RWM (resistive wall mode) | Y | N | N | N | N | N |
| Peeling-ballooning (ELM boundary) | Y | N | Y | N | N | N |
| **Infrastructure** | | | | | | |
| Rust native backend | Y (12 crates) | N | N | N | N | N |
| GPU acceleration (JAX XLA) | Y | Y | Y (Julia) | N | N | N |
| GPU compute shader (wgpu) | Y | N | N | N | N | N |
| JAX autodifferentiation | Y | Y | Y (Julia AD) | N | N | N |
| Digital twin (real-time) | **Y** | N | N | N | N | N |
| Simulated HIL scaffold | **Y** | N | N | N | N | N |
| Deterministic replay | **Y** | N | N | N | N | N |
| SCPN phase dynamics (Kuramoto/UPDE) | **Y** | N | N | N | N | N |
| Compact reactor optimizer | Y | N | Y (DEMO) | N | N | Y |
| Neutronics / TBR | Y (1-D slab) | N | Y | N | N | Y |
| Research validation (opt-in) | SPARC, ITPA, JET | DIII-D | DIII-D | JET | ITER | ITER, DEMO |

---

## 5. Where Competitors Lead

| Area | Detail | Who Does It Better |
|------|--------|-------------------|
| JAX GPU training | TORAX trains end-to-end on GPU with JAX; SCPN uses JAX solvers but training was done offline on L40S | TORAX |
| Production gyrokinetics | GENE and GS2 are gold-standard nonlinear 5D Vlasov codes with decades of validation against experiment. SCPN's native production path remains linear/reduced-order; full nonlinear production fidelity is reached through external solver decks and site installations. | GENE, GS2, CGYRO |
| Runaway electron physics | DREAM solves the full kinetic RE distribution including hot-tail, Dreicer, avalanche, and synchrotron radiation loss. SCPN's Fokker-Planck RE model covers the primary mechanisms but with reduced-order approximations. | DREAM |
| Stellarator geometry | FUSE supports stellarator equilibria natively. SCPN Fusion Core is tokamak-only in the current public scope. | FUSE |
| QLKNN accuracy | SCPN test_rel_L2 = 0.094. TORAX trains deeper networks with more data. | TORAX, FUSE |
| Community size | Single-team development vs DeepMind (TORAX), General Atomics (FUSE), or IPP (GENE) resources. | All listed codes |
| Reference-data validation | Real data: 8 SPARC GEQDSKs + 53 ITPA discharges + DIII-D disruption templates. TORAX/FUSE have broader machine cross-validation. | TORAX, FUSE |

### Previously Listed Gaps -- Now Closed

| Former Weakness | Resolution | Module |
|----------------|-----------|--------|
| No autodiff | JAX-differentiable GS solver with `jax.grad` through Picard+SOR | `core/jax_equilibrium_solver.py` |
| No GPU equilibrium | JAX solver targets GPU via XLA; wgpu compute shader for Rust path | `core/jax_equilibrium_solver.py`, `fusion-gpu/gs_solver.wgsl` |
| No RL integration | Gymnasium v0.29+ compliant `TokamakEnv` with SB3/RLlib support | `control/gym_tokamak_env.py` |
| Single controller | PID, H-inf, LQR, NMPC, SNN all verified non-disrupting | `control/` suite |
| No gyrokinetics | Native linear GK eigenvalue + 5 external GK interfaces + hybrid scheduler | `core/gk_*.py` (23 modules) |
| No free-boundary tracking | Direct kernel + supervisor + EKF + bounded recovery mode | `control/free_boundary_tracking.py` + supervisory |

---

## 6. Codebase Metrics

| Metric | Value |
|--------|-------|
| Python modules (src/) | 277 |
| Lines of Python (src/) | 73,552 |
| Rust crates | 12 |
| Test files | 382 |
| Individual test functions | 3,817 |
| Validation scripts | 74 |
| CI jobs (ci.yml) | 16 + 11 auxiliary workflows |
| CI workflows total | 12 |
| Supported Python versions | 3.10, 3.11, 3.12 |

---

## 7. Unique Competitive Position

Six capabilities that no other open-source fusion code combines:

1. **Native linear GK eigenvalue + 5 external GK interfaces + QLKNN hybrid.**
   The only code that ships its own ballooning-representation eigenvalue solver
   (Miller geometry, Sugama collisions), interfaces to GENE/CGYRO/GS2/TGLF/QuaLiKiz,
   and adaptively selects fidelity via OOD detection and online learning.

2. **Neuro-symbolic SPN-to-SNN compiler with formal contract checking.**
   Stochastic Petri Net structure compiles to spiking neural network artifacts
   via sc-neurocore, with pre/post-condition contracts verified at compile time.
   No other fusion code has a formal neuro-symbolic compilation pipeline.

3. **GK-to-SCPN phase bridge (Kuramoto/UPDE coupling).**
   Gyrokinetic growth rates and frequencies map into the 16-layer SCPN phase
   dynamics via Kuramoto coupling, enabling physics-informed state-space
   representation that bridges turbulence to control.

4. **Fast reduced-order control kernel: 0.52 us P50 Rust PID.**
   Measured in the committed Rust-extension campaign artifact. This is a
   reduced-order kernel metric, not a same-work speedup over Python
   Grad-Shafranov solving. Same-harness controller-loop timings are reported
   by `validation/reports/scpn_end_to_end_latency.md`.

5. **Free-boundary tracking controller (direct kernel + supervisor + EKF).**
   Closed-loop shape control re-identifies coil-to-objective response directly
from repeated GS solves, with supervisory bounded recovery, disturbance
   observer, latency injection/compensation, and deterministic replay.
   No other open-source code ships a comparable free-boundary tracking stack.

6. **Full-stack breadth in one package.**
   Equilibrium (fixed + free boundary + neural) + transport (1.5D + neural +
   GK three-path) + control (PID + H-inf + NMPC + SNN + burn + RZIP + EKF) +
   disruption (prediction + SPI + ELM/RMP + RE + halo + pellet) + MHD (7
   criteria) + digital twin + simulated HIL scaffold + deterministic replay.
   Live inventory counts are generated by `tools/capability_manifest.py`, not
   hand-maintained in this analysis.

---

## 8. Three-Path Fidelity Architecture

The GK three-path design is central to SCPN's transport strategy:

```
                    ┌─────────────────────────────────────────────┐
                    │           gk_scheduler                      │
                    │  Selects fidelity per surface per timestep  │
                    └────┬──────────────┬───────────────┬─────────┘
                         │              │               │
                    Path A          Path B          Path C
                  (Surrogate)    (Native GK)    (External GK)
                         │              │               │
               neural_transport   gk_eigenvalue    gk_gene
               ~24 ns/point      ~0.3 s/surface   gk_cgyro
                                 gk_quasilinear   gk_gs2
                                                  gk_tglf
                                                  gk_qualikiz
                         │              │               │
                    gk_ood_detector ◄───┘               │
                    (triggers B/C if surrogate OOD)     │
                         │                              │
                    gk_online_learner ◄─────────────────┘
                    (retrains surrogate from new GK data)
                         │
                    gk_corrector
                    (delta correction when surrogate drifts)
```

**Path A** runs in real-time loops. **Path B** runs for offline verification or
when the OOD detector flags surrogate unreliability. **Path C** dispatches to
external production GK codes when available on the system. All three paths share
the same `GKLocalParams`/`GKOutput` interface (`gk_interface.py`), including
the `physics_model` and 5D grid metadata needed to request nonlinear
electrostatic or electromagnetic external runs without changing transport-solver
call sites.

---

## References

- Lao, L.L. et al. (1985). *Nucl. Fusion* 25, 1611 (EFIT).
- Sabbagh, S.A. et al. (2023). GPU-accelerated EFIT (P-EFIT). ACM SC23.
- Lutjens, H. et al. (1996). *Comput. Phys. Commun.* 97, 219 (CHEASE).
- Huysmans, G.T.A. et al. (1991). *Proc. CP90* (HELENA).
- Romanelli, M. et al. (2014). *Plasma Fusion Res.* 9, 3403023 (JINTRAC).
- Citrin, J. et al. (2024). *arXiv:2406.06718* (TORAX).
- Meneghini, O. et al. (2024). *arXiv:2409.05894* (FUSE).
- Jenko, F. et al. (2000). *Phys. Plasmas* 7, 1904 (GENE).
- Belli, E.A. & Candy, J. (2008). *Phys. Plasmas* 15, 092510 (CGYRO).
- Hoppe, M. et al. (2021). *Comput. Phys. Commun.* 268, 108098 (DREAM).
- van de Plassche, K.L. et al. (2020). *Phys. Plasmas* 27, 022310 (QLKNN).
- Penaflor, B.G. et al. (2024). DIII-D PCS. *Fus. Eng. Des.*
- Barr, J.L. et al. (2024). *arXiv:2511.11964* (Parallelised RT physics on DIII-D).
- Dimits, A.M. et al. (2000). *Phys. Plasmas* 7, 969 (Cyclone Base Case).
- Kotschenreuther, M. et al. (1995). *Comput. Phys. Commun.* 88, 128 (GS2).
- Staebler, G.M. et al. (2007). *Phys. Plasmas* 14, 055909 (TGLF).
- Bourdelle, C. et al. (2007). *Phys. Plasmas* 14, 112501 (QuaLiKiz).
- Candy, J. & Waltz, R.E. (2003). *J. Comput. Phys.* 186, 545 (CGYRO).
- FreeGS: https://github.com/freegs-plasma/freegs
- FreeGSNKE: https://docs.freegsnke.com/
- Gym-TORAX: *arXiv:2510.11283*
