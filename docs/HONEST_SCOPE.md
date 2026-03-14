# Honest Scope & Limitations

SCPN Fusion Core is a **control-algorithm development framework** with enough
physics fidelity to validate reactor control strategies against real equilibrium
data. It is **not** a replacement for TRANSP, JINTRAC, GENE, or any
first-principles transport/gyrokinetic code.

## What it does

| Capability | Evidence |
|-----------|---------|
| Petri net → SNN compilation with formal verification | 37 hardening tasks, deterministic replay |
| Sub-microsecond Rust control kernel (0.52 µs P50) and closed-loop latency (23.8 µs P50 / 122 µs P99) | `docs/PERFORMANCE_METRIC_TAXONOMY.md`, `validation/verify_10khz_rust.py`, Criterion benches |
| QLKNN-10D real-gyrokinetic transport surrogate | test_rel_L2 = 0.094 (1024×512×256 gated MLP, 500K samples, GPU L40S), Zenodo DOI 10.5281/zenodo.3497066 |
| IPB98(y,2) confinement scaling on 53 shots / 24 machines | `validation/reference_data/itpa/hmode_confinement.csv` |
| 8 SPARC EFIT GEQDSK equilibrium validation | `validation/reference_data/sparc/` (MIT, CFS) |
| Solov'ev manufactured-source parity | **PASS** — ψ NRMSE 0.000 across 5 tokamak geometries (v3.9.4). 1/R stencil sign error fixed. |
| 0% disruption rate across 1,000-shot stress campaigns | `validation/stress_test_campaign.py` |
| JAX-differentiable GS equilibrium (autodiff through Picard+SOR) | `core/jax_equilibrium_solver.py`, 9 tests |
| Gymnasium RL environment for controller training | `control/gym_tokamak_env.py`, Stable-Baselines3/RLlib compatible |
| Multi-controller suite (PID, H-inf, LQR) | All 3 non-disrupting on ITER config; H-inf v2 corrected integrator plant model |
| GPU-accelerated equilibrium via JAX XLA + wgpu compute shader | Auto-targets GPU when available; `fusion-gpu/gs_solver.wgsl` |
| Graceful degradation (no Rust / no GPU / no SC-NeuroCore) | Every module has a pure-Python fallback |

## What it does not do

| Gap | Why | Alternative |
|-----|-----|-------------|
| 5D gyrokinetic turbulence | Deliberately reduced-order for real-time control | Use GENE/GS2; couple via surrogate training |
| Full 3D nonlinear MHD | Out of scope for real-time loop | Use NIMROD/M3D-C1 externally |
| Complete impurity transport | Simple diffusion only | Use JINTRAC/STRAHL |

## Physics model fidelity

| Module | Actual fidelity | Known limitations |
|--------|----------------|-------------------|
| Equilibrium | Picard + SOR/multigrid, converges on SPARC GEQDSKs; default 129×129 grid; free-boundary via Green's function + coil optimisation | Not EFIT-quality inverse reconstruction |
| Transport | 1.5D Bohm/gyro-Bohm + Chang-Hinton neoclassical | No ITG/TEM/ETG channels; no NBI slowing-down |
| Neural equilibrium | PCA+MLP on 18 GEQDSK files (SPARC+DIII-D+JET) × 25 perturbations | Useful only for equilibrium families it was trained on |
| FNO turbulence | QLKNN-oracle-trained (val_rel_L2 = 0.055); validated against QLKNN-10D test set | No direct gyrokinetic (GENE/CGYRO) validation |
| Neural transport MLP | 53-row ITPA illustrative dataset | Cannot capture full H-mode parameter space |
| Stability | 7-criterion suite: Mercier, ballooning, Kruskal-Shafranov, Troyon, NTM, RWM, peeling-ballooning | Reduced-order; no full eigenvalue PB code (ELITE/MISHKA) |

## Pretrained surrogate status

5 of 8 surrogate lanes ship pretrained weights:

| Surrogate | Status | Evidence |
|-----------|--------|---------|
| MLP ITPA confinement | Shipped | 13.5% RMSE on training set |
| FNO EUROfusion-proxy | Retired (v3.9) | rel_L2 = 0.79 (synthetic only) |
| Neural equilibrium (multi-machine) | Shipped | PCA+MLP, 1818 samples (SPARC+DIII-D+JET), max rel_L2 < 0.001 |
| QLKNN-10D transport | Shipped | test_rel_L2 = 0.094 (GPU L40S, 500K samples, 1024×512×256) |
| FNO turbulence (JAX) | Shipped | val_rel_L2 = 0.055 (4-layer FNO, modes=24, 2000 QLKNN-oracle equilibria) |
| Heat ML shadow | Requires user training | No pretrained weights |
| Gyro-Swin | Requires user training | No pretrained weights |
| Turbulence oracle | Requires user training | No pretrained weights |

## Validation is mostly synthetic

The validation pipeline uses real SPARC GEQDSK files (8 shots, MIT license from
CFS) and a 53-entry ITPA confinement subset, but the bulk of testing uses
synthetic Solov'ev equilibria and template-generated profiles. The DIII-D
disruption shots are reference profiles reconstructed from published parameters,
not raw MDSplus data.

Full claims-to-evidence audit: [`docs/CLAIMS_EVIDENCE_MAP.md`](CLAIMS_EVIDENCE_MAP.md)

## Phase 5+6+GK Scope Boundaries

v3.9.4 added 69 modules across Phase 5, Phase 6, and GK three-path subsystems.
Each carries explicit fidelity limitations.

| Module Area | What We Implement | What We Do Not |
|-------------|-------------------|----------------|
| Native linear GK | Simplified linear eigenvalue solver for ITG/TEM/ETG | Full nonlinear gyrokinetic (GENE, GS2, CGYRO solve 5D Vlasov-Maxwell) |
| Free-boundary tracking | Direct coil-response identification from equilibrium sensitivity | Inverse reconstruction from magnetic probes (EFIT, LIUQE) |
| Disruption predictor | ML classifier trained on reconstructed profile features | Bayesian credibility intervals or physics-based disruption chain models |
| Impurity transport | Banana-regime neoclassical (Pfirsch-Schluter + banana plateau) | Full Hirshman-Sigmar multi-species collisional operator |
| VMEC-lite | Reduced-order Fourier representation of 3D equilibria | Full VMEC variational energy minimization with free-boundary |
| Runaway electrons | Dreicer + hot-tail generation rates, 0D avalanche model | Kinetic runaway distribution (CODE/DREAM-level Fokker-Planck) |
| ELM model | Peeling-ballooning stability proxy with crash operator | Nonlinear MHD ELM simulation (JOREK, BOUT++) |
| Neural turbulence | MLP surrogate trained on QLKNN-10D oracle data | Direct coupling to first-principles GK turbulence codes |
| Orbit following | Guiding-centre Boris push in axisymmetric fields | Full-orbit tracking with 3D perturbation fields and collisions |
| L-H transition | Martin scaling threshold with hysteresis | First-principles edge turbulence suppression (XGC, GENE-X) |

## Underdeveloped flags

The auto-generated [`UNDERDEVELOPED_REGISTER.md`](../UNDERDEVELOPED_REGISTER.md)
tracks the current flag count across the codebase (see Executive Summary for
live totals). The register is regenerated on each release and CI-gated.
