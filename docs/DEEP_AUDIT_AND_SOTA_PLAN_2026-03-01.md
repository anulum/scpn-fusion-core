# Deep Audit and SOTA Gap-Closure Plan (2026-03-01)

## 1) Audit Baseline (Current Repo State)

Snapshot date: 2026-03-01 (local audit + live repo scans)

### Core metrics

| Metric | Current |
|---|---:|
| Underdeveloped total flags | 297 |
| Underdeveloped P0/P1 flags | 92 |
| Test files (`tests/test_*.py`) | 170 |
| Test functions (`def test_*`) | 875 |
| Source modules (`src/scpn_fusion`, non-`__init__`) | 93 |
| Rust repo files (`scpn-fusion-rs`) | 113 |

### Underdeveloped marker distribution

| Marker | Count |
|---|---:|
| `FALLBACK` | 172 |
| `EXPERIMENTAL` | 55 |
| `SIMPLIFIED` | 27 |
| `PLANNED` | 22 |
| `DEPRECATED` | 17 |
| `NOT_VALIDATED` | 4 |

### Domain distribution (all flags)

| Domain | Count |
|---|---:|
| `docs_claims` | 139 |
| `core_physics` | 44 |
| `other` | 36 |
| `validation` | 35 |
| `control` | 28 |
| `compiler_runtime` | 8 |
| `nuclear` | 4 |
| `diagnostics_io` | 3 |

### What is actually underdeveloped in production source

After de-duplicating top/full register sections and filtering to `src/`:
- Unique source flags (all priorities): 42
- Source P0/P1 flags: 17
- Source P0/P1 mix: 12 core physics, 2 control, 2 nuclear, 1 cli maturity-gate

Key implication:
- The largest risk concentration is no longer broad code quality drift.
- The critical residual gaps are targeted: physics fidelity upgrades, deprecated FNO lane retirement, and validation/claim rigor.

## 2) Code Hotspots and Underdeveloped Areas

### 2.1 P0/P1 hotspots (source)

| File | P0/P1 markers | Why it matters |
|---|---:|---|
| `src/scpn_fusion/core/fno_turbulence_suppressor.py` | 3 (`DEPRECATED`, `NOT_VALIDATED`, `SIMPLIFIED`) | Highest-credibility physics lane remains deprecated/not validated |
| `src/scpn_fusion/core/integrated_transport_solver.py` | 1 (`SIMPLIFIED`) | Largest physics file, central to transport claims |
| `src/scpn_fusion/core/eped_pedestal.py` | 1 (`SIMPLIFIED`) | Pedestal closure fidelity bottleneck |
| `src/scpn_fusion/core/stability_mhd.py` | 1 (`SIMPLIFIED`) | NTM/MHD reliability for disruption-oriented claims |
| `src/scpn_fusion/control/fokker_planck_re.py` | 1 (`SIMPLIFIED`) | Runaway electron fidelity limits hard-safety narratives |
| `src/scpn_fusion/control/spi_ablation.py` | 1 (`SIMPLIFIED`) | SPI mitigation realism in disruption campaigns |
| `src/scpn_fusion/nuclear/blanket_neutronics.py` | 1 (`SIMPLIFIED`) | TBR confidence boundary |
| `src/scpn_fusion/nuclear/nuclear_wall_interaction.py` | 1 (`SIMPLIFIED`) | PWI claims bounded by reduced model assumptions |

### 2.2 Fallback-heavy runtime surfaces

Raw marker scan in `src/`:
- `FALLBACK`: 67 mentions
- `SIMPLIFIED`: 19 mentions
- `EXPERIMENTAL`: 15 mentions

Top fallback concentration:
1. `src/scpn_fusion/control/disruption_predictor.py` (13)
2. `src/scpn_fusion/core/integrated_transport_solver.py` (7)
3. `src/scpn_fusion/core/_rust_compat.py` (6)
4. `src/scpn_fusion/scpn/compiler.py` (5)

Implication:
- Fallbacks are a strength for resilience, but now require hard telemetry gates:
  hit-rate, reason, deterministic parity, and strict-backend CI mode.

### 2.3 Large-complexity modules (coverage priority)

Top source file sizes:
1. `integrated_transport_solver.py` (2140 lines)
2. `fusion_kernel.py` (1808)
3. `imas_connector.py` (1185)
4. `fno_training.py` (1170)
5. `disruption_predictor.py` (965)

Implication:
- These files dominate regression risk and should receive branch-coverage-first treatment.

### 2.4 Modules with no direct test mention (heuristic scan)

15 modules appear without direct test linkage:
- `src/scpn_fusion/control/nengo_snn_wrapper.py`
- `src/scpn_fusion/control/rust_flight_sim_wrapper.py`
- `src/scpn_fusion/core/compact_reactor_optimizer.py`
- `src/scpn_fusion/core/config_schema.py`
- `src/scpn_fusion/core/force_balance.py`
- `src/scpn_fusion/core/hall_mhd_discovery.py`
- `src/scpn_fusion/core/lazarus_bridge.py`
- `src/scpn_fusion/core/quantum_bridge.py`
- `src/scpn_fusion/core/state_space.py`
- `src/scpn_fusion/core/turbulence_oracle.py`
- `src/scpn_fusion/core/vibrana_bridge.py`
- `src/scpn_fusion/engineering/balance_of_plant.py`
- `src/scpn_fusion/engineering/thermal_hydraulics.py`
- `src/scpn_fusion/io/logging_config.py`
- `src/scpn_fusion/ui/dashboard_launcher.py`

These are the immediate candidates for coverage expansion or explicit de-scoping.

## 3) Coverage and Quality-Gate Audit

### 3.1 Release-lane test inventory

Live release-lane run completed:
- 2074 collected
- 50 deselected
- 2024 selected (`-m "not experimental"`)
- 2003 passed, 21 skipped, 0 failed
- Runtime: 1777.75s (29m 37s)

Coverage output (`coverage-python.xml`):
- Total line coverage: **73.78%** (13124/17789)
- Branch coverage: not currently measured in this lane (`0/0`)
- Files analyzed: 103
- Files <80%: 53
- Files <50%: 20
- Files at 0%: 9

### 3.2 Type-system coverage state

`pyproject.toml` currently uses pragmatic mypy scope:
- Global strict mode is disabled.
- Selected critical modules are checked.
- Several subsystem-wide ignores remain for optional/third-party-heavy areas.
- Local strict-gate execution timed out at 180s in this audit pass, indicating
  scale/performance tuning is needed before making strict typing a hard gate.

Implication:
- Type coverage is strong in crown-jewel modules but not yet project-wide strict.

### 3.3 Coverage deficit concentration

Lowest-coverage modules from the release lane:
- 0%: `control/rust_flight_sim_wrapper.py`, `core/compact_reactor_optimizer.py`,
  `core/force_balance.py`, `core/lazarus_bridge.py`, `core/state_space.py`,
  `core/turbulence_oracle.py`, `core/vibrana_bridge.py`,
  `nuclear/blanket_neutronics.py`, `ui/app.py`
- <25%: `engineering/balance_of_plant.py`, `core/fno_jax_training.py`,
  `core/gyro_swin_surrogate.py`, `engineering/thermal_hydraulics.py`,
  `core/fno_turbulence_suppressor.py`, `core/stability_analyzer.py`

Coverage by major domain (line hits):
- `scpn`: 88.3%
- `control`: 80.4%
- `io`: 80.7%
- `core`: 71.6%
- `diagnostics`: 74.6%
- `hpc`: 71.2%
- `engineering`: 68.3%
- `nuclear`: 36.0%
- `ui`: 24.8%

## 4) External SOTA Baseline (Primary Sources)

### 4.1 Control and transport SOTA signals

1. **TORAX** documents a differentiable tokamak transport simulator verified against RAPTOR and JETTO baselines.  
   Relevance: differentiable simulation + surrogate integration is now mainstream for fast control workflows.

2. **FreeGSNKE** positions free-boundary tokamak simulation with active coil/vessel dynamics and DINA benchmark comparison.  
   Relevance: free-boundary parity is a mandatory credibility checkpoint.

3. **Deep RL in tokamak control** (DeepMind + TCV, then tearing-mode suppression in Nature 2024) demonstrates robust, real-time, high-power operation gains from learning-based control.  
   Relevance: data-driven controllers are now accepted when bounded by rigorous safety and validation gates.

4. **MAST-U real-time current profile estimation (2025)** uses force-balance + EKF to infer profiles from magnetic diagnostics.  
   Relevance: estimator integration is SOTA for physics-constrained real-time state reconstruction.

5. **MAST-U Newton-Krylov Grad-Shafranov solver (2025)** reports >3x speedup and improved robustness over classic Picard-style approaches.  
   Relevance: modern equilibrium lanes are converging to faster, more robust nonlinear solvers.

6. **Disruption prediction transfer learning** (PPPL/APS feature) highlights cross-machine portability as key for deployment realism.  
   Relevance: single-machine synthetic-heavy pipelines are no longer enough for competitiveness.

7. **ITER CODAC / real-time framework focus** emphasizes deterministic real-time software architecture and strict integration discipline.  
   Relevance: governance and deterministic runtime contracts are strategic, not optional.

8. **Neuromorphic hardware trajectory (Intel Loihi 2)** reports major efficiency/speed scaling versus first-generation neuromorphic processors.  
   Relevance: SCPN’s Petri->SNN compiler path is well-aligned with hardware trends, but needs stronger deployment evidence.

9. **QLKNN-10D benchmark baseline** remains a key reference for fast transport surrogates with substantial speedups over direct QuaLiKiz solves.  
   Relevance: replacing deprecated synthetic FNO defaults with validated QLKNN-first lanes is aligned with field expectations.

## 5) Gap Matrix: Current vs SOTA

| Area | Current SCPN state | SOTA bar | Gap |
|---|---|---|---|
| Free-boundary equilibrium | Hardening in progress, fixed-boundary legacy still dominant | Free-boundary + robust nonlinear solve + benchmark parity | Medium-High |
| Turbulence surrogate | Deprecated FNO lane; QLKNN available | Validated non-deprecated transport surrogate in default lane | Medium |
| Real-data breadth | SPARC/ITPA subset + synthetic heavy lanes | Multi-machine, wider-shot real-data validation | High |
| Cross-machine generalization | Partial validation; limited transfer-learning evidence | Explicit transfer-learning/generalization benchmarks | High |
| Deterministic runtime contracts | Strong and improving | Strict backend/timeout/provenance gates everywhere | Medium |
| Hardware-targeted SNN evidence | Rust + fallback strong; FPGA path emerging | Public reproducible HW parity demos | Medium-High |
| Static quality gates | Strong tests, selective typing | Near-complete typing + branch coverage discipline | Medium |

## 6) Plan to Fill Gaps and Drive Toward 100% (Where Physically/Practically Possible)

## Principle
Absolute 100% across all optional hardware/data/network branches is not physically meaningful.  
Target instead: **100% of all feasible deterministic production pathways**, with explicit waivers for externally constrained branches.

### 6.1 Coverage model (what “100%” means here)

1. **Code-path coverage**  
   100% of non-experimental, non-hardware-gated Python production code paths.
2. **Branch coverage**  
   100% for critical safety/runtime modules (`scpn/`, `control/`, `integrated_transport_solver`, `disruption_predictor`, `cli` gates).
3. **Contract coverage**  
   100% of documented domain contracts tested (input validation, fallback provenance, timeout behavior).
4. **Claim-evidence coverage**  
   100% of headline README/RESULTS claims mapped in `CLAIMS_EVIDENCE_MAP`.
5. **Data provenance coverage**  
   100% of validation artifacts carry provenance (source, split, calibration holdout).
6. **Type coverage**  
   100% strict typing for target critical modules; explicit waivers for optional third-party integrations.

### 6.1.1 Numeric coverage targets (v3.9.3 -> v4.0 runway)

| Domain | Current | Target |
|---|---:|---:|
| `scpn` | 88.3% | 95%+ |
| `control` | 80.4% | 92%+ |
| `core` | 71.6% | 88%+ |
| `io` | 80.7% | 92%+ |
| `diagnostics` | 74.6% | 90%+ |
| `engineering` | 68.3% | 85%+ |
| `nuclear` | 36.0% | 80%+ |
| `ui` | 24.8% | 80%+ |
| **Total** | **73.8%** | **90%+** |

Hard 100% requirement in v4.0:
- 100% for critical deterministic safety/runtime paths:
  `scpn/contracts.py`, `scpn/controller.py`, `scpn/compiler.py`,
  `control/disruption_contracts.py`, `core/integrated_transport_solver.py`,
  `core/quasi_3d_contracts.py`, `cli.py` maturity/timeout gates.
- Explicit waiver list required for any excluded line/branch.

### 6.2 Execution waves

### Wave A (Days 1-7): Hard baseline and gate instrumentation
- Freeze and publish a machine-readable underdeveloped snapshot (`json`) each CI release run.
- Add branch coverage gate for critical modules only (start with >=95%).
- Add “untested module guard” for the 15 no-direct-test modules.
- Exit gate: no silent drift in underdeveloped counts or claim mappings.

### Wave B (Days 8-14): Physics P0/P1 closure
- Remove deprecated FNO from any release-default path.
- Make QLKNN/validated transport path the only default in release lane.
- Expand free-boundary parity lane and publish error thresholds.
- Exit gate: source P0 flags reduced; physics-critical P1 list materially reduced.

### Wave C (Days 15-21): Test and typing saturation
- Add tests for all 15 currently unlinked modules (or explicitly retire/de-scope).
- Tighten mypy scope: core/control/scpn + io/diagnostics critical paths in strict mode.
- Add explicit branch tests for fallback reason/provenance contracts.
- Exit gate: zero untested production module left without rationale.

### Wave D (Days 22-30): SOTA parity and “beyond” track activation
- Add cross-machine transfer-learning benchmark lane for disruption prediction.
- Add EKF-style real-time profile reconstruction prototype lane (MAST-U-inspired path).
- Add Newton-style GS solver benchmark lane and report convergence/robustness deltas.
- Publish reproducibility-stamped benchmark pack for v3.9.3 release.
- Exit gate: v3.9.3 includes measurable SOTA-parity evidence and roadmap to beyond-SOTA items.

## 7) “Beyond SOTA” Targeted Research Tracks

1. **Proof-carrying control artifacts**  
   Compile-time formal guarantees + runtime verifier attestation bundle.
2. **Differentiable control co-design lane**  
   Couple differentiable transport approximations with policy-gradient/MPC hybrid tuning.
3. **Cross-machine adaptation benchmark suite**  
   Standardized transfer-learning scorecard across SPARC/ITPA/DIII-D/JET-accessible subsets.
4. **Neuromorphic HIL parity pack**  
   Bit-exact replay across NumPy, Rust, and hardware-backed SNN lanes with signed artifacts.

## 8) Immediate Next Actions (Autonomous)

1. Convert the 17 source P0/P1 entries into tracked issues with owners and measurable closure criteria.
2. Add tests for the 15 no-direct-test modules and enforce “no unlinked module” CI guard.
3. Add a branch-coverage gate for critical modules and publish weekly trend deltas.
4. Push free-boundary and transfer-learning lanes into release-adjacent validation reports.
5. Release v3.9.3 only when underdeveloped and claim-evidence deltas are positive versus previous baseline.

## 9) External Research References

1. TORAX docs: https://torax.readthedocs.io/en/latest/  
2. TORAX GitHub: https://github.com/google-deepmind/torax  
3. FreeGSNKE GitHub: https://github.com/FusionComputingLab/freegsnke  
4. DeepMind (Nature 2024 tearing control): https://deepmind.google/discover/blog/how-ai-could-help-fusion-reactors-control-plasma/  
5. DeepMind (Nature 2022 TCV control): https://deepmind.google/discover/blog/accelerating-fusion-science-through-learned-plasma-control/  
6. DOE DIII-D robust controller news: https://www.energy.gov/science/fes/articles/artificial-intelligence-advances-fusion-research-and-development  
7. PPPL transfer-learning disruption prediction feature: https://www.pppl.gov/news/2024/artificial-intelligence-predicts-disruptions-fusion-experiments  
8. QLKNN-10D publication (Phys. Plasmas, 2020): https://www.osti.gov/biblio/1604324  
9. MAST-U real-time EKF profile estimation (2025): https://www.osti.gov/biblio/2503499  
10. MAST-U Newton-Krylov free-boundary GS solver (2025): https://www.osti.gov/pages/biblio/2534725  
11. ITER CODAC architecture overview: https://www.iter.org/fr/node/20687/codac  
12. ITER real-time framework case study (Fus. Eng. Des., 2023): https://www.osti.gov/pages/biblio/2247748  
13. Intel Loihi 2 release: https://www.intc.com/news-events/press-releases/detail/1684/intel-builds-worlds-largest-neuromorphic-system-to
