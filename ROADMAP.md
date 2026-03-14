# Roadmap

> Last updated: 2026-03-14. This roadmap reflects current priorities and may
> change based on community feedback and validation results.

Execution detail: [`docs/HARDENING_30_DAY_EXECUTION_PLAN.md`](docs/HARDENING_30_DAY_EXECUTION_PLAN.md)

## Current Snapshot (2026-03-14)

| Area | Current State | Tracking / Gate |
|---|---|---|
| Source modules | **234 Python files**, 62,570 lines | `src/scpn_fusion/` |
| Tests | **2859 passing** | `pytest tests/ -v` |
| Underdeveloped register | **115 entries** (96 → 115 after GK port), **0 P0/P1** | `UNDERDEVELOPED_REGISTER.md` (generated) |
| Pretrained surrogates | **5/8 shipped** (62.5%): ITPA MLP, Neural EQ, QLKNN, FNO JAX, FNO legacy (deprecated) | `weights/pretrained_surrogates_manifest.json` |
| QLKNN transport surrogate | **test_rel_L2 = 0.094** (GPU L40S, 500K samples, gated 1024×512×256, 911 epochs) | `weights/neural_transport_qlknn.metrics.json` |
| FNO turbulence (JAX) | **val_rel_L2 = 0.055** (4-layer, modes=24, width=128, 5-channel input, 5000 equilibria) | `weights/fno_turbulence_jax.metrics.json` |
| Validation pipeline | **24/24 benchmarks passing** (15 legacy + 9 new) | `python validation/collect_results.py` |
| Real-data roadmap progress | 18 equilibrium files, 8 SPARC, 53 transport shots, 24 machines, 16 disruption shots, 1 JET-DT | `tools/real_data_roadmap_progress.py` |
| Enterprise hardening | **19/20 sections passing** (branch protection, labels, tool config, Docker healthcheck) | `.coordination/ENTERPRISE_REPO_HARDENING_CHECKLIST.md` |
| DIII-D raw ingestion readiness | **Not ready yet** (strict lane blocks promotion) | `tools/run_real_data_strict_gate.py` + `real-data-strict.yml` |
| FreeGS strict parity | Dedicated strict no-fallback lane available | `.github/workflows/freegs-strict.yml` |

Roadmap KPI commands:

```bash
python tools/real_data_roadmap_progress.py \
  --report artifacts/real_shot_validation.json \
  --targets tools/real_data_roadmap_targets.json
python tools/real_data_roadmap_non_regression_guard.py \
  --progress-json artifacts/real_data_roadmap_progress.json \
  --baseline-json tools/real_data_roadmap_baseline.json
```

## Shipped

### v3.9.4 (current)

- [x] **Phase 5 physics port** (8 modules): impurity transport, momentum transport, runaway electrons, Alfven eigenmodes, ELM model, pellet injection, plasma-wall interaction, kinetic EFIT
- [x] **Phase 5 control port** (5 modules): free-boundary tracking, state estimator (EKF), volt-second manager, RWM feedback, mu-synthesis
- [x] **Phase 6 physics port** (10 modules): disruption sequence, locked mode, plasma startup, L-H transition, MARFE, neural turbulence, orbit following, tearing mode coupling, VMEC-lite, blob transport
- [x] **Phase 6 control port** (2 modules): detachment controller, density controller
- [x] **GK three-path** (18 modules): native linear eigenvalue solver, quasilinear flux model, 5 external GK interfaces (TGLF, GENE, GS2, CGYRO, QuaLiKiz), OOD detection, correction, scheduling, online learning, verification reporting
- [x] **JAX differentiable solvers** (3 modules): jax_gs_solver, jax_neural_equilibrium, jax_solvers (Thomas + Crank-Nicolson)
- [x] Integrated scenario simulator, neoclassical transport, vessel model, tokamak config presets, IMAS adapter
- [x] **Phase dynamics subpackage** (10 modules): Kuramoto UPDE, adaptive K_nm, GK-to-UPDE bridge, plasma K_nm, Lyapunov guard
- [x] 9 new validation benchmarks
- [x] CoilSet extended with `x_point_target`, `divertor_strike_points` fields
- [x] 69 new modules total, 2859 tests, 234 source files

### v3.9.3

- [x] Hash-pinned deps across all CI workflows
- [x] CII best-practices badge earned
- [x] Per-Python-version lock files (`ci-py39.txt` .. `ci-py312.txt`)
- [x] CodeQL v4 migration

## v4.0 — Validation-First Release (target: Q2 2026)

### Retrain surrogates to elite accuracy

GPU-trained weights shipped in v3.9.2, retrained in v3.9.3. Status:

| Surrogate | Current | Target | Status |
|-----------|---------|--------|--------|
| QLKNN transport | **test_rel_L2 = 0.094** | < 0.10 | **PASS** (L40S GPU, gated 1024×512×256, 911 epochs) |
| FNO turbulence (JAX) | **val_rel_L2 = 0.055** | < 0.20 | **PASS** (L40S GPU, 5-channel [psi,Te,Ti,q,grad_Ti], 267 epochs) |
| Neural equilibrium | psi NRMSE = 0.076 | < 0.05 | Blocked on GEQDSK data (need 50+ equilibria) |
| Manufactured-source parity | ψ NRMSE = 0.000 | < 0.005 | **PASS** (v3.9.3 GS stencil sign fix, 5/5 tokamak cases) |

Next targets for v4.0:
- QLKNN: ensemble of 3 models for UQ, 10M sample retrain
- FNO: retrain on real gyrokinetic spatial data when available
- Neural equilibrium: acquire 50+ GEQDSK equilibria, retrain

The legacy NumPy FNO (rel_L2 = 0.79) is DEPRECATED and will be removed in v4.0.

Runbook: [`docs/FNO_EXTERNAL_RETRAIN_RUNBOOK.md`](docs/FNO_EXTERNAL_RETRAIN_RUNBOOK.md)

### Expand real-data validation

| Target | Current | Goal |
|--------|---------|------|
| SPARC GEQDSK equilibria | 8 shots (MIT/CFS) | 20+ (pending CFS data release) |
| ITPA confinement entries | 53 / 24 machines | 100+ / 30+ machines |
| DIII-D disruption shots | 16 reconstructed profiles | Raw MDSplus data (pending GA access) |
| JET DT campaign | None | 5+ shots (pending EUROfusion access) |

### Reduce underdeveloped flag count

Current totals are tracked in `UNDERDEVELOPED_REGISTER.md` (auto-generated each
hardening wave). As of 2026-03-14: **115 total flags**, 0 P0/P1.
Target for v4.0: resolve all P0/P1 flags, reduce total below 80.

### FPGA deployment path

The Petri net -> SNN compiler targets NumPy today. v4.0 adds:
- SC-NeuroCore bitstream export (Xilinx Zynq / Intel Cyclone)
- Deterministic replay verification: FPGA output matches Python bit-for-bit
- Latency target: < 1 us end-to-end on FPGA

### Free-boundary equilibrium

- [x] CoilSet extended for free-boundary tracking (v3.9.4)
- [x] Direct coil-response identification (v3.9.4)
- [ ] FreeGS-compatible coil model
- [ ] Inverse reconstruction mode (fit to magnetic probes)
- [ ] Benchmark: NRMSE < 5% against EFIT on 10+ shots

## v4.1 — Community & Integration (target: Q3 2026)

### TORAX coupling

Bidirectional coupling with the JAX-based TORAX integrated modelling code:
- Export SCPN equilibria as TORAX initial conditions
- Import TORAX transport profiles as plant model for SNN controller

### Gym-compatible RL environment

`gym_tokamak_env.py` exists but is experimental. v4.1 targets:
- Stable Gym API (observation/action spaces documented)
- Baseline PPO/SAC agents with published reward curves
- Comparison against PID and SNN controllers

### Multi-ion transport

- [x] Extend 1.5D transport solver beyond D-T (v3.9.4)
- [x] Impurity transport (C, W, Ar) with banana-regime neoclassical model (v3.9.4)
- [x] Pedestal model coupling (EPED-like) (v3.9.4)
- [ ] D-T-He3 and D-D-He3 fuel cycles

## v5.0 — Production Hardening (target: Q4 2026)

- Safety-certified control loop (IEC 61508 SIL-2 target)
- Real-time telemetry streaming (gRPC)
- Multi-device support: tokamak + stellarator equilibrium
- Zenodo dataset DOI for all validation reference data

## How to influence the roadmap

Open a [Discussion](https://github.com/anulum/scpn-fusion-core/discussions)
in the **Ideas** category, or comment on existing issues. Contributions to
validation data (real equilibria, experimental profiles) are especially valued.
