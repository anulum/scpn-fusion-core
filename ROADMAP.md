# Roadmap

> Last updated: 2026-03-07. This roadmap reflects current priorities and may
> change based on community feedback and validation results.

Execution detail: [`docs/HARDENING_30_DAY_EXECUTION_PLAN.md`](docs/HARDENING_30_DAY_EXECUTION_PLAN.md)

## Current Hardening Snapshot (2026-03-08)

| Area | Current State | Tracking / Gate |
|---|---|---|
| Underdeveloped register | 68 total flags, **0 P0/P1** | `UNDERDEVELOPED_REGISTER.md` (generated) |
| Pretrained surrogates | **5/8 shipped** (62.5%): ITPA MLP, Neural EQ, QLKNN, FNO JAX, FNO legacy (deprecated) | `weights/pretrained_surrogates_manifest.json` |
| QLKNN transport surrogate | test_rel_L2 = 0.201 (GPU L40S, 500K samples, gated 1024×512×256) | `weights/neural_transport_qlknn.metrics.json` |
| FNO turbulence (JAX) | val_rel_L2 = 0.356 (4-layer, modes=24, width=128, 2000 equilibria) | `weights/fno_turbulence_jax.metrics.json` |
| Real-data roadmap progress | 18 equilibrium files, 8 SPARC, 53 transport shots, 24 machines, 16 disruption shots, 1 JET-DT | `tools/real_data_roadmap_progress.py` + `tools/real_data_roadmap_non_regression_guard.py` |
| DIII-D raw ingestion readiness | **Not ready yet** (strict lane blocks promotion) | `tools/run_real_data_strict_gate.py` + `real-data-strict.yml` |
| FreeGS strict parity | Dedicated strict no-fallback lane available | `.github/workflows/freegs-strict.yml` + `tools/check_freegs_strict_artifact.py` |

Roadmap KPI commands:

```bash
python tools/real_data_roadmap_progress.py \
  --report artifacts/real_shot_validation.json \
  --targets tools/real_data_roadmap_targets.json
python tools/real_data_roadmap_non_regression_guard.py \
  --progress-json artifacts/real_data_roadmap_progress.json \
  --baseline-json tools/real_data_roadmap_baseline.json
```

## v4.0 — Validation-First Release (target: Q2 2026)

### Retrain surrogates to elite accuracy

GPU-trained weights shipped in v3.9.2. Next targets:

| Surrogate | Current | Elite Target | Method |
|-----------|---------|-------------|--------|
| QLKNN transport | test_rel_L2 = 0.201 | < 0.10 | Next: 10M samples, dropout, ensemble |
| FNO turbulence (JAX) | val_rel_L2 = 0.356 | < 0.20 | Bottleneck: oracle noise floor; needs real gyrokinetic spatial data |
| Neural equilibrium | psi NRMSE = 0.076 | < 0.05 | Blocked on GEQDSK data (need 50+ equilibria). Retrain needed after A2 vacuum-field fix (v3.9.3). |
| FreeGS parity | ψ NRMSE = 1.80 | < 0.005 | FAIL after v3.9.3 vacuum-field correction. GS solver boundary handling needs rework. |

The legacy NumPy FNO (rel_L2 = 0.79) is DEPRECATED and superseded by the JAX
FNO. It will be removed in v4.0 if not retrained on real gyrokinetic data.

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
hardening wave). As of 2026-03-08: 68 total flags, 0 P0/P1.
Target for v4.0: reduce remaining docs-claims flags below 30.

### FPGA deployment path

The Petri net -> SNN compiler targets NumPy today. v4.0 adds:
- SC-NeuroCore bitstream export (Xilinx Zynq / Intel Cyclone)
- Deterministic replay verification: FPGA output matches Python bit-for-bit
- Latency target: < 1 us end-to-end on FPGA

### Free-boundary equilibrium

Current GS solver uses fixed-boundary (coil currents as inputs). v4.0 goal:
- FreeGS-compatible coil model
- Inverse reconstruction mode (fit to magnetic probes)
- Benchmark: NRMSE < 5% against EFIT on 10+ shots

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

Extend 1.5D transport solver beyond D-T to support:
- D-T-He3 and D-D-He3 fuel cycles
- Impurity transport (C, W, Ar) with simple coronal model
- Pedestal model coupling (EPED-like)

## v5.0 — Production Hardening (target: Q4 2026)

- Safety-certified control loop (IEC 61508 SIL-2 target)
- Real-time telemetry streaming (gRPC)
- Multi-device support: tokamak + stellarator equilibrium
- Zenodo dataset DOI for all validation reference data

## How to influence the roadmap

Open a [Discussion](https://github.com/anulum/scpn-fusion-core/discussions)
in the **Ideas** category, or comment on existing issues. Contributions to
validation data (real equilibria, experimental profiles) are especially valued.
