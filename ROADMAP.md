# Roadmap

> Last updated: 2026-02-27. This roadmap reflects current priorities and may
> change based on community feedback and validation results.

Execution detail: [`docs/HARDENING_30_DAY_EXECUTION_PLAN.md`](docs/HARDENING_30_DAY_EXECUTION_PLAN.md)

## v4.0 — Validation-First Release (target: Q2 2026)

### Remove or retrain FNO turbulence surrogate

The JAX FNO module (rel_L2 = 0.79, synthetic Hasegawa-Wakatani only) is
DEPRECATED in v3.9. In v4.0 it will be either:

- **Retrained** on real gyrokinetic data (GENE/CGYRO flux-tube runs) if
  collaboration access materializes, or
- **Removed entirely** if no real training data is available.

Status: DEPRECATED. Tracking: `src/scpn_fusion/core/fno_turbulence_suppressor.py`

### Expand real-data validation

| Target | Current | Goal |
|--------|---------|------|
| SPARC GEQDSK equilibria | 8 shots (MIT/CFS) | 20+ (pending CFS data release) |
| ITPA confinement entries | 53 / 24 machines | 100+ / 30+ machines |
| DIII-D disruption shots | 16 reconstructed profiles | Raw MDSplus data (pending GA access) |
| JET DT campaign | None | 5+ shots (pending EUROfusion access) |

### Reduce underdeveloped flag count

Current totals are tracked in `UNDERDEVELOPED_REGISTER.md` (auto-generated each
hardening wave). Target for v4.0: reduce P0/P1 count below 50.

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
