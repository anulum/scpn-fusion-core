# SCPN Fusion Core — Benchmark Results (v3.9.2)

> **Auto-generated** by `validation/collect_results.py` on 2026-02-27 11:12 UTC.
> Re-run the script to refresh these numbers on your hardware.

## Environment

- **CPU:** Intel64 Family 6 Model 167 Stepping 1, GenuineIntel
- **Architecture:** AMD64
- **OS:** Windows-11-10.0.26200-SP0
- **Python:** 3.12.5
- **NumPy:** 1.26.4
- **RAM:** 31.8 GB
- **Version:** 3.9.2
- **Generated:** 2026-02-27 11:12 UTC
- **Wall-clock:** 62s

## Equilibrium & Transport

| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| 3D Force-Balance initial residual | 3.8002e+05 | — | Spectral variational method |
| 3D Force-Balance final residual | 1.0706e+05 | — | After 20 iterations |
| 3D Force-Balance reduction factor | 3.5× | — | initial / final |
| Neural Equilibrium inference (mean) | 1.62 | ms | PCA+MLP surrogate on 129x129 grid |
| Neural Equilibrium inference (P95) | 4.07 | ms | 129x129 grid |

## QLKNN Neural Transport Surrogate

| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| Test relative L2 | 0.0943 | — | Hard-fail gate < 0.10 |
| Val relative L2 | 0.0954 | — | |
| Train relative L2 | 0.0917 | — | val/train = 1.04 |
| Best val MSE | 0.002566 | — | |
| Architecture | 1024×512×256 | — | MLP hidden dims |
| Epochs | 911 | — | Early-stopped |
| Training time | 3.6 | h | GPU |
| Data source | QLKNN-10D (Zenodo DOI 10.5281/zenodo.3497066) | — | |
| Backup test relative L2 | 0.0949 | — | 512×256×128 architecture |

## Confinement Scaling (ITPA H-mode)

| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| Machines validated | 53 | — | ITER, JET, DIII-D, ASDEX-U, C-Mod, JT-60U, NSTX, MAST, KSTAR, EAST, SPARC, ARC, TFTR, WEST, TCV, HL-2A, HL-2M, COMPASS, JT-60SA, SST-1, Aditya-U, Globus-M2, NSTX-U, MAST-U |
| tau_E RMSE | 0.0969 | s | |
| tau_E relative RMSE | 50.1 | % | |
| H98 RMSE | 0.6518 | — | |
| ITER_15MA_baseline τ_E error | -1.0 | % | τ_pred=3.664 s |
| SPARC_V2C τ_E error | 2.9 | % | τ_pred=0.793 s |
| β_N RMSE | 0.0417 | — | |
| ITER_15MA_baseline β_N error | -2.8 | % | Q=15, P_fus=1981 MW |
| SPARC_V2C β_N error | 3.0 | % | Q=15, P_fus=493 MW |
| Interferometer phase RMSE | 0.003379 | rad | 3 channels |
| Neutron rate relative error | 3.0 | % | |
| Thomson voltage RMSE | 6.11e-07 | V | 3 channels |

## Heating & Neutronics

| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| Best Q (ITER-like scan) | 15.00 | — | Target: Q ≥ 10 |
| Q ≥ 10 achieved | Yes | — | 0.80 × 10²⁰ m⁻³ |
| P_aux at best Q | 10.0 | MW | Auxiliary heating |
| P_fus at best Q | 1061.3 | MW | Fusion power |
| T at best Q | 21.5 | keV | Ion temperature |
| ECRH absorption efficiency | 99.0 | % | 170 GHz, 1st harmonic, 20 MW |
| Tritium Breeding Ratio (total) | 1.1409 | — | 3-group, 80 cm, 90% ⁶Li |
| TBR fast group | 0.0278 | — | 14.1 MeV neutrons |
| TBR epithermal group | 0.2257 | — | Slowed neutrons |
| TBR thermal group | 0.8875 | — | Thermalized |

## Disruption & Control

| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| Disruption prevention rate (SNN) | 0.0 | % | 50-run ensemble |
| Mean halo current peak | 2.242 | MA | |
| P95 halo current peak | 3.328 | MA | |
| Mean RE current peak | 13.726 | MA | |
| P95 RE current peak | 15.495 | MA | |
| Passes ITER limits | No | — | Halo + RE constraints |
| HIL control-loop P50 latency | 27.4 | μs | 1000 iterations |
| HIL control-loop P95 latency | 103.0 | μs | |
| HIL control-loop P99 latency | 317.2 | μs | |
| Sub-ms achieved | Yes | — | Total loop: 42.8 μs |

## Real-Shot Validation

| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| Disruption recall | 1.00 | — | 6/6 disruptions detected |
| Disruption FPR | 0.00 | — | 0/10 false alarms |
| Disruption detection | Yes | — | recall ≥ 0.6 and FPR ≤ 0.4 |
| Transport tau_E RMSE | 0.1287 | s | 20 shots |
| Transport within 2σ | 95 | % | Gate ≥ 80% |
| Transport validation | Yes | — | |
| Equilibrium ψ pass fraction | 67 | % | 12/18 files |
| Equilibrium q95 pass fraction | 100 | % | 18/18 files |
| Equilibrium validation | Yes | — | |
| Overall real-shot pass | Yes | — | |

## Disturbance Rejection

| Controller | Scenario | ISE | Settling (s) | Overshoot | Stable |
|-----------|----------|-----|-------------|-----------|--------|
| SNN (H-inf) | VDE | 4.84e-06 | 0.0545 | 0.0167 | Yes |
| PID | VDE | 1.36e+01 | 0.1999 | 10.0383 | No |
| MPC | VDE | 1.37e+01 | 0.1999 | 10.0568 | No |
| SNN (H-inf) | Density ramp | 4.46e-05 | 2.9990 | 0.0079 | Yes |
| PID | Density ramp | 2.64e+02 | 2.9990 | 10.4110 | No |
| MPC | Density ramp | 2.48e+02 | 2.9990 | 10.1085 | No |
| SNN (H-inf) | ELM pacing | 2.07e-07 | 0.4597 | 0.0015 | Yes |
| PID | ELM pacing | 4.07e+01 | 0.4999 | 10.0673 | No |
| MPC | ELM pacing | 4.09e+01 | 0.4999 | 10.0894 | No |

## FreeGS Equilibrium Benchmark

| Case | ψ NRMSE | q NRMSE | Axis error (m) | Passes |
|------|---------|---------|---------------|--------|
| ITER-like | 8.856 | 0.544 | 5.683 | No |
| SPARC-like | 2.166 | 0.539 | 1.754 | No |
| Spherical-tokamak | 0.916 | 0.711 | 1.829 | No |

*Overall ψ NRMSE: 3.979 (threshold: 0.1). Overall: FAIL*

## Disruption Threshold Optimization

| Metric | Value | Notes |
|--------|-------|-------|
| Optimal bias | -5.0 | |
| Optimal threshold | 0.99 | |
| Recall | 1.00 | 6/6 |
| FPR | 0.50 | 5/10 |
| Pareto score | 0.60 | recall − FPR |
| Shots evaluated | 16 | 6 disruptions, 10 safe |

## Legacy Surrogates

| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| Neural transport MLP surrogate tau_E RMSE | 0.0607 | s | ITPA H-mode confinement time |
| Neural transport MLP surrogate tau_E RMSE % | 13.5 | % | 20 samples |
| JAX FNO turbulence surrogate relative L2 (mean) | 0.7925 | — | DEPRECATED — synthetic-only, removal in v4.0 |
| JAX FNO turbulence surrogate relative L2 (P95) | 0.7933 | — | DEPRECATED — use QLKNN-10D instead |

## Validation Summary

| Lane | Status | Key metric |
|------|--------|------------|
| QLKNN Transport | PASS | test_rel_l2 = 0.0943 |
| Real-shot validation | PASS | recall=100%, FPR=0% |
| Confinement ITPA | RUN | RMSE = 0.0969 s |
| 3D Force Balance | RUN | reduction = 3.5× |
| Q ≥ 10 | PASS | Q = 15.0 |
| TBR > 1.05 | PASS | TBR = 1.1409 |
| ECRH absorption | RUN | 99.0% |
| Disruption detection | PASS | recall=100% |
| HIL sub-ms | PASS | P50 = 27.4 μs |
| FreeGS analytic | FAIL | ψ NRMSE = 3.979 |
| FNO EUROfusion | DEPRECATED | rel_L2 = 0.7925 (synthetic-only, removal in v4.0) |

## Documentation & Hero Notebooks

Official performance demonstrations and tutorial paths:
- `examples/neuro_symbolic_control_demo_v2.ipynb` (Golden Base v2)
- `examples/platinum_standard_demo_v1.ipynb` (Platinum Standard - Project TOKAMAK-MASTER)

Legacy frozen notebooks:
- `examples/neuro_symbolic_control_demo.ipynb` (v1)

---

*All benchmarks run on the environment listed above.
Artifact-based lanes load pre-computed JSON from `artifacts/` and `weights/`.
Timings are wall-clock and may vary between machines.
Re-run with `python validation/collect_results.py` to reproduce.*
