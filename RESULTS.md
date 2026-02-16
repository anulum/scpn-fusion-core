# SCPN Fusion Core — Benchmark Results

> **Auto-generated** by `validation/collect_results.py` on 2026-02-16 00:46 UTC.
> Re-run the script to refresh these numbers on your hardware.

## Environment

- **CPU:** Intel64 Family 6 Model 167 Stepping 1, GenuineIntel
- **Architecture:** AMD64
- **OS:** Windows-11-10.0.26200-SP0
- **Python:** 3.12.5
- **NumPy:** 1.26.4
- **RAM:** 31.8 GB
- **Generated:** 2026-02-16 00:46 UTC
- **Wall-clock:** 14s

## Equilibrium & Transport

| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| 3D Force-Balance initial residual | 3.8002e+05 | — | Spectral variational method |
| 3D Force-Balance final residual | 1.0706e+05 | — | After 20 iterations |
| 3D Force-Balance reduction factor | 3.5× | — | initial / final |
| Neural Equilibrium inference (mean) | 0.86 | ms | PCA+MLP surrogate on 129x129 grid |
| Neural Equilibrium inference (P95) | 2.58 | ms | 129x129 grid |

## Heating & Neutronics

| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| Best Q (ITER-like scan) | 98.07 | — | Target: Q ≥ 10 |
| Q ≥ 10 achieved | Yes | — | 1.00 × 10²⁰ m⁻³ |
| P_aux at best Q | 20.0 | MW | Auxiliary heating |
| P_fus at best Q | 1785.9 | MW | Fusion power |
| T at best Q | 22.4 | keV | Ion temperature |
| ECRH absorption efficiency | 99.0 | % | 170 GHz, 1st harmonic, 20 MW |
| Tritium Breeding Ratio (total) | 1.6684 | — | 3-group, 80 cm, 90% ⁶Li |
| TBR fast group | 0.0406 | — | 14.1 MeV neutrons |
| TBR epithermal group | 0.3300 | — | Slowed neutrons |
| TBR thermal group | 1.2978 | — | Thermalized |

## Disruption & Control

| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| Disruption prevention rate | 0.0 | % | 10-run ensemble |
| Mean halo current peak | 1.402 | MA | |
| P95 halo current peak | 2.139 | MA | |
| Mean RE current peak | 13.388 | MA | |
| P95 RE current peak | 15.270 | MA | |
| Passes ITER limits | No | — | Halo + RE constraints |
| HIL control-loop P50 latency | 12.6 | μs | 200 iterations |
| HIL control-loop P95 latency | 22.2 | μs | |
| HIL control-loop P99 latency | 35.0 | μs | |
| Sub-ms achieved | Yes | — | Total loop: 16.2 μs |

## Surrogates

| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| MLP (ITPA H-mode) RMSE | 0.0607 | s | τ_E confinement time |
| MLP (ITPA H-mode) RMSE % | 13.5 | % | 20 samples |
| FNO (EUROfusion JET) relative L2 (mean) | 0.7925 | — | ψ(R,Z) reconstruction |
| FNO (EUROfusion JET) relative L2 (P95) | 0.7933 | — | 16 samples |

---

*All benchmarks run on the environment listed above.
Timings are wall-clock and may vary between machines.
Re-run with `python validation/collect_results.py` to reproduce.*
