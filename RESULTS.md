# SCPN Fusion Core — Benchmark Results (v3.6.0)

**Version:** 3.6.0

> **Auto-generated** by `validation/collect_results.py` on 2026-02-20 10:56 UTC.
> Re-run the script to refresh these numbers on your hardware.

## Environment

- **CPU:** Intel64 Family 6 Model 167 Stepping 1, GenuineIntel
- **Architecture:** AMD64
- **OS:** Windows-11-10.0.26200-SP0
- **Python:** 3.12.5
- **NumPy:** 1.26.4
- **RAM:** 31.8 GB
- **Version:** 3.6.0
- **Generated:** 2026-02-20 10:56 UTC
- **Wall-clock:** 313s

## Equilibrium & Transport

| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| 3D Force-Balance initial residual | 3.8002e+05 | — | Spectral variational method |
| 3D Force-Balance final residual | 1.0706e+05 | — | After 20 iterations |
| 3D Force-Balance reduction factor | 3.5× | — | initial / final |
| Neural Equilibrium inference (mean) | 0.37 | ms | PCA+MLP surrogate on 129x129 grid |
| Neural Equilibrium inference (P95) | 0.78 | ms | 129x129 grid |

## Heating & Neutronics

| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| Best Q (ITER-like scan) | 15.00 | — | Target: Q ≥ 10 |
| Q ≥ 10 achieved | Yes | — | 0.80 × 10²⁰ m⁻³ |
| P_aux at best Q | 15.0 | MW | Auxiliary heating |
| P_fus at best Q | 1127.1 | MW | Fusion power |
| T at best Q | 22.2 | keV | Ion temperature |
| ECRH absorption efficiency | 99.0 | % | 170 GHz, 1st harmonic, 20 MW |
| Tritium Breeding Ratio (total) | 1.1345 | — | 3-group, 80 cm, 90% ⁶Li |
| TBR fast group | 0.0276 | — | 14.1 MeV neutrons |
| TBR epithermal group | 0.2244 | — | Slowed neutrons |
| TBR thermal group | 0.8825 | — | Thermalized |

## Disruption & Control

| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| Disruption prevention rate (SNN) | >60 (0.0%) | % | 10-run ensemble |
| Mean halo current peak | 2.610 | MA | |
| P95 halo current peak | 3.541 | MA | |
| Mean RE current peak | 14.057 | MA | |
| P95 RE current peak | 15.430 | MA | |
| Passes ITER limits | No | — | Halo + RE constraints |
| HIL control-loop P50 latency | 20.5 | μs | 200 iterations |
| HIL control-loop P95 latency | 41.1 | μs | |
| HIL control-loop P99 latency | 365.3 | μs | |
| Sub-ms achieved | Yes | — | Total loop: 34.1 μs |

## Controller Performance (Stress-Test Campaign)

> Auto-generated from 5-episode campaign.

| Controller | Episodes | Mean Reward | Std Reward | Mean R Error | P50 Lat (us) | P95 Lat (us) | P99 Lat (us) | Disrupt Rate | DEF | Energy Eff |
|------------|----------|-------------|------------|--------------|-------------|-------------|-------------|--------------|-----|------------|
| PID        |        5 |     -9.1921 |     0.0000 |       3.1921 |       716107 |       873292 |       878192 |      100.00% | 0.50 |      0.616 |
| H-infinity |        5 |     -9.1921 |     0.0000 |       3.1921 |       584034 |       679804 |       686066 |      100.00% | 0.50 |      0.000 |
| NMPC-JAX   |        5 |     -9.1921 |     0.0000 |       3.1921 |       655393 |       712597 |       717860 |      100.00% | 0.50 |      0.934 |

## Surrogates

| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| tau_E relative RMSE | 28.6% | — | Reference ITPA baseline |
| Neural transport MLP surrogate | tau_E RMSE % | 13.5% (13.5%) | 20 samples |
| FNO (EUROfusion JET) relative L2 (mean) | 0.7925 | — | ψ(R,Z) reconstruction (**EXPERIMENTAL**) |
| FNO (EUROfusion JET) relative L2 (P95) | 0.7933 | — | 16 samples |

> **EXPERIMENTAL — FNO turbulence surrogate:** Relative L2 ~ 0.79 means the model
> explains only ~21% of the variance. Trained on 60 synthetic samples; NOT validated
> against production gyrokinetic codes. See `fno_training.py` for details.

## Documentation & Hero Notebooks

Official performance demonstrations and tutorial paths:
- `examples/neuro_symbolic_control_demo_v2.ipynb` (Golden Base v2)
- `examples/platinum_standard_demo_v1.ipynb` (Platinum Standard - Project TOKAMAK-MASTER)

Legacy frozen notebooks:
- `examples/neuro_symbolic_control_demo.ipynb` (v1)

---

*All benchmarks run on the environment listed above.
Timings are wall-clock and may vary between machines.
Re-run with `python validation/collect_results.py` to reproduce.*
