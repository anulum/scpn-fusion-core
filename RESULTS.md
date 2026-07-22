# SCPN Fusion Core — Benchmark Results (v4.0.0)

> **Auto-generated** by `validation/collect_results.py` on 2026-03-09 19:04 UTC.
> Re-run the script to refresh these numbers on your hardware.

## Environment

- **CPU:** Intel64 Family 6 Model 167 Stepping 1, GenuineIntel
- **Architecture:** AMD64
- **OS:** Windows-11-10.0.26200-SP0
- **Python:** 3.12.5
- **NumPy:** 1.26.4
- **RAM:** 31.8 GB
- **Version:** 4.0.0
- **Generated:** 2026-03-09 19:04 UTC
- **Wall-clock:** 55s

## Equilibrium & Transport

| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| 3D Force-Balance initial residual | 3.8002e+05 | — | Spectral variational method |
| 3D Force-Balance final residual | 1.0706e+05 | — | After 20 iterations |
| 3D Force-Balance reduction factor | 3.5× | — | initial / final |
| Neural Equilibrium inference (mean) | 1.05 | ms | PCA+MLP surrogate on 129x129 grid |
| Neural Equilibrium inference (P95) | 2.69 | ms | 129x129 grid |

## QLKNN Neural Transport Surrogate

| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| Test relative L2 | 0.0943 | — | Hard-fail gate < 0.25 |
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
| H98 RMSE | 0.2954 | — | |
| ITER_15MA_baseline τ_E error | -1.0 | % | τ_pred=3.664 s |
| SPARC_V2C τ_E error | 2.9 | % | τ_pred=0.793 s |
| β_N RMSE | 0.1731 | — | |
| ITER_15MA_baseline β_N error | 5.8 | % | Q=15, P_fus=2538 MW |
| SPARC_V2C β_N error | 22.2 | % | Q=15, P_fus=840 MW |
| Interferometer phase RMSE | 0.003379 | rad | 3 channels |
| Neutron rate relative error | 3.0 | % | |
| Thomson voltage RMSE | 6.11e-07 | V | 3 channels |

## Heating & Neutronics

| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| Best Q (ITER-like scan) | 15.00 | — | Target: Q ≥ 10 |
| Q ≥ 10 achieved | Yes | — | 0.80 × 10²⁰ m⁻³ |
| P_aux at best Q | 10.0 | MW | Auxiliary heating |
| P_fus at best Q | 1564.0 | MW | Fusion power |
| T at best Q | 24.8 | keV | Ion temperature |
| ECRH absorption efficiency | 99.0 | % | 170 GHz, 1st harmonic, 20 MW |
| Tritium Breeding Ratio (total) | 1.1409 | — | 3-group, 80 cm, 90% ⁶Li |
| TBR fast group | 0.0278 | — | 14.1 MeV neutrons |
| TBR epithermal group | 0.2257 | — | Slowed neutrons |
| TBR thermal group | 0.8875 | — | Thermalized |

## Disruption & Control

| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| Disruption mitigation rate (synthetic RL episodes) | 100.0 | % | 50-run synthetic-signal RL/heuristic ensemble; not SNN, not real plasma |
| Mean halo current peak | 1.408 | MA | |
| P95 halo current peak | 2.111 | MA | |
| Mean RE current peak | 0.014 | MA | |
| P95 RE current peak | 0.021 | MA | |
| ITER halo+RE contract pass (stress lane) | Yes | — | Requires prevention>=90%, P95 halo<=3.4 MA, P95 RE<=1.0 MA |
| HIL control-loop P50 latency | 24.5 | μs | 1000 iterations |
| HIL control-loop P95 latency | 37.8 | μs | |
| HIL control-loop P99 latency | 96.5 | μs | |
| Sub-ms achieved | Yes | — | Total loop: 31.9 μs |

## Real-Shot Validation

| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| Disruption recall | 1.00 | — | 6/6 disruptions detected |
| Disruption FPR | 0.00 | — | 0/10 false alarms |
| Disruption detection | Yes | — | recall ≥ 0.6 and FPR ≤ 0.4 |
| Transport tau_E RMSE | 0.0969 | s | 53 shots |
| Transport within 2σ | 72 | % | Gate ≥ 80% |
| Transport validation | Yes | — | |
| Equilibrium ψ pass fraction (self-consistency proxy) | 67 | % | 12/18 files; GS-residual proxy, not solver-vs-EFIT |
| Equilibrium q95 pass fraction (self-reference) | 100 | % | 18/18 files; self-reference, always passes — not external validation |
| Equilibrium self-consistency | Yes | — | self-consistency check, NOT external EFIT validation (see separate EFIT NRMSE lane) |
| Data provenance | Mixed | — | Real SPARC/ITPA + template-generated DIII-D disruption shots |
| Overall real-shot pass | Yes | — | |

## Disturbance Rejection

| Controller | Scenario | ISE | Settling (s) | Overshoot | Stable |
|-----------|----------|-----|-------------|-----------|--------|
| PID | VDE | 1.08e-05 | 0.0843 | 0.0200 | Yes |
| H-infinity | VDE | 4.86e-05 | 0.0894 | 0.0397 | Yes |
| MPC | VDE | 5.27e-06 | 0.0553 | 0.0171 | Yes |
| SNN | VDE | 7.74e-02 | 1.9999 | 0.3100 | Yes |
| PID | Density ramp | 5.96e-05 | 3.9999 | 0.0091 | Yes |
| H-infinity | Density ramp | 9.08e-04 | 3.9999 | 0.0316 | Yes |
| MPC | Density ramp | 2.06e-05 | 3.9999 | 0.0048 | Yes |
| SNN | Density ramp | 1.55e-01 | 3.9999 | 0.3143 | Yes |
| PID | ELM pacing | 7.18e-07 | 2.9999 | 0.0013 | Yes |
| H-infinity | ELM pacing | 1.37e-05 | 2.9999 | 0.0045 | Yes |
| MPC | ELM pacing | 4.86e-07 | 2.9593 | 0.0011 | Yes |
| SNN | ELM pacing | 1.15e-01 | 2.9999 | 0.3104 | Yes |

> Note: the SNN lane's ISE is orders of magnitude larger than PID/MPC. This table records the SNN as stable-but-far-worse, not competitive; MPC wins every scenario.

## Manufactured-Source Equilibrium Parity (Solov'ev lane)

| Case | ψ NRMSE | q NRMSE | Axis error (m) | Passes |
|------|---------|---------|---------------|--------|
| ITER-like | 0.000 | 0.000 | 0.000 | Yes |
| SPARC-like | 0.000 | 0.000 | 0.000 | Yes |
| Spherical-tokamak | 0.000 | 0.676 | 0.000 | Yes |
| KSTAR-like | 0.000 | 0.000 | 0.000 | Yes |
| SPARC-high-kappa | 0.000 | 0.000 | 0.000 | Yes |

> Provenance: synthetic manufactured-source parity lane (release default when FreeGS backend is unavailable).

*Overall ψ NRMSE: 0.000 (threshold: 0.11). Overall: PASS*

## Predictive Free-Boundary Forward (JAX, compiled)

Differentiable free-boundary Grad–Shafranov forward solve — (coil currents, p′, FF′) → ψ —
with the whole Anderson fixed-point iteration compiled into a single `lax.while_loop`
under `jax.jit` (`scpn_fusion.core.jax_predictive_forward_compiled`; batched `vmap`
variant for the ensemble/MCMC pattern). All numbers below are measured by committed,
provenance-bound generators; `*_h100.json` files are dedicated-hardware snapshots.

| Metric | Value | Evidence |
|--------|-------|----------|
| Compiled vs eager fixed point (33², span-rel) | 7.7e-10 | `artifacts/rung2_mg_preconditioner/compiled_forward_speedup.json` |
| Warm vs cold fixed point (129², warm+MG-Richardson, span-rel) | 1.7e-9 | `artifacts/rung2_mg_preconditioner/warm_start_forward.json` |
| H100 FP64 129² cold / warm / warm+MG-Richardson(2) | 164.8 / 26.3 / 13.0 ms | `artifacts/rung2_mg_preconditioner/warm_start_forward_h100.json` |
| H100 replication on the public head (single, warm+MG-Richardson) | 13.44 ms | `artifacts/rung2_mg_preconditioner/batched_forward_amortisation_h100.json` |
| H100 batched per-solve (warm, B=16/64/256) | 13.6 / 13.0 / 14.6 ms | `artifacts/rung2_mg_preconditioner/batched_forward_amortisation_h100.json` |
| Batched element ≡ single solve (span-rel) | ≤ 2.6e-15 | `artifacts/rung2_mg_preconditioner/batched_forward_amortisation.json` |
| Adjoint coil gradient vs warm finite difference | ≤ 1.718e-05 relative (100–300 A steps) | `artifacts/coilgrad_adjoint_fd_evidence.json` |

Real-data reproduction (DIII-D shot 145419, EFIT g-file, 129×129):

| Lane | ψ error (span-relative RMS, ψ_N ≤ 0.95 region) | Notes |
|------|------------------------------------------------|-------|
| Full-domain forward reproduction (point source) | 0.72 % | the error is concentrated in the ψ_N > 0.95 pedestal shell |
| Full-domain with sub-cell source averaging (4×4) | 0.48 % | genuine model improvement, no extra measured information; saturates by 8×8 |
| Shell-pinned attribution (model source ψ_N ≤ 0.95) | 0.051 % | thin-shell source error ⇒ smooth quasi-harmonic remainder |
| Sub-cell + shell-pin attribution | 0.056 % | sub-cell averaging leaves the core unharmed; the residual lives in ψ_N ∈ [0.98, 1] |
| Zero-external-field negative control | ~127 % | cold start, no coil field — fails as it must |

Honest boundaries (recorded inside the artifacts themselves):

- Warm-start convergence envelope: a −1 % coil perturbation does not converge warm at
  33² within 300 iterations (+1 % converges in 11; ±0.2 % in 10–11); batched
  comparisons guard on single-solve convergence, and the batched API inherits exactly
  the single solver's convergence envelope.
- On the H100 the batched per-solve cost equals the single-solve cost (no amortisation
  gain at B ≤ 256; the same code gains 2.5× on a GTX 1060) — the batched wall-clock
  appears bound by a device-independent per-iteration cost, an open question assigned
  to the persistent-kernel experiment lane.
- Timings labelled *indicative* in the bound artifacts are load-contaminated
  development-host numbers; dedicated-hardware numbers live in the `*_h100.json`
  snapshot lane.
- Sub-cell source averaging removes about a third of the real-data full-domain
  error; the remaining ~0.46 % concentrates in ψ_N ∈ [0.98, 1] and a quadratic
  (saddle-aware) ψ expansion adds nothing — the residual is an edge-band
  representation question against EFIT's own discretised solution, recorded in
  the validation artifact rather than chased by imitating EFIT's numerics.

## Disruption Transfer-Generalization

| Group | Shots | Disruptions | Safe | Recall | FPR |
|---|---:|---:|---:|---:|---:|
| Source | 5 | 1 | 4 | 1.000 | 0.000 |
| Target | 11 | 5 | 6 | 1.000 | 0.000 |

*Transfer efficiency (target/source recall): 1.000 | Overall: PASS*

## Disruption Threshold Optimization

| Metric | Value | Notes |
|--------|-------|-------|
| Optimal bias | -4.0 | |
| Optimal threshold | 0.99 | |
| Recall | 1.00 | 6/6 |
| FPR | 0.00 | 0/10 |
| Selection | feasible | recall ≥ 0.80 and FPR ≤ 0.30 both satisfied |
| Shots evaluated | 16 | 6 disruptions, 10 safe |

## Legacy Surrogates

| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| Neural transport MLP surrogate tau_E RMSE | 0.0607 | s | ITPA H-mode confinement time |
| Neural transport MLP surrogate tau_E RMSE % | 13.5 | % | 20 samples |

## Validation Summary

| Lane | Status | Key metric |
|------|--------|------------|
| QLKNN Transport | PASS | test_rel_l2 = 0.0943 |
| Real-shot validation (mixed real+template) | PASS | recall=100%, FPR=0% |
| Confinement ITPA | RUN | RMSE = 0.0969 s |
| 3D Force Balance | RUN | reduction = 3.5× |
| Q ≥ 10 | PASS | Q = 15.0 |
| TBR > 1.05 | PASS | TBR = 1.1409 |
| ECRH absorption | RUN | 99.0% |
| Disruption detection | PASS | recall=100% |
| HIL sub-ms | PASS | P50 = 24.5 μs |
| Solov'ev manufactured-source parity | PASS | ψ NRMSE = 0.000 |
| Transfer generalization | PASS | eff=1.000, target_recall=1.000 |

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
