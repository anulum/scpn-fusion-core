# SCPN Fusion Core — Benchmark Results (v2.0.0)

> **Auto-generated** by `validation/collect_results.py` on 2026-02-17 UTC.
> Re-run the script to refresh these numbers on your hardware.

## Environment

- **CPU:** Intel64 Family 6 Model 167 Stepping 1, GenuineIntel
- **Architecture:** AMD64
- **OS:** Windows-11-10.0.26200-SP0
- **Python:** 3.12.5
- **NumPy:** 1.26.4
- **RAM:** 31.8 GB
- **Version:** 2.0.0

## What Changed in v2.0.0

| Area | v1.0.2 | v2.0.0 | Impact |
|------|--------|--------|--------|
| Equilibrium solver | SOR only | Multigrid V-cycle (default) | 3-5x faster convergence |
| Transport model | Constant chi_base=0.5 | Gyro-Bohm + EPED pedestal | Physics-based, calibrated |
| H-infinity controller | Fake (fixed gains) | Riccati ARE synthesis | Proven robust stability |
| Disruption predictor | 500 synthetic shots | 10,000 synthetic + 10 reference shots | Higher recall |
| Disruption prevention | 0% | >60% (SNN on reference data) | First nonzero rate |
| Real-shot validation | None | 5-shot validation gate in CI | Externally defensible |
| GEQDSK dataset | 8 SPARC only | 8 SPARC + 100 multi-machine | Broader coverage |
| IPB98 uncertainty | None | Log-linear error propagation | 95% CI quantified |
| HIL demo | None | Register map + TMR simulation | FPGA-ready path |
| Petri net verification | Informal | Constructive boundedness + liveness | Documented proofs |

---

## Equilibrium & Transport

| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| Default solver | Multigrid V-cycle | — | Full-weighting restrict, bilinear prolongate, RB-SOR smooth |
| Multigrid convergence (129x129) | <500 | V-cycles | To residual <1e-6 |
| 3D Force-Balance initial residual | 3.8002e+05 | — | Spectral variational method |
| 3D Force-Balance final residual | 1.0706e+05 | — | After 20 iterations |
| 3D Force-Balance reduction factor | 3.5x | — | initial / final |
| EPED pedestal width (DIII-D) | within 30% | — | Snyder (2009) scaling |
| Gyro-Bohm c_gB (calibrated) | See gyro_bohm_coefficients.json | — | Against 20 ITPA shots |
| Neural Equilibrium inference (mean) | 0.21 | ms | PCA+MLP surrogate on 129x129 grid |
| Neural Equilibrium inference (P95) | 0.30 | ms | 129x129 grid |

## Heating & Neutronics

| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| Best Q (ITER-like scan) | 98.07 | — | Target: Q >= 10 |
| Q >= 10 achieved | Yes | — | 1.00 x 10^20 m^-3 |
| P_aux at best Q | 20.0 | MW | Auxiliary heating |
| P_fus at best Q | 1785.9 | MW | Fusion power |
| T at best Q | 22.4 | keV | Ion temperature |
| ECRH absorption efficiency | 99.0 | % | 170 GHz, 1st harmonic, 20 MW |
| Tritium Breeding Ratio (total) | 1.6684 | — | 3-group, 80 cm, 90% Li-6 |
| TBR fast group | 0.0406 | — | 14.1 MeV neutrons |
| TBR epithermal group | 0.3300 | — | Slowed neutrons |
| TBR thermal group | 1.2978 | — | Thermalized |

## Disruption & Control (v2.0.0)

| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| Disruption prevention rate (SNN) | >60 | % | 10-shot reference replay |
| Disruption prevention rate (PID) | ~40 | % | Baseline comparison |
| Mean halo current peak | 2.547 | MA | |
| P95 halo current peak | 3.541 | MA | |
| Mean RE current peak | 5.792 | MA | |
| P95 RE current peak | 13.865 | MA | |
| Passes ITER limits | No | — | Halo + RE constraints |
| HIL control-loop P50 latency | 26.8 | us | 200 iterations |
| HIL control-loop P95 latency | 147.4 | us | |
| HIL control-loop P99 latency | 520.6 | us | |
| Sub-ms achieved | Yes | — | Total loop: 54.2 us |
| HIL demo FPGA latency (simulated) | 380 | ns | Alveo U250 @ 250 MHz |
| TMR bit-flip recovery | <16 | ns | 4 clock cycles |

## H-Infinity Controller (v2.0.0)

| Metric | Value | Notes |
|--------|-------|-------|
| Synthesis method | Doyle-Glover-Khargonekar ARE | Two Riccati equations via scipy.linalg.solve_continuous_are |
| Plant model | Linearised vertical stability | A = [[0,1],[gamma^2,0]], gamma from published data |
| Guaranteed robustness | <=20% multiplicative plant uncertainty | Verified by perturbation campaign |
| Gamma (attenuation level) | Bisection search | Optimal gamma found automatically |
| Outperforms PID on VDE | Yes | Lower ISE, faster settling |
| Verification artifact | `tests/test_h_infinity_controller.py` | Riccati residual + stability regression lock |

## Transport Metrics (Disambiguated)

| Lane | Metric | Value | Notes |
|------|--------|-------|-------|
| Physics transport (Gyro-Bohm + Chang-Hinton + EPED-like pedestal) | tau_E RMSE | 0.1287 s | 20-shot ITPA run (`validation/validate_transport_itpa.py`) |
| Physics transport (Gyro-Bohm + Chang-Hinton + EPED-like pedestal) | tau_E relative RMSE | 28.6% | Same 20-shot ITPA run |
| Physics transport (Gyro-Bohm + Chang-Hinton + EPED-like pedestal) | tau_E mean absolute relative error | 32.5% | RMSE dashboard aggregate (`validation/reports/rmse_dashboard.json`) |
| Neural transport MLP surrogate | tau_E RMSE % | 13.5% | Surrogate regression lane only (not full physics transport) |

> **Important:** the 13.5% value belongs to the neural surrogate fit, while
> 28.6%/32.5% are full physics-transport validation metrics. They are not
> interchangeable.

## Surrogates

| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| MLP (ITPA H-mode) RMSE | 0.0607 | s | tau_E confinement time |
| MLP (ITPA H-mode) RMSE % | 13.5 | % | 20 samples |
| FNO (EUROfusion JET) relative L2 (mean) | 0.7925 | — | psi(R,Z) reconstruction (**EXPERIMENTAL**) |
| FNO (EUROfusion JET) relative L2 (P95) | 0.7933 | — | 16 samples |

> **EXPERIMENTAL — FNO turbulence surrogate:** Relative L2 = 0.79 means the model
> explains only ~21% of the variance. Trained on 60 synthetic Hasegawa-Wakatani samples;
> NOT validated against production gyrokinetic codes (GENE, GS2, QuaLiKiz). A runtime
> `warnings.warn()` fires on import. Use for exploratory/research purposes only.
> Retraining on real gyrokinetic data or retirement planned for v3.0.

---

## External Validation

Comparison of our solver outputs against published ITER/DIII-D/JET reference values.

| Metric | Our Value | Published | Source | Agreement |
|--------|-----------|-----------|--------|-----------|
| ITER beta_N (0-D estimator) | 0.070 | 1.8 | ITER Physics Basis (1999) | **-96% (miscalibrated)** |
| SPARC beta_N (0-D estimator) | 0.583 | 1.0 | Creely et al., JPP 86 (2020) | **-42% (miscalibrated)** |
| ITER q95 | 3.0 | 3.0 | Shimada et al., NF 47 (2007) | Exact |
| DIII-D elongation kappa | 1.80 | 1.80 | Luxon, NF 42 (2002) | Exact |
| JET DTE2 Pfus | 58 MW (scaled) | 59 MW | JET Team, NF (2022) | 1.7% |
| Bootstrap fraction (ITER) | 0.34 | 0.30-0.40 | Sauter et al., PP 6 (1999) | Within range |
| Spitzer eta at 1keV | 1.65e-8 Ohm.m | 1.65e-8 Ohm.m | Spitzer (1962) | Exact |
| TBR (Li-ceramic) | 1.67 | 1.15-1.35 | Fischer et al., FED (2015) | High (ideal geometry) |

> **Known issue — beta_N:** The 0-D `FusionBurnPhysics` estimator catastrophically
> underestimates beta_N for both ITER (-96%) and SPARC (-42%). Root cause: the
> transport solver runs one-shot without GS-Transport self-consistency, producing an
> unconverged pressure profile that yields a beta_t far too low. Fix requires the
> outer self-consistency loop (v2.1, Task 2.3) and recalibration of the beta_N
> formula against the ITPA dataset.

## IPB98(y,2) Confinement Scaling (v2.0.0)

| Metric | Value | Notes |
|--------|-------|-------|
| Scaling law | IPB98(y,2) | ITER Physics Basis, NF 39 (1999) |
| Uncertainty method | Log-linear error propagation | Verdoolaege et al., NF 61 (2021) |
| ITPA H-mode shots | 20 | Across 11 machines |
| ITER design-point relative error | -0.96% | Single best-matched point |
| ITPA 20-shot mean abs. relative error | 32.5% | Full dataset; dominated by edge cases |
| tau_E 2-sigma coverage | >=80% | Measured values within 95% CI band |

> **Honesty note:** The 32.5% overall RMSE is the honest figure across all 20 ITPA
> shots. The ITER design point (-0.96% error) is excellent but not representative of
> the full dataset. Gyro-Bohm calibration improves ITER-class conditions but does not
> fully capture the multi-machine scatter. Improvement planned for v2.1 via GS-Transport
> self-consistency loop (see CHANGELOG).

## Solver Performance

Rust vs Python timing comparison for key solvers.

| Solver | Grid | Python (ms) | Rust (ms) | Speedup |
|--------|------|-------------|-----------|---------|
| Vacuum field | 65x65 | 45.2 | 2.1 | 21.5x |
| Vacuum field | 129x129 | 178.5 | 7.8 | 22.9x |
| GS Picard (10 iter) | 65x65 | 312.0 | 14.5 | 21.5x |
| Transport step | 50 radial | 0.85 | 0.04 | 21.2x |
| Hall-MHD step | 64x64 | 23.4 | 1.1 | 21.3x |
| Multigrid V-cycle | 129x129 | ~800 | ~35 | ~23x |

## Neural Surrogate Accuracy

PCA+MLP surrogate model metrics after hardening (12 features, physics-informed loss).

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| MSE | 0.0023 | 0.0031 | 0.0035 |
| Max Error | 0.082 | 0.098 | 0.105 |
| GS Residual | 0.0015 | 0.0019 | 0.0021 |
| R-squared | 0.997 | 0.995 | 0.994 |

- Features: R0, a, B0, Ip, beta_p, l_i, kappa, delta_upper, delta_lower, q95, Z_eff, n_e0
- Split: 70% train / 15% val / 15% test
- Early stopping: patience=20 on validation loss

## Bootstrap Current Validation

Sauter model coefficients compared to published values.

| Coefficient | eps=0.1 (ours) | eps=0.1 (Sauter) | eps=0.3 (ours) | eps=0.3 (Sauter) |
|-------------|-------------|----------------|-------------|----------------|
| f_t | 0.462 | 0.46 | 0.800 | 0.80 |
| L31 (low nu*) | 0.72 | 0.71 | 0.85 | 0.85 |
| L32 (low nu*) | 0.46 | 0.46 | 0.62 | 0.62 |
| L34 (low nu*) | 0.31 | 0.30 | 0.42 | 0.41 |

Reference: Sauter et al., Phys. Plasmas 6, 2834 (1999)

## EPED Pedestal Model (v2.0.0)

| Metric | Value | Notes |
|--------|-------|-------|
| Scaling | Snyder (2009) simplified | Delta_ped ~ 0.076 * beta_p_ped^0.5 * nu_star_ped^-0.2 |
| Pedestal width accuracy | Within 30% of published DIII-D | Snyder et al., PoP 18 (2011) |
| Integration | Boundary condition for transport solver | Replaces hardcoded chi suppression |
| Limitation | Not full EPED (no peeling-ballooning stability) | Documented as "EPED-like scaling" |

## Controller Performance (v2.0.0)

4-way controller comparison (100-episode campaign).

| Controller | Mean Reward | P95 Latency (us) | Disruption Rate | DEF | Energy Eff |
|------------|-------------|-------------------|-----------------|-----|------------|
| PID | -0.052 | 145 | 2.0% | 0.99 | 0.92 |
| H-infinity | -0.038 | 162 | 1.0% | 0.99 | 0.90 |
| MPC | -0.029 | 890 | 0.5% | 1.00 | 0.94 |
| SNN | -0.045 | 78 | 3.0% | 0.98 | 0.88 |

DEF = Disruption Extension Factor (controlled t_disruption / uncontrolled t_disruption).

No single controller is a decisive winner across all objectives:
- Best latency: `SNN` (78 us)
- Best disruption rate and reward: `MPC` (0.5%, -0.029)
- Best balanced robust baseline: `H-infinity` (1.0%, 162 us)
- Baseline classical reference: `PID`

### Disturbance Rejection Benchmark (v2.0.0)

Three scenarios: VDE (gamma=100/s), density ramp (0.5->1.2 Greenwald), ELM pacing (10 Hz).
See `artifacts/benchmark_disturbance_rejection.json` for detailed results.

## Disruption Predictor (v2.0.0)

Enhanced predictor with real + synthetic shot data.

### Synthetic Test Set (held-out from training distribution)

| Metric | Value |
|--------|-------|
| Training data | 10,000 synthetic + 10 reference shots |
| Split | 80/10/10 train/val/test |
| Recall@30ms | 0.87 |
| Recall@50ms | 0.92 |
| Recall@100ms | 0.96 |
| False positive rate | 0.08 |
| AUC-ROC | 0.94 |

### Real-Shot Replay (16 DIII-D reference profiles)

| Metric | Value |
|--------|-------|
| Shots tested | 16 (6 disruptions, 10 safe) |
| Recall | >= 60% |
| **False positive rate** | **90%** |
| Prevention rate (SNN) | >60% |
| Prevention rate (PID) | ~40% |
| Validation status | PARTIAL_PASS |

> **Honesty note:** The synthetic-test FPR (0.08) does NOT transfer to real shots.
> On the 16-shot DIII-D reference set, the predictor fires on 9 of 10 safe shots
> (FPR = 90%), indicating the risk threshold is too aggressive. The predictor's recall
> on actual disruptions is acceptable, but its specificity is operationally unusable.
> Threshold tuning (target: FPR < 30%) is planned for v2.1.

### Disruption Shot Database (v2.0.0)

| Shot | Type | Machine | Outcome |
|------|------|---------|---------|
| 155916 | Locked mode | DIII-D (ref) | Disruption |
| 160409 | Density limit | DIII-D (ref) | Disruption |
| 161598 | VDE | DIII-D (ref) | Disruption |
| 164965 | Tearing | DIII-D (ref) | Disruption |
| 166000 | Beta limit | DIII-D (ref) | Disruption |
| 163303 | H-mode | DIII-D (ref) | Safe |
| 154406 | Hybrid | DIII-D (ref) | Safe |
| 175970 | Neg-delta | DIII-D (ref) | Safe |
| 166549 | Snowflake | DIII-D (ref) | Safe |
| 176673 | High-beta | DIII-D (ref) | Safe |

## Uncertainty Quantification

Monte Carlo parameter perturbation analysis (N=200 samples).

| Parameter | Perturbation | Effect on Q | Effect on beta_N |
|-----------|-------------|-------------|---------------|
| T_i +/-10% | Uniform | Delta_Q = +/-2.1 | Delta_beta_N = +/-0.12 |
| n_e +/-10% | Uniform | Delta_Q = +/-1.8 | Delta_beta_N = +/-0.09 |
| Z_eff +/-20% | Uniform | Delta_Q = +/-0.9 | Delta_beta_N = +/-0.05 |
| B0 +/-5% | Uniform | Delta_Q = +/-1.2 | Delta_beta_N = +/-0.07 |

## GEQDSK Dataset (v2.0.0)

Equilibrium database: 100+ shots across 5 tokamaks + 8 SPARC EFIT.

| Machine | Shots | R0 (m) | a (m) | B0 (T) | Ip range (MA) | Source |
|---------|-------|--------|-------|--------|---------------|--------|
| SPARC | 8 | 1.85 | 0.57 | 12.2 | 8.7 | CFS SPARCPublic (MIT license) |
| DIII-D | 20 | 1.67 | 0.67 | 2.19 | 0.75-2.25 | Synthetic Solov'ev |
| JET | 20 | 2.96 | 1.25 | 3.45 | 1.50-4.80 | Synthetic Solov'ev |
| EAST | 20 | 1.85 | 0.45 | 3.50 | 0.25-0.75 | Synthetic Solov'ev |
| KSTAR | 20 | 1.80 | 0.50 | 3.50 | 0.35-1.05 | Synthetic Solov'ev |
| ASDEX-U | 20 | 1.65 | 0.50 | 2.50 | 0.50-1.50 | Synthetic Solov'ev |

Each synthetic shot: Solov'ev analytic equilibrium with self-consistent psi(R,Z), p'(psi), FF'(psi), q(psi).
SPARC shots: Real EFIT reconstructions from CFS public dataset.

> **Note on SPARC axis RMSE:** The reported value of ~1.595 m is dominated by synthetic
> L-mode GEQDSK files where the analytic axis differs from the EFIT convention. On the
> 8 real SPARC EFIT files, axis location RMSE is 2-9 mm.

## TGLF Comparison

Interface for comparing our critical-gradient transport model against TGLF v2.0.

| Regime | Our chi_i (m^2/s) | TGLF chi_i (m^2/s) | Our chi_e (m^2/s) | TGLF chi_e (m^2/s) |
|--------|---------------|------------------|---------------|------------------|
| ITG-dominated | 1.45 | 1.52 | 0.58 | 0.60 |
| TEM-dominated | 0.63 | 0.66 | 1.61 | 1.68 |
| ETG-dominated | 0.12 | 0.13 | 2.55 | 2.69 |

Reference data from TGLF v2.0: Staebler et al., Phys. Plasmas 14, 055909 (2007).

## HIL Demo (v2.0.0)

Hardware-in-the-Loop demonstration targeting Alveo U250 FPGA.

| Metric | Value | Notes |
|--------|-------|-------|
| Target FPGA | Xilinx Alveo U250 | UltraScale+ XCU250 |
| Fabric clock | 250 MHz | |
| Total inference latency | 380 ns | 95 clock cycles |
| TMR recovery latency | <16 ns | 4 clock cycles |
| Fixed-point format | Q16.16 | 16 int + 16 frac bits |
| Neuron count | 8 LIF | In TMR (24 total) |
| Register interface | AXI-Lite compatible | See docs/hil_demo.md |

Note: Actual bitstream generation requires Vivado + hardware. The demo provides a
software simulation of the register-mapped SNN controller.

## Formal Verification (v2.0.0)

Petri net verification results for the SCPN controller compilation.

| Property | Status | Method |
|----------|--------|--------|
| Boundedness | Verified | Constructive: clamping to [0,1] at every step |
| Liveness | >=99% coverage | Exhaustive attempt-to-fire campaign |
| Reachability | Argued | Continuous marking -> connected subset |
| Machine-checked | No | Documented as future work (Coq formalization) |

See `docs/formal_verification.md` for full proofs and arguments.

## IMAS Conformance (v2.0.0)

| IDS | Fields present | Schema validated | Notes |
|-----|---------------|-----------------|-------|
| equilibrium | 15+ | Yes | time_slice, profiles_1d, boundary, global_quantities |
| core_profiles | 10+ | Yes | grid, electrons, ion, q, pressure, j_total, zeff |
| summary | 8+ | Yes | ip, li, beta_pol, beta_tor, q95, elongation |

## Test Coverage

| Suite | Count | Framework | Notes |
|-------|-------|-----------|-------|
| Python unit tests | 859+ | pytest | All passing |
| Python property tests | 15+ | Hypothesis | Seed=0 deterministic |
| Rust unit tests | 200+ | cargo test | All passing |
| Rust property tests | 30+ | proptest | Deterministic |
| IMAS conformance | 4+ | pytest | Schema validation |
| Rust/Python parity | 4+ | pytest | rtol=1e-3, skip if no Rust |

---

## Known Limitations (v2.0.0)

| Issue | Severity | Status | Fix Target |
|-------|----------|--------|------------|
| beta_N estimator: ITER -96%, SPARC -42% | Critical | Known | v2.1 (GS-Transport loop) |
| Disruption predictor FPR = 90% on real shots | High | PARTIAL_PASS | v2.1 (threshold tuning) |
| tau_E ITPA 20-shot RMSE = 32.5% | Medium | Known | v2.1 (self-consistent transport) |
| FNO L2 = 0.79 (21% variance explained) | Medium | Experimental | v3.0 (retrain or retire) |
| GS-Transport: one-shot, no outer loop | Medium | Known | v2.1 |
| Rust SNN not exposed via PyO3 | Low | Documented | v3.0 |
| MHD stability: Mercier + ballooning only | Low | Known | v2.1 (add kink, NTM, Troyon) |
| SPARC axis RMSE: 1.595m (synthetic-dominated) | Low | Known | Real EFIT: 2-9mm |

*All benchmarks run on the environment listed above.
Timings are wall-clock and may vary between machines.
Re-run with `python validation/collect_results.py` to reproduce.*
