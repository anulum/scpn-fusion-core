# Physics Validation Status

> Last updated: 2026-03-07. Comprehensive audit of all 4 validation areas.

## 1. FreeGS Parity

### Current State

The benchmark (`validation/benchmark_vs_freegs.py`) operates in **two modes**:

| Mode | Gate | Status |
|------|------|--------|
| Solov'ev (analytic) | psi_nrmse < 0.11 | **PASS** (avg 0.076) |
| FreeGS strict | psi_nrmse < 0.005 + 5 more gates | **Not run on CI** (opt-in) |

Latest artifact (`artifacts/freegs_benchmark.json`, 2026-03-02):

| Case | Psi NRMSE | q NRMSE | Axis Err (m) | Sep NRMSE | Converged |
|------|-----------|---------|--------------|-----------|-----------|
| ITER-like | 0.074 | 0.181 | 0.500 | 0.152 | Yes |
| SPARC-like | 0.072 | 0.180 | 0.143 | 0.151 | Yes |
| Spherical | 0.102 | 0.626 | 0.327 | 0.148 | **No** |
| KSTAR-like | 0.061 | 0.152 | 0.094 | 0.133 | Yes |
| SPARC-high-k | 0.073 | 0.189 | 0.143 | 0.155 | Yes |

### FreeGS Strict Thresholds (6 conjunctive gates)

- `psi_nrmse < 0.005` (22x tighter than Solov'ev)
- `psi_nrmse_normalized < 0.06`
- `q_nrmse < 0.10`
- `axis_error_m < 0.10`
- `separatrix_nrmse < 0.05`
- `flux_area_rel_error < 0.12`

### Root Cause of Gap

Solov'ev lane compares against an analytic solution (self-referential: analytic BC +
analytic source). FreeGS lane compares against an iterative nonlinear solver solving the
full GS equation with pressure/current profiles — a fundamentally different problem class.

Current Psi NRMSE values (0.06-0.10) exceed the 0.005 FreeGS threshold by 12-20x.
The threshold appears empirically set without a convergence study.

### Known Issues

1. Spherical-tokamak case passes Solov'ev gate despite `converged=false`.
2. FreeGS strict workflow is manual-dispatch only (`.github/workflows/freegs-strict.yml`).
3. Normalization is correct (scale-invariant NRMSE), but the two lanes compare different
   problems — Solov'ev is a manufactured solution, FreeGS is a free-boundary solve.

### Roadmap Actions

- Calibrate FreeGS thresholds with convergence study (vary grid resolution, iterations).
- Add convergence-required gate to Solov'ev lane (reject `converged=false` cases).
- Run FreeGS strict lane on a machine with FreeGS installed to establish realistic baselines.
- Target: NRMSE < 5% against EFIT on 10+ shots (v4.0 goal per ROADMAP.md).

### Key Files

| File | Purpose |
|------|---------|
| `validation/benchmark_vs_freegs.py` | Benchmark driver (1153 lines) |
| `tests/test_freegs_benchmark.py` | Unit tests (401 lines) |
| `tests/test_freegs_benchmark_enhanced.py` | Enhanced tests (261 lines) |
| `tools/check_freegs_strict_artifact.py` | Strict artifact validator |
| `.github/workflows/freegs-strict.yml` | Manual-dispatch strict lane |
| `artifacts/freegs_benchmark.json` | Latest results (Solov'ev mode) |

---

## 2. Transport / ITPA Validation

### Current State: ALL GATES PASS

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Shots validated | 53 | — | — |
| Machines | 20 | — | — |
| tau_E RMSE | 0.097 s | — | — |
| Relative RMSE | 37.4% | — | — |
| Within 2-sigma | 72% | >= 70% | PASS |
| Abs relative error p95 | 2.162 | <= 2.50 | PASS |
| Z-score p95 | 3.792 | <= 4.00 | PASS |

### IPB98(y,2) Implementation

Source: `src/scpn_fusion/core/scaling_laws.py`

Coefficients (Verdoolaege et al., NF 61 076006, 2021):
- C=0.0562, alpha_Ip=0.93, alpha_BT=0.15, alpha_n=0.41, alpha_P=-0.69,
  alpha_R=1.97, alpha_kappa=0.78, alpha_eps=0.58, alpha_M=0.19

Training domain: Ip 0.4-22 MA, BT 0.3-14 T, ne 0.5-25 x10^19 m^-3,
P_loss 0.5-220 MW, R 0.45-8.5 m, kappa 1.0-3.0, eps 0.08-0.9, M 1.0-3.0 AMU.

### ITPA Dataset

- 53 shots across 20 machines (ITER, JET, DIII-D, ASDEX-U, C-Mod, JT-60U,
  NSTX, MAST, KSTAR, EAST, SPARC, ARC, and 8 others).
- Source: `validation/reference_data/itpa/hmode_confinement.csv`
- Provenance: mixed (real ITPA published data + synthetic DIII-D disruption profiles).

### Gyro-Bohm Calibration

- c_gB = 2.58 +/- 2.19 (fitted to 20-shot subset).
- Transport model: gyro-Bohm + Chang-Hinton neoclassical.
- RMSE: 0.27 s (59.8% relative) — higher uncertainty than IPB98 fit.

### Known Gaps

1. 53-point subset is illustrative (full ITPA DB has 5920 shots across 18 machines).
2. SPARC BT=12.2 T approaches training domain boundary (max 14 T).
3. IPB98 is purely empirical — no separation of neoclassical/anomalous transport.
4. No explicit pedestal structure treatment (parabolic profile assumption).
5. Gyro-Bohm model shows much higher uncertainty than IPB98 scaling.

### Roadmap Actions

- Expand to 100+ shots / 30+ machines (v4.0 goal per ROADMAP.md).
- Pending: CFS data release for additional SPARC shots.
- Pending: GA access for DIII-D raw MDSplus data.
- Pending: EUROfusion access for JET DT campaign shots.

### Key Files

| File | Purpose |
|------|---------|
| `validation/validate_transport_itpa.py` | Standalone ITPA validation |
| `validation/validate_real_shots.py` | Three-lane validation (eq, transport, disruption) |
| `validation/benchmark_transport_uncertainty_envelope.py` | Contract benchmark |
| `validation/reports/transport_uncertainty_envelope_benchmark.json` | Latest results |
| `validation/reference_data/itpa/hmode_confinement.csv` | 53-shot dataset |
| `validation/reference_data/itpa/ipb98y2_coefficients.json` | Scaling law coefficients |
| `src/scpn_fusion/core/scaling_laws.py` | IPB98(y,2) implementation |

---

## 3. Neural Equilibrium Surrogate

### Architecture

PCA (20 components) + MLP (12 -> 128 -> 64 -> 32 -> 20), pure NumPy.

Input: 12D feature vector (Ip, Bt, R_axis, Z_axis, pprime_scale, ffprime_scale,
simag, sibry, kappa, delta_upper, delta_lower, q95).

Output: 129x129 poloidal flux psi(R,Z) grid.

### Training Data

- 3 SPARC GEQDSK files (lmode_hv, lmode_vh, lmode_vv).
- 25 perturbations per file = 78 effective samples.
- Split: 54 train / 12 val / 12 test.
- Training time: 3.1 seconds on CPU.

### Current Accuracy

| Metric | Value |
|--------|-------|
| Test NRMSE (best) | 3.1e-5 |
| Test NRMSE (worst) | 1.03e-4 |
| Gate threshold | 0.05 (5%) |
| All cases | PASS (6/6) |
| Explained variance | 100% (20 PCA components) |
| Inference latency | 0.28 ms mean, 0.54 ms P95 |
| Speedup vs FusionKernel | ~1000x |

### SOR Solver Fix (This Session)

`validation/psi_pointwise_rmse.py`: replaced hardcoded `omega=1.3, max_iter=200`
with optimal SOR omega `2/(1 + sin(pi/N))` and `max_iter=5000`.

All 8 SPARC files now converge (480-540 iters, res ~5e-9).
High NRMSE (0.4-1.5) remains in the manufactured-source solve — root cause is
reference data GS residuals (2.5-65.7), not solver error.

### Weight File

`weights/neural_equilibrium_sparc.npz` (2.8 MB, 2026-02-16).
Format: `np.savez(allow_pickle=False)` with PCA + MLP weights.
Does NOT depend on UPDE Kuramoto coupling (unaffected by prior bug fixes).

### Known Gaps

1. Only 3 GEQDSK files — 78 samples is very small.
2. All training data is SPARC L-mode — no H-mode, no other machines.
3. No uncertainty quantification (point estimates only).
4. GS residual loss is approximate penalty, not exact autodiff.
5. Fixed 129x129 grid resolution.

### Roadmap Actions

- Expand to 50-200 GEQDSK files across multiple machines and regimes.
- Add H-mode equilibria, DIII-D, JET, KSTAR data when available.
- Consider larger MLP (256, 128, 64) for larger datasets.
- Add MC dropout or Bayesian variant for uncertainty.
- GPU retraining estimated at 5-15 min for 10K samples.

### Key Files

| File | Purpose |
|------|---------|
| `src/scpn_fusion/core/neural_equilibrium.py` | Architecture + train/predict (605 lines) |
| `src/scpn_fusion/core/neural_equilibrium_training.py` | Training entrypoint (125 lines) |
| `src/scpn_fusion/core/neural_equilibrium_kernel.py` | FusionKernel drop-in replacement |
| `validation/benchmark_sparc_geqdsk_rmse.py` | Benchmark script (386 lines) |
| `validation/psi_pointwise_rmse.py` | Point-wise psi(R,Z) validation |
| `weights/neural_equilibrium_sparc.npz` | Pretrained weights |
| `weights/pretrained_surrogates_manifest.json` | Training metadata |
| `artifacts/sparc_geqdsk_rmse_benchmark.json` | Latest benchmark results |

---

## 4. GPU Training Infrastructure

### Available Training Pipelines

| Model | Script | Framework | Data Source | GPU Time | Status |
|-------|--------|-----------|-------------|----------|--------|
| Neural Transport (QLKNN) | `tools/train_neural_transport_qlknn.py` | JAX | Zenodo QLKNN-10D (500K samples) | 1-2 h | Production |
| FNO Spatial | `tools/train_fno_qlknn_spatial.py` | JAX | QLKNN oracle (200 equilibria) | 30-60 min | Production |
| GS-Transport Surrogate | `src/.../gs_transport_surrogate_training.py` | NumPy | TransportSolver (5K samples) | 10-30 min | Production |
| Neural Equilibrium | `src/.../neural_equilibrium_training.py` | NumPy | SPARC GEQDSK (78 samples) | 5-15 min | Production |
| Multi-Regime FNO | `src/.../fno_training_multi_regime.py` | NumPy | Synthetic H-W | 30-90 min | Research-only |
| Advanced FNO (velocity) | `src/.../fno_jax_training.py` | JAX | GENE data (not available) | TBD | Scaffold only |

### GPU Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| VRAM | 6 GB | 24 GB (RTX A4000/A5000) |
| CUDA | 12.x | 12.4 |
| cuDNN | 9.x | 9.x |
| RAM | 16 GB | 32 GB |
| Storage | 50 GB | 200 GB (includes QLKNN-10D download) |

Install: `pip install -e ".[gpu,ml,dev]"`

### UpCloud GPU Deployment

```bash
# 1. Clone + install
git clone https://github.com/anulum/scpn-fusion-core.git
cd scpn-fusion-core
pip install -e ".[gpu,ml,dev]"

# 2. Verify GPU
python tools/check_gpu.py

# 3. Download data (QLKNN-10D, ~12 GB)
python tools/download_qlknn10d.py
python tools/qlknn10d_to_npz.py --max-samples 500000

# 4. Train neural transport (primary model)
python tools/train_neural_transport_qlknn.py

# 5. Train FNO spatial
python tools/train_fno_qlknn_spatial.py

# 6. Validate
python validation/full_validation_pipeline.py
```

### External Service Workflow

For production FNO retraining on real gyrokinetic data:
1. Export request: `python tools/export_fno_external_retrain_request.py`
2. Service trains on GENE/CGYRO flux-tube runs externally.
3. Import: `python tools/import_external_fno_weights.py --manifest ... --weights ...`
4. Validate: re-run full pipeline.
Runbook: `docs/FNO_EXTERNAL_RETRAIN_RUNBOOK.md`

### Pretrained Weights Status

| Weight | Date | Accuracy | Depends on Corrected Physics? |
|--------|------|----------|-------------------------------|
| `mlp_itpa` | 2026-02-16 | RMSE 13.5% | No (ITPA scaling fit) |
| `fno_eurofusion_jet` | 2026-02-16 | rel_L2=0.79 | No (Retired in v3.9) |
| `neural_equilibrium_sparc` | 2026-02-16 | NRMSE <1e-4 | No (GS-only, no UPDE) |

All weights generated before Gemini's physics hardening (2026-02-21) but none depend
on the corrected subsystems (Kuramoto coupling, MRE disruption, Bosch-Hale reactivity).
The FNO is already retired from the default lane (v4.0 decision: retrain on real data or remove).

### Priority for GPU Training

1. **Neural Transport (QLKNN)** — most useful, cleanest pipeline, public data.
2. **FNO Spatial** — quick win, JAX-accelerated, improves turbulence modelling.
3. **Neural Equilibrium** — needs more GEQDSK files first (3 files insufficient).
4. **Multi-Regime FNO** — research-only, synthetic data only.

---

## 5. Reference Data Provenance

Source: `validation/reference_data/provenance_manifest.json` (v2, 2026-03-05).

### Real Data (from published sources)

| Dataset | Files | Source |
|---------|-------|--------|
| SPARC equilibria | 8 EQDSK + 3 lmode GEQDSK | CFS/MIT public design data |
| SPARC POPCON | 1 CSV | Published design tables |
| SPARC device description | 1 JSON | Public design parameters |
| ITPA confinement | 53 shots, 1 CSV | Published ITPA H-mode database subset |
| ITPA coefficients | 2 JSON | Derived from published ITPA data |
| QLKNN-10D benchmarks | 1 JSON (summary) | Published QLKNN paper values |
| QLKNN-10D training data | 500K samples (Zenodo) | Real gyrokinetic GENE/QuaLiKiz runs |

### Synthetic Data (generated in-repo)

| Dataset | Files | Provenance Label |
|---------|-------|------------------|
| DIII-D equilibria | 5 GEQDSK | `synthetic-v1` ("DIII-D-like") |
| DIII-D disruption shots | 16 NPZ | `synthetic-v1` ("synthetic DIII-D-like") |
| JET equilibria | 5 GEQDSK | `synthetic-v1` ("JET-like") |
| Blind validation suite | 2 JSON | `synthetic-v1` |

### Training Data Summary

| Model | Trains On | Real? |
|-------|-----------|-------|
| Neural Transport (QLKNN) | Zenodo QLKNN-10D (500K samples) | **Yes** (real gyrokinetic) |
| Neural Equilibrium | 3 SPARC lmode GEQDSK | **Yes** (real CFS/MIT) |
| MLP ITPA | 53-shot ITPA CSV | **Yes** (real multi-machine) |
| FNO turbulence | Synthetic Hasegawa-Wakatani | **No** (Retired in v3.9) |
| GS-Transport surrogate | TransportSolver oracle | **No** (synthetic) |
| Disruption predictor | 16 DIII-D-like NPZ | **No** (synthetic) |

### Data Access Blockers

- **DIII-D raw data**: Pending GA MDSplus access (would replace synthetic GEQDSK + disruption shots).
- **JET DT campaign**: Pending EUROfusion data access (would replace synthetic JET GEQDSK).
- **Additional SPARC**: Pending CFS data release (currently 8 equilibria; target 20+).
