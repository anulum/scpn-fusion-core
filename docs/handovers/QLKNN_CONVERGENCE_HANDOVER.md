# Claude Handover: QLKNN-10D Neural Transport Convergence

**Date**: 2026-02-22
**Branch**: `main`
**Latest commit**: `6b38608` (fix(qlknn): harden data pipeline + training for real Zenodo data)
**Objective**: Achieve val_relative_l2 < 0.10 on real QLKNN-10D gyrokinetic data
**Current best**: val_relative_l2 ~ 0.33 (from previous 22h Gemini session)
**Priority**: This is the single blocking task. Do not work on anything else.

---

## CRITICAL CONTEXT: Why This Is Hard

A previous 22-hour Gemini session identified fundamental dataset issues that make naive training impossible. The pipeline scripts (`tools/qlknn10d_to_npz.py` and `tools/train_neural_transport_qlknn.py`) have already been hardened to address these issues but **convergence to the target accuracy has not been achieved**.

### Dataset Issues (Already Mitigated in Pipeline)

1. **It's 9D, not 10D**: The Zenodo record is called "QLKNN10D" but the primary HDF5 file `gen5_9D_nions0_flat_filter10.h5` contains only 9 input dimensions. `alpha` (alpha_MHD) and `Machtor` are missing; `Machtor` is hardcoded to `0.0` in file metadata.

2. **Zeff Interference**: The 10D MLP contract excludes `Zeff`. Variations in `Zeff` in the dataset appear as irreducible noise. **Fix already applied**: pipeline filters to `Zeff = 1.0` only.

3. **Grid Bias**: Data is stored in strict grid order (by `rho`, then `Zeff`). Loading first N samples gives heavily biased subset. **Fix already applied**: global shuffle before slicing.

4. **Pandas HDFStore Layout**: Non-standard `block0_values` layout requiring `pd.read_hdf()`. **Fix already applied** in loader.

5. **Sharp On/Off Transitions**: The model learns general gyro-Bohm scaling but struggles with the sharp threshold transitions at critical gradients (ITG onset, TEM onset). This is the **primary convergence blocker**.

---

## WHAT YOU MUST NOT DO

1. **DO NOT** lower the verification hard-fail threshold below 0.10
2. **DO NOT** generate synthetic data to substitute for real QLKNN-10D data
3. **DO NOT** modify `src/scpn_fusion/core/neural_transport.py` (the inference module)
4. **DO NOT** modify any files in `tests/`
5. **DO NOT** push to remote without explicit user permission
6. **DO NOT** delete existing weight files until replacements pass all gates
7. **DO NOT** modify the 10D input / 3D output contract

---

## CURRENT STATE

### Data: Already Downloaded
```
data/qlknn10d/
├── gen5_9D_nions0_flat_filter10.h5   (primary — ~2.8 GB)
├── Zeffcombo_prepared.nc
├── Zeffcombo_prepared_grow.nc
├── derived_sets.md5
└── README.md
```
**You do NOT need to re-download.** Step 1 is done.

### Pipeline Scripts: Already Hardened (commit 6b38608)
- `tools/qlknn10d_to_npz.py` — Zeff=1.0 filter, global shuffle, log-uniform Te sampling
- `tools/train_neural_transport_qlknn.py` — Physical output_scale init, exact GELU, grad clip=10.0

### Processed Data: May Need Regeneration
If `data/qlknn10d_processed/{train,val,test}.npz` exist from a previous run, they may use old parameters. **Regenerate with the commands below.**

---

## EXECUTION PLAN

### Step 1: Verify Environment

```bash
cd <repo root>   # C:\aaa_God_of_the_Math_Collection\03_CODE\SCPN-Fusion-Core
python --version  # Need 3.9+
pip install -e ".[dev]"
python -c "import jax; print(jax.devices())"  # Check GPU
python -m pytest tests/test_neural_transport.py -v --tb=short  # Must pass
```

### Step 2: Regenerate Processed Data (1M samples, Zeff=1.0 filter)

```bash
python tools/qlknn10d_to_npz.py --max-samples 1000000
```

**Expected**: After Zeff=1.0 filtering, this yields ~200,000 high-quality samples.
**Verify**: `ls -la data/qlknn10d_processed/` shows `train.npz`, `val.npz`, `test.npz`.

The `--max-samples 1000000` means load up to 1M raw rows from HDF5, then filter to Zeff=1.0. The actual yield depends on Zeff distribution (~20% of rows have Zeff=1.0).

### Step 3: Train with Increased Capacity

**First attempt** — wider network, lower learning rate, more patience:

```bash
python tools/train_neural_transport_qlknn.py \
    --data-dir data/qlknn10d_processed \
    --output weights/neural_transport_qlknn.npz \
    --hidden-dims "512,512,256" \
    --epochs 1000 \
    --lr 1e-4 \
    --batch-size 8192 \
    --patience 100
```

**Monitor**: The script prints every epoch. Watch for:
- `val_mse` decreasing over first 100 epochs
- `val_mse` eventually plateauing
- `best_val` metric improving

**If val_relative_l2 converges to 0.15-0.25** (better than 0.33 but not < 0.10):

The problem is the sharp on/off threshold physics. You need to modify the loss function. See "Convergence Strategies" below.

**If val_relative_l2 converges to 0.25-0.35** (no improvement from previous):

The model capacity or data volume is insufficient. Try:
```bash
python tools/qlknn10d_to_npz.py --max-samples 5000000
python tools/train_neural_transport_qlknn.py \
    --data-dir data/qlknn10d_processed \
    --output weights/neural_transport_qlknn.npz \
    --hidden-dims "512,512,512,256" \
    --epochs 1500 \
    --lr 5e-5 \
    --batch-size 16384 \
    --patience 150
```

### Step 4: Verification Gate

The training script auto-runs verification. It will print:
- `PASS` if test_relative_l2 < 0.05
- `WARN` if 0.05 <= test_relative_l2 < 0.10 (weights still saved)
- `FAIL` if test_relative_l2 >= 0.10 (weights NOT saved)

If WARN (0.05-0.10): **This is acceptable for now.** The weights are saved. Proceed to Step 5.

If FAIL (>= 0.10): Apply convergence strategies below, retrain.

### Step 5: Post-Training Validation

```bash
# Verify model loads
python -c "
import sys; sys.path.insert(0, 'src')
from scpn_fusion.core.neural_transport import NeuralTransportModel, TransportInputs
model = NeuralTransportModel('weights/neural_transport_qlknn.npz')
assert model.is_neural, 'FAIL: model did not load as neural'
result = model.predict(TransportInputs(
    rho=0.5, te_kev=10.0, ti_kev=10.0, ne_19=5.0,
    grad_te=8.0, grad_ti=8.0, grad_ne=2.0,
    q=1.5, s_hat=0.8, beta_e=0.02))
print(f'chi_e={result.chi_e:.4f}, chi_i={result.chi_i:.4f}, d_e={result.d_e:.4f}')
assert all(v > 0 for v in [result.chi_e, result.chi_i, result.d_e])
print('PASS')
"

# Full validation
python validation/validate_transport_qlknn.py \
    --weights weights/neural_transport_qlknn.npz \
    --data-dir data/qlknn10d_processed

# All tests still pass
python -m pytest tests/test_neural_transport.py -v --tb=short
```

### Step 6: Update Manifest and Report

If training succeeds (val_relative_l2 < 0.10):

1. Update `weights/pretrained_surrogates_manifest.json` with actual measured numbers
2. Run `python validation/collect_results.py --quick` to regenerate RESULTS.md
3. Report results to the user — include exact val_relative_l2, per-output L2, training time

---

## CONVERGENCE STRATEGIES

If Step 3 stalls at val_relative_l2 > 0.15, the problem is threshold physics.
Apply these strategies **in order**, retraining after each change.

### Strategy A: Hybrid Log-MSE Loss (RECOMMENDED FIRST)

The sharp ITG/TEM onset creates a bimodal distribution: near-zero fluxes below threshold, large fluxes above. Standard MSE is dominated by large-flux errors and ignores the threshold structure.

In `tools/train_neural_transport_qlknn.py`, replace the `loss_fn`:

```python
@jit
def loss_fn(params, x_batch, y_batch):
    preds = vmap(lambda x: forward(params, x))(x_batch)
    # Hybrid: MSE on raw values + MSE on log(1 + value)
    mse_raw = jnp.mean((preds - y_batch) ** 2)
    mse_log = jnp.mean((jnp.log1p(preds) - jnp.log1p(y_batch)) ** 2)
    return 0.5 * mse_raw + 0.5 * mse_log
```

This gives equal weight to getting the threshold location right (log space) and the magnitude right (linear space).

### Strategy B: Regime-Weighted Loss

Weight samples by regime difficulty. Near-threshold samples (where flux transitions from ~0 to large) get higher weight:

```python
@jit
def loss_fn(params, x_batch, y_batch):
    preds = vmap(lambda x: forward(params, x))(x_batch)
    errors = (preds - y_batch) ** 2
    # Higher weight for samples near threshold (small but non-zero flux)
    flux_mag = jnp.sum(y_batch, axis=1)
    near_threshold = jnp.exp(-0.5 * ((flux_mag - 1.0) / 0.5) ** 2)  # Gaussian bump around flux~1
    weights = 1.0 + 2.0 * near_threshold
    return jnp.mean(weights[:, None] * errors)
```

### Strategy C: Mixture of Experts / Gating Network

If A and B don't work, the MLP may need structural changes to handle the bimodal distribution. Add a soft gating mechanism:

Replace the single MLP with two parallel heads (stable/unstable) and a learned gate:
- Head 1: specializes in near-zero fluxes (stable regime)
- Head 2: specializes in large fluxes (unstable regime)
- Gate: sigmoid(linear(features)) blends the two

This is a larger change. Only attempt if A and B fail.

### Strategy D: Input Feature Engineering

Add derived features that make the threshold explicit:
- `ati_minus_threshold`: where threshold ~ 4.0 for ITG
- `ate_minus_threshold`: where threshold ~ 5.0 for TEM
- `max(0, ati - threshold)`: ReLU-like critical gradient excess

This changes `INPUT_DIM` from 10 to 12-13. Requires updates to:
- `tools/qlknn10d_to_npz.py` (add columns to X)
- `tools/train_neural_transport_qlknn.py` (update INPUT_DIM)
- `src/scpn_fusion/core/neural_transport.py` (update inference)
Only attempt as last resort.

---

## KEY FILES

| File | Role | Modify? |
|------|------|---------|
| `tools/qlknn10d_to_npz.py` | HDF5 → .npz conversion | Only if data issues found |
| `tools/train_neural_transport_qlknn.py` | JAX MLP training | Yes — loss function tuning |
| `src/scpn_fusion/core/neural_transport.py` | Inference module | NO (unless Strategy D) |
| `weights/neural_transport_qlknn.npz` | Output weights | Overwritten by training |
| `weights/pretrained_surrogates_manifest.json` | Metadata | Update after success |
| `data/qlknn10d/` | Raw HDF5 data | Already downloaded, don't touch |
| `data/qlknn10d_processed/` | Processed .npz | Regenerate in Step 2 |
| `validation/validate_transport_qlknn.py` | Post-training validation | NO |
| `tests/test_neural_transport.py` | Unit tests | NO |

---

## SUCCESS CRITERIA

| Metric | Hard Fail | Acceptable | Good | Excellent |
|--------|-----------|-----------|------|-----------|
| val_relative_l2 | >= 0.10 | 0.05-0.10 | 0.03-0.05 | < 0.03 |
| Per-output chi_e L2 | >= 0.15 | 0.08-0.15 | < 0.08 | < 0.05 |
| Per-output chi_i L2 | >= 0.15 | 0.06-0.15 | < 0.06 | < 0.04 |
| Per-output D_e L2 | >= 0.20 | 0.10-0.20 | < 0.10 | < 0.06 |
| All predictions finite | Required | - | - | - |
| All predictions >= 0 | Required | - | - | - |

The "acceptable" range (0.05-0.10) is a significant improvement from the current 0.33 and would be a publishable result. Don't chase "excellent" if it means spending 20+ hours — get to "acceptable" first, then iterate.

---

## REPORTING

When done, write results to:
`.coordination/handovers/CLAUDE_QLKNN_TRAINING_RESULTS_<DATE>.md`

Include:
1. Final val_relative_l2 and per-output breakdown
2. Which convergence strategy worked (or which ones failed and why)
3. Exact command used for final successful training
4. Training time and platform (GPU/CPU)
5. Any pipeline changes made (with diffs)
6. Whether tests still pass

---

## HARDWARE NOTE

- **GPU strongly recommended** for convergence tuning (many retrain cycles needed)
- CPU training at 512,512,256 with 200K samples takes ~2-4 hours per run
- GPU training same config takes ~15-30 minutes per run
- To check: `python -c "import jax; print(jax.devices())"`
- For CUDA: `pip install "jax[cuda12]>=0.4.20"`

---

## CITATION

Data source: van de Plassche, K.L. et al. (2020). "Fast modeling of turbulent transport in fusion plasmas using neural networks." *Phys. Plasmas* 27, 022310. DOI: 10.1063/1.5134126. Dataset: https://doi.org/10.5281/zenodo.3497066
