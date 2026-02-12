# Neural Transport Training

**Date**: 2026-02-12  
**Status**: Active (`WP-E2`)

---

## 1. Goal

Train and validate a neural transport surrogate that matches the analytic
critical-gradient fallback and exports weights in the `.npz` format expected by:

- Python: `scpn_fusion.core.neural_transport.NeuralTransportModel`
- Rust: `fusion-ml/src/neural_transport.rs`

Architecture:

- Input: `10`
- Hidden 1: `64` (ReLU)
- Hidden 2: `32` (ReLU)
- Output: `3` (Softplus, scaled)

Output channels:

1. `chi_i`
2. `chi_e`
3. `d_eff`

---

## 2. Data Generation

Training data is generated synthetically from the analytic fallback model:

`critical_gradient_model(inputs) -> [chi_i, chi_e, d_eff]`

Input ordering:

`[grad_ti, grad_te, grad_ne, shear, collisionality, zeff, q95, beta_n, rho, aspect]`

The ranges are sampled uniformly and cover a practical operating envelope for
rapid bootstrap training.

---

## 3. Training Command

From `03_CODE/SCPN-Fusion-Core`:

```bash
python examples/train_neural_transport.py
```

Useful options:

```bash
python examples/train_neural_transport.py --samples 12000 --epochs 200 --lr 1e-3
python examples/train_neural_transport.py --quick
python examples/train_neural_transport.py --output weights/neural_transport_weights.npz
```

---

## 4. Weight Format

The training script writes:

- `w1` shape `(10, 64)`
- `b1` shape `(64,)`
- `w2` shape `(64, 32)`
- `b2` shape `(32,)`
- `w3` shape `(32, 3)`
- `b3` shape `(3,)`
- `input_mean` shape `(10,)`
- `input_std` shape `(10,)`
- `output_scale` shape `(3,)`

This is the exact key set loaded by the Rust `NeuralTransportModel::from_npz`.

---

## 5. Validation Command

```bash
python examples/validate_neural_transport.py --weights weights/neural_transport_weights.npz
```

Validation reports:

- MAE
- RMSE
- Relative L2
- global R²

Default pass criterion:

- `relative L2 <= 0.20`

Adjust threshold if needed:

```bash
python examples/validate_neural_transport.py --max-rel-l2 0.15
```

---

## 6. Notes

- Training is NumPy-only (no PyTorch dependency).
- The saved model is suitable for direct Rust inference.
- `--quick` mode is intended for CI/smoke checks, not final quality training.
