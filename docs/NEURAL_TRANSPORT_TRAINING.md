# Neural Transport Training (QLKNN-10D)

This guide reproduces the `weights/neural_transport_qlknn.npz` artifact used by
`scpn_fusion.core.neural_transport.NeuralTransportModel`.

## Scope

- Dataset: QLKNN-10D public record (Zenodo DOI `10.5281/zenodo.3497066`)
- Pipeline scripts:
  - `tools/download_qlknn10d.py`
  - `tools/qlknn10d_to_npz.py`
  - `tools/train_neural_transport_qlknn.py`
- Output artifact:
  - `weights/neural_transport_qlknn.npz`
  - `weights/neural_transport_qlknn.metrics.json`

## 1. Download + integrity check

```bash
python tools/download_qlknn10d.py
python tools/download_qlknn10d.py --check
```

Notes:
- Download path defaults to `data/qlknn10d/`.
- The downloader verifies Zenodo-provided MD5 checksums.
- A provenance file is written to `data/qlknn10d/README.md`.

## 2. Convert raw data to train/val/test NPZ

```bash
python tools/qlknn10d_to_npz.py --max-samples 500000
```

Outputs (default):
- `data/qlknn10d_processed/train.npz`
- `data/qlknn10d_processed/val.npz`
- `data/qlknn10d_processed/test.npz`

Schema:
- `X`: input features
- `Y`: target fluxes
- `regimes`: regime labels
- `gb_normalized`: metadata flag (1/0)

## 3. Train surrogate

Quick smoke run:

```bash
python tools/train_neural_transport_qlknn.py --quick
```

Full training (recommended baseline):

```bash
python tools/train_neural_transport_qlknn.py \
  --epochs 500 \
  --lr 3e-4 \
  --batch-size 4096 \
  --hidden-dims 256,128
```

Optional physics-aware flags:
- `--log-transform`
- `--gb-scale`
- `--gated`
- `--hybrid-log`
- `--align-metric`
- `--residual`

## 4. Verify the produced weight artifact

The training script already verifies saved weights before returning success.
Manual smoke check:

```bash
python - <<'PY'
from scpn_fusion.core.neural_transport import NeuralTransportModel, TransportInputs
m = NeuralTransportModel("weights/neural_transport_qlknn.npz")
print("is_neural:", m.is_neural, "checksum:", m.weights_checksum)
print(m.predict(TransportInputs()))
PY
```

Expected:
- `is_neural: True`
- non-empty `checksum`
- finite non-negative flux outputs

## 5. Weight file format contract

`NeuralTransportModel` expects:
- `w1..wN`, `b1..bN` with `N >= 2`
- `input_mean`
- `input_std`
- `output_scale`

Optional keys:
- `version` (must equal loader format, currently `1`)
- `log_transform` (0/1)
- `gb_scale` (0/1)
- `gated` (0/1)
- `residual` (0/1)
- `stiff_coeffs` (used with residual mode)

Hardening checks enforced by loader:
- `.npz` extension only
- `allow_pickle=False`
- bounded file size
- required-key and version validation
- revert to analytic critical-gradient model on any contract failure

## 6. Reproducibility notes

- Pin a seed (`--seed`, default `42`) for deterministic split/shuffle behavior.
- Archive both files for claims/evidence:
  - `weights/neural_transport_qlknn.npz`
  - `weights/neural_transport_qlknn.metrics.json`
- Record command line and git SHA in release artifacts when publishing results.
