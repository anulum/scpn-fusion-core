# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Neural Transport Training Guide
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────

This document describes how to retrain the Neural Transport Surrogate using the QLKNN-10D dataset.

## 1. Dataset Acquisition
The SCPN Neural Transport module is designed for the QLKNN-10D public dataset:
- **DOI:** [10.5281/zenodo.3700755](https://doi.org/10.5281/zenodo.3700755)
- **Files:** `QLKNN-10D_lowflux_regular_10000.csv` (or full version)

Download the CSV files and place them in `validation/reference_data/qlknn/`.

## 2. Training with JAX
Use the provided utility `tools/train_neural_transport.py` to fit the MLP.

```bash
# Set PYTHONPATH to include src/
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Run training
python tools/train_neural_transport.py --output weights/neural_transport_qlknn.npz
```

The script currently uses a high-fidelity synthetic generator if the CSV is missing. To use the real dataset, modify the `generate_synthetic_data` call in the script to load your downloaded CSV.

## 3. Architecture Details
- **Inputs (10):** `[rho, Te, Ti, ne, R/LTe, R/LTi, R/Lne, q, s_hat, beta_e]`
- **Outputs (3):** `[chi_e, chi_i, D_e]`
- **Hidden Layers:** 64 -> 32 (ReLU activation)
- **Output Activation:** Softplus (ensures non-negative diffusivities)

## 4. Integration
Once trained, the `NeuralTransportModel` in `scpn_fusion.core.neural_transport` will automatically load the weights from `weights/neural_transport_qlknn.npz` by default.

To verify the model performance:
```python
from scpn_fusion.core.neural_transport import NeuralTransportModel
model = NeuralTransportModel()
print(f"Neural model active: {model.is_neural}")
```
