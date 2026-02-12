# Neural Transport Training Guide

Training recipe for the QLKNN-style MLP surrogate used by
`scpn_fusion.core.neural_transport.NeuralTransportModel`.

## Overview

The neural transport surrogate replaces the analytic critical-gradient
model with a trained MLP that reproduces gyrokinetic-level predictions
at millisecond inference speeds.  The architecture follows the QLKNN
paradigm (van de Plassche et al., *Phys. Plasmas* 27, 022310, 2020).

## 1. Obtain Training Data

Download the QLKNN-10D public dataset from Zenodo:

```
https://doi.org/10.5281/zenodo.3700755
```

The dataset contains ~300 million QuaLiKiz evaluations across 10
normalised plasma parameters.  Extract the archive to a working
directory:

```bash
mkdir -p data/qlknn
cd data/qlknn
wget https://zenodo.org/records/3700755/files/qlknn_10d_data.tar.gz
tar xzf qlknn_10d_data.tar.gz
```

## 2. Input / Output Specification

### Inputs (10 features)

| Index | Symbol  | Description                          | Range (typical) |
|-------|---------|--------------------------------------|-----------------|
| 0     | ρ       | Normalised toroidal flux coordinate  | 0.0 – 1.0      |
| 1     | T_e     | Electron temperature [keV]           | 0.1 – 30       |
| 2     | T_i     | Ion temperature [keV]                | 0.1 – 30       |
| 3     | n_e     | Electron density [10^19 m^-3]        | 0.5 – 15       |
| 4     | R/L_Te  | Normalised electron temp. gradient   | 0 – 50         |
| 5     | R/L_Ti  | Normalised ion temp. gradient        | 0 – 50         |
| 6     | R/L_ne  | Normalised density gradient          | -10 – 30       |
| 7     | q       | Safety factor                        | 1 – 6          |
| 8     | ŝ       | Magnetic shear                       | 0 – 5          |
| 9     | β_e     | Electron beta                        | 0 – 0.05       |

### Outputs (3 targets)

| Index | Symbol | Description                      | Unit    |
|-------|--------|----------------------------------|---------|
| 0     | χ_e    | Electron thermal diffusivity     | m²/s    |
| 1     | χ_i    | Ion thermal diffusivity          | m²/s    |
| 2     | D_e    | Particle diffusivity             | m²/s    |

## 3. Architecture

```
Input(10) → Dense(hidden) → ReLU → Dense(hidden) → ReLU → Dense(3) → Softplus
```

Recommended hidden size: 64 or 128.  Softplus on the output layer
ensures all transport coefficients are strictly positive.

Input normalisation: subtract `input_mean`, divide by `input_std`
(computed from the training set).

Output scaling: multiply by `output_scale` (per-channel standard
deviation of the training targets).

## 4. Training Script

```python
#!/usr/bin/env python3
"""Train QLKNN-style MLP for neural transport surrogate."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Hyperparameters ──────────────────────────────────────────────
HIDDEN = 128
EPOCHS = 100
BATCH = 4096
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Load data ────────────────────────────────────────────────────
# Adapt paths to your extracted QLKNN-10D dataset.
X_train = np.load("data/qlknn/X_train.npy").astype(np.float64)
Y_train = np.load("data/qlknn/Y_train.npy").astype(np.float64)
X_val   = np.load("data/qlknn/X_val.npy").astype(np.float64)
Y_val   = np.load("data/qlknn/Y_val.npy").astype(np.float64)

# ── Normalisation ────────────────────────────────────────────────
input_mean = X_train.mean(axis=0)
input_std  = X_train.std(axis=0)
input_std[input_std < 1e-8] = 1.0

output_scale = Y_train.std(axis=0)
output_scale[output_scale < 1e-8] = 1.0

X_train_n = (X_train - input_mean) / input_std
X_val_n   = (X_val   - input_mean) / input_std

# ── Model ────────────────────────────────────────────────────────
class TransportMLP(nn.Module):
    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x)

model = TransportMLP().double()
opt   = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

train_dl = DataLoader(
    TensorDataset(
        torch.from_numpy(X_train_n),
        torch.from_numpy(Y_train / output_scale),
    ),
    batch_size=BATCH, shuffle=True,
)

# ── Training loop ────────────────────────────────────────────────
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(xb)

    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(torch.from_numpy(X_val_n))
        val_loss = loss_fn(val_pred, torch.from_numpy(Y_val / output_scale))

    print(f"Epoch {epoch+1:3d}  train={total_loss/len(X_train):.6f}  val={val_loss:.6f}")

# ── Export weights as .npz ───────────────────────────────────────
state = model.state_dict()
np.savez(
    "weights/neural_transport_v1.npz",
    version=np.array(1),
    w1=state["net.0.weight"].numpy().T,   # (10, hidden)
    b1=state["net.0.bias"].numpy(),        # (hidden,)
    w2=state["net.2.weight"].numpy().T,    # (hidden, hidden)
    b2=state["net.2.bias"].numpy(),        # (hidden,)
    w3=state["net.4.weight"].numpy().T,    # (hidden, 3)
    b3=state["net.4.bias"].numpy(),        # (3,)
    input_mean=input_mean,
    input_std=input_std,
    output_scale=output_scale,
)
print("Exported weights/neural_transport_v1.npz")
```

## 5. Export Format

The `.npz` file must contain these arrays:

| Key            | Shape            | Description                    |
|----------------|------------------|--------------------------------|
| `version`      | scalar           | Format version (must be `1`)   |
| `w1`           | `(10, hidden)`   | First layer weights            |
| `b1`           | `(hidden,)`      | First layer biases             |
| `w2`           | `(hidden, hidden)`| Second layer weights          |
| `b2`           | `(hidden,)`      | Second layer biases            |
| `w3`           | `(hidden, 3)`    | Output layer weights           |
| `b3`           | `(3,)`           | Output layer biases            |
| `input_mean`   | `(10,)`          | Per-feature input mean         |
| `input_std`    | `(10,)`          | Per-feature input std          |
| `output_scale` | `(3,)`           | Per-channel output scaling     |

**Note:** PyTorch stores weights as `(out, in)`, so transpose with
`.T` when exporting (the inference engine expects `(in, out)`).

## 6. Loading Weights

```python
from scpn_fusion.core.neural_transport import NeuralTransportModel

model = NeuralTransportModel("weights/neural_transport_v1.npz")
assert model.is_neural
print(f"Checksum: {model.weights_checksum}")
```

The loader performs:
- Version check (`version == 1`)
- Key validation (all 9 arrays present)
- SHA-256 checksum for reproducibility tracking

## 7. Validation

After training, verify the surrogate against the analytic fallback:

```python
import numpy as np
from scpn_fusion.core.neural_transport import (
    NeuralTransportModel, TransportInputs, critical_gradient_model,
)

model = NeuralTransportModel("weights/neural_transport_v1.npz")

# Sweep R/L_Ti and compare
for grad_ti in [3.0, 5.0, 8.0, 12.0, 20.0]:
    inp = TransportInputs(grad_ti=grad_ti)
    neural = model.predict(inp)
    analytic = critical_gradient_model(inp)
    print(f"R/L_Ti={grad_ti:5.1f}  neural chi_i={neural.chi_i:.3f}  "
          f"analytic chi_i={analytic.chi_i:.3f}")
```

The neural model should agree qualitatively with the critical-gradient
model (zero below threshold, monotonically increasing above) but with
smoother, more physical transport coefficients.

## References

1. van de Plassche, K.L. et al. (2020). "Fast modeling of turbulent
   transport in fusion plasmas using neural networks." *Phys. Plasmas*
   27, 022310. doi:10.1063/1.5134126

2. Citrin, J. et al. (2015). "Real-time capable first-principles based
   modelling of tokamak turbulent transport." *Nucl. Fusion* 55, 092001.

3. QLKNN-10D Dataset: https://doi.org/10.5281/zenodo.3700755
