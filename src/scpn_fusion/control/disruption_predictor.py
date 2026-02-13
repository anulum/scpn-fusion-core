# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Disruption Predictor
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:  # pragma: no cover - optional dependency path
    torch = None
    nn = None
    optim = None

# --- PHYSICS: MODIFIED RUTHERFORD EQUATION ---
def simulate_tearing_mode(steps=1000):
    """
    Generates synthetic shot data.
    Returns:
        signal (array): Magnetic sensor data (dB/dt)
        label (int): 1 if disrupted, 0 if safe
        time_to_disruption (array): Time remaining (or -1 if safe)
    """
    dt = 0.01
    w = 0.01 # Island width
    
    # Physics Parameters
    # Stable shot: Delta' < 0
    # Disruptive shot: Delta' > 0 (Triggered at random time)
    is_disruptive = np.random.rand() > 0.5
    trigger_time = np.random.randint(200, 800) if is_disruptive else 9999
    
    delta_prime = -0.5
    w_history = []
    
    for t in range(steps):
        # Trigger Instability
        if t > trigger_time:
            delta_prime = 0.5 # Become unstable
            
        # Rutherford Equation: dw/dt = Delta' + Const/w (simplified)
        # Saturated growth: dw/dt = Delta' * (1 - w/w_sat)
        dw = (delta_prime * (1 - w/10.0)) * dt
        w += dw
        w += np.random.normal(0, 0.05) # Measurement noise
        w = max(w, 0.01)
        
        w_history.append(w)
        
        # Disruption Condition: Mode Lock
        if w > 8.0:
            # Locked mode! Signal goes silent or explodes
            return np.array(w_history), 1, (t - trigger_time)
            
    return np.array(w_history), 0, -1


def build_disruption_feature_vector(signal, toroidal_observables=None):
    """
    Build a compact feature vector for control-oriented disruption scoring.

    Feature layout:
      [mean, std, max, slope, energy, last,
       toroidal_n1_amp, toroidal_n2_amp, toroidal_n3_amp,
       toroidal_asymmetry_index, toroidal_radial_spread]
    """
    sig = np.asarray(signal, dtype=float).reshape(-1)
    if sig.size == 0:
        raise ValueError("signal must contain at least one sample")

    mean = float(np.mean(sig))
    std = float(np.std(sig))
    max_val = float(np.max(sig))
    slope = float((sig[-1] - sig[0]) / max(sig.size - 1, 1))
    energy = float(np.mean(sig**2))
    last = float(sig[-1])

    obs = toroidal_observables or {}
    n1 = float(obs.get("toroidal_n1_amp", 0.0))
    n2 = float(obs.get("toroidal_n2_amp", 0.0))
    n3 = float(obs.get("toroidal_n3_amp", 0.0))
    asym = float(obs.get("toroidal_asymmetry_index", np.sqrt(n1 * n1 + n2 * n2 + n3 * n3)))
    spread = float(obs.get("toroidal_radial_spread", 0.0))

    return np.array(
        [mean, std, max_val, slope, energy, last, n1, n2, n3, asym, spread],
        dtype=float,
    )


def predict_disruption_risk(signal, toroidal_observables=None):
    """
    Lightweight deterministic disruption risk estimator (0..1) for control loops.

    This supplements the Transformer pathway by explicitly consuming toroidal
    asymmetry observables from 3D diagnostics.
    """
    features = build_disruption_feature_vector(signal, toroidal_observables)
    mean, std, max_val, slope, energy, last, n1, n2, n3, asym, spread = features

    thermal_term = 0.55 * max_val + 0.35 * std + 0.10 * energy + 0.25 * slope
    asym_term = 1.10 * n1 + 0.70 * n2 + 0.45 * n3 + 0.50 * asym + 0.15 * spread
    state_term = 0.15 * mean + 0.20 * last

    logits = -4.0 + thermal_term + asym_term + state_term
    return float(1.0 / (1.0 + np.exp(-logits)))

# --- AI: TRANSFORMER MODEL ---
if torch is not None:
    class DisruptionTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Linear(1, 32)
            self.pos_encoder = nn.Parameter(torch.zeros(1, 100, 32))
            
            encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            
            self.classifier = nn.Linear(32, 1) # Output: Probability of Disruption
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, src):
            # src shape: [Batch, Seq_Len, 1]
            x = self.embedding(src) + self.pos_encoder[:, :src.shape[1], :]
            x = x.permute(1, 0, 2) # [Seq, Batch, Dim] for Transformer
            
            output = self.transformer(x)
            
            # Take the last time step embedding for classification
            last_step = output[-1, :, :]
            return self.sigmoid(self.classifier(last_step))
else:  # pragma: no cover - only used without torch installed
    class DisruptionTransformer:  # type: ignore[no-redef]
        def __init__(self):
            raise RuntimeError("Torch is required for DisruptionTransformer.")

def train_predictor():
    if torch is None or optim is None:
        raise RuntimeError("Torch is required for train_predictor().")

    print("--- SCPN SAFETY AI: Disruption Prediction (Transformer) ---")
    
    # 1. Generate Data
    print("Generating synthetic shots (Rutherford Physics)...")
    X_train = []
    y_train = []
    
    for _ in range(500):
        sig, label, _ = simulate_tearing_mode()
        # Crop/Pad to length 100
        if len(sig) > 100: sig = sig[:100]
        else: sig = np.pad(sig, (0, 100-len(sig)))
        
        X_train.append(sig.reshape(-1, 1))
        y_train.append(label)
        
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    # 2. Train Model
    model = DisruptionTransformer()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    print("Training Transformer...")
    losses = []
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}")
            
    # 3. Validate
    print("Validating on a new shot...")
    test_sig, test_lbl, _ = simulate_tearing_mode()
    # Prepare for AI
    real_len = min(len(test_sig), 100)
    input_sig = test_sig[:real_len]
    input_tensor = torch.tensor(input_sig, dtype=torch.float32).reshape(1, -1, 1)
    
    with torch.no_grad():
        risk = model(input_tensor).item()
        
    print(f"Test Shot Ground Truth: {'DISRUPTIVE' if test_lbl else 'SAFE'}")
    print(f"AI Prediction Risk: {risk*100:.1f}%")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(losses)
    ax1.set_title("Transformer Training Loss")
    ax1.set_xlabel("Epoch")
    
    ax2.plot(test_sig, 'r-' if test_lbl else 'g-')
    ax2.set_title(f"Diagnostic Signal (AI Risk: {risk:.2f})")
    
    plt.savefig("Disruption_AI_Result.png")
    print("Saved: Disruption_AI_Result.png")

if __name__ == "__main__":
    train_predictor()
