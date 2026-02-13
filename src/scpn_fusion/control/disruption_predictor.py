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


def apply_bit_flip_fault(value, bit_index):
    """Inject a deterministic single-bit fault into a float."""
    bit = int(bit_index) % 64
    raw = np.array([float(value)], dtype=np.float64).view(np.uint64)[0]
    flipped = np.uint64(raw ^ (np.uint64(1) << np.uint64(bit)))
    out = np.array([flipped], dtype=np.uint64).view(np.float64)[0]
    return float(out if np.isfinite(out) else value)


def _synthetic_control_signal(rng, length):
    t = np.linspace(0.0, 1.0, int(length), dtype=float)
    base = 0.7 + 0.15 * np.sin(2.0 * np.pi * 3.0 * t) + 0.05 * np.cos(2.0 * np.pi * 7.0 * t)
    ramp = np.where(t > 0.65, (t - 0.65) * 0.9, 0.0)
    noise = rng.normal(0.0, 0.01, size=t.shape)
    return np.clip(base + ramp + noise, 0.01, None)


def run_fault_noise_campaign(
    seed=42,
    episodes=64,
    window=128,
    noise_std=0.03,
    bit_flip_interval=11,
    recovery_window=6,
    recovery_epsilon=0.03,
):
    """
    Run deterministic synthetic fault/noise campaign for disruption-risk resilience.
    """
    rng = np.random.default_rng(int(seed))
    episodes = max(int(episodes), 1)
    window = max(int(window), 16)
    bit_flip_interval = max(int(bit_flip_interval), 1)
    recovery_window = max(int(recovery_window), 1)
    recovery_epsilon = max(float(recovery_epsilon), 1e-9)

    abs_errors = []
    recovery_steps = []
    n_faults = 0

    for _ in range(episodes):
        signal = _synthetic_control_signal(rng, window)
        toroidal = {
            "toroidal_n1_amp": float(rng.uniform(0.05, 0.25)),
            "toroidal_n2_amp": float(rng.uniform(0.03, 0.18)),
            "toroidal_n3_amp": float(rng.uniform(0.01, 0.12)),
            "toroidal_radial_spread": float(rng.uniform(0.01, 0.08)),
        }
        toroidal["toroidal_asymmetry_index"] = float(
            np.sqrt(
                toroidal["toroidal_n1_amp"] ** 2
                + toroidal["toroidal_n2_amp"] ** 2
                + toroidal["toroidal_n3_amp"] ** 2
            )
        )

        baseline = np.array(
            [predict_disruption_risk(signal[: i + 1], toroidal) for i in range(window)],
            dtype=float,
        )

        faulty_signal = signal.copy()
        faulty_indices = []
        for i in range(window):
            faulty_signal[i] += float(rng.normal(0.0, noise_std))
            if i % bit_flip_interval == 0:
                faulty_signal[i] = apply_bit_flip_fault(
                    faulty_signal[i], int(rng.integers(0, 52))
                )
                faulty_indices.append(i)

        faulty_toroidal = dict(toroidal)
        faulty_toroidal["toroidal_n1_amp"] = max(
            0.0, faulty_toroidal["toroidal_n1_amp"] + float(rng.normal(0.0, noise_std * 0.5))
        )
        faulty_toroidal["toroidal_n2_amp"] = max(
            0.0, faulty_toroidal["toroidal_n2_amp"] + float(rng.normal(0.0, noise_std * 0.4))
        )
        faulty_toroidal["toroidal_asymmetry_index"] = float(
            np.sqrt(
                faulty_toroidal["toroidal_n1_amp"] ** 2
                + faulty_toroidal["toroidal_n2_amp"] ** 2
                + faulty_toroidal["toroidal_n3_amp"] ** 2
            )
        )

        perturbed = np.array(
            [
                predict_disruption_risk(faulty_signal[: i + 1], faulty_toroidal)
                for i in range(window)
            ],
            dtype=float,
        )

        err = np.abs(perturbed - baseline)
        abs_errors.extend(err.tolist())

        for idx in faulty_indices:
            n_faults += 1
            stop = min(window, idx + recovery_window + 1)
            recover_idx = stop
            for j in range(idx, stop):
                if err[j] <= recovery_epsilon:
                    recover_idx = j
                    break
            recovery_steps.append(int(recover_idx - idx))

    errors_arr = np.asarray(abs_errors, dtype=float)
    rec_arr = np.asarray(recovery_steps if recovery_steps else [recovery_window + 1], dtype=float)

    mean_abs_err = float(np.mean(errors_arr))
    p95_abs_err = float(np.percentile(errors_arr, 95))
    p95_recovery = float(np.percentile(rec_arr, 95))
    success_rate = float(np.mean(rec_arr <= recovery_window))

    thresholds = {
        "max_mean_abs_risk_error": 0.08,
        "max_p95_abs_risk_error": 0.22,
        "max_recovery_steps_p95": float(recovery_window),
        "min_recovery_success_rate": 0.80,
    }
    passes = bool(
        mean_abs_err <= thresholds["max_mean_abs_risk_error"]
        and p95_abs_err <= thresholds["max_p95_abs_risk_error"]
        and p95_recovery <= thresholds["max_recovery_steps_p95"]
        and success_rate >= thresholds["min_recovery_success_rate"]
    )

    return {
        "seed": int(seed),
        "episodes": episodes,
        "window": window,
        "noise_std": float(noise_std),
        "bit_flip_interval": bit_flip_interval,
        "fault_count": int(n_faults),
        "mean_abs_risk_error": mean_abs_err,
        "p95_abs_risk_error": p95_abs_err,
        "recovery_steps_p95": p95_recovery,
        "recovery_success_rate": success_rate,
        "thresholds": thresholds,
        "passes_thresholds": passes,
    }


class HybridAnomalyDetector:
    """
    Lightweight supervised+unsupervised anomaly detector for early alarms.

    - Supervised term: `predict_disruption_risk`.
    - Unsupervised term: online z-score novelty on recent risk stream.
    """

    def __init__(self, threshold=0.65, ema=0.05):
        self.threshold = float(np.clip(threshold, 0.0, 1.0))
        self.ema = float(np.clip(ema, 1e-4, 1.0))
        self.mean = 0.0
        self.var = 1.0
        self.initialized = False

    def score(self, signal, toroidal_observables=None):
        supervised = predict_disruption_risk(signal, toroidal_observables)
        value = float(supervised)

        if self.initialized:
            z = abs(value - self.mean) / float(np.sqrt(self.var + 1e-9))
            unsupervised = float(1.0 - np.exp(-0.5 * z))
        else:
            unsupervised = 0.0
            self.initialized = True

        alpha = self.ema
        delta = value - self.mean
        self.mean += alpha * delta
        self.var = (1.0 - alpha) * self.var + alpha * delta * delta
        self.var = float(max(self.var, 1e-9))

        anomaly_score = float(np.clip(0.7 * supervised + 0.3 * unsupervised, 0.0, 1.0))
        return {
            "supervised_score": float(supervised),
            "unsupervised_score": float(unsupervised),
            "anomaly_score": anomaly_score,
            "alarm": bool(anomaly_score >= self.threshold),
        }


def run_anomaly_alarm_campaign(
    seed=42,
    episodes=32,
    window=128,
    threshold=0.65,
):
    """
    Deterministic anomaly-alarm campaign under random perturbations.
    """
    rng = np.random.default_rng(int(seed))
    episodes = max(int(episodes), 1)
    window = max(int(window), 16)
    detector = HybridAnomalyDetector(threshold=threshold)

    true_positives = 0
    false_positives = 0
    positives = 0
    negatives = 0
    latencies = []

    for ep in range(episodes):
        np.random.seed(int(seed + ep))
        signal, label, _ = simulate_tearing_mode(steps=window)
        if signal.size < window:
            signal = np.pad(signal, (0, window - signal.size), mode="edge")
        signal = np.asarray(signal[:window], dtype=float)

        n1 = float(rng.uniform(0.03, 0.24))
        n2 = float(rng.uniform(0.02, 0.16))
        n3 = float(rng.uniform(0.01, 0.10))
        toroidal = {
            "toroidal_n1_amp": n1,
            "toroidal_n2_amp": n2,
            "toroidal_n3_amp": n3,
            "toroidal_asymmetry_index": float(np.sqrt(n1 * n1 + n2 * n2 + n3 * n3)),
            "toroidal_radial_spread": float(rng.uniform(0.01, 0.08)),
        }

        first_alarm = None
        for k in range(window):
            perturbed_signal = signal[: k + 1].copy()
            perturbed_signal[-1] += float(rng.normal(0.0, 0.02))
            score = detector.score(perturbed_signal, toroidal)
            if score["alarm"] and first_alarm is None:
                first_alarm = k

        is_positive = bool(label == 1)
        if is_positive:
            positives += 1
            if first_alarm is not None:
                true_positives += 1
                latencies.append(first_alarm)
        else:
            negatives += 1
            if first_alarm is not None:
                false_positives += 1

    tpr = float(true_positives / max(positives, 1))
    fpr = float(false_positives / max(negatives, 1))
    p95_latency = float(np.percentile(latencies, 95)) if latencies else float(window)
    passes = bool(tpr >= 0.75 and fpr <= 0.35)

    return {
        "seed": int(seed),
        "episodes": episodes,
        "window": window,
        "threshold": float(threshold),
        "true_positive_rate": tpr,
        "false_positive_rate": fpr,
        "p95_alarm_latency_steps": p95_latency,
        "passes_thresholds": passes,
    }

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
