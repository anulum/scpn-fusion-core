# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Disruption Predictor
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Any, Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:  # pragma: no cover - optional dependency path
    torch = None
    nn = None
    optim = None

DEFAULT_SEQ_LEN = 100
DEFAULT_MODEL_FILENAME = "disruption_model.pth"


def _require_int(name: str, value: object, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        if minimum is None:
            raise ValueError(f"{name} must be an integer.")
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    parsed = int(value)
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return parsed


# --- PHYSICS: MODIFIED RUTHERFORD EQUATION ---
def simulate_tearing_mode(
    steps=1000,
    *,
    rng: Optional[np.random.Generator] = None,
):
    """
    Generates synthetic shot data.
    Returns:
        signal (array): Magnetic sensor data (dB/dt)
        label (int): 1 if disrupted, 0 if safe
        time_to_disruption (array): Time remaining (or -1 if safe)
    """
    steps = _require_int("steps", steps, 1)
    dt = 0.01
    w = 0.01 # Island width
    local_rng = rng if rng is not None else np.random.default_rng()
    
    # Physics Parameters
    # Stable shot: Delta' < 0
    # Disruptive shot: Delta' > 0 (Triggered at random time)
    is_disruptive = float(local_rng.random()) > 0.5
    trigger_time = int(local_rng.integers(200, 800)) if is_disruptive else 9999
    
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
        w += float(local_rng.normal(0.0, 0.05)) # Measurement noise
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
    if not np.all(np.isfinite(sig)):
        raise ValueError("signal must be finite.")

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
    if not np.all(np.isfinite([n1, n2, n3, asym, spread])):
        raise ValueError("toroidal observables must be finite.")

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
    if isinstance(bit_index, bool) or not isinstance(bit_index, (int, np.integer)):
        raise ValueError("bit_index must be an integer in [0, 63].")
    bit = int(bit_index)
    if bit < 0 or bit > 63:
        raise ValueError("bit_index must be an integer in [0, 63].")
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


def _normalize_fault_campaign_inputs(
    seed,
    episodes,
    window,
    noise_std,
    bit_flip_interval,
    recovery_window,
    recovery_epsilon,
):
    seed_i = _require_int("seed", seed, 0)
    episodes_i = _require_int("episodes", episodes, 1)
    window_i = _require_int("window", window, 16)
    noise = float(noise_std)
    bit_flip_i = _require_int("bit_flip_interval", bit_flip_interval, 1)
    recovery_window_i = _require_int("recovery_window", recovery_window, 1)
    recovery_eps = float(recovery_epsilon)
    if not np.isfinite(noise) or noise < 0.0:
        raise ValueError("noise_std must be finite and >= 0.")
    if not np.isfinite(recovery_eps) or recovery_eps <= 0.0:
        raise ValueError("recovery_epsilon must be finite and > 0.")

    return (
        seed_i,
        episodes_i,
        window_i,
        noise,
        bit_flip_i,
        recovery_window_i,
        recovery_eps,
    )


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
    (
        seed_i,
        episodes_i,
        window_i,
        noise,
        bit_flip_i,
        recovery_window_i,
        recovery_eps,
    ) = _normalize_fault_campaign_inputs(
        seed,
        episodes,
        window,
        noise_std,
        bit_flip_interval,
        recovery_window,
        recovery_epsilon,
    )
    rng = np.random.default_rng(seed_i)

    abs_errors = []
    recovery_steps = []
    n_faults = 0

    for _ in range(episodes_i):
        signal = _synthetic_control_signal(rng, window_i)
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
            [predict_disruption_risk(signal[: i + 1], toroidal) for i in range(window_i)],
            dtype=float,
        )

        faulty_signal = signal.copy()
        faulty_indices = []
        for i in range(window_i):
            faulty_signal[i] += float(rng.normal(0.0, noise))
            if i % bit_flip_i == 0:
                faulty_signal[i] = apply_bit_flip_fault(
                    faulty_signal[i], int(rng.integers(0, 52))
                )
                faulty_indices.append(i)

        faulty_toroidal = dict(toroidal)
        faulty_toroidal["toroidal_n1_amp"] = max(
            0.0, faulty_toroidal["toroidal_n1_amp"] + float(rng.normal(0.0, noise * 0.5))
        )
        faulty_toroidal["toroidal_n2_amp"] = max(
            0.0, faulty_toroidal["toroidal_n2_amp"] + float(rng.normal(0.0, noise * 0.4))
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
                for i in range(window_i)
            ],
            dtype=float,
        )

        err = np.abs(perturbed - baseline)
        abs_errors.extend(err.tolist())

        for idx in faulty_indices:
            n_faults += 1
            stop = min(window_i, idx + recovery_window_i + 1)
            recover_idx = stop
            for j in range(idx, stop):
                if err[j] <= recovery_eps:
                    recover_idx = j
                    break
            recovery_steps.append(int(recover_idx - idx))

    errors_arr = np.asarray(abs_errors, dtype=float)
    rec_arr = np.asarray(
        recovery_steps if recovery_steps else [recovery_window_i + 1],
        dtype=float,
    )

    mean_abs_err = float(np.mean(errors_arr))
    p95_abs_err = float(np.percentile(errors_arr, 95))
    p95_recovery = float(np.percentile(rec_arr, 95))
    success_rate = float(np.mean(rec_arr <= recovery_window_i))

    thresholds = {
        "max_mean_abs_risk_error": 0.08,
        "max_p95_abs_risk_error": 0.22,
        "max_recovery_steps_p95": float(recovery_window_i),
        "min_recovery_success_rate": 0.80,
    }
    passes = bool(
        mean_abs_err <= thresholds["max_mean_abs_risk_error"]
        and p95_abs_err <= thresholds["max_p95_abs_risk_error"]
        and p95_recovery <= thresholds["max_recovery_steps_p95"]
        and success_rate >= thresholds["min_recovery_success_rate"]
    )

    return {
        "seed": seed_i,
        "episodes": episodes_i,
        "window": window_i,
        "noise_std": noise,
        "bit_flip_interval": bit_flip_i,
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
        threshold_f = float(threshold)
        ema_f = float(ema)
        if not np.isfinite(threshold_f) or threshold_f < 0.0 or threshold_f > 1.0:
            raise ValueError("threshold must be finite and in [0, 1].")
        if not np.isfinite(ema_f) or ema_f <= 0.0 or ema_f > 1.0:
            raise ValueError("ema must be finite and in (0, 1].")
        self.threshold = threshold_f
        self.ema = ema_f
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
    seed_i = _require_int("seed", seed, 0)
    episodes_i = _require_int("episodes", episodes, 1)
    window_i = _require_int("window", window, 16)
    threshold_f = float(threshold)
    if not np.isfinite(threshold_f) or threshold_f < 0.0 or threshold_f > 1.0:
        raise ValueError("threshold must be finite and in [0, 1].")

    rng = np.random.default_rng(seed_i)
    detector = HybridAnomalyDetector(threshold=threshold_f)

    true_positives = 0
    false_positives = 0
    positives = 0
    negatives = 0
    latencies = []

    for ep in range(episodes_i):
        sim_rng = np.random.default_rng(int(seed_i + ep))
        signal, label, _ = simulate_tearing_mode(steps=window_i, rng=sim_rng)
        if signal.size < window_i:
            signal = np.pad(signal, (0, window_i - signal.size), mode="edge")
        signal = np.asarray(signal[:window_i], dtype=float)

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
        for k in range(window_i):
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
    p95_latency = float(np.percentile(latencies, 95)) if latencies else float(window_i)
    passes = bool(tpr >= 0.75 and fpr <= 0.35)

    return {
        "seed": seed_i,
        "episodes": episodes_i,
        "window": window_i,
        "threshold": threshold_f,
        "true_positive_rate": tpr,
        "false_positive_rate": fpr,
        "p95_alarm_latency_steps": p95_latency,
        "passes_thresholds": passes,
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_model_path() -> Path:
    return _repo_root() / "artifacts" / DEFAULT_MODEL_FILENAME


def _normalize_seq_len(seq_len):
    return _require_int("seq_len", seq_len, 8)


def _prepare_signal_window(signal, seq_len):
    seq_len = _normalize_seq_len(seq_len)
    flat = np.asarray(signal, dtype=float).reshape(-1)
    if flat.size >= seq_len:
        return flat[:seq_len]
    return np.pad(flat, (0, seq_len - flat.size), mode="edge")

# --- AI: TRANSFORMER MODEL ---
if torch is not None:
    class DisruptionTransformer(nn.Module):
        def __init__(self, seq_len=DEFAULT_SEQ_LEN):
            super().__init__()
            self.seq_len = _normalize_seq_len(seq_len)
            self.embedding = nn.Linear(1, 32)
            self.pos_encoder = nn.Parameter(torch.zeros(1, self.seq_len, 32))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=32,
                nhead=4,
                dim_feedforward=64,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.classifier = nn.Linear(32, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, src):
            if src.shape[1] > self.seq_len:
                raise ValueError(
                    f"Input sequence length {src.shape[1]} exceeds configured seq_len {self.seq_len}."
                )
            x = self.embedding(src) + self.pos_encoder[:, :src.shape[1], :]
            output = self.transformer(x)
            last_step = output[:, -1, :]
            return self.sigmoid(self.classifier(last_step))
else:  # pragma: no cover - only used without torch installed
    class DisruptionTransformer:  # type: ignore[no-redef]
        def __init__(self):
            raise RuntimeError("Torch is required for DisruptionTransformer.")

def train_predictor(
    seq_len=DEFAULT_SEQ_LEN,
    n_shots=500,
    epochs=50,
    model_path=None,
    seed=42,
    save_plot=True,
):
    if torch is None or optim is None:
        raise RuntimeError("Torch is required for train_predictor().")

    seq_len = _normalize_seq_len(seq_len)
    n_shots = _require_int("n_shots", n_shots, 8)
    epochs = _require_int("epochs", epochs, 1)
    seed = _require_int("seed", seed, 0)
    torch.manual_seed(seed)
    data_rng = np.random.default_rng(seed)
    eval_rng = np.random.default_rng(seed + 1000003)

    model_path = Path(model_path) if model_path is not None else default_model_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print("--- SCPN SAFETY AI: Disruption Prediction (Transformer) ---")
    print(f"Sequence length: {seq_len} | Shots: {n_shots} | Epochs: {epochs}")

    print("Generating synthetic shots (Rutherford Physics)...")
    X_train = []
    y_train = []

    sim_steps = max(1000, seq_len + 16)
    for _ in range(n_shots):
        sig, label, _ = simulate_tearing_mode(steps=sim_steps, rng=data_rng)
        sig_window = _prepare_signal_window(sig, seq_len)
        X_train.append(sig_window.reshape(-1, 1))
        y_train.append(label)

    X_tensor = torch.tensor(np.asarray(X_train, dtype=np.float32), dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    model = DisruptionTransformer(seq_len=seq_len)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    print("Training Transformer...")
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}")

    print("Validating on a new shot...")
    test_sig, test_lbl, _ = simulate_tearing_mode(steps=sim_steps, rng=eval_rng)
    input_sig = _prepare_signal_window(test_sig, seq_len)
    input_tensor = torch.tensor(input_sig, dtype=torch.float32).reshape(1, -1, 1)

    model.eval()
    with torch.no_grad():
        risk = model(input_tensor).item()

    print(f"Test Shot Ground Truth: {'DISRUPTIVE' if test_lbl else 'SAFE'}")
    print(f"AI Prediction Risk: {risk * 100:.1f}%")

    torch.save({"state_dict": model.state_dict(), "seq_len": int(seq_len)}, model_path)
    print(f"Saved model: {model_path}")

    plot_path = _repo_root() / "artifacts" / "Disruption_AI_Result.png"
    if save_plot:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(losses)
        ax1.set_title("Transformer Training Loss")
        ax1.set_xlabel("Epoch")
        ax2.plot(test_sig, "r-" if test_lbl else "g-")
        ax2.set_title(f"Diagnostic Signal (AI Risk: {risk:.2f})")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"Saved: {plot_path}")

    return model, {
        "seq_len": int(seq_len),
        "shots": int(n_shots),
        "epochs": int(epochs),
        "model_path": str(model_path),
        "risk": float(risk),
        "test_label": int(test_lbl),
    }


def load_or_train_predictor(
    model_path=None,
    seq_len=DEFAULT_SEQ_LEN,
    force_retrain=False,
    train_kwargs=None,
    train_if_missing=True,
    allow_fallback=True,
):
    if torch is None:
        if not allow_fallback:
            raise RuntimeError("Torch is required for load_or_train_predictor().")
        return None, {
            "trained": False,
            "fallback": True,
            "reason": "torch_unavailable",
            "model_path": str(model_path) if model_path is not None else str(default_model_path()),
            "seq_len": int(_normalize_seq_len(seq_len)),
        }

    path = Path(model_path) if model_path is not None else default_model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    seq_len = _normalize_seq_len(seq_len)
    kwargs = dict(train_kwargs or {})

    if path.exists() and not force_retrain:
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            loaded_seq_len = _normalize_seq_len(checkpoint.get("seq_len", seq_len))
        else:
            state_dict = checkpoint
            loaded_seq_len = seq_len

        model = DisruptionTransformer(seq_len=loaded_seq_len)
        model.load_state_dict(state_dict)
        model.eval()
        return model, {
            "trained": False,
            "fallback": False,
            "model_path": str(path),
            "seq_len": int(loaded_seq_len),
        }

    if not train_if_missing and not force_retrain:
        if allow_fallback:
            return None, {
                "trained": False,
                "fallback": True,
                "reason": "checkpoint_missing",
                "model_path": str(path),
                "seq_len": int(seq_len),
            }
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    kwargs.setdefault("seq_len", seq_len)
    kwargs.setdefault("model_path", path)
    try:
        model, info = train_predictor(**kwargs)
    except Exception as exc:
        if not allow_fallback:
            raise
        return None, {
            "trained": False,
            "fallback": True,
            "reason": f"train_failed:{exc.__class__.__name__}",
            "model_path": str(path),
            "seq_len": int(seq_len),
        }
    info["trained"] = True
    info["fallback"] = False
    return model, info


def predict_disruption_risk_safe(
    signal,
    toroidal_observables=None,
    *,
    model_path=None,
    seq_len=DEFAULT_SEQ_LEN,
    train_if_missing=False,
) -> tuple[float, dict[str, Any]]:
    """
    Predict disruption risk with checkpoint path if available, else deterministic fallback.

    Returns
    -------
    risk, metadata
        ``risk`` is always a bounded float in ``[0, 1]``.
        ``metadata`` includes whether fallback mode was used.
    """
    base_risk = float(np.clip(predict_disruption_risk(signal, toroidal_observables), 0.0, 1.0))

    model, meta = load_or_train_predictor(
        model_path=model_path,
        seq_len=seq_len,
        force_retrain=False,
        train_kwargs={"seq_len": _normalize_seq_len(seq_len), "save_plot": False},
        train_if_missing=bool(train_if_missing),
        allow_fallback=True,
    )

    if model is None or torch is None:
        out_meta = dict(meta)
        out_meta["mode"] = "fallback"
        out_meta["risk_source"] = "predict_disruption_risk"
        return base_risk, out_meta

    try:
        model.eval()
        model_seq_len = int(meta.get("seq_len", _normalize_seq_len(seq_len)))
        input_sig = _prepare_signal_window(signal, model_seq_len)
        input_tensor = torch.tensor(input_sig, dtype=torch.float32).reshape(1, -1, 1)
        with torch.no_grad():
            model_risk = float(np.clip(float(model(input_tensor).item()), 0.0, 1.0))
        out_meta = dict(meta)
        out_meta["mode"] = "checkpoint"
        out_meta["risk_source"] = "transformer"
        return model_risk, out_meta
    except Exception as exc:
        out_meta = dict(meta)
        out_meta["mode"] = "fallback"
        out_meta["risk_source"] = "predict_disruption_risk"
        out_meta["reason"] = f"inference_failed:{exc.__class__.__name__}"
        return base_risk, out_meta

def evaluate_predictor(
    model: Any,
    X_test: Any,
    y_test: Any,
    times_test: Any = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate disruption predictor on test set.

    Returns dict with accuracy, precision, recall, F1, confusion matrix,
    and recall@T for T in [10, 20, 30, 50, 100] ms.
    """
    pred_list: list[int] = []
    for seq in X_test:
        pred = model.predict(seq)
        pred_list.append(1 if pred > threshold else 0)
    predictions = np.array(pred_list)
    y_true = np.array(y_test)

    tp = np.sum((predictions == 1) & (y_true == 1))
    fp = np.sum((predictions == 1) & (y_true == 0))
    tn = np.sum((predictions == 0) & (y_true == 0))
    fn = np.sum((predictions == 0) & (y_true == 1))

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    fpr = fp / max(fp + tn, 1)

    result = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'false_positive_rate': float(fpr),
        'confusion_matrix': {'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)},
    }

    # Recall@T metrics
    if times_test is not None:
        for T_ms in [10, 20, 30, 50, 100]:
            T_s = T_ms / 1000.0
            early_enough = np.array(times_test) >= T_s
            mask = (y_true == 1) & early_enough
            if mask.sum() > 0:
                recall_at_t = np.sum(predictions[mask] == 1) / mask.sum()
            else:
                recall_at_t = 0.0
            result[f'recall_at_{T_ms}ms'] = float(recall_at_t)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or load disruption Transformer predictor.")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--shots", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    _, meta = load_or_train_predictor(
        model_path=args.model_path,
        seq_len=args.seq_len,
        force_retrain=bool(args.force_retrain),
        train_kwargs={
            "seq_len": args.seq_len,
            "n_shots": args.shots,
            "epochs": args.epochs,
            "seed": args.seed,
            "save_plot": not args.no_plot,
        },
    )
    print(
        f"Predictor ready | trained={meta.get('trained')} | seq_len={meta.get('seq_len')} "
        f"| model_path={meta.get('model_path')}"
    )
