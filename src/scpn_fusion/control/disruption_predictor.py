# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Disruption Predictor
from __future__ import annotations

import logging
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Any

from scpn_fusion.control.disruption_checkpoint_policy import (
    _augment_with_fallback_telemetry,
    _normalize_seq_len,
    _prepare_signal_window,
    _record_recovery_event,
    _repo_root,
    _resolve_allow_fallback,
    _safe_torch_checkpoint_load,
    _validated_checkpoint_state_dict,
    default_model_path,
)
from scpn_fusion.control.disruption_risk_runtime import (
    DEFAULT_DISRUPTION_RISK_BIAS,
    DEFAULT_DISRUPTION_RISK_THRESHOLD,
    DISRUPTION_RISK_LINEAR_WEIGHTS,
    HybridAnomalyDetector,
    _require_int,
    apply_bit_flip_fault,
    apply_disruption_logit_bias,
    build_disruption_feature_vector,
    predict_disruption_risk,
    run_anomaly_alarm_campaign,
    run_fault_noise_campaign,
    simulate_tearing_mode,
)

__all__ = [
    "DEFAULT_DISRUPTION_RISK_BIAS",
    "DEFAULT_DISRUPTION_RISK_THRESHOLD",
    "DISRUPTION_RISK_LINEAR_WEIGHTS",
    "HybridAnomalyDetector",
    "apply_bit_flip_fault",
    "apply_disruption_logit_bias",
    "build_disruption_feature_vector",
    "predict_disruption_risk",
    "run_anomaly_alarm_campaign",
    "run_fault_noise_campaign",
    "DisruptionTransformer",
    "train_predictor",
    "load_or_train_predictor",
    "predict_disruption_risk_safe",
    "evaluate_predictor",
]

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except (ImportError, OSError):  # pragma: no cover - optional dependency path
    torch = None
    nn = None
    optim = None

DEFAULT_SEQ_LEN = 100
DEFAULT_MODEL_FILENAME = "disruption_model.pth"
_CHECKPOINT_LOAD_EXCEPTIONS = (
    RuntimeError,
    ValueError,
    TypeError,
    OSError,
    KeyError,
    AttributeError,
    IndexError,
    EOFError,
    pickle.UnpicklingError,
)
_CHECKPOINT_TRAIN_EXCEPTIONS = (RuntimeError, ValueError, TypeError, OSError, AttributeError)
_INFERENCE_FALLBACK_EXCEPTIONS = (RuntimeError, ValueError, TypeError, OSError, AttributeError)

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
            if src.ndim != 3:
                raise ValueError(
                    f"Input tensor must have shape [batch, seq, 1]; got rank {src.ndim}."
                )
            if src.shape[1] < 1:
                raise ValueError("Input sequence length must be >= 1.")
            if src.shape[2] != 1:
                raise ValueError(f"Input feature dimension must be 1; got {src.shape[2]}.")
            if src.shape[1] > self.seq_len:
                raise ValueError(
                    f"Input sequence length {src.shape[1]} exceeds configured seq_len {self.seq_len}."
                )
            x = self.embedding(src) + self.pos_encoder[:, : src.shape[1], :]
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

    model_path = (
        Path(model_path) if model_path is not None else default_model_path(DEFAULT_MODEL_FILENAME)
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("--- SCPN SAFETY AI: Disruption Prediction (Transformer) ---")
    logger.info("Sequence length: %d | Shots: %d | Epochs: %d", seq_len, n_shots, epochs)

    logger.info("Generating synthetic shots (Rutherford Physics)...")
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

    logger.info("Training Transformer...")
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info("Epoch %d: Loss=%.4f", epoch, loss.item())

    logger.info("Validating on a new shot...")
    test_sig, test_lbl, _ = simulate_tearing_mode(steps=sim_steps, rng=eval_rng)
    input_sig = _prepare_signal_window(test_sig, seq_len)
    input_tensor = torch.tensor(input_sig, dtype=torch.float32).reshape(1, -1, 1)

    model.eval()
    with torch.no_grad():
        risk = model(input_tensor).item()

    logger.info("Test Shot Ground Truth: %s", "DISRUPTIVE" if test_lbl else "SAFE")
    logger.info("AI Prediction Risk: %.1f%%", risk * 100)

    torch.save({"state_dict": model.state_dict(), "seq_len": int(seq_len)}, model_path)
    logger.info("Saved model: %s", model_path)

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
        logger.info("Saved: %s", plot_path)

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
    recovery_allowed = _resolve_allow_fallback(bool(allow_fallback))
    if torch is None:
        if not recovery_allowed:
            raise RuntimeError("Torch is required for load_or_train_predictor().")
        _record_recovery_event(
            "torch_unavailable_fallback",
            context={"model_path": str(model_path) if model_path is not None else None},
        )
        return None, _augment_with_fallback_telemetry(
            {
                "trained": False,
                "fallback": True,
                "reason": "torch_unavailable",
                "model_path": (
                    str(model_path)
                    if model_path is not None
                    else str(default_model_path(DEFAULT_MODEL_FILENAME))
                ),
                "seq_len": int(_normalize_seq_len(seq_len)),
            }
        )

    path = (
        Path(model_path) if model_path is not None else default_model_path(DEFAULT_MODEL_FILENAME)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    seq_len = _normalize_seq_len(seq_len)
    kwargs = dict(train_kwargs or {})

    if path.exists() and not force_retrain:
        try:
            checkpoint = _safe_torch_checkpoint_load(path)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = _validated_checkpoint_state_dict(checkpoint["state_dict"])
                loaded_seq_len = _normalize_seq_len(checkpoint.get("seq_len", seq_len))
            else:
                state_dict = _validated_checkpoint_state_dict(checkpoint)
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
        except _CHECKPOINT_LOAD_EXCEPTIONS as exc:
            if not recovery_allowed:
                raise
            _record_recovery_event(
                "checkpoint_load_failed_fallback",
                context={"error": exc.__class__.__name__, "model_path": str(path)},
            )
            return None, _augment_with_fallback_telemetry(
                {
                    "trained": False,
                    "fallback": True,
                    "reason": f"checkpoint_load_failed:{exc.__class__.__name__}",
                    "model_path": str(path),
                    "seq_len": int(seq_len),
                }
            )

    if not train_if_missing and not force_retrain:
        if recovery_allowed:
            _record_recovery_event(
                "checkpoint_missing_fallback",
                context={"model_path": str(path)},
            )
            return None, _augment_with_fallback_telemetry(
                {
                    "trained": False,
                    "fallback": True,
                    "reason": "checkpoint_missing",
                    "model_path": str(path),
                    "seq_len": int(seq_len),
                }
            )
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    kwargs.setdefault("seq_len", seq_len)
    kwargs.setdefault("model_path", path)
    try:
        model, info = train_predictor(**kwargs)
    except _CHECKPOINT_TRAIN_EXCEPTIONS as exc:
        if not recovery_allowed:
            raise
        _record_recovery_event(
            "train_failed_fallback",
            context={"error": exc.__class__.__name__, "model_path": str(path)},
        )
        return None, _augment_with_fallback_telemetry(
            {
                "trained": False,
                "fallback": True,
                "reason": f"train_failed:{exc.__class__.__name__}",
                "model_path": str(path),
                "seq_len": int(seq_len),
            }
        )
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
    allow_fallback: bool = True,
) -> tuple[float, dict[str, Any]]:
    """
    Predict disruption risk with checkpoint path if available, else deterministic
    compatibility estimator.

    Returns
    -------
    risk, metadata
        ``risk`` is always a bounded float in ``[0, 1]``.
        ``metadata`` includes whether compatibility mode was used.
    allow_fallback
        If ``False``, this API raises on missing/broken checkpoints or inference
        failures instead of returning compatibility risk from
        ``predict_disruption_risk``.
    """
    recovery_allowed = _resolve_allow_fallback(bool(allow_fallback))
    base_risk = float(np.clip(predict_disruption_risk(signal, toroidal_observables), 0.0, 1.0))

    model, meta = load_or_train_predictor(
        model_path=model_path,
        seq_len=seq_len,
        force_retrain=False,
        train_kwargs={"seq_len": _normalize_seq_len(seq_len), "save_plot": False},
        train_if_missing=bool(train_if_missing),
        allow_fallback=bool(recovery_allowed),
    )

    if model is None or torch is None:
        if not recovery_allowed:
            reason = meta.get("reason", "model_unavailable")
            raise RuntimeError(
                "predict_disruption_risk_safe fallback disabled: "
                f"disruption model unavailable ({reason})."
            )
        _record_recovery_event(
            "inference_model_unavailable_fallback",
            context={"reason": str(meta.get("reason", "model_unavailable"))},
        )
        out_meta = _augment_with_fallback_telemetry(dict(meta))
        out_meta["mode"] = "fallback"
        out_meta["risk_source"] = "predict_disruption_risk"
        return base_risk, out_meta

    try:
        model.train()  # Enable Dropout for Monte Carlo sampling
        model_seq_len = int(meta.get("seq_len", _normalize_seq_len(seq_len)))
        input_sig = _prepare_signal_window(signal, model_seq_len)
        input_tensor = torch.tensor(input_sig, dtype=torch.float32).reshape(1, -1, 1)

        # MC-Dropout: Run 10 forward passes
        n_mc = 10
        mc_risks = []
        with torch.no_grad():
            for _ in range(n_mc):
                mc_risks.append(float(model(input_tensor).item()))

        risk_mean = float(np.mean(mc_risks))
        risk_std = float(np.std(mc_risks))
        confidence = float(np.clip(1.0 - 2.0 * risk_std, 0.0, 1.0))

        out_meta = dict(meta)
        out_meta["mode"] = "checkpoint"
        out_meta["risk_source"] = "transformer_mc_dropout"
        out_meta["uncertainty_std"] = risk_std
        out_meta["confidence_score"] = confidence

        return float(np.clip(risk_mean, 0.0, 1.0)), out_meta
    except _INFERENCE_FALLBACK_EXCEPTIONS as exc:
        if not recovery_allowed:
            raise RuntimeError(
                "predict_disruption_risk_safe fallback disabled: "
                f"transformer inference failed ({exc.__class__.__name__})."
            ) from exc
        _record_recovery_event(
            "inference_failed_fallback",
            context={"error": exc.__class__.__name__},
        )
        out_meta = _augment_with_fallback_telemetry(dict(meta))
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
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
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
            result[f"recall_at_{T_ms}ms"] = float(recall_at_t)

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
