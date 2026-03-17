# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Federated Learning for Multi-Machine Disruption Prediction
"""Federated learning framework for cross-machine disruption prediction.

Trains a shared MLP disruption classifier across heterogeneous tokamak
datasets (DIII-D, JET, KSTAR, EAST, SPARC) without centralising raw data.
Supports FedAvg (McMahan et al., AISTATS 2017) and FedProx (Li et al.,
MLSys 2020) aggregation strategies.

Disruption features: Ip, beta_N, q95, n/n_GW, li, dBp/dt,
locked_mode_amplitude, n1_rms — 8-dimensional input space whose
distributions differ across machines (JET: higher Ip; DIII-D: more
shaping; KSTAR: longer pulses).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

N_FEATURES = 8  # Ip, beta_N, q95, n/n_GW, li, dBp/dt, locked_mode_amp, n1_rms
FEATURE_NAMES = ("Ip", "beta_N", "q95", "n_nGW", "li", "dBp_dt", "locked_mode_amp", "n1_rms")

# Machine-specific feature distribution parameters (mean, std) per feature.
# Derived from ITPA global confinement database ranges.
# Greenwald, NF 42 (2002); de Vries, NF 51 (2011).
MACHINE_PROFILES: dict[str, dict[str, tuple[float, float]]] = {
    "DIII-D": {
        "Ip": (1.2, 0.3),
        "beta_N": (2.2, 0.6),
        "q95": (4.5, 1.0),
        "n_nGW": (0.55, 0.15),
        "li": (0.90, 0.10),
        "dBp_dt": (0.8, 0.5),
        "locked_mode_amp": (0.3, 0.25),
        "n1_rms": (0.15, 0.12),
    },
    "JET": {
        "Ip": (2.8, 0.6),
        "beta_N": (1.8, 0.4),
        "q95": (3.8, 0.8),
        "n_nGW": (0.65, 0.12),
        "li": (0.85, 0.08),
        "dBp_dt": (1.2, 0.7),
        "locked_mode_amp": (0.5, 0.35),
        "n1_rms": (0.20, 0.15),
    },
    "KSTAR": {
        "Ip": (0.6, 0.15),
        "beta_N": (2.5, 0.7),
        "q95": (5.0, 1.2),
        "n_nGW": (0.45, 0.12),
        "li": (0.95, 0.12),
        "dBp_dt": (0.5, 0.3),
        "locked_mode_amp": (0.2, 0.15),
        "n1_rms": (0.10, 0.08),
    },
    "EAST": {
        "Ip": (0.5, 0.12),
        "beta_N": (2.0, 0.5),
        "q95": (5.5, 1.5),
        "n_nGW": (0.50, 0.18),
        "li": (0.88, 0.09),
        "dBp_dt": (0.4, 0.25),
        "locked_mode_amp": (0.18, 0.12),
        "n1_rms": (0.08, 0.06),
    },
    "SPARC": {
        "Ip": (8.7, 1.0),
        "beta_N": (1.5, 0.3),
        "q95": (3.5, 0.6),
        "n_nGW": (0.75, 0.10),
        "li": (0.80, 0.06),
        "dBp_dt": (2.0, 1.0),
        "locked_mode_amp": (0.6, 0.40),
        "n1_rms": (0.25, 0.18),
    },
}


# ── MLP (numpy-only, same pattern as neural_transport.py) ────────────


def _relu(x: np.ndarray) -> np.ndarray:
    return np.asarray(np.maximum(0.0, x), dtype=x.dtype)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0))), dtype=x.dtype)


def _binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    p = np.clip(y_pred, 1e-7, 1.0 - 1e-7)
    return -float(np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def _init_mlp_weights(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Xavier initialisation for 8→32→16→1 MLP."""
    return {
        "w1": rng.normal(0, np.sqrt(2.0 / (N_FEATURES + 32)), (N_FEATURES, 32)),
        "b1": np.zeros(32),
        "w2": rng.normal(0, np.sqrt(2.0 / (32 + 16)), (32, 16)),
        "b2": np.zeros(16),
        "w3": rng.normal(0, np.sqrt(2.0 / (16 + 1)), (16, 1)),
        "b3": np.zeros(1),
    }


def _mlp_forward(x: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    """Forward pass: 8→32→16→1 with ReLU hidden, sigmoid output."""
    h1 = _relu(x @ weights["w1"] + weights["b1"])
    h2 = _relu(h1 @ weights["w2"] + weights["b2"])
    return _sigmoid(h2 @ weights["w3"] + weights["b3"]).ravel()


def _mlp_gradients(
    x: np.ndarray, y: np.ndarray, weights: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], float]:
    """Backprop for BCE loss. Returns (grads_dict, loss)."""
    n = x.shape[0]
    h1_pre = x @ weights["w1"] + weights["b1"]
    h1 = np.maximum(0.0, h1_pre)
    h2_pre = h1 @ weights["w2"] + weights["b2"]
    h2 = np.maximum(0.0, h2_pre)
    logits = (h2 @ weights["w3"] + weights["b3"]).ravel()
    y_pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))

    loss = _binary_cross_entropy(y_pred, y)

    # dL/d_logits for BCE with sigmoid output
    dl = (y_pred - y) / n  # (n,)

    grads: dict[str, np.ndarray] = {}
    grads["b3"] = dl.sum(axis=0, keepdims=True).ravel()
    grads["w3"] = h2.T @ dl.reshape(-1, 1)

    dh2 = dl.reshape(-1, 1) @ weights["w3"].T
    dh2 = dh2 * (h2_pre > 0).astype(float)
    grads["b2"] = dh2.sum(axis=0)
    grads["w2"] = h1.T @ dh2

    dh1 = dh2 @ weights["w2"].T
    dh1 = dh1 * (h1_pre > 0).astype(float)
    grads["b1"] = dh1.sum(axis=0)
    grads["w1"] = x.T @ dh1

    return grads, loss


# ── Differential privacy ─────────────────────────────────────────────


def differential_privacy_clip(
    gradients: dict[str, np.ndarray],
    max_norm: float,
    noise_sigma: float,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Clip per-parameter gradient norms and add Gaussian noise (DP-SGD).

    Reference: Abadi et al., "Deep Learning with Differential Privacy",
    CCS 2016.
    """
    rng = rng or np.random.default_rng()
    total_norm = np.sqrt(sum(float(np.sum(g**2)) for g in gradients.values()))
    clip_factor = min(1.0, max_norm / max(total_norm, 1e-12))
    clipped: dict[str, np.ndarray] = {}
    for key, grad in gradients.items():
        clipped[key] = grad * clip_factor + rng.normal(0.0, noise_sigma, grad.shape)
    return clipped


# ── Data generation ──────────────────────────────────────────────────


def _generate_disruption_data(
    machine: str,
    n_samples: int,
    disruption_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic disruption dataset for a tokamak.

    Safe shots: features sampled from machine profile.
    Disruptive shots: elevated locked_mode_amp, dBp/dt, lower q95, higher n/n_GW.
    """
    if machine not in MACHINE_PROFILES:
        raise ValueError(f"Unknown machine {machine!r}; available: {sorted(MACHINE_PROFILES)}")
    profile = MACHINE_PROFILES[machine]
    X = np.empty((n_samples, N_FEATURES))
    y = np.zeros(n_samples)

    n_disrupt = int(n_samples * disruption_fraction)
    y[:n_disrupt] = 1.0

    for i, feat in enumerate(FEATURE_NAMES):
        mu, sigma = profile[feat]
        X[:, i] = rng.normal(mu, sigma, n_samples)

    # Shift disruptive shots toward instability boundaries
    X[:n_disrupt, 2] -= 0.8  # q95 drops (de Vries et al., NF 51 2011)
    X[:n_disrupt, 3] += 0.25  # n/n_GW rises toward Greenwald limit
    X[:n_disrupt, 5] *= 2.5  # dBp/dt spike
    X[:n_disrupt, 6] += 0.6  # locked mode growth
    X[:n_disrupt, 7] += 0.3  # n=1 RMS rise

    # Clamp non-negative features
    X[:, 0] = np.maximum(X[:, 0], 0.01)  # Ip > 0
    X[:, 3] = np.clip(X[:, 3], 0.01, 1.5)
    X[:, 4] = np.maximum(X[:, 4], 0.3)
    X[:, 6] = np.maximum(X[:, 6], 0.0)
    X[:, 7] = np.maximum(X[:, 7], 0.0)

    # Shuffle
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


# ── Config ───────────────────────────────────────────────────────────


@dataclass
class FederatedConfig:
    """Configuration for federated disruption prediction training."""

    n_rounds: int = 10
    local_epochs: int = 5
    learning_rate: float = 0.01
    aggregation: str = "fedavg"  # "fedavg" or "fedprox"
    mu_proximal: float = 0.01  # FedProx proximal term weight; Li et al. MLSys 2020
    min_clients: int = 2
    machines: list[str] = field(default_factory=lambda: ["DIII-D", "JET", "KSTAR"])

    def __post_init__(self) -> None:
        if self.n_rounds < 1:
            raise ValueError("n_rounds must be >= 1")
        if self.local_epochs < 1:
            raise ValueError("local_epochs must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.aggregation not in ("fedavg", "fedprox"):
            raise ValueError(f"aggregation must be 'fedavg' or 'fedprox', got {self.aggregation!r}")
        if self.mu_proximal < 0:
            raise ValueError("mu_proximal must be >= 0")
        if self.min_clients < 1:
            raise ValueError("min_clients must be >= 1")
        for m in self.machines:
            if m not in MACHINE_PROFILES:
                raise ValueError(f"Unknown machine {m!r}; available: {sorted(MACHINE_PROFILES)}")


# ── Client ───────────────────────────────────────────────────────────


class MachineClient:
    """Local training client for a single tokamak."""

    def __init__(
        self,
        machine: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        learning_rate: float = 0.01,
    ) -> None:
        if machine not in MACHINE_PROFILES:
            raise ValueError(f"Unknown machine {machine!r}")
        self.machine = machine
        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.y_train = np.asarray(y_train, dtype=np.float64).ravel()
        self.X_test = np.asarray(X_test, dtype=np.float64)
        self.y_test = np.asarray(y_test, dtype=np.float64).ravel()
        self.learning_rate = learning_rate
        self._weights: dict[str, np.ndarray] = {}

    def get_data_size(self) -> int:
        return int(self.X_train.shape[0])

    def local_train(
        self,
        global_weights: dict[str, np.ndarray],
        n_epochs: int,
        mu_proximal: float = 0.0,
    ) -> dict[str, np.ndarray]:
        """SGD on local data, starting from global_weights.

        When mu_proximal > 0, adds the FedProx penalty
        (mu/2)||w - w_global||^2 to the loss gradient.
        """
        w = {k: v.copy() for k, v in global_weights.items()}

        for _ in range(n_epochs):
            grads, _ = _mlp_gradients(self.X_train, self.y_train, w)
            for key in w:
                g = grads[key]
                if mu_proximal > 0:
                    g = g + mu_proximal * (w[key] - global_weights[key])
                w[key] = w[key] - self.learning_rate * g

        self._weights = w
        return {k: v.copy() for k, v in w.items()}

    def local_evaluate(self, weights: dict[str, np.ndarray]) -> dict[str, float]:
        """Binary classification metrics on local test set."""
        y_pred_prob = _mlp_forward(self.X_test, weights)
        y_pred = (y_pred_prob >= 0.5).astype(float)
        y = self.y_test

        tp = float(np.sum((y_pred == 1) & (y == 1)))
        fp = float(np.sum((y_pred == 1) & (y == 0)))
        fn = float(np.sum((y_pred == 0) & (y == 1)))
        tn = float(np.sum((y_pred == 0) & (y == 0)))
        n = max(len(y), 1)

        accuracy = (tp + tn) / n
        precision = tp / max(tp + fp, 1e-12)
        recall = tp / max(tp + fn, 1e-12)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        loss = _binary_cross_entropy(y_pred_prob, y)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "loss": loss,
            "n_samples": int(n),
        }


# ── Server ───────────────────────────────────────────────────────────


class FederatedServer:
    """Orchestrates federated training across machine clients."""

    def __init__(self, config: FederatedConfig, seed: int = 42) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.global_weights = _init_mlp_weights(self.rng)

    def aggregate(self, client_updates: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        """FedAvg: weighted average of model weights by dataset size.

        McMahan et al., "Communication-Efficient Learning of Deep Networks
        from Decentralized Data", AISTATS 2017.
        """
        if not client_updates:
            raise ValueError("aggregate requires at least one client update")
        total = sum(u["n_samples"] for u in client_updates)
        avg: dict[str, np.ndarray] = {}
        for key in client_updates[0]["weights"]:
            avg[key] = sum(u["weights"][key] * (u["n_samples"] / total) for u in client_updates)
        return avg

    def fedprox_aggregate(
        self,
        client_updates: list[dict[str, Any]],
        global_weights: dict[str, np.ndarray],
        mu: float,
    ) -> dict[str, np.ndarray]:
        """FedProx aggregation with proximal regularisation.

        The proximal term is applied during local training (not aggregation),
        so aggregation itself is weighted averaging — the difference from
        FedAvg is in the client-side gradient update. This method exists
        for API symmetry; the mu parameter documents the proximal weight used.
        """
        _ = mu  # applied during local_train, not aggregation
        return self.aggregate(client_updates)

    def run_round(self, clients: list[MachineClient]) -> dict[str, Any]:
        """Single federated round: distribute → local train → aggregate."""
        if len(clients) < self.config.min_clients:
            raise ValueError(f"Need >= {self.config.min_clients} clients, got {len(clients)}")

        mu = self.config.mu_proximal if self.config.aggregation == "fedprox" else 0.0
        updates: list[dict[str, Any]] = []
        client_metrics: list[dict[str, Any]] = []

        for client in clients:
            local_w = client.local_train(self.global_weights, self.config.local_epochs, mu)
            metrics = client.local_evaluate(local_w)
            updates.append({"weights": local_w, "n_samples": client.get_data_size()})
            client_metrics.append({"machine": client.machine, **metrics})

        if self.config.aggregation == "fedprox":
            self.global_weights = self.fedprox_aggregate(updates, self.global_weights, mu)
        else:
            self.global_weights = self.aggregate(updates)

        return {"client_metrics": client_metrics}

    def train(
        self, clients: list[MachineClient], n_rounds: int | None = None
    ) -> list[dict[str, Any]]:
        """Full federated training loop.

        Returns per-round metrics including per-client accuracy, loss, n_samples.
        """
        rounds = n_rounds if n_rounds is not None else self.config.n_rounds
        history: list[dict[str, Any]] = []

        for r in range(rounds):
            round_result = self.run_round(clients)
            mean_loss = float(np.mean([m["loss"] for m in round_result["client_metrics"]]))
            mean_acc = float(np.mean([m["accuracy"] for m in round_result["client_metrics"]]))
            round_result["round"] = r
            round_result["mean_loss"] = mean_loss
            round_result["mean_accuracy"] = mean_acc
            history.append(round_result)
            logger.info("round %d  mean_loss=%.4f  mean_acc=%.3f", r, mean_loss, mean_acc)

        return history

    def get_state(self) -> dict[str, Any]:
        """Serialisable snapshot of server state."""
        return {
            "config": {
                "n_rounds": self.config.n_rounds,
                "local_epochs": self.config.local_epochs,
                "learning_rate": self.config.learning_rate,
                "aggregation": self.config.aggregation,
                "mu_proximal": self.config.mu_proximal,
                "min_clients": self.config.min_clients,
                "machines": list(self.config.machines),
            },
            "weights": {k: v.tolist() for k, v in self.global_weights.items()},
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> FederatedServer:
        """Reconstruct server from serialised state."""
        cfg = FederatedConfig(**state["config"])
        server = cls(cfg)
        server.global_weights = {k: np.asarray(v) for k, v in state["weights"].items()}
        return server


# ── Factory ──────────────────────────────────────────────────────────


def create_machine_clients(
    machine_configs: list[dict[str, Any]],
    seed: int = 42,
) -> list[MachineClient]:
    """Create MachineClient instances with synthetic disruption data.

    Parameters
    ----------
    machine_configs : list of dicts
        Each dict must have "machine" (str) and optionally
        "n_train" (int, default 200), "n_test" (int, default 50),
        "disruption_fraction" (float, default 0.4),
        "learning_rate" (float, default 0.01).
    seed : int
        Base RNG seed; each machine gets seed + index.
    """
    clients: list[MachineClient] = []
    for i, cfg in enumerate(machine_configs):
        machine = cfg["machine"]
        n_train = cfg.get("n_train", 200)
        n_test = cfg.get("n_test", 50)
        frac = cfg.get("disruption_fraction", 0.4)
        lr = cfg.get("learning_rate", 0.01)
        rng = np.random.default_rng(seed + i)

        X_train, y_train = _generate_disruption_data(machine, n_train, frac, rng)
        X_test, y_test = _generate_disruption_data(machine, n_test, frac, rng)
        clients.append(MachineClient(machine, X_train, y_train, X_test, y_test, lr))

    return clients
