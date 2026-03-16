# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Neural Turbulence Surrogate (QLKNN-class)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class QLKNNSurrogate:
    """
    Pure NumPy inference for a QLKNN-like neural network.
    Predicts turbulent fluxes [Q_i, Q_e, Gamma_e] from 10 parameters.
    van de Plassche et al., Phys. Plasmas 27, 022310 (2020).

    Default construction auto-trains on a Jenko et al. (2001) critical gradient
    model so predictions are physically meaningful out of the box.
    """

    def __init__(
        self,
        hidden_layers: list[int] | None = None,
        activation: str = "elu",
        pretrained: bool = True,
    ):
        if hidden_layers is None:
            hidden_layers = [128, 128, 64]

        self.hidden_layers = hidden_layers
        self.activation = activation

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        rng = np.random.RandomState(42) if pretrained else np.random.RandomState()

        layers = [10] + hidden_layers + [3]
        for i in range(len(layers) - 1):
            n_in = layers[i]
            n_out = layers[i + 1]
            w = rng.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            b = np.zeros(n_out)
            self.weights.append(w)
            self.biases.append(b)

        if pretrained:
            self._pretrain(rng)

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "elu":
            return np.asarray(np.where(x > 0, x, np.exp(x) - 1.0))
        if self.activation == "relu":
            return np.asarray(np.maximum(0, x))
        if self.activation == "tanh":
            return np.asarray(np.tanh(x))
        return x

    def _activate_deriv(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "elu":
            return np.where(x > 0, 1.0, np.exp(x))
        if self.activation == "relu":
            return np.where(x > 0, 1.0, 0.0)
        if self.activation == "tanh":
            return np.asarray(1.0 - np.tanh(x) ** 2)
        return np.ones_like(x)

    def _pretrain(self, rng: np.random.RandomState) -> None:
        """Train on Jenko et al. (2001) analytic critical gradient model."""
        X = TrainingDataGenerator.generate_parameter_scan(500, rng=rng)
        y = TrainingDataGenerator.generate_analytic_targets(X)

        n_val = 50
        X_train, y_train = X[:-n_val], y[:-n_val]

        for _ in range(100):
            activations = [X_train]
            pre_acts: list[np.ndarray] = []
            out = X_train
            for i in range(len(self.weights) - 1):
                z = out @ self.weights[i] + self.biases[i]
                pre_acts.append(z)
                out = self._activate(z)
                activations.append(out)
            z_last = out @ self.weights[-1] + self.biases[-1]
            pred = z_last

            n_train = X_train.shape[0]
            delta = 2.0 * (pred - y_train) / (n_train * y_train.shape[1])

            for i in range(len(self.weights) - 1, -1, -1):
                dW = activations[i].T @ delta
                db = np.sum(delta, axis=0)
                dW_norm = float(np.linalg.norm(dW))
                if dW_norm > 1.0:
                    dW = dW / dW_norm
                    db = db / max(float(np.linalg.norm(db)), 1e-8)
                self.weights[i] -= 1e-3 * dW
                self.biases[i] -= 1e-3 * db
                if i > 0:
                    delta = (delta @ self.weights[i].T) * self._activate_deriv(pre_acts[i - 1])
                    np.clip(delta, -1e6, 1e6, out=delta)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x shape: (batch_size, 10)
        returns shape: (batch_size, 3)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        out = x
        for i in range(len(self.weights) - 1):
            out = out @ self.weights[i] + self.biases[i]
            out = self._activate(out)

        out = out @ self.weights[-1] + self.biases[-1]
        return np.asarray(out)

    def load_weights(self, path: str) -> None:
        data = np.load(path, allow_pickle=False)
        self.weights = [data[f"w{i}"] for i in range(len(self.weights))]
        self.biases = [data[f"b{i}"] for i in range(len(self.biases))]

    def save_weights(self, path: str) -> None:
        arrays: dict[str, Any] = {}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"w{i}"] = w
            arrays[f"b{i}"] = b
        np.savez(path, **arrays)


class TransportInputNormalizer:
    @staticmethod
    def from_profiles(
        Te: np.ndarray,
        Ti: np.ndarray,
        ne: np.ndarray,
        q: np.ndarray,
        R0: float,
        a: float,
        B0: float,
        r: np.ndarray,
    ) -> np.ndarray:
        """
        Convert physical profiles into the 10 dimensionless QLKNN inputs.
        """
        dr = r[1] - r[0] if len(r) > 1 else 0.1

        # Gradients (simple central difference)
        grad_Te = np.gradient(Te, dr)
        grad_Ti = np.gradient(Ti, dr)
        grad_ne = np.gradient(ne, dr)
        grad_q = np.gradient(q, dr)

        # 1. R/L_Ti
        R_L_Ti = -R0 / np.maximum(Ti, 1e-3) * grad_Ti
        # 2. R/L_Te
        R_L_Te = -R0 / np.maximum(Te, 1e-3) * grad_Te
        # 3. R/L_ne
        R_L_ne = -R0 / np.maximum(ne, 1e-3) * grad_ne
        # 4. q
        q_norm = q
        # 5. s_hat (shear)
        s_hat = r / np.maximum(q, 1e-3) * grad_q
        # 6. alpha_MHD (pressure gradient)
        # p = 2 * n_e * T_e (roughly)
        p = 2.0 * ne * 1e19 * Te * 1e3 * 1.602e-19
        grad_p = np.gradient(p, dr)
        mu_0 = 4.0 * np.pi * 1e-7
        alpha_MHD = -(q**2) * R0 * grad_p * 2.0 * mu_0 / (B0**2)
        # 7. Ti/Te
        Ti_Te = Ti / np.maximum(Te, 1e-3)
        # 8. nu_star (collisionality)
        epsilon = r / R0
        # highly simplified nu_star
        Te_safe_nu = np.maximum(Te, 1e-3)
        nu_star = 0.1 * ne / (Te_safe_nu**2 * np.maximum(epsilon, 1e-3) ** 1.5)
        # 9. Z_eff (assumed flat 1.5)
        Z_eff = np.ones_like(r) * 1.5
        # 10. epsilon
        eps = epsilon

        inputs = np.vstack(
            [R_L_Ti, R_L_Te, R_L_ne, q_norm, s_hat, alpha_MHD, Ti_Te, nu_star, Z_eff, eps]
        ).T

        return inputs


class TrainingDataGenerator:
    @staticmethod
    def generate_parameter_scan(
        n_samples: int, rng: np.random.RandomState | None = None
    ) -> np.ndarray:
        """Uniform random sampling in 10D QLKNN parameter space."""
        if rng is None:
            rng = np.random.RandomState()
        bounds = np.array(
            [
                [0.0, 15.0],
                [0.0, 15.0],
                [-5.0, 10.0],
                [0.5, 5.0],
                [-1.0, 3.0],
                [0.0, 2.0],
                [0.1, 2.0],
                [1e-3, 1.0],
                [1.0, 3.0],
                [0.01, 0.3],
            ]
        )

        X = np.zeros((n_samples, 10))
        for i in range(10):
            X[:, i] = rng.uniform(bounds[i, 0], bounds[i, 1], n_samples)

        return X

    @staticmethod
    def generate_analytic_targets(inputs: np.ndarray) -> np.ndarray:
        """
        Compute flux targets from simplified analytical quasilinear model.
        Returns [Q_i, Q_e, Gamma_e] in gyro-Bohm units.
        """
        n_samples = inputs.shape[0]
        y = np.zeros((n_samples, 3))

        for i in range(n_samples):
            R_L_Ti = inputs[i, 0]
            R_L_Te = inputs[i, 1]
            R_L_ne = inputs[i, 2]
            q = inputs[i, 3]
            s_hat = inputs[i, 4]
            eps = inputs[i, 9]
            nu_star = inputs[i, 7]

            # Jenko et al. critical gradient formula
            R_L_Ti_crit = (
                (1.0 + inputs[i, 6]) * max(1.33 + 1.91 * s_hat / q, 0.0) * (1.0 - 1.5 * eps)
            )
            R_L_Ti_crit = max(R_L_Ti_crit, 0.0)

            # ITG Flux
            Q_i = 0.0
            if R_L_Ti > R_L_Ti_crit:
                # Q_i ~ (R/L_Ti - R/L_Ti_crit)^1.5
                Q_i = 5.0 * (R_L_Ti - R_L_Ti_crit) ** 1.5

            # TEM Flux
            Q_e = 0.0
            Gamma_e = 0.0
            R_L_ne_crit = 2.0
            if R_L_ne > R_L_ne_crit:
                # TEM driven flux, collisionality dampens it
                drive = R_L_ne - R_L_ne_crit
                Q_e = 2.0 * drive * np.sqrt(max(nu_star, 1e-4))
                Gamma_e = 1.0 * drive * np.sqrt(max(nu_star, 1e-4))

            y[i, 0] = Q_i
            y[i, 1] = Q_e
            y[i, 2] = Gamma_e

        return y


class NeuralTransportTrainer:
    def _activate_deriv(self, x: np.ndarray, activation: str) -> np.ndarray:
        if activation == "elu":
            return np.where(x > 0, 1.0, np.exp(x))
        if activation == "relu":
            return np.where(x > 0, 1.0, 0.0)
        if activation == "tanh":
            return np.asarray(1.0 - np.tanh(x) ** 2)
        return np.ones_like(x)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        lr: float = 1e-3,
        val_frac: float = 0.2,
    ) -> dict:
        n_samples = X.shape[0]
        n_val = max(int(n_samples * val_frac), 1)

        X_train, y_train = X[:-n_val], y[:-n_val]
        X_val, y_val = X[-n_val:], y[-n_val:]

        model = QLKNNSurrogate(pretrained=False)

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Forward pass — store pre-activations for backprop
            activations = [X_train]
            pre_acts = []
            out = X_train
            for i in range(len(model.weights) - 1):
                z = out @ model.weights[i] + model.biases[i]
                pre_acts.append(z)
                out = model._activate(z)
                activations.append(out)
            z_last = out @ model.weights[-1] + model.biases[-1]
            pre_acts.append(z_last)
            pred = z_last
            activations.append(pred)

            loss_train = float(np.mean((pred - y_train) ** 2))

            # Backprop
            n_train = X_train.shape[0]
            delta = 2.0 * (pred - y_train) / (n_train * y_train.shape[1])

            for i in range(len(model.weights) - 1, -1, -1):
                dW = activations[i].T @ delta
                db = np.sum(delta, axis=0)
                # Gradient clipping to prevent explosion
                dW_norm = np.linalg.norm(dW)
                if dW_norm > 1.0:
                    dW = dW / dW_norm
                    db = db / max(float(np.linalg.norm(db)), 1e-8)
                model.weights[i] -= lr * dW
                model.biases[i] -= lr * db
                if i > 0:
                    delta = (delta @ model.weights[i].T) * self._activate_deriv(
                        pre_acts[i - 1], model.activation
                    )
                    np.clip(delta, -1e6, 1e6, out=delta)

            pred_val = model.forward(X_val)
            loss_val = float(np.mean((pred_val - y_val) ** 2))

            history["train_loss"].append(loss_train)
            history["val_loss"].append(loss_val)

        self._model = model
        return history


@dataclass
class TransportFluxes:
    Q_i_W_m2: np.ndarray
    Q_e_W_m2: np.ndarray
    Gamma_e_inv_m2_s: np.ndarray


class QLKNNTransportModel:
    def __init__(self, surrogate: QLKNNSurrogate):
        self.surrogate = surrogate
        self.normalizer = TransportInputNormalizer()

    def compute_fluxes(
        self,
        Te: np.ndarray,
        Ti: np.ndarray,
        ne: np.ndarray,
        q: np.ndarray,
        R0: float,
        a: float,
        B0: float,
        r: np.ndarray,
    ) -> TransportFluxes:
        inputs = self.normalizer.from_profiles(Te, Ti, ne, q, R0, a, B0, r)

        # Predict gyro-Bohm fluxes
        gB_fluxes = self.surrogate.forward(inputs)

        Q_i_gB = gB_fluxes[:, 0]
        Q_e_gB = gB_fluxes[:, 1]
        Gamma_e_gB = gB_fluxes[:, 2]

        # De-normalize
        e_charge = 1.602e-19
        m_i = 2.0 * 1.67e-27

        Te_safe = np.maximum(Te, 1e-3)
        Te_J = Te_safe * 1e3 * e_charge
        c_s = np.sqrt(Te_J / m_i)
        rho_s = np.maximum(m_i * c_s / (e_charge * B0), 1e-6)

        ne_safe = np.maximum(ne, 1e-3)
        ne_m3 = ne_safe * 1e19
        Q_gB_phys = ne_m3 * Te_J * c_s * (rho_s / a) ** 2
        Gamma_gB_phys = ne_m3 * c_s * (rho_s / a) ** 2

        Q_i_phys = Q_i_gB * Q_gB_phys
        Q_e_phys = Q_e_gB * Q_gB_phys
        Gamma_e_phys = Gamma_e_gB * Gamma_gB_phys

        return TransportFluxes(Q_i_phys, Q_e_phys, Gamma_e_phys)
