# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — GS-Transport Surrogate Training
"""Dedicated training module for the GS-transport MLP surrogate.

This module was split out of ``fno_training.py`` to reduce monolithic risk
surface while keeping the same public API and behavior.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GS_TRANSPORT_WEIGHTS_PATH = REPO_ROOT / "weights" / "gs_transport_surrogate.npz"

# MOCK_CONFIG template matching FusionKernel / TransportSolver expectations.
_GS_TRANSPORT_MOCK_CONFIG_TEMPLATE: Dict[str, object] = {
    "reactor_name": "GS-Transport-Surrogate",
    "grid_resolution": [20, 20],
    "dimensions": {"R_min": 4.0, "R_max": 8.0, "Z_min": -4.0, "Z_max": 4.0},
    "physics": {"plasma_current_target": 15.0, "vacuum_permeability": 1.0},
    "coils": [
        {"name": "CS", "r": 1.7, "z": 0.0, "current": 0.15},
    ],
    "solver": {
        "max_iterations": 10,
        "convergence_threshold": 1e-4,
        "relaxation_factor": 0.1,
    },
}


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x**3))))


def _relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    denom = np.linalg.norm(target) + 1e-8
    return float(np.linalg.norm(pred - target) / denom)


class _AdamOptimizer:
    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray], lr: float) -> None:
        self.t += 1
        for key, param in params.items():
            grad = grads[key]
            if key not in self.m:
                self.m[key] = np.zeros_like(param)
                self.v[key] = np.zeros_like(param)
            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (grad * grad)

            m_hat = self.m[key] / (1.0 - self.beta1**self.t)
            v_hat = self.v[key] / (1.0 - self.beta2**self.t)
            param -= lr * m_hat / (np.sqrt(v_hat) + self.eps)


def _generate_gs_transport_pairs(
    n_samples: int = 5000,
    grid_size: int = 50,
    seed: int = 20260218,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    """Generate training data using TransportSolver as a physics oracle."""
    from scpn_fusion.core.integrated_transport_solver import TransportSolver

    rng = np.random.default_rng(seed)

    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    metadata: List[Dict[str, object]] = []

    for i in range(n_samples):
        if i > 0 and i % 100 == 0:
            logger.info(
                "GS-transport pair generation: %d / %d (collected %d)",
                i,
                n_samples,
                len(x_list),
            )

        Ip = float(rng.uniform(1.0, 15.0))  # MA
        BT = float(rng.uniform(2.0, 12.5))  # T
        kappa = float(rng.uniform(1.5, 2.1))  # elongation
        n_e20 = float(rng.uniform(0.3, 1.2))  # 10^20 m^-3
        P_aux = float(rng.uniform(10.0, 120.0))  # MW
        T0 = float(rng.uniform(3.0, 25.0))  # keV (peak temperature)

        try:
            config = dict(_GS_TRANSPORT_MOCK_CONFIG_TEMPLATE)
            config["physics"] = {
                "plasma_current_target": Ip,
                "vacuum_permeability": 1.0,
            }

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
                encoding="utf-8",
            ) as f:
                json.dump(config, f)
                tmp_path = f.name

            try:
                solver = TransportSolver(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            if solver.nr != grid_size:
                solver.nr = grid_size
                solver.rho = np.linspace(0, 1, grid_size)
                solver.drho = 1.0 / (grid_size - 1)
                solver.Te = T0 * (1 - solver.rho**2)
                solver.Ti = T0 * (1 - solver.rho**2)
                solver.ne = n_e20 * 10.0 * (1 - solver.rho**2) ** 0.5
                solver.chi_e = np.ones(grid_size)
                solver.chi_i = np.ones(grid_size)
                solver.D_n = np.ones(grid_size)
                solver.n_impurity = np.zeros(grid_size)
            else:
                solver.Ti = T0 * (1 - solver.rho**2)
                solver.Te = T0 * (1 - solver.rho**2)
                solver.ne = n_e20 * 10.0 * (1 - solver.rho**2) ** 0.5

            Ti_initial = solver.Ti.copy()
            for _ in range(5):
                solver.update_transport_model(P_aux)
                solver.evolve_profiles(dt=0.1, P_aux=P_aux)
            Ti_final = solver.Ti.copy()

            if not (np.all(np.isfinite(Ti_initial)) and np.all(np.isfinite(Ti_final))):
                continue

            x_list.append(Ti_initial)
            y_list.append(Ti_final)
            metadata.append(
                {
                    "Ip": Ip,
                    "BT": BT,
                    "kappa": kappa,
                    "n_e20": n_e20,
                    "P_aux": P_aux,
                    "T0": T0,
                }
            )

        except Exception as exc:
            logger.debug("Sample %d failed: %s", i, exc)
            continue

    if len(x_list) == 0:
        raise RuntimeError(
            f"GS-transport pair generation produced 0 valid samples out of {n_samples} attempts."
        )

    logger.info(
        "GS-transport pair generation complete: %d / %d valid samples",
        len(x_list),
        n_samples,
    )

    x = np.stack(x_list, axis=0)
    y = np.stack(y_list, axis=0)
    return x, y, metadata


class MLPSurrogate:
    """Simple MLP surrogate for 1D transport profile prediction."""

    def __init__(self, input_dim: int = 50, hidden_dim: int = 128, seed: int = 42) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(seed)

        def _xavier(fan_in: int, fan_out: int) -> np.ndarray:
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float64)

        self.W1 = _xavier(input_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.W2 = _xavier(hidden_dim, hidden_dim)
        self.b2 = np.zeros(hidden_dim, dtype=np.float64)
        self.W3 = _xavier(hidden_dim, input_dim)
        self.b3 = np.zeros(input_dim, dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = _gelu(x @ self.W1 + self.b1)
        h = _gelu(h @ self.W2 + self.b2)
        return h @ self.W3 + self.b3

    def _params_dict(self) -> Dict[str, np.ndarray]:
        return {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "W3": self.W3,
            "b3": self.b3,
        }

    def save_weights(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self._params_dict())

    def load_weights(self, path: str | Path) -> None:
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            self.W1 = np.array(data["W1"], dtype=np.float64)
            self.b1 = np.array(data["b1"], dtype=np.float64)
            self.W2 = np.array(data["W2"], dtype=np.float64)
            self.b2 = np.array(data["b2"], dtype=np.float64)
            self.W3 = np.array(data["W3"], dtype=np.float64)
            self.b3 = np.array(data["b3"], dtype=np.float64)
            self.input_dim = self.W1.shape[0]
            self.hidden_dim = self.W1.shape[1]


def train_gs_transport_surrogate(
    n_samples: int = 5000,
    epochs: int = 200,
    lr: float = 1e-3,
    save_path: str | Path | None = None,
    seed: int = 42,
    patience: int = 30,
) -> Dict[str, object]:
    """Train an MLP surrogate on GS-transport oracle data."""
    if save_path is None:
        save_path = DEFAULT_GS_TRANSPORT_WEIGHTS_PATH
    save_path = Path(save_path)

    logger.info("Generating %d GS-transport oracle samples...", n_samples)
    x, y, meta = _generate_gs_transport_pairs(
        n_samples=n_samples,
        grid_size=50,
        seed=seed,
    )
    n_valid = len(x)
    logger.info("Collected %d valid samples", n_valid)

    rng = np.random.default_rng(seed + 789)
    perm = rng.permutation(n_valid)
    n_train = max(1, int(0.8 * n_valid))
    n_val = max(1, int(0.1 * n_valid))

    x_train, y_train = x[perm[:n_train]], y[perm[:n_train]]
    x_val, y_val = x[perm[n_train : n_train + n_val]], y[perm[n_train : n_train + n_val]]
    x_test, y_test = x[perm[n_train + n_val :]], y[perm[n_train + n_val :]]

    grid_size = x.shape[1]
    model = MLPSurrogate(input_dim=grid_size, hidden_dim=128, seed=seed)
    optimizer = _AdamOptimizer()

    history: Dict[str, object] = {
        "train_loss": [],
        "val_loss": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "n_samples_generated": n_valid,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": len(x_test),
        "epochs_requested": epochs,
        "data_mode": "gs_transport_oracle",
    }

    best_params = {k: v.copy() for k, v in model._params_dict().items()}
    best_val = float("inf")
    wait = 0

    for epoch in range(epochs):
        pred_train = model.forward(x_train)
        error_train = pred_train - y_train
        mse_train = float(np.mean(error_train**2))

        z1 = x_train @ model.W1 + model.b1
        h1 = _gelu(z1)
        z2 = h1 @ model.W2 + model.b2
        h2 = _gelu(z2)

        n_points = float(x_train.shape[0] * grid_size)
        d_out = (2.0 / n_points) * error_train

        grad_W3 = h2.T @ d_out
        grad_b3 = np.sum(d_out, axis=0)
        d_h2 = d_out @ model.W3.T

        eps_fd = 1e-5
        gelu_deriv_z2 = (_gelu(z2 + eps_fd) - _gelu(z2 - eps_fd)) / (2.0 * eps_fd)
        d_z2 = d_h2 * gelu_deriv_z2

        grad_W2 = h1.T @ d_z2
        grad_b2 = np.sum(d_z2, axis=0)
        d_h1 = d_z2 @ model.W2.T

        gelu_deriv_z1 = (_gelu(z1 + eps_fd) - _gelu(z1 - eps_fd)) / (2.0 * eps_fd)
        d_z1 = d_h1 * gelu_deriv_z1

        grad_W1 = x_train.T @ d_z1
        grad_b1 = np.sum(d_z1, axis=0)

        params = {
            "W1": model.W1,
            "b1": model.b1,
            "W2": model.W2,
            "b2": model.b2,
            "W3": model.W3,
            "b3": model.b3,
        }
        grads = {
            "W1": grad_W1,
            "b1": grad_b1,
            "W2": grad_W2,
            "b2": grad_b2,
            "W3": grad_W3,
            "b3": grad_b3,
        }
        optimizer.step(params, grads, lr=lr)

        pred_val = model.forward(x_val)
        mse_val = float(np.mean((pred_val - y_val) ** 2))

        history["train_loss"].append(mse_train)  # type: ignore[attr-defined]
        history["val_loss"].append(mse_val)  # type: ignore[attr-defined]

        if mse_val < best_val:
            best_val = mse_val
            best_params = {k: v.copy() for k, v in model._params_dict().items()}
            history["best_epoch"] = epoch + 1
            history["best_val_loss"] = mse_val
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        if epoch % 50 == 0:
            logger.info(
                "Epoch %d: train_mse=%.6f val_mse=%.6f",
                epoch,
                mse_train,
                mse_val,
            )

    for k, v in best_params.items():
        setattr(model, k, v)

    if len(x_test) > 0:
        pred_test = model.forward(x_test)
        test_mse = float(np.mean((pred_test - y_test) ** 2))
        test_rel_l2 = _relative_l2(pred_test.ravel(), y_test.ravel())
    else:
        test_mse = float("nan")
        test_rel_l2 = float("nan")

    model.save_weights(save_path)

    history["saved_path"] = str(save_path)
    history["epochs_completed"] = len(history["train_loss"])  # type: ignore[arg-type]
    history["final_train_loss"] = (
        float(history["train_loss"][-1]) if history["train_loss"] else None  # type: ignore[index]
    )
    history["final_val_loss"] = (
        float(history["val_loss"][-1]) if history["val_loss"] else None  # type: ignore[index]
    )
    history["test_mse"] = test_mse
    history["test_rel_l2"] = test_rel_l2

    machine_counts: Dict[str, int] = {"ITER": 0, "SPARC": 0, "DIII-D": 0, "other": 0}
    for m in meta:
        ip = float(m["Ip"])  # type: ignore[arg-type]
        bt = float(m["BT"])  # type: ignore[arg-type]
        if ip > 10 and bt > 5:
            machine_counts["ITER"] += 1
        elif bt > 10:
            machine_counts["SPARC"] += 1
        elif ip < 3:
            machine_counts["DIII-D"] += 1
        else:
            machine_counts["other"] += 1
    history["machine_class_counts"] = machine_counts

    logger.info(
        "GS-transport surrogate training complete: %d epochs, best_val_mse=%.6f, test_rel_l2=%.4f",
        history["epochs_completed"],
        best_val,
        test_rel_l2,
    )
    logger.info("Machine-class distribution: %s", machine_counts)

    return history
