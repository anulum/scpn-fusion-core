# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Online Surrogate Retraining
"""
Online learning layer for surrogate transport model improvement.

Accumulates GK spot-check results as training data and periodically
fine-tunes the QLKNN surrogate.  Includes validation holdout and
automatic rollback if performance degrades.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    """Single (input, target) pair from a GK spot-check."""

    input_10d: NDArray[np.float64]  # shape (10,)
    target_3d: NDArray[np.float64]  # shape (3,): chi_e, chi_i, D_e


@dataclass
class LearnerConfig:
    """Online learner parameters."""

    buffer_size: int = 100  # trigger retraining when buffer reaches this
    validation_fraction: float = 0.2
    n_epochs: int = 10
    learning_rate: float = 1e-4
    max_generations: int = 50  # max retraining cycles before stopping


class OnlineLearner:
    """Accumulate GK data and fine-tune the surrogate when ready."""

    def __init__(self, config: LearnerConfig | None = None) -> None:
        """Initialize validation-guarded online retraining state."""
        self.config = config or LearnerConfig()
        if self.config.buffer_size < 2:
            raise ValueError("buffer_size must be at least 2 to allow a validation holdout")
        if not 0.0 < self.config.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be strictly between 0 and 1")
        if self.config.n_epochs < 1:
            raise ValueError("n_epochs must be at least 1")
        if self.config.learning_rate <= 0.0 or not np.isfinite(self.config.learning_rate):
            raise ValueError("learning_rate must be a finite positive value")
        if self.config.max_generations < 1:
            raise ValueError("max_generations must be at least 1")
        self.buffer: list[TrainingSample] = []
        self.generation: int = 0
        self._best_val_loss: float = float("inf")
        self._weights_backup: dict | None = None
        self.retrain_history: list[dict] = []

    def add_sample(self, input_10d: NDArray[np.float64], target_3d: NDArray[np.float64]) -> None:
        """Validate and append one 10D GK training sample to the buffer."""
        input_array = np.asarray(input_10d, dtype=np.float64)
        target_array = np.asarray(target_3d, dtype=np.float64)
        if input_array.shape != (10,):
            raise ValueError(f"input_10d must have shape (10,), got {input_array.shape}")
        if target_array.shape != (3,):
            raise ValueError(f"target_3d must have shape (3,), got {target_array.shape}")
        if not np.all(np.isfinite(input_array)):
            raise ValueError("input_10d must contain only finite values")
        if not np.all(np.isfinite(target_array)):
            raise ValueError("target_3d must contain only finite values")
        self.buffer.append(
            TrainingSample(input_10d=input_array.copy(), target_3d=target_array.copy())
        )

    @property
    def buffer_full(self) -> bool:
        """Return whether buffered samples have reached the retraining trigger."""
        return len(self.buffer) >= self.config.buffer_size

    def try_retrain(
        self,
        current_weights: dict | None = None,
    ) -> dict | None:
        """Attempt retraining if buffer is full.

        Parameters
        ----------
        current_weights : dict or None
            Current MLP weight arrays (w1, b1, ...). If None, trains
            from scratch on the buffer data.

        Returns
        -------
        dict or None
            Updated weights if retraining succeeded, None if skipped or
            rolled back due to validation loss increase.
        """
        if not self.buffer_full:
            return None

        if self.generation >= self.config.max_generations:
            _logger.info("Max retraining generations (%d) reached", self.config.max_generations)
            return None

        n = len(self.buffer)
        n_val = max(int(n * self.config.validation_fraction), 1)
        n_train = n - n_val
        if n_train < 1:
            raise ValueError("buffer_size and validation_fraction leave no training samples")

        # Shuffle and split
        rng = np.random.default_rng(self.generation)
        indices = rng.permutation(n)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        X_train = np.array([self.buffer[i].input_10d for i in train_idx])
        Y_train = np.array([self.buffer[i].target_3d for i in train_idx])
        X_val = np.array([self.buffer[i].input_10d for i in val_idx])
        Y_val = np.array([self.buffer[i].target_3d for i in val_idx])

        # Simple gradient descent on MSE (no external ML framework needed)
        if current_weights is None:
            # Initialise small random weights
            w1 = rng.normal(0, 0.1, (10, 64))
            b1 = np.zeros(64)
            w2 = rng.normal(0, 0.1, (64, 32))
            b2 = np.zeros(32)
            w3 = rng.normal(0, 0.1, (32, 3))
            b3 = np.zeros(3)
        else:
            w1, b1 = current_weights["w1"].copy(), current_weights["b1"].copy()
            w2, b2 = current_weights["w2"].copy(), current_weights["b2"].copy()
            w3, b3 = current_weights["w3"].copy(), current_weights["b3"].copy()

        self._weights_backup = {
            "w1": w1.copy(),
            "b1": b1.copy(),
            "w2": w2.copy(),
            "b2": b2.copy(),
            "w3": w3.copy(),
            "b3": b3.copy(),
        }

        lr = self.config.learning_rate
        best_val = float("inf")
        best_weights = None

        for _ in range(self.config.n_epochs):
            # Forward pass (train)
            z1 = X_train @ w1 + b1
            h1 = np.maximum(0, z1)
            z2 = h1 @ w2 + b2
            h2 = np.maximum(0, z2)
            pred = h2 @ w3 + b3

            # Full dense-network MSE backpropagation through both hidden layers.
            grad_out = 2.0 * (pred - Y_train) / pred.size
            grad_w3 = h2.T @ grad_out
            grad_b3 = grad_out.sum(axis=0)
            grad_h2 = grad_out @ w3.T
            grad_z2 = grad_h2 * (z2 > 0.0)
            grad_w2 = h1.T @ grad_z2
            grad_b2 = grad_z2.sum(axis=0)
            grad_h1 = grad_z2 @ w2.T
            grad_z1 = grad_h1 * (z1 > 0.0)
            grad_w1 = X_train.T @ grad_z1
            grad_b1 = grad_z1.sum(axis=0)

            w1 -= lr * grad_w1
            b1 -= lr * grad_b1
            w2 -= lr * grad_w2
            b2 -= lr * grad_b2
            w3 -= lr * grad_w3
            b3 -= lr * grad_b3

            # Validation loss
            h1_v = np.maximum(0, X_val @ w1 + b1)
            h2_v = np.maximum(0, h1_v @ w2 + b2)
            pred_v = h2_v @ w3 + b3
            loss_val = float(np.mean((pred_v - Y_val) ** 2))

            if loss_val < best_val:
                best_val = loss_val
                best_weights = {
                    "w1": w1.copy(),
                    "b1": b1.copy(),
                    "w2": w2.copy(),
                    "b2": b2.copy(),
                    "w3": w3.copy(),
                    "b3": b3.copy(),
                }

        # Rollback check
        if best_val >= self._best_val_loss:
            _logger.info(
                "Validation loss did not improve (%.4f >= %.4f), rolling back",
                best_val,
                self._best_val_loss,
            )
            self.retrain_history.append(
                {"generation": self.generation, "accepted": False, "val_loss": best_val}
            )
            return None

        self._best_val_loss = best_val
        self.generation += 1
        self.buffer.clear()
        self.retrain_history.append(
            {"generation": self.generation, "accepted": True, "val_loss": best_val}
        )
        _logger.info("Retraining gen %d accepted, val_loss=%.6f", self.generation, best_val)
        return best_weights

    def reset(self) -> None:
        """Clear buffered data, accepted-generation state, and retraining history."""
        self.buffer.clear()
        self.generation = 0
        self._best_val_loss = float("inf")
        self._weights_backup = None
        self.retrain_history.clear()
