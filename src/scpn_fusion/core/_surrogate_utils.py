# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Shared Surrogate Training Utilities
"""
Shared helpers for surrogate training (optimizers, activations, metrics).
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


def gelu(x: FloatArray) -> FloatArray:
    """Gaussian Error Linear Unit activation."""
    return np.asarray(
        0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))),
        dtype=np.float64,
    )


def relative_l2(y_pred: FloatArray, y_true: FloatArray) -> float:
    """Calculate relative L2 error norm."""
    denom = np.linalg.norm(y_true) + 1e-8
    return float(np.linalg.norm(y_pred - y_true) / denom)


class AdamOptimizer:
    """
    Minimal NumPy-based Adam optimizer for surrogate training.

    Supports both list and dict parameter structures.
    """

    def __init__(
        self,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: dict[Any, FloatArray] = {}
        self.v: dict[Any, FloatArray] = {}
        self.t = 0

    def step(
        self,
        params: dict[Any, FloatArray] | list[FloatArray],
        grads: dict[Any, FloatArray] | list[FloatArray],
        lr: float,
    ) -> None:
        """Update parameters in place using bias-corrected Adam moments."""
        self.t += 1

        if isinstance(params, dict):
            if not isinstance(grads, dict):
                raise TypeError("AdamOptimizer grads must be a dict when params is a dict.")
            for key, p in params.items():
                g = grads[key]
                self._step_one(key, p, g, lr)
        else:
            if not isinstance(grads, list):
                raise TypeError("AdamOptimizer grads must be a list when params is a list.")
            for key, p in enumerate(params):
                self._step_one(key, p, grads[key], lr)

    def _step_one(self, key: Any, param: FloatArray, grad: FloatArray, lr: float) -> None:
        if key not in self.m:
            self.m[key] = np.zeros_like(param)
            self.v[key] = np.zeros_like(param)

        self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * grad
        self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (grad**2)

        m_hat = self.m[key] / (1.0 - self.beta1**self.t)
        v_hat = self.v[key] / (1.0 - self.beta2**self.t)

        param -= lr * m_hat / (np.sqrt(v_hat) + self.eps)
