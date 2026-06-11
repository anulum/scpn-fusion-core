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

from typing import Any, Dict, List, Union

import numpy as np
from numpy.typing import NDArray


def gelu(x: NDArray) -> NDArray:
    """Gaussian Error Linear Unit activation."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def relative_l2(y_pred: NDArray, y_true: NDArray) -> float:
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
        self.m: Dict[Any, NDArray] = {}
        self.v: Dict[Any, NDArray] = {}
        self.t = 0

    def step(
        self,
        params: Union[Dict[Any, NDArray], List[NDArray]],
        grads: Union[Dict[Any, NDArray], List[NDArray]],
        lr: float,
    ) -> None:
        """Update parameters in place using bias-corrected Adam moments."""
        self.t += 1

        if isinstance(params, dict):
            keys = params.keys()
        else:
            keys = range(len(params)) # type: ignore[assignment]

        for key in keys:
            p = params[key] # type: ignore[index]
            g = grads[key] # type: ignore[index]

            if key not in self.m:
                self.m[key] = np.zeros_like(p)
                self.v[key] = np.zeros_like(p)

            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (g**2)

            m_hat = self.m[key] / (1.0 - self.beta1**self.t)
            v_hat = self.v[key] / (1.0 - self.beta2**self.t)

            params[key] -= lr * m_hat / (np.sqrt(v_hat) + self.eps) # type: ignore[index]
