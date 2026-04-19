# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Neural Transport Math
"""Math kernels extracted from ``neural_transport`` monolith."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _relu(x: FloatArray) -> FloatArray:
    return np.maximum(0.0, x)


def _softplus(x: FloatArray) -> FloatArray:
    return np.log1p(np.exp(np.clip(x, -20.0, 20.0)))


def _gelu(x: FloatArray) -> FloatArray:
    """Gaussian Error Linear Unit (matches JAX/PyTorch training)."""
    return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def _mlp_forward(x: FloatArray, weights: Any) -> FloatArray:
    """Forward pass through a variable-depth MLP."""
    h = (x - weights.input_mean) / np.maximum(weights.input_std, 1e-8)
    for i in range(weights.n_layers - 1):
        h = _gelu(h @ weights.layers_w[i] + weights.layers_b[i])

    raw = h @ weights.layers_w[-1] + weights.layers_b[-1]
    if weights.gated:
        flux = _softplus(raw[..., :3]) * weights.output_scale
        gate = 1.0 / (1.0 + np.exp(-np.clip(raw[..., 3:], -20.0, 20.0)))
        out = gate * flux
    else:
        out = _softplus(raw) * weights.output_scale

    if weights.log_transform:
        out = np.expm1(np.clip(out, 0.0, 20.0))

    if weights.gb_scale:
        te = x[..., 1]
        _m_D = 3.344e-27  # deuterium mass [kg]
        _e = 1.602e-19  # elementary charge [C]
        _B0_SPARC = 5.3  # reference B0 [T], matched to QLKNN training
        _R0_ITER = 6.2  # reference R0 [m], gyro-Bohm normalisation
        te_j = te * 1e3 * _e
        cs = np.sqrt(te_j / _m_D)
        rho_s = np.sqrt(_m_D * te_j) / (_e * _B0_SPARC)
        chi_gb = rho_s**2 * cs / _R0_ITER
        if chi_gb.ndim == 0:
            out = out * float(chi_gb)
        else:
            out = out * chi_gb[..., np.newaxis]

    return out


def _compute_nustar(
    te_kev: float,
    ne_19: float,
    q: float,
    rho: float,
    r_major: float = 6.2,
    a_minor: float = 2.0,
    z_eff: float = 1.0,
) -> float:
    """Electron collisionality nu_star (Wesson Ch.14, Eq.14.5.4).

    eps = rho * a_minor / r_major  (local inverse aspect ratio).
    """
    ln_lambda = 15.2
    ne_m3 = ne_19 * 1e19
    te_ev = te_kev * 1e3
    eps = max(rho * a_minor / r_major, 1e-4)
    return (
        6.921e-18 * ne_m3 * q * r_major * z_eff**2 * ln_lambda / (max(te_ev, 1.0) ** 2 * eps**1.5)
    )


__all__ = ["_relu", "_softplus", "_gelu", "_mlp_forward", "_compute_nustar"]
