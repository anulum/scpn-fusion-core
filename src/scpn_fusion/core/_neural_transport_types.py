# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Shared data contracts for the neural transport surrogate."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

# Weight file format version expected by the loader.
_WEIGHTS_FORMAT_VERSION = 1
_MAX_WEIGHTS_FILE_BYTES = 128 * 1024 * 1024


@dataclass
class TransportInputs:
    """Local plasma parameters at a single radial location.

    All quantities are in SI / conventional tokamak units.
    """

    rho: float = 0.5
    te_kev: float = 5.0
    ti_kev: float = 5.0
    ne_19: float = 5.0
    grad_te: float = 6.0
    grad_ti: float = 6.0
    grad_ne: float = 2.0
    q: float = 1.5
    s_hat: float = 0.8
    beta_e: float = 0.01
    r_major_m: float = 6.2
    a_minor_m: float = 2.0
    b_tesla: float = 5.3
    z_eff: float = 1.0


@dataclass
class TransportFluxes:
    """Predicted turbulent transport fluxes."""

    chi_e: float = 0.0
    chi_i: float = 0.0
    d_e: float = 0.0
    channel: str = "stable"
    chi_e_itg: float = 0.0
    chi_e_tem: float = 0.0
    chi_e_etg: float = 0.0
    chi_i_itg: float = 0.0


@dataclass
class MLPWeights:
    """Stored weights for a variable-depth feedforward MLP."""

    layers_w: list[FloatArray] = field(default_factory=list)
    layers_b: list[FloatArray] = field(default_factory=list)
    input_mean: FloatArray = field(default_factory=lambda: np.zeros(10))
    input_std: FloatArray = field(default_factory=lambda: np.ones(10))
    output_scale: FloatArray = field(default_factory=lambda: np.ones(3))
    log_transform: bool = False
    gb_scale: bool = False
    gated: bool = False

    @property
    def w1(self) -> FloatArray:
        return self.layers_w[0] if self.layers_w else np.zeros((0, 0))

    @property
    def b1(self) -> FloatArray:
        return self.layers_b[0] if self.layers_b else np.zeros(0)

    @property
    def w2(self) -> FloatArray:
        return self.layers_w[1] if len(self.layers_w) > 1 else np.zeros((0, 0))

    @property
    def b2(self) -> FloatArray:
        return self.layers_b[1] if len(self.layers_b) > 1 else np.zeros(0)

    @property
    def w3(self) -> FloatArray:
        return self.layers_w[2] if len(self.layers_w) > 2 else np.zeros((0, 0))

    @property
    def b3(self) -> FloatArray:
        return self.layers_b[2] if len(self.layers_b) > 2 else np.zeros(0)

    @property
    def n_layers(self) -> int:
        return len(self.layers_w)


__all__ = [
    "FloatArray",
    "MLPWeights",
    "TransportFluxes",
    "TransportInputs",
    "_MAX_WEIGHTS_FILE_BYTES",
    "_WEIGHTS_FORMAT_VERSION",
]
