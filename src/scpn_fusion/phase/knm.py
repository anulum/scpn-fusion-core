# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Knm Coupling Matrix (Paper 27)
"""
Paper 27 Knm specification.

K[n, m] encodes coupling from source layer n to target layer m.
Diagonal: intra-layer synchronisation strength.
Off-diagonal: inter-layer bidirectional causality (bottom-up / top-down).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# Canonical 16-layer natural frequencies (rad/s).
# Paper 27, Table 1 — SCPN Kuramoto calibration.
OMEGA_N_16 = np.array(
    [
        1.329,
        2.610,
        0.844,
        1.520,
        0.710,
        3.780,
        1.055,
        0.625,
        2.210,
        1.740,
        0.480,
        3.210,
        0.915,
        1.410,
        2.830,
        0.991,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class KnmSpec:
    """Paper 27 coupling specification.

    K      : (L, L) coupling matrix.  K[n, m] = source n -> target m.
    alpha  : (L, L) Sakaguchi phase-lag (optional).
    zeta   : (L,) per-layer global-driver gain ζ_m (optional).
    """

    K: FloatArray
    alpha: FloatArray | None = None
    zeta: FloatArray | None = None
    layer_names: Sequence[str] | None = None

    def __post_init__(self) -> None:
        K = np.asarray(self.K, dtype=np.float64)
        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError("K must be square (L, L)")
        L = K.shape[0]
        if self.alpha is not None:
            a = np.asarray(self.alpha, dtype=np.float64)
            if a.shape != (L, L):
                raise ValueError(f"alpha shape {a.shape} != ({L}, {L})")
        if self.zeta is not None:
            z = np.asarray(self.zeta, dtype=np.float64)
            if z.shape != (L,):
                raise ValueError(f"zeta shape {z.shape} != ({L},)")
        if self.layer_names is not None and len(self.layer_names) != L:
            raise ValueError("layer_names length must equal L")

    @property
    def L(self) -> int:
        return int(np.asarray(self.K).shape[0])


def build_knm_paper27(
    L: int = 16,
    K_base: float = 0.45,
    K_alpha: float = 0.3,
    zeta_uniform: float = 0.0,
) -> KnmSpec:
    """Build the canonical Paper 27 Knm with exponential distance decay.

    K[i,j] = K_base · exp(−K_alpha · |i − j|),  diag(K) kept for
    intra-layer sync (unlike the inter-oscillator Knm which zeros diag).

    K_base=0.45 and K_alpha=0.3 from Paper 27 §3.2, Eq. 12.
    Calibration anchors from Paper 27, Table 2.
    Cross-hierarchy boosts from Paper 27 §4.3.
    """
    idx = np.arange(L)
    dist = np.abs(idx[:, None] - idx[None, :])
    K = K_base * np.exp(-K_alpha * dist)

    # Calibration anchors — Paper 27 Table 2
    anchors = [(0, 1, 0.302), (1, 2, 0.201), (2, 3, 0.252), (3, 4, 0.154)]
    for i, j, val in anchors:
        if i < L and j < L:
            K[i, j] = val
            K[j, i] = val

    # Cross-hierarchy boosts — Paper 27 §4.3
    if L >= 16:
        K[0, 15] = max(K[0, 15], 0.05)
        K[15, 0] = max(K[15, 0], 0.05)
    if L >= 7:
        K[4, 6] = max(K[4, 6], 0.15)
        K[6, 4] = max(K[6, 4], 0.15)

    zeta = np.full(L, zeta_uniform, dtype=np.float64) if zeta_uniform != 0.0 else None

    return KnmSpec(K=K, zeta=zeta)
