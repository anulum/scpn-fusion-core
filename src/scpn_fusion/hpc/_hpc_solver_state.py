# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Native Solver Checkpoint
"""Owned native solver history with immutable array storage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class HPCSolverState:
    """Complete mutable native history and its originating solver context.

    Parameters
    ----------
    psi : NDArray[np.float64]
        Row-major reduced flux with immutable owned byte storage.
    j_phi : NDArray[np.float64]
        Toroidal source history with immutable owned byte storage.
    boundary_value : float
        Native fixed Dirichlet wall value.
    context : object
        Opaque identity of the originating initialization.
    grid : tuple[int, int, float, float, float, float]
        Grid counts and radial/vertical coordinate endpoints.
    library_sha256 : str or None
        Digest of the originating trusted library.
    """

    psi: NDArray[np.float64]
    j_phi: NDArray[np.float64]
    boundary_value: float
    context: object
    grid: tuple[int, int, float, float, float, float]
    library_sha256: str | None


def immutable_field(field: NDArray[np.float64]) -> NDArray[np.float64]:
    """Copy a field into immutable bytes, preventing write-flag reactivation.

    Parameters
    ----------
    field : NDArray[np.float64]
        Array to preserve independently of native work buffers.

    Returns
    -------
    NDArray[np.float64]
        Read-only array backed by a new immutable byte string.
    """
    return np.frombuffer(field.tobytes(order="C"), dtype=np.float64).reshape(field.shape)
