# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Conservative Runaway Kinetic Grid
"""Physical radius-momentum-pitch grids for runaway kinetic evolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _strict_faces(name: str, values: FloatArray, *, lower: float | None = None) -> FloatArray:
    faces = np.asarray(values, dtype=np.float64)
    if faces.ndim != 1 or faces.size < 2:
        raise ValueError(f"{name} must be a one-dimensional face array with at least two entries")
    if not np.all(np.isfinite(faces)):
        raise ValueError(f"{name} contains a non-finite coordinate")
    if np.any(np.diff(faces) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    if lower is not None and faces[0] < lower:
        raise ValueError(f"{name} starts below {lower}")
    result = np.array(faces, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RunawayKineticGrid:
    """Tensor-product grid whose three axes are all evolved physical axes.

    The distribution layout is ``(radius, pitch, momentum)``.  Faces are
    stored explicitly so the same grid can be reconstructed from a DREAM HDF5
    output without inventing projected axes.
    """

    radius_faces_m: FloatArray
    pitch_faces: FloatArray
    momentum_faces_mc: FloatArray

    def __post_init__(self) -> None:
        """Validate and freeze all phase-space cell-face coordinates."""
        radius = _strict_faces("radius_faces_m", self.radius_faces_m, lower=0.0)
        pitch = _strict_faces("pitch_faces", self.pitch_faces)
        momentum = _strict_faces("momentum_faces_mc", self.momentum_faces_mc, lower=0.0)
        if pitch[0] < -1.0 or pitch[-1] > 1.0:
            raise ValueError("pitch_faces must stay within [-1, 1]")
        object.__setattr__(self, "radius_faces_m", radius)
        object.__setattr__(self, "pitch_faces", pitch)
        object.__setattr__(self, "momentum_faces_mc", momentum)

    @property
    def nr(self) -> int:
        """Number of evolved radial cells."""
        return self.radius_faces_m.size - 1

    @property
    def nxi(self) -> int:
        """Number of evolved pitch cells."""
        return self.pitch_faces.size - 1

    @property
    def np(self) -> int:
        """Number of evolved momentum cells."""
        return self.momentum_faces_mc.size - 1

    @property
    def shape(self) -> tuple[int, int, int]:
        """Distribution shape in radius-pitch-momentum order."""
        return (self.nr, self.nxi, self.np)

    @property
    def radius_m(self) -> FloatArray:
        """Radial cell centres."""
        return 0.5 * (self.radius_faces_m[1:] + self.radius_faces_m[:-1])

    @property
    def pitch(self) -> FloatArray:
        """Pitch cell centres."""
        return 0.5 * (self.pitch_faces[1:] + self.pitch_faces[:-1])

    @property
    def momentum_mc(self) -> FloatArray:
        """Momentum cell centres in units of electron rest momentum."""
        return 0.5 * (self.momentum_faces_mc[1:] + self.momentum_faces_mc[:-1])

    @property
    def radial_shell_measure_m2(self) -> FloatArray:
        """Cylindrical radial measure ``(r_f[i+1]^2-r_f[i]^2)/2``."""
        return 0.5 * (self.radius_faces_m[1:] ** 2 - self.radius_faces_m[:-1] ** 2)

    @property
    def momentum_shell_measure(self) -> FloatArray:
        """Relativistic spherical momentum measure ``(p_f^3)/3``."""
        return (self.momentum_faces_mc[1:] ** 3 - self.momentum_faces_mc[:-1] ** 3) / 3.0

    @property
    def phase_space_cell_measure(self) -> FloatArray:
        """Axisymmetric cylindrical phase-space measure without ``4π²R``.

        The omitted constant cancels from normalized moments and conservation
        residuals.  Geometry imported from DREAM can instead supply its exact
        bounce-averaged cell volumes to the operator.
        """
        return cast(
            FloatArray,
            self.radial_shell_measure_m2[:, None, None]
            * np.diff(self.pitch_faces)[None, :, None]
            * (2.0 * np.pi * self.momentum_shell_measure)[None, None, :],
        )

    def require_state(self, name: str, values: FloatArray) -> FloatArray:
        """Validate and copy a finite tensor defined on every physical cell."""
        state = np.asarray(values, dtype=np.float64)
        if state.shape != self.shape:
            raise ValueError(f"{name} must have shape {self.shape}, got {state.shape}")
        if not np.all(np.isfinite(state)):
            raise ValueError(f"{name} contains a non-finite value")
        return np.array(state, dtype=np.float64, copy=True)


__all__ = ["FloatArray", "RunawayKineticGrid"]
