# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Free-Boundary Tracking Control
"""Private typed records used by the free-boundary tracking mixins."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class _ObjectiveBlock:
    name: str
    start: int
    stop: int


@dataclass(frozen=True)
class _ActuatorSnapshot:
    state: float
    delay_buffer: tuple[float, ...]


@dataclass(frozen=True)
class _ObservationSnapshot:
    true: FloatArray
    measured: FloatArray
    delayed: FloatArray
    effective: FloatArray
