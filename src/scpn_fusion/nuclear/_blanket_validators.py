# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Blanket Neutronics Input Validators
"""Shared numeric input validators for the blanket-neutronics models.

Imported by both :class:`~scpn_fusion.nuclear.blanket_neutronics.BreedingBlanket`
and :class:`~scpn_fusion.nuclear.multigroup_blanket.MultiGroupBlanket`.
"""

from __future__ import annotations

import numpy as np


def _require_finite_float(
    name: str,
    value: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    if min_value is not None and out < min_value:
        raise ValueError(f"{name} must be >= {min_value}.")
    if max_value is not None and out > max_value:
        raise ValueError(f"{name} must be <= {max_value}.")
    return out


def _require_int(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    out = int(value)
    if out < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return out
