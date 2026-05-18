# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Configuration-to-CoilSet construction for free-boundary kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from scpn_fusion.core.fusion_kernel_numerics import FloatArray

if TYPE_CHECKING:
    from scpn_fusion.core.fusion_kernel import CoilSet


def _as_finite_vector(value: Any, *, name: str, length: int | None = None) -> FloatArray:
    """Return ``value`` as a one-dimensional finite ``float64`` vector."""
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if length is not None and arr.shape != (length,):
        raise ValueError(f"{name} must have length {length}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values only.")
    return arr


def _as_finite_points(value: Any, *, name: str) -> FloatArray:
    """Return validated ``(R, Z)`` control or diagnostic points."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 1:
        raise ValueError(f"{name} must have shape (n_points, 2) with n_points > 0.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values only.")
    return arr


def build_coilset_from_config(kernel: Any) -> CoilSet:
    """Build a validated :class:`~scpn_fusion.core.fusion_kernel.CoilSet`.

    The configuration contract is intentionally strict because these values
    feed the free-boundary least-squares and Green's-function paths directly.
    Coil positions must be finite cylindrical ``(R, Z)`` coordinates with
    ``R > 0``; turns must be positive integers; optional current limits must be
    finite and positive; and every optional target or diagnostic vector must
    have a length consistent with its associated point array.
    """
    from scpn_fusion.core.fusion_kernel import CoilSet

    coil_cfg = kernel.cfg.get("coils", [])
    if not isinstance(coil_cfg, list) or len(coil_cfg) < 1:
        raise ValueError("coils must contain at least one external coil.")

    positions: list[tuple[float, float]] = []
    currents: list[float] = []
    turns: list[int] = []
    for idx, coil in enumerate(coil_cfg):
        if not isinstance(coil, dict):
            raise ValueError(f"coils[{idx}] must be an object.")
        try:
            r = float(coil["r"])
            z = float(coil["z"])
            current = float(coil.get("current", 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"coils[{idx}] must define finite r, z, and current fields.") from exc
        if not np.isfinite(r) or r <= 0.0 or not np.isfinite(z) or not np.isfinite(current):
            raise ValueError(f"coils[{idx}] must define finite r > 0, finite z, and current.")
        raw_turns = coil.get("turns", 1)
        if isinstance(raw_turns, bool):
            raise ValueError(f"coils[{idx}].turns must be a positive integer.")
        try:
            n_turns = int(raw_turns)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"coils[{idx}].turns must be a positive integer.") from exc
        if n_turns < 1 or float(n_turns) != float(raw_turns):
            raise ValueError(f"coils[{idx}].turns must be a positive integer.")
        positions.append((r, z))
        currents.append(current)
        turns.append(n_turns)

    n_coils = len(positions)
    fb_cfg = kernel.cfg.get("free_boundary", {})
    if fb_cfg is None:
        fb_cfg = {}
    if not isinstance(fb_cfg, dict):
        raise ValueError("free_boundary must be an object when provided.")

    current_limits = None
    if "current_limits" in fb_cfg and fb_cfg["current_limits"] is not None:
        current_limits = _as_finite_vector(
            fb_cfg["current_limits"], name="current_limits", length=n_coils
        )
        if np.any(current_limits <= 0.0):
            raise ValueError("current_limits must contain finite positive values only.")

    target_flux_points = None
    if "target_flux_points" in fb_cfg and fb_cfg["target_flux_points"] is not None:
        target_flux_points = _as_finite_points(
            fb_cfg["target_flux_points"], name="target_flux_points"
        )

    target_flux_values = None
    if "target_flux_values" in fb_cfg and fb_cfg["target_flux_values"] is not None:
        if target_flux_points is None:
            raise ValueError("target_flux_points must be set when target_flux_values is provided.")
        target_flux_values = _as_finite_vector(
            fb_cfg["target_flux_values"],
            name="target_flux_values",
            length=int(target_flux_points.shape[0]),
        )

    x_point_target = None
    if "x_point_target" in fb_cfg and fb_cfg["x_point_target"] is not None:
        x_point_target = _as_finite_vector(
            fb_cfg["x_point_target"], name="x_point_target", length=2
        )

    x_point_flux_target = None
    if "x_point_flux_target" in fb_cfg and fb_cfg["x_point_flux_target"] is not None:
        x_point_flux_target = float(fb_cfg["x_point_flux_target"])
        if not np.isfinite(x_point_flux_target):
            raise ValueError("x_point_flux_target must be finite.")

    divertor_strike_points = None
    if "divertor_strike_points" in fb_cfg and fb_cfg["divertor_strike_points"] is not None:
        divertor_strike_points = _as_finite_points(
            fb_cfg["divertor_strike_points"], name="divertor_strike_points"
        )

    divertor_flux_values = None
    if "divertor_flux_values" in fb_cfg and fb_cfg["divertor_flux_values"] is not None:
        if divertor_strike_points is None:
            raise ValueError(
                "divertor_strike_points must be set when divertor_flux_values is provided."
            )
        divertor_flux_values = _as_finite_vector(
            fb_cfg["divertor_flux_values"],
            name="divertor_flux_values",
            length=int(divertor_strike_points.shape[0]),
        )

    return CoilSet(
        positions=positions,
        currents=np.asarray(currents, dtype=np.float64),
        turns=turns,
        current_limits=current_limits,
        target_flux_points=target_flux_points,
        target_flux_values=target_flux_values,
        x_point_target=x_point_target,
        x_point_flux_target=x_point_flux_target,
        divertor_strike_points=divertor_strike_points,
        divertor_flux_values=divertor_flux_values,
    )
