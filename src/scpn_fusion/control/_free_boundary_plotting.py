# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core - Free-Boundary Plotting
"""Plotting helper for free-boundary supervisory simulation traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, cast

import numpy as np

from scpn_fusion.control._free_boundary_supervisory_types import (
    FloatArray,
    FreeBoundaryTarget,
)


def _as_finite_vector(name: str, values: FloatArray) -> FloatArray:
    """Return ``values`` as a finite one-dimensional float64 array."""
    array = cast(FloatArray, np.asarray(values, dtype=np.float64))
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return array


def _as_finite_matrix(name: str, values: FloatArray, *, min_columns: int) -> FloatArray:
    """Return ``values`` as a finite two-dimensional float64 history matrix."""
    array = cast(FloatArray, np.asarray(values, dtype=np.float64))
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < min_columns:
        raise ValueError(
            f"{name} must be a non-empty 2-D array with at least {min_columns} columns.",
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return array


def _normalise_output_path(output_path: str) -> Path:
    """Return a concrete plot file path and reject empty destinations."""
    stripped = output_path.strip()
    if not stripped:
        raise ValueError("output_path must not be empty.")
    path = Path(stripped)
    if not path.name:
        raise ValueError("output_path must point to a file.")
    return path


def plot_free_boundary_control(
    *,
    time_axis: FloatArray,
    states: FloatArray,
    target: FreeBoundaryTarget,
    actions: FloatArray,
    output_path: str,
) -> tuple[bool, Optional[str]]:
    """Render axis, X-point, and coil-command histories to a plot file.

    Parameters
    ----------
    time_axis : FloatArray
        Non-empty one-dimensional simulation times.
    states : FloatArray
        State history with at least four columns in ``R_axis``, ``Z_axis``,
        ``X_point_R``, and ``X_point_Z`` order.
    target : FreeBoundaryTarget
        Reference magnetic-axis and X-point coordinates drawn as dashed lines.
    actions : FloatArray
        Coil-command history with one row per time sample and at least one
        command column.
    output_path : str
        Destination image path. Missing parent directories are created.

    Returns
    -------
    tuple of bool and str or None
        ``(True, None)`` when the plot is written. On invalid inputs, missing
        Matplotlib, or backend I/O failures, returns ``(False, reason)``.
    """
    try:
        time = _as_finite_vector("time_axis", time_axis)
        state_history = _as_finite_matrix("states", states, min_columns=4)
        action_history = _as_finite_matrix("actions", actions, min_columns=1)
        target_vector = target.as_vector()
        if not np.all(np.isfinite(target_vector)):
            raise ValueError("target coordinates must be finite.")
        if (
            state_history.shape[0] != time.shape[0]
            or action_history.shape[0] != time.shape[0]
        ):
            raise ValueError("time_axis, states, and actions must have matching row counts.")
        destination = _normalise_output_path(output_path)
    except ValueError as exc:
        return False, f"invalid free-boundary plot input: {exc}"

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return False, f"matplotlib unavailable: {exc}"

    fig: Any | None = None
    try:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].set_title("Axis Tracking")
        axes[0].plot(time, state_history[:, 0], label="R axis")
        axes[0].plot(time, state_history[:, 1], label="Z axis")
        axes[0].axhline(target.r_axis_m, color="C0", linestyle="--", alpha=0.5)
        axes[0].axhline(target.z_axis_m, color="C1", linestyle="--", alpha=0.5)
        axes[0].grid(True)
        axes[0].legend()

        axes[1].set_title("X-Point Tracking")
        axes[1].plot(time, state_history[:, 2], label="X-point R")
        axes[1].plot(time, state_history[:, 3], label="X-point Z")
        axes[1].axhline(target.x_point_r_m, color="C0", linestyle="--", alpha=0.5)
        axes[1].axhline(target.x_point_z_m, color="C1", linestyle="--", alpha=0.5)
        axes[1].grid(True)
        axes[1].legend()

        axes[2].set_title("Coil Commands")
        for idx in range(action_history.shape[1]):
            axes[2].plot(time, action_history[:, idx], label=f"coil_{idx}")
        axes[2].grid(True)
        axes[2].legend()

        plt.tight_layout()
        if destination.parent != Path("."):
            destination.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(destination))
        return True, None
    except Exception as exc:
        return False, f"plot render failed: {exc}"
    finally:
        if fig is not None:
            plt.close(fig)
