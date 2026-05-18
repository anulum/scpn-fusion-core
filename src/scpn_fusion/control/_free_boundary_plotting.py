# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core - Free-Boundary Plotting
"""Plotting helper for free-boundary supervisory simulation traces."""

from __future__ import annotations

from typing import Optional

from scpn_fusion.control._free_boundary_supervisory_types import (
    FloatArray,
    FreeBoundaryTarget,
)


def plot_free_boundary_control(
    *,
    time_axis: FloatArray,
    states: FloatArray,
    target: FreeBoundaryTarget,
    actions: FloatArray,
    output_path: str,
) -> tuple[bool, Optional[str]]:
    """Render axis, X-point, and coil-command histories to a plot file."""

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return False, f"matplotlib unavailable: {exc}"

    try:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].set_title("Axis Tracking")
        axes[0].plot(time_axis, states[:, 0], label="R axis")
        axes[0].plot(time_axis, states[:, 1], label="Z axis")
        axes[0].axhline(target.r_axis_m, color="C0", linestyle="--", alpha=0.5)
        axes[0].axhline(target.z_axis_m, color="C1", linestyle="--", alpha=0.5)
        axes[0].grid(True)
        axes[0].legend()

        axes[1].set_title("X-Point Tracking")
        axes[1].plot(time_axis, states[:, 2], label="X-point R")
        axes[1].plot(time_axis, states[:, 3], label="X-point Z")
        axes[1].axhline(target.x_point_r_m, color="C0", linestyle="--", alpha=0.5)
        axes[1].axhline(target.x_point_z_m, color="C1", linestyle="--", alpha=0.5)
        axes[1].grid(True)
        axes[1].legend()

        axes[2].set_title("Coil Commands")
        for idx in range(actions.shape[1]):
            axes[2].plot(time_axis, actions[:, idx], label=f"coil_{idx}")
        axes[2].grid(True)
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        return True, None
    except Exception as exc:
        return False, str(exc)
