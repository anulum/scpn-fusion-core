# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Fusion Kernel Coilset Config Tests
"""Tests for free-boundary coilset configuration contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_fusion.core.fusion_kernel import FusionKernel


def _base_config() -> dict[str, Any]:
    """Return a minimal FusionKernel config with a valid free-boundary coil set."""
    return {
        "reactor_name": "Coilset-Config-Test",
        "grid_resolution": [12, 12],
        "dimensions": {"R_min": 4.0, "R_max": 8.0, "Z_min": -3.0, "Z_max": 3.0},
        "physics": {"plasma_current_target": 12.0, "vacuum_permeability": 1.0},
        "coils": [
            {"name": "PF1", "r": 3.0, "z": 1.2, "current": 2.5, "turns": 10},
            {"name": "PF2", "r": 5.0, "z": -1.2, "current": -1.5, "turns": 8},
        ],
        "free_boundary": {
            "current_limits": [6.0, 7.0],
            "target_flux_points": [[4.0, 0.0], [4.5, 0.2]],
            "target_flux_values": [0.1, 0.2],
            "x_point_target": [4.2, -1.1],
            "x_point_flux_target": 0.05,
            "divertor_strike_points": [[3.2, -2.2]],
            "divertor_flux_values": [0.03],
        },
        "solver": {
            "max_iterations": 3,
            "convergence_threshold": 1e-4,
            "relaxation_factor": 0.1,
        },
    }


def _kernel(tmp_path: Path, config: dict[str, Any] | None = None) -> FusionKernel:
    """Instantiate ``FusionKernel`` from a real JSON config file."""
    cfg = tmp_path / "fusion_config.json"
    cfg.write_text(json.dumps(_base_config() if config is None else config), encoding="utf-8")
    return FusionKernel(str(cfg))


def test_build_coilset_accepts_absent_free_boundary_block(tmp_path: Path) -> None:
    """A missing optional free-boundary block still builds the required coil contract."""
    config = _base_config()
    config["free_boundary"] = None

    coils = _kernel(tmp_path, config).build_coilset_from_config()

    assert coils.positions == [(3.0, 1.2), (5.0, -1.2)]
    np.testing.assert_allclose(coils.currents, np.array([2.5, -1.5], dtype=np.float64))
    assert coils.turns == [10, 8]
    assert coils.current_limits is None
    assert coils.target_flux_points is None


@pytest.mark.parametrize(
    ("coils", "message"),
    [
        ([], "at least one external coil"),
        ("not-a-list", "at least one external coil"),
        ([None], r"coils\[0\] must be an object"),
        ([{"z": 0.0, "current": 1.0}], "must define finite r, z, and current"),
        ([{"r": 0.0, "z": 0.0, "current": 1.0}], "finite r > 0"),
        ([{"r": 3.0, "z": float("nan"), "current": 1.0}], "finite r > 0"),
        ([{"r": 3.0, "z": 0.0, "current": float("inf")}], "finite r > 0"),
        ([{"r": 3.0, "z": 0.0, "current": 1.0, "turns": True}], "positive integer"),
        ([{"r": 3.0, "z": 0.0, "current": 1.0, "turns": "bad"}], "positive integer"),
        ([{"r": 3.0, "z": 0.0, "current": 1.0, "turns": 1.5}], "positive integer"),
        ([{"r": 3.0, "z": 0.0, "current": 1.0, "turns": 0}], "positive integer"),
    ],
)
def test_build_coilset_rejects_invalid_coil_definitions(
    tmp_path: Path,
    coils: object,
    message: str,
) -> None:
    """Coil definitions fail closed before free-boundary response construction."""
    kernel = _kernel(tmp_path)
    kernel.cfg["coils"] = coils

    with pytest.raises(ValueError, match=message):
        kernel.build_coilset_from_config()


@pytest.mark.parametrize(
    ("free_boundary", "message"),
    [
        ("invalid", "free_boundary must be an object"),
        ({"current_limits": [5.0, 0.0]}, "current_limits"),
        ({"current_limits": [5.0, float("inf")]}, "current_limits"),
        ({"target_flux_points": []}, "target_flux_points"),
        ({"target_flux_points": [[4.0, 0.0, 1.0]]}, "target_flux_points"),
        ({"target_flux_points": [[4.0, float("nan")]]}, "target_flux_points"),
        ({"target_flux_values": [0.1]}, "target_flux_points must be set"),
        ({"x_point_target": [4.2, float("inf")]}, "x_point_target"),
        ({"x_point_flux_target": float("nan")}, "x_point_flux_target"),
        ({"divertor_strike_points": [[3.2, float("inf")]]}, "divertor_strike_points"),
        ({"divertor_flux_values": [0.03]}, "divertor_strike_points must be set"),
    ],
)
def test_build_coilset_rejects_invalid_free_boundary_contracts(
    tmp_path: Path,
    free_boundary: object,
    message: str,
) -> None:
    """Optional free-boundary targets and limits must stay finite and aligned."""
    config = _base_config()
    config["free_boundary"] = free_boundary

    with pytest.raises(ValueError, match=message):
        _kernel(tmp_path, config).build_coilset_from_config()
