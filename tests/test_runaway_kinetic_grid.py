# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Conservative Runaway Kinetic Grid Tests
"""Tests for the unprojected radius-momentum-pitch grid."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_fusion.core.runaway_kinetic_grid import RunawayKineticGrid


def test_grid_exposes_all_three_physical_axes_and_measure() -> None:
    grid = RunawayKineticGrid(
        radius_faces_m=np.array([0.0, 0.1, 0.2]),
        pitch_faces=np.array([-1.0, 0.0, 1.0]),
        momentum_faces_mc=np.array([0.0, 1.0, 2.0]),
    )

    assert grid.shape == (2, 2, 2)
    assert np.allclose(grid.radius_m, [0.05, 0.15], rtol=0.0, atol=1.0e-15)
    assert np.array_equal(grid.pitch, [-0.5, 0.5])
    assert np.array_equal(grid.momentum_mc, [0.5, 1.5])
    assert grid.phase_space_cell_measure.shape == grid.shape
    assert np.all(grid.phase_space_cell_measure > 0.0)


@pytest.mark.parametrize(
    ("axis", "message"),
    [
        (np.array([0.0, 0.0]), "strictly increasing"),
        (np.array([0.0, np.nan]), "non-finite"),
    ],
)
def test_grid_rejects_invalid_radius_faces(axis: NDArray[np.float64], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RunawayKineticGrid(
            radius_faces_m=axis,
            pitch_faces=np.array([-1.0, 1.0]),
            momentum_faces_mc=np.array([0.0, 1.0]),
        )


@pytest.mark.parametrize(
    ("radius", "pitch", "momentum", "message"),
    [
        ([-0.1, 0.1], [-1.0, 1.0], [0.0, 1.0], "starts below"),
        ([0.0, 0.1], [-1.1, 1.0], [0.0, 1.0], "within"),
        ([0.0, 0.1], [-1.0, 1.0], [-0.1, 1.0], "starts below"),
        ([0.0], [-1.0, 1.0], [0.0, 1.0], "at least two"),
    ],
)
def test_grid_rejects_out_of_domain_or_incomplete_axes(
    radius: list[float],
    pitch: list[float],
    momentum: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RunawayKineticGrid(
            radius_faces_m=np.asarray(radius),
            pitch_faces=np.asarray(pitch),
            momentum_faces_mc=np.asarray(momentum),
        )


def test_state_validation_rejects_projection_and_nonfinite_data() -> None:
    grid = RunawayKineticGrid(
        radius_faces_m=np.array([0.0, 0.1, 0.2]),
        pitch_faces=np.array([-1.0, 0.0, 1.0]),
        momentum_faces_mc=np.array([0.0, 1.0, 2.0]),
    )

    with pytest.raises(ValueError, match="shape"):
        grid.require_state("projection", np.ones((2, 2)))
    state = np.ones(grid.shape)
    state[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        grid.require_state("state", state)
