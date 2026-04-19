# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.state_space import FusionState


class _DummyKernel:
    def __init__(self) -> None:
        self.Psi = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 2.0, 0.6],
            ],
            dtype=np.float64,
        )
        self.R = np.array([5.8, 6.2, 6.6], dtype=np.float64)
        self.Z = np.array([-0.2, 0.2], dtype=np.float64)
        self.cfg = {"physics": {"plasma_current_target": 8.7}}

    def find_x_point(self, _psi: np.ndarray) -> tuple[tuple[float, float], None]:
        return (6.4, -0.1), None


def test_compute_bootstrap_fraction_clips_to_bounds() -> None:
    s = FusionState(beta_p=10.0)
    f_bs = s.compute_bootstrap_fraction(epsilon=1.0)
    assert 0.0 <= f_bs <= 0.9
    assert f_bs == 0.9


def test_to_vector_preserves_feature_order() -> None:
    s = FusionState(
        r_axis=6.2,
        z_axis=0.1,
        x_point_r=6.5,
        x_point_z=-0.2,
        ip_ma=8.7,
        q95=3.3,
    )
    vec = s.to_vector()
    assert vec.shape == (6,)
    assert np.allclose(vec, np.array([6.2, 0.1, 6.5, -0.2, 8.7, 3.3], dtype=np.float64))


def test_from_kernel_constructs_state() -> None:
    kernel = _DummyKernel()
    s = FusionState.from_kernel(kernel, time_s=0.125)
    assert s.r_axis == 6.2
    assert s.z_axis == 0.2
    assert s.x_point_r == 6.4
    assert s.x_point_z == -0.1
    assert s.ip_ma == 8.7
    assert s.time_s == 0.125


def test_construction_defaults() -> None:
    state = FusionState()
    assert state.ip_ma == 0.0
    assert state.beta_p == 0.0
    assert state.f_bs == 0.0


def test_to_vector_dtype() -> None:
    vec = FusionState(ip_ma=15.0, q95=3.0).to_vector()
    assert vec.shape == (6,)
    assert vec.dtype == np.float64


def test_compute_bootstrap_fraction_analytic() -> None:
    """f_bs = clip(0.4 * beta_p * sqrt(epsilon), 0, 0.9). Sauter-like scaling."""
    state = FusionState(beta_p=1.0)
    eps = 0.32
    result = state.compute_bootstrap_fraction(epsilon=eps)
    expected = 0.4 * 1.0 * np.sqrt(eps)
    assert result == pytest.approx(expected, rel=1e-10)
    assert state.f_bs == pytest.approx(expected, rel=1e-10)


def test_compute_bootstrap_fraction_beta_p_zero() -> None:
    state = FusionState(beta_p=0.0)
    assert state.compute_bootstrap_fraction() == 0.0
