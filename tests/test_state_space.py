from __future__ import annotations

import numpy as np

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
