"""Direct tests for extracted neural transport math helpers."""

from __future__ import annotations

import numpy as np

from scpn_fusion.core.neural_transport import MLPWeights
from scpn_fusion.core.neural_transport_math import _compute_nustar, _mlp_forward


def _weights() -> MLPWeights:
    rng = np.random.RandomState(7)
    hidden = 8
    return MLPWeights(
        layers_w=[
            rng.randn(10, hidden) * 0.05,
            rng.randn(hidden, hidden) * 0.05,
            rng.randn(hidden, 3) * 0.05,
        ],
        layers_b=[np.zeros(hidden), np.zeros(hidden), np.zeros(3)],
        input_mean=np.zeros(10),
        input_std=np.ones(10),
        output_scale=np.ones(3),
    )


def test_mlp_forward_batch_shape_and_nonnegative_output() -> None:
    out = _mlp_forward(np.ones((4, 10), dtype=np.float64), _weights())
    assert out.shape == (4, 3)
    assert np.all(out >= 0.0)


def test_compute_nustar_increases_with_density() -> None:
    low = _compute_nustar(te_kev=5.0, ne_19=2.0, q=1.6, rho=0.5)
    high = _compute_nustar(te_kev=5.0, ne_19=8.0, q=1.6, rho=0.5)
    assert low > 0.0
    assert high > low
