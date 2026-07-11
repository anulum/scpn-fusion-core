# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests: neural-transport MLP weight accessors

from __future__ import annotations

import numpy as np

from scpn_fusion.core._neural_transport_types import MLPWeights


class TestMLPWeightsAccessors:
    """The w1/b1..w3/b3 convenience properties index or fall back to empty arrays."""

    def test_empty_weights_return_empty_arrays(self) -> None:
        w = MLPWeights()

        assert w.n_layers == 0
        for matrix in (w.w1, w.w2, w.w3):
            assert matrix.shape == (0, 0)
        for vector in (w.b1, w.b2, w.b3):
            assert vector.shape == (0,)

    def test_populated_weights_expose_each_layer(self) -> None:
        layers_w = [np.ones((4, 3)), np.ones((2, 4)), np.ones((1, 2))]
        layers_b = [np.ones(3), np.ones(2), np.ones(1)]
        w = MLPWeights(layers_w=layers_w, layers_b=layers_b)

        assert w.n_layers == 3
        np.testing.assert_array_equal(w.w1, layers_w[0])
        np.testing.assert_array_equal(w.w2, layers_w[1])
        np.testing.assert_array_equal(w.w3, layers_w[2])
        np.testing.assert_array_equal(w.b1, layers_b[0])
        np.testing.assert_array_equal(w.b2, layers_b[1])
        np.testing.assert_array_equal(w.b3, layers_b[2])

    def test_single_layer_falls_back_for_deeper_accessors(self) -> None:
        w = MLPWeights(layers_w=[np.ones((4, 3))], layers_b=[np.ones(3)])

        assert w.n_layers == 1
        np.testing.assert_array_equal(w.w1, np.ones((4, 3)))
        # The second and third layers are absent, so their accessors fall back.
        assert w.w2.shape == (0, 0)
        assert w.w3.shape == (0, 0)
        assert w.b2.shape == (0,)
        assert w.b3.shape == (0,)
