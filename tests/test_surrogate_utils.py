# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests: shared surrogate training utilities

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core._surrogate_utils import AdamOptimizer, gelu, relative_l2


class TestActivationsAndMetrics:
    """The GELU activation and relative-L2 metric behave as documented."""

    def test_gelu_is_odd_symmetric_around_zero(self) -> None:
        x = np.array([-1.0, 0.0, 1.0])
        y = gelu(x)
        assert y[1] == pytest.approx(0.0)
        assert y[2] > 0.0 > y[0]

    def test_relative_l2_is_zero_for_identical_arrays(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert relative_l2(y, y) == pytest.approx(0.0, abs=1e-6)


class TestAdamOptimizerParameterStructures:
    """AdamOptimizer.step accepts matched dict or list params/grads, and rejects mismatches."""

    def test_dict_params_require_dict_grads(self) -> None:
        opt = AdamOptimizer()
        params = {"w": np.ones(2)}
        with pytest.raises(TypeError, match="grads must be a dict"):
            opt.step(params, [np.ones(2)], lr=0.01)

    def test_list_params_require_list_grads(self) -> None:
        opt = AdamOptimizer()
        params = [np.ones(2)]
        with pytest.raises(TypeError, match="grads must be a list"):
            opt.step(params, {"w": np.ones(2)}, lr=0.01)

    def test_list_params_are_updated_in_place(self) -> None:
        opt = AdamOptimizer()
        param = np.ones(3)
        opt.step([param], [np.full(3, 0.5)], lr=0.1)
        # A positive gradient drives the parameter downward from its initial value.
        assert np.all(param < 1.0)

    def test_dict_params_are_updated_in_place(self) -> None:
        opt = AdamOptimizer()
        param = np.ones(3)
        opt.step({"w": param}, {"w": np.full(3, 0.5)}, lr=0.1)
        assert np.all(param < 1.0)
