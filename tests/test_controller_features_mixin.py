# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the neuro-symbolic controller feature/action mixin.

The mixin is designed to be composed onto a controller that supplies the
pre-sized scratch buffers and axis/injection/action configuration. These tests
drive it through a controlled state double to exercise feature extraction,
feature-dict assembly, place injection, action decoding, and the deterministic
bit-flip fault model, including every empty-configuration and missing-key guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.scpn.controller_features_mixin import NeuroSymbolicControllerFeaturesMixin


class _Controller(NeuroSymbolicControllerFeaturesMixin):
    """Minimal state double supplying the buffers the mixin operates on."""

    def __init__(self) -> None:
        self._axis_count = 2
        self._empty = np.zeros(0, dtype=np.float64)
        self._tmp_obs_vals = np.zeros(2, dtype=np.float64)
        self._axis_obs_keys = ["a", "b"]
        self._axis_targets = np.array([1.0, 2.0], dtype=np.float64)
        self._axis_scales = np.array([1.0, 1.0], dtype=np.float64)
        self._tmp_feature_err = np.zeros(2, dtype=np.float64)
        self._tmp_feature_pos = np.zeros(2, dtype=np.float64)
        self._tmp_feature_neg = np.zeros(2, dtype=np.float64)
        self._axis_pos_keys = ["a_pos", "b_pos"]
        self._axis_neg_keys = ["a_neg", "b_neg"]
        self._passthrough_sources = ["c"]

        # One axis-sourced injection and one passthrough injection.
        self._inj_count = 2
        self._tmp_inj_values = np.zeros(2, dtype=np.float64)
        self._inj_source_axis_idx = np.array([0, -1], dtype=np.int64)
        self._inj_source_axis_pos = np.array([True, False], dtype=np.bool_)
        self._inj_passthrough_pairs = [(1, "c")]
        self._inj_scales = np.array([1.0, 1.0], dtype=np.float64)
        self._inj_offsets = np.array([0.0, 0.0], dtype=np.float64)
        self._inj_has_clamp = True
        self._inj_clamp_idx = np.array([0, 1], dtype=np.int64)
        self._inj_place_ids = np.array([0, 1], dtype=np.int64)

        self._action_names = ["act0", "act1"]
        self._action_count = 2
        self._tmp_actions = np.zeros(2, dtype=np.float64)
        self._action_pos_idx = np.array([0, 1], dtype=np.int64)
        self._action_neg_idx = np.array([2, 3], dtype=np.int64)
        self._action_gains = np.array([1.0, 1.0], dtype=np.float64)
        self._prev_actions = np.zeros(2, dtype=np.float64)
        self._action_max_delta = np.array([10.0, 10.0], dtype=np.float64)
        self._action_abs_max = np.array([1.0, 1.0], dtype=np.float64)
        self._sc_bitflip_rate = 0.0


class TestFeatureComponents:
    """Positive/negative error-feature extraction."""

    def test_from_obs_map_splits_pos_neg(self) -> None:
        """Observations below/above target map to positive/negative features."""
        ctrl = _Controller()
        pos, neg = ctrl._compute_feature_components({"a": 0.0, "b": 3.0})
        # a is below its target (error +1 -> pos), b is above (error -1 -> neg).
        assert pos[0] == 1.0
        assert neg[1] == 1.0

    def test_missing_obs_key_raises(self) -> None:
        """A missing observation key is rejected."""
        ctrl = _Controller()
        with pytest.raises(KeyError, match="feature extraction"):
            ctrl._compute_feature_components({"a": 0.0})

    def test_zero_axis_returns_empty(self) -> None:
        """A controller with no axes returns the shared empty buffers."""
        ctrl = _Controller()
        ctrl._axis_count = 0
        pos, neg = ctrl._compute_feature_components({})
        assert pos.size == 0
        assert neg.size == 0

    def test_vector_zero_axis_returns_empty(self) -> None:
        """The vector path also short-circuits when there are no axes."""
        ctrl = _Controller()
        ctrl._axis_count = 0
        pos, neg = ctrl._compute_feature_components_vector(np.zeros(0))
        assert pos.size == 0
        assert neg.size == 0

    def test_vector_shape_mismatch_raises(self) -> None:
        """A vector whose length disagrees with the axis count is rejected."""
        ctrl = _Controller()
        with pytest.raises(ValueError, match="must have length"):
            ctrl._compute_feature_components_vector(np.zeros(3))


class TestFeatureDict:
    """Feature-dictionary assembly with passthrough sources."""

    def test_build_dict_includes_axes_and_passthrough(self) -> None:
        """The feature dict carries every pos/neg axis key plus clamped passthroughs."""
        ctrl = _Controller()
        pos = np.array([0.2, 0.4], dtype=np.float64)
        neg = np.array([0.1, 0.3], dtype=np.float64)
        feats = ctrl._build_feature_dict({"c": 2.0}, pos, neg)
        assert feats["a_pos"] == 0.2
        assert feats["b_neg"] == 0.3
        # Passthrough value is clamped into [0, 1].
        assert feats["c"] == 1.0

    def test_build_dict_missing_passthrough_raises(self) -> None:
        """A missing passthrough source is rejected."""
        ctrl = _Controller()
        with pytest.raises(KeyError, match="passthrough"):
            ctrl._build_feature_dict({}, np.zeros(2), np.zeros(2))


class TestInjectPlaces:
    """Place-injection into a marking vector."""

    def test_inject_writes_axis_and_passthrough_values(self) -> None:
        """Injection writes axis-sourced and passthrough features into the marking."""
        ctrl = _Controller()
        marking = np.zeros(4, dtype=np.float64)
        pos = np.array([0.6, 0.0], dtype=np.float64)
        neg = np.array([0.0, 0.0], dtype=np.float64)
        ctrl._inject_places(marking, {"c": 0.5}, pos, neg)
        assert marking[0] == 0.6
        assert marking[1] == 0.5

    def test_inject_missing_passthrough_raises(self) -> None:
        """A missing passthrough observation for injection is rejected."""
        ctrl = _Controller()
        marking = np.zeros(4, dtype=np.float64)
        with pytest.raises(KeyError, match="passthrough"):
            ctrl._inject_places(marking, {}, np.zeros(2), np.zeros(2))

    def test_inject_noop_when_no_injections(self) -> None:
        """With no injections configured the marking is left untouched."""
        ctrl = _Controller()
        ctrl._inj_count = 0
        marking = np.ones(4, dtype=np.float64)
        ctrl._inject_places(marking, {}, np.zeros(2), np.zeros(2))
        assert np.all(marking == 1.0)


class TestDecodeActions:
    """Action decoding from a marking vector."""

    def test_decode_returns_named_actions(self) -> None:
        """Decoding returns one entry per configured action name."""
        ctrl = _Controller()
        marking = np.array([1.0, 0.5, 0.0, 0.0], dtype=np.float64)
        actions = ctrl._decode_actions(marking)
        assert set(actions) == {"act0", "act1"}
        assert actions["act0"] == 1.0  # clamped to abs-max

    def test_decode_empty_when_no_actions(self) -> None:
        """A controller with no actions decodes to an empty mapping."""
        ctrl = _Controller()
        ctrl._action_count = 0
        actions = ctrl._decode_actions(np.zeros(4))
        assert actions == {}


class TestBitFlipFaults:
    """Deterministic bit-flip fault model."""

    def test_zero_rate_returns_copy_unchanged(self) -> None:
        """A non-positive flip rate returns the values unchanged."""
        ctrl = _Controller()
        values = np.array([0.2, 0.4], dtype=np.float64)
        out = ctrl._apply_bit_flip_faults(values, np.random.default_rng(0))
        assert np.array_equal(out, values)

    def test_full_rate_flips_within_unit_interval(self) -> None:
        """A unit flip rate perturbs every entry but keeps the output in [0, 1]."""
        ctrl = _Controller()
        ctrl._sc_bitflip_rate = 1.0
        values = np.array([0.2, 0.4, 0.6], dtype=np.float64)
        out = ctrl._apply_bit_flip_faults(values, np.random.default_rng(1))
        assert out.shape == values.shape
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)

    def test_rate_without_flips_returns_copy(self) -> None:
        """A tiny flip rate that selects no bits returns the values unchanged."""
        ctrl = _Controller()
        ctrl._sc_bitflip_rate = 1e-12
        values = np.array([0.3, 0.7], dtype=np.float64)
        out = ctrl._apply_bit_flip_faults(values, np.random.default_rng(2))
        assert np.array_equal(out, values)
