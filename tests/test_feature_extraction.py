# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests: observation → unipolar feature extraction guards

from __future__ import annotations

import math

import pytest

from scpn_fusion.scpn.contracts import (
    ControlScales,
    ControlTargets,
    FeatureAxisSpec,
    extract_features,
)


class TestExtractFeaturesDefaults:
    """The default axes emit the four unipolar R/Z error features."""

    def test_returns_unipolar_axis_features(self) -> None:
        obs = {"R_axis_m": 6.2, "Z_axis_m": 0.0}
        features = extract_features(obs, ControlTargets(), ControlScales())
        assert set(features) == {"x_R_pos", "x_R_neg", "x_Z_pos", "x_Z_neg"}
        # On-target observation yields zero error on both polarities.
        assert all(0.0 <= v <= 1.0 for v in features.values())
        assert features["x_R_pos"] == pytest.approx(0.0)
        assert features["x_R_neg"] == pytest.approx(0.0)


class TestExtractFeaturesGuards:
    """Feature extraction rejects missing keys and non-finite axis targets."""

    def test_missing_observation_key_raises(self) -> None:
        with pytest.raises(
            KeyError, match=r"Missing observation key for feature extraction: R_axis_m"
        ):
            extract_features({}, ControlTargets(), ControlScales())

    def test_non_finite_axis_target_raises(self) -> None:
        axes = [
            FeatureAxisSpec(
                obs_key="k",
                target=math.inf,
                scale=1.0,
                pos_key="x_k_pos",
                neg_key="x_k_neg",
            )
        ]
        with pytest.raises(ValueError, match=r"Feature axis target must be finite: k"):
            extract_features({"k": 0.0}, ControlTargets(), ControlScales(), feature_axes=axes)

    def test_missing_passthrough_key_raises(self) -> None:
        obs = {"R_axis_m": 6.2, "Z_axis_m": 0.0}
        with pytest.raises(KeyError, match=r"Missing observation key for passthrough: extra"):
            extract_features(obs, ControlTargets(), ControlScales(), passthrough_keys=["extra"])

    def test_passthrough_key_is_clipped_into_features(self) -> None:
        obs = {"R_axis_m": 6.2, "Z_axis_m": 0.0, "extra": 0.4}
        features = extract_features(
            obs, ControlTargets(), ControlScales(), passthrough_keys=["extra"]
        )
        assert features["extra"] == pytest.approx(0.4)
