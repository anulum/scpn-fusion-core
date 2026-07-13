# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — HEAT-ML Shadow Surrogate (GAI-03)
"""Unit guards and fit/predict sanity for the HEAT-ML shadow surrogate.

The GAI-03 campaign test exercises the surrogate through the design scanner;
these tests pin the surrogate's own validation branches and the deterministic
ridge fit/predict contract directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.heat_ml_shadow_surrogate import (
    HeatMLShadowSurrogate,
    generate_shadow_dataset,
    rmse_percent,
    synthetic_shadow_reference,
)


class TestInputGuards:
    """Reject malformed feature/sample/shape inputs."""

    def test_reference_rejects_wrong_feature_width(self) -> None:
        with pytest.raises(ValueError, match=r"Expected shape \(N, 7\)"):
            synthetic_shadow_reference(np.zeros((2, 5), dtype=np.float64))

    def test_generate_dataset_rejects_too_few_samples(self) -> None:
        with pytest.raises(ValueError, match="samples must be >= 8"):
            generate_shadow_dataset(seed=0, samples=4)

    def test_fit_rejects_row_count_mismatch(self) -> None:
        dataset = generate_shadow_dataset(seed=1, samples=8)
        model = HeatMLShadowSurrogate()
        with pytest.raises(ValueError, match="row count mismatch"):
            model.fit(dataset.features, np.zeros(5, dtype=np.float64))

    def test_predict_before_fit_raises(self) -> None:
        model = HeatMLShadowSurrogate()
        with pytest.raises(RuntimeError, match="not fit"):
            model.predict_shadow_fraction(np.zeros((1, 7), dtype=np.float64))

    def test_rmse_percent_rejects_empty_or_mismatched(self) -> None:
        with pytest.raises(ValueError, match="non-empty and same shape"):
            rmse_percent(np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64))
        with pytest.raises(ValueError, match="non-empty and same shape"):
            rmse_percent(
                np.asarray([0.1, 0.2], dtype=np.float64),
                np.asarray([0.1], dtype=np.float64),
            )


class TestFitPredictContract:
    """The ridge fit is deterministic and predictions stay in the clipped band."""

    def test_fit_synthetic_predicts_within_band(self) -> None:
        model = HeatMLShadowSurrogate()
        model.fit_synthetic(seed=42, samples=256)
        dataset = generate_shadow_dataset(seed=7, samples=32)
        preds = model.predict_shadow_fraction(dataset.features)
        assert preds.shape == (32,)
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 0.85)
        # A ridge fit on the synthetic law tracks it to single-digit RMSE percent.
        assert rmse_percent(dataset.shadow_fraction, preds) < 25.0

    def test_divertor_flux_attenuates_and_stays_positive(self) -> None:
        model = HeatMLShadowSurrogate()
        model.fit_synthetic(seed=3, samples=128)
        dataset = generate_shadow_dataset(seed=9, samples=16)
        baseline = 1.0e7
        flux = model.predict_divertor_flux(baseline, dataset.features)
        assert flux.shape == (16,)
        assert np.all(flux > 0.0)
        assert np.all(flux <= baseline)

    def test_fit_is_deterministic(self) -> None:
        dataset = generate_shadow_dataset(seed=11, samples=64)
        preds = []
        for _ in range(2):
            model = HeatMLShadowSurrogate()
            model.fit(dataset.features, dataset.shadow_fraction)
            preds.append(model.predict_shadow_fraction(dataset.features))
        assert np.array_equal(preds[0], preds[1])
