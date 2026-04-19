# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests
"""MC-Dropout uncertainty tests for predict_disruption_risk_safe."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch required for MC-dropout tests")

from scpn_fusion.control.disruption_predictor import predict_disruption_risk_safe


def test_mc_dropout_confidence_bounds(tmp_path):
    """predict_disruption_risk_safe returns bounded confidence and non-negative uncertainty."""
    signal = np.random.default_rng(42).standard_normal(200)
    model_path = tmp_path / "mc_test_model.pth"
    risk, meta = predict_disruption_risk_safe(
        signal,
        model_path=model_path,
        seq_len=50,
        train_if_missing=True,
        allow_fallback=True,
    )
    if meta.get("mode") == "fallback":
        pytest.skip("Model training fell back; cannot test MC-dropout path")
    assert 0.0 <= risk <= 1.0
    assert 0.0 <= meta["confidence_score"] <= 1.0
    assert meta["uncertainty_std"] >= 0.0
    assert meta["risk_source"] == "transformer_mc_dropout"


def test_mc_dropout_no_checkpoint_raises(tmp_path):
    """FileNotFoundError when allow_fallback=False and no checkpoint exists."""
    signal = np.random.default_rng(7).standard_normal(200)
    bogus_path = tmp_path / "nonexistent_model.pth"
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        predict_disruption_risk_safe(
            signal,
            model_path=bogus_path,
            train_if_missing=False,
            allow_fallback=False,
        )
