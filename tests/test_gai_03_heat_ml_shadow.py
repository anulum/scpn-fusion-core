# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GAI-03 HEAT-ML Shadow Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for GAI-03 HEAT-ML shadow surrogate and scanner integration."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from scpn_fusion.core.global_design_scanner import GlobalDesignExplorer
from scpn_fusion.core.heat_ml_shadow_surrogate import generate_shadow_dataset


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "gai_03_heat_ml_shadow.py"
SPEC = importlib.util.spec_from_file_location("gai_03_heat_ml_shadow", MODULE_PATH)
assert SPEC and SPEC.loader
gai_03_heat_ml_shadow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gai_03_heat_ml_shadow)


def test_shadow_dataset_is_deterministic() -> None:
    a = generate_shadow_dataset(seed=123, samples=64)
    b = generate_shadow_dataset(seed=123, samples=64)
    np.testing.assert_allclose(a.features, b.features, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(a.shadow_fraction, b.shadow_fraction, rtol=0.0, atol=0.0)


def test_scanner_integration_reduces_divertor_load() -> None:
    explorer = GlobalDesignExplorer("dummy")
    out = explorer.evaluate_design(R_maj=1.9, B_field=8.6, I_plasma=15.0)
    assert out["Shadow_Fraction"] >= 0.0
    assert out["Shadow_Fraction"] <= 0.85
    assert out["Div_Load_Optimized"] <= out["Div_Load_Baseline"]


def test_campaign_meets_thresholds_smoke() -> None:
    out = gai_03_heat_ml_shadow.run_campaign(
        seed=42, train_samples=1024, eval_samples=320, scan_samples=260
    )
    assert out["rmse_pct"] <= 10.0
    assert out["inference_seconds_200k"] <= 1.0
    assert out["mean_divertor_reduction_pct"] >= 8.0
    assert out["passes_thresholds"] is True
