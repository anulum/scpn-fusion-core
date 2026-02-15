# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Task 4 Quasi-3D Modeling Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for validation/task4_quasi_3d_modeling.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "task4_quasi_3d_modeling.py"
SPEC = importlib.util.spec_from_file_location("task4_quasi_3d_modeling", MODULE_PATH)
assert SPEC and SPEC.loader
task4_quasi_3d_modeling = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(task4_quasi_3d_modeling)


def test_task4_campaign_passes_thresholds_smoke() -> None:
    out = task4_quasi_3d_modeling.run_campaign(
        seed=42,
        quasi_3d_samples=256,
        hall_grid=16,
        hall_steps=24,
        toroidal_points=24,
        tbr_thickness_cm=260.0,
        asdex_erosion_ref_mm_year=0.25,
    )
    assert out["passes_thresholds"] is True
    assert out["quasi_3d"]["force_balance_rmse_pct"] <= 8.0
    assert out["quasi_3d"]["force_residual_p95_pct"] <= 12.0
    assert out["divertor_two_fluid"]["two_fluid_index"] >= 0.10
    assert out["divertor_two_fluid"]["two_fluid_temp_split_index"] > 0.0
    assert out["jet_heat_flux_validation"]["rmse_pct"] <= 15.0
    assert out["jet_heat_flux_validation"]["jet_file_count"] >= 1
    assert out["pwi_tbr_calibration"]["erosion_curve_rmse_pct"] <= 35.0
    assert out["pwi_tbr_calibration"]["calibrated_tbr"] <= 1.10


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"quasi_3d_samples": 32}, "quasi_3d_samples"),
        ({"hall_grid": 4}, "hall_grid"),
        ({"hall_steps": 0}, "hall_steps"),
        ({"toroidal_points": 8}, "toroidal_points"),
        ({"tbr_thickness_cm": 0.0}, "tbr_thickness_cm"),
        ({"asdex_erosion_ref_mm_year": 0.0}, "asdex_erosion_ref_mm_year"),
    ],
)
def test_task4_campaign_rejects_invalid_inputs(
    kwargs: dict[str, float | int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        task4_quasi_3d_modeling.run_campaign(**kwargs)


def test_task4_render_markdown_contains_required_sections() -> None:
    report = task4_quasi_3d_modeling.generate_report(
        seed=7,
        quasi_3d_samples=128,
        hall_grid=12,
        hall_steps=16,
        toroidal_points=20,
        tbr_thickness_cm=260.0,
        asdex_erosion_ref_mm_year=0.25,
    )
    text = task4_quasi_3d_modeling.render_markdown(report)
    assert "# Task 4 Quasi-3D Modeling Report" in text
    assert "Quasi-3D Force Balance" in text
    assert "Hall-MHD + TEMHD Coupling" in text
    assert "JET / SOLPS-ITER Proxy Heat Flux" in text
    assert "Erosion-Calibrated TBR Guard" in text
