# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — RMSE Dashboard Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Unit tests for validation/rmse_dashboard.py helper functions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "rmse_dashboard.py"
SPEC = importlib.util.spec_from_file_location("rmse_dashboard", MODULE_PATH)
assert SPEC and SPEC.loader
rmse_dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rmse_dashboard)


def test_ipb98_regression_point() -> None:
    tau = rmse_dashboard.ipb98_tau_e(
        ip_ma=15.0,
        b_t=5.3,
        n_e19=10.1,
        p_loss_mw=85.0,
        r_m=6.2,
        kappa=1.7,
        epsilon=2.0 / 6.2,
        a_eff_amu=2.5,
    )
    assert abs(tau - 3.6643409641578857) < 1e-9


def test_rmse_basic() -> None:
    assert rmse_dashboard.rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    val = rmse_dashboard.rmse([1.0, 3.0], [2.0, 5.0])
    assert val == pytest.approx((2.5**0.5))


def test_rmse_raises_on_invalid_input() -> None:
    with pytest.raises(ValueError):
        rmse_dashboard.rmse([], [])
    with pytest.raises(ValueError):
        rmse_dashboard.rmse([1.0], [1.0, 2.0])


def test_sparc_axis_rmse_smoke() -> None:
    sparc_dir = ROOT / "validation" / "reference_data" / "sparc"
    result = rmse_dashboard.sparc_axis_rmse(sparc_dir)
    assert result["count"] >= 1
    assert result["axis_rmse_m"] >= 0.0
    assert len(result["rows"]) == result["count"]


def test_render_markdown_contains_sections() -> None:
    report = {
        "generated_at_utc": "2026-02-12T00:00:00+00:00",
        "runtime_seconds": 1.23,
        "confinement_itpa": {
            "count": 2,
            "tau_rmse_s": 0.1,
            "tau_mae_rel_pct": 5.0,
            "h98_rmse": 0.2,
        },
        "confinement_iter_sparc": {
            "count": 2,
            "tau_rmse_s": 0.3,
        },
        "beta_iter_sparc": {
            "count": 2,
            "beta_n_rmse": 0.4,
        },
        "sparc_axis": {
            "count": 5,
            "axis_rmse_m": 0.01,
        },
    }
    text = rmse_dashboard.render_markdown(report)
    assert "# SCPN RMSE Dashboard" in text
    assert "Confinement RMSE (ITPA H-mode)" in text
    assert "Beta_N RMSE (ITER + SPARC references)" in text
