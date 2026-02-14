# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Fueling Mode Runtime Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Runtime summary and guard tests for fueling_mode entrypoint."""

from __future__ import annotations

import pytest

from scpn_fusion.control.fueling_mode import run_fueling_mode


def test_run_fueling_mode_returns_expected_summary() -> None:
    summary = run_fueling_mode(
        target_density=1.0,
        initial_density=0.84,
        steps=1600,
        dt_s=1e-3,
    )
    for key in (
        "target_density",
        "initial_density",
        "steps",
        "dt_s",
        "final_density",
        "final_abs_error",
        "rmse",
        "max_abs_command",
        "passes_thresholds",
    ):
        assert key in summary
    assert summary["steps"] == 1600
    assert summary["dt_s"] == 1e-3
    assert summary["max_abs_command"] <= 2.0 + 1e-12
    assert summary["min_density"] >= 0.0


def test_run_fueling_mode_is_deterministic_for_fixed_inputs() -> None:
    kwargs = dict(
        target_density=1.0,
        initial_density=0.83,
        steps=1200,
        dt_s=1e-3,
    )
    a = run_fueling_mode(**kwargs)
    b = run_fueling_mode(**kwargs)
    for key in (
        "final_density",
        "final_abs_error",
        "rmse",
        "max_abs_command",
        "min_density",
        "max_density",
    ):
        assert a[key] == pytest.approx(b[key], rel=0.0, abs=0.0)


def test_run_fueling_mode_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="target_density"):
        run_fueling_mode(target_density=0.0)
    with pytest.raises(ValueError, match="initial_density"):
        run_fueling_mode(initial_density=-0.1)
    with pytest.raises(ValueError, match="dt_s"):
        run_fueling_mode(dt_s=0.0)
