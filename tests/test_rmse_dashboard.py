# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — RMSE Dashboard Tests
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
    """The IPB98(y,2) scaling reproduces its regression value at the ITER point."""
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
    """RMSE is zero for equal series and matches a hand value otherwise."""
    assert rmse_dashboard.rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    val = rmse_dashboard.rmse([1.0, 3.0], [2.0, 5.0])
    assert val == pytest.approx(2.5**0.5)


def test_rmse_raises_on_invalid_input() -> None:
    """RMSE refuses empty or unequal-length series."""
    with pytest.raises(ValueError):
        rmse_dashboard.rmse([], [])
    with pytest.raises(ValueError):
        rmse_dashboard.rmse([1.0], [1.0, 2.0])


def test_sparc_axis_rmse_smoke() -> None:
    """The bundled SPARC files give a finite axis RMSE with one row each."""
    sparc_dir = ROOT / "validation" / "reference_data" / "sparc"
    result = rmse_dashboard.sparc_axis_rmse(sparc_dir)
    assert result["count"] >= 1
    assert result["axis_rmse_m"] >= 0.0
    assert len(result["rows"]) == result["count"]


def test_forward_diagnostics_rmse_includes_thomson_metrics() -> None:
    """Forward diagnostics report interferometer, Thomson and neutron errors."""
    result = rmse_dashboard.forward_diagnostics_rmse()
    assert result["count_interferometer_channels"] >= 1
    assert result["count_thomson_channels"] >= 1
    assert result["phase_rmse_rad"] >= 0.0
    assert result["neutron_rate_rel_error_pct"] >= 0.0
    assert result["thomson_voltage_rmse_v"] >= 0.0


def test_render_markdown_contains_sections() -> None:
    """The rendered report has its title and the confinement, beta_N and diagnostics sections."""
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
        "forward_diagnostics": {
            "count_interferometer_channels": 3,
            "count_thomson_channels": 3,
            "phase_rmse_rad": 1e-4,
            "neutron_rate_rel_error_pct": 2.0,
            "thomson_voltage_rmse_v": 2e-3,
        },
    }
    text = rmse_dashboard.render_markdown(report)
    assert "# SCPN RMSE Dashboard" in text
    assert "Confinement RMSE (ITPA H-mode)" in text
    assert "Beta_N RMSE (ITER + SPARC references)" in text
    assert "Forward Diagnostics RMSE" in text


def test_beta_rmse_rows_name_the_dynamic_model() -> None:
    """The maintained ITER and SPARC rows record the model that produced beta_N."""
    result = rmse_dashboard.beta_rmse_iter_sparc(
        ROOT / "validation" / "reference_data", ROOT / "validation"
    )

    assert [row["beta_n_model"] for row in result["rows"]] == ["dynamic_burn_model"] * 2


def test_beta_fallback_records_the_legacy_model_and_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing dynamic model leaves a record of the fallback instead of a silent switch."""
    import scpn_fusion.core.fusion_ignition_sim as ignition

    class FailingModel:
        """Stand-in for the dynamic burn model that refuses to build."""

        def __init__(self, **_: object) -> None:
            """Refuse every construction."""
            raise RuntimeError("dynamic model unavailable")

    monkeypatch.setattr(ignition, "DynamicBurnModel", FailingModel)
    reference = rmse_dashboard.load_json(
        ROOT / "validation" / "reference_data" / "iter_reference.json"
    )

    beta_n, metrics = rmse_dashboard.estimate_beta_n_from_burn(
        reference, ROOT / "validation" / "iter_validated_config.json"
    )

    assert metrics["beta_n_model"] == "legacy_fusion_burn_physics"
    assert metrics["fallback_reason"] == "RuntimeError: dynamic model unavailable"
    assert beta_n >= 0.0


def test_report_carries_a_real_fallback_into_json_and_markdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Through ``main``, a real legacy fallback keeps its reason and honest wording."""
    import json
    import sys

    import scpn_fusion.core.fusion_ignition_sim as ignition

    class FailingModel:
        """Stand-in for the dynamic burn model that refuses to build."""

        def __init__(self, **_: object) -> None:
            """Refuse every construction."""
            raise RuntimeError("dynamic model unavailable")

    monkeypatch.setattr(ignition, "DynamicBurnModel", FailingModel)
    monkeypatch.setattr(rmse_dashboard, "render_plots", lambda report, output_dir: [])
    out_json, out_md = tmp_path / "dashboard.json", tmp_path / "dashboard.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["rmse_dashboard.py", "--output-json", str(out_json), "--output-md", str(out_md)],
    )

    assert rmse_dashboard.main() == 0

    rows = json.loads(out_json.read_text(encoding="utf-8"))["beta_iter_sparc"]["rows"]
    assert [row["beta_n_model"] for row in rows] == ["legacy_fusion_burn_physics"] * 2
    assert [row["beta_n_fallback_reason"] for row in rows] == [
        "RuntimeError: dynamic model unavailable"
    ] * 2
    markdown = out_md.read_text(encoding="utf-8")
    assert (
        "legacy `FusionBurnPhysics` fallback, uncalibrated "
        "(RuntimeError: dynamic model unavailable)"
    ) in markdown
    assert "is not the calibrated model's fit residual" in markdown
    assert "is a fit residual, not a validation result" not in markdown


def test_beta_n_model_label_names_every_recorded_state() -> None:
    """Calibrated, fallback (with or without a reason) and unrecorded rows read differently."""
    label = rmse_dashboard._beta_n_model_label

    assert label({"beta_n_model": "dynamic_burn_model"}) == "`DynamicBurnModel` (calibrated)"
    assert (
        label({"beta_n_model": "legacy_fusion_burn_physics", "beta_n_fallback_reason": "E: x"})
        == "legacy `FusionBurnPhysics` fallback, uncalibrated (E: x)"
    )
    assert label({"beta_n_model": "legacy_fusion_burn_physics"}).endswith("(reason not recorded)")
    assert label({}) == "model not recorded"


def test_healthy_rows_keep_the_calibrated_wording() -> None:
    """With only dynamic-model rows the report keeps the fit-residual note and names the model."""
    report = rmse_dashboard.beta_rmse_iter_sparc(
        ROOT / "validation" / "reference_data", ROOT / "validation"
    )

    assert [row["beta_n_fallback_reason"] for row in report["rows"]] == [None, None]
    lines = []
    for row in report["rows"]:
        lines.append(rmse_dashboard._beta_n_model_label(row))
    assert lines == ["`DynamicBurnModel` (calibrated)"] * 2


def _report_with_beta(beta: dict[str, object]) -> dict[str, object]:
    """Build the smallest report the renderer accepts, around one beta_N section."""
    return {
        "generated_at_utc": "2026-11-19T00:00:00+00:00",
        "runtime_seconds": 1.0,
        "confinement_itpa": {
            "count": 2,
            "tau_rmse_s": 0.1,
            "tau_mae_rel_pct": 5.0,
            "h98_rmse": 0.2,
        },
        "confinement_iter_sparc": {"count": 2, "tau_rmse_s": 0.3},
        "beta_iter_sparc": beta,
        "sparc_axis": {"count": 5, "axis_rmse_m": 0.01},
    }


CALIBRATED_NOTE = "is a fit residual, not a validation result"
FALLBACK_ONLY_NOTE = "tracks the uncalibrated fallback only"
MIXED_NOTE = "is neither the calibrated model's fit residual nor a fallback-only figure"
UNKNOWN_NOTE = "cannot be attributed to a model"


def test_beta_n_aggregate_model_classifies_every_state() -> None:
    """All-dynamic, all-fallback, mixed and unknown aggregates are told apart."""
    classify = rmse_dashboard.beta_n_aggregate_model
    dynamic = {"beta_n_model": "dynamic_burn_model"}
    fallback = {"beta_n_model": "legacy_fusion_burn_physics"}

    assert classify([dynamic, dynamic]) == "all_dynamic"
    assert classify([fallback, fallback]) == "all_fallback"
    assert classify([dynamic, fallback]) == "mixed"
    assert classify([]) == "unknown"
    assert classify([{}, {}]) == "unknown"
    assert classify([dynamic, {}]) == "unknown"
    assert classify([dynamic, {"beta_n_model": "other"}]) == "unknown"
    assert classify([dynamic, "not a row"]) == "unknown"


def test_mixed_models_reach_the_report_entry_and_its_markdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One calibrated and one fallback scenario: pooled RMSE kept, wording names the mix."""
    estimates = {
        "iter_validated_config.json": (1.8, {"beta_n_model": "dynamic_burn_model"}),
        "sparc_config.json": (
            2.4,
            {"beta_n_model": "legacy_fusion_burn_physics", "fallback_reason": "E: down"},
        ),
    }

    def fake_estimate(
        reference: dict[str, object], config_path: Path
    ) -> tuple[float, dict[str, object]]:
        """Return a fixed estimate per configuration, with its recorded model."""
        beta_n, metrics = estimates[config_path.name]
        return beta_n, {"Q": 1.0, "P_fusion_MW": 1.0, **metrics}

    monkeypatch.setattr(rmse_dashboard, "estimate_beta_n_from_burn", fake_estimate)
    beta = rmse_dashboard.beta_rmse_iter_sparc(
        ROOT / "validation" / "reference_data", ROOT / "validation"
    )

    measured = [row["beta_n_measured"] for row in beta["rows"]]
    assert beta["beta_n_rmse"] == rmse_dashboard.rmse(measured, [1.8, 2.4])
    assert beta["beta_n_aggregate_model"] == "mixed"
    markdown = rmse_dashboard.render_markdown(_report_with_beta(beta))
    assert "- Aggregate model attribution: `mixed`" in markdown
    assert MIXED_NOTE in markdown
    assert (
        "mix models: `ITER_15MA_baseline` from the calibrated `DynamicBurnModel` and "
        "`SPARC_V2C` from the uncalibrated legacy `FusionBurnPhysics` fallback"
    ) in markdown
    assert "(E: down)" in markdown
    assert CALIBRATED_NOTE not in markdown
    assert FALLBACK_ONLY_NOTE not in markdown


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"scenario": "ITER_15MA_baseline"}, {"scenario": "SPARC_V2C"}],
        [
            {"scenario": "ITER_15MA_baseline", "beta_n_model": "dynamic_burn_model"},
            {"scenario": "SPARC_V2C"},
        ],
    ],
)
def test_unrecorded_models_never_get_the_calibrated_wording(rows: list[dict[str, str]]) -> None:
    """Rows without a recorded model, the tracked report's old shape, are not called calibrated."""
    markdown = rmse_dashboard.render_markdown(
        _report_with_beta({"count": 2, "beta_n_rmse": 0.09, "rows": rows})
    )

    assert "- Aggregate model attribution: `unknown`" in markdown
    assert UNKNOWN_NOTE in markdown
    assert CALIBRATED_NOTE not in markdown
    if rows:
        assert "for `SPARC_V2C`" in markdown
    else:
        assert "(no beta_N rows were recorded)" in markdown


def test_only_an_all_dynamic_aggregate_keeps_the_calibrated_wording() -> None:
    """The fit-residual wording appears for a nonempty, fully dynamic aggregate and no other."""
    dynamic = [
        {"scenario": "ITER_15MA_baseline", "beta_n_model": "dynamic_burn_model"},
        {"scenario": "SPARC_V2C", "beta_n_model": "dynamic_burn_model"},
    ]
    fallback = [{**row, "beta_n_model": "legacy_fusion_burn_physics"} for row in dynamic]

    calibrated = rmse_dashboard.render_markdown(
        _report_with_beta({"count": 2, "beta_n_rmse": 0.09, "rows": dynamic})
    )
    fallback_only = rmse_dashboard.render_markdown(
        _report_with_beta({"count": 2, "beta_n_rmse": 0.09, "rows": fallback})
    )

    assert "- Aggregate model attribution: `all_dynamic`" in calibrated
    assert CALIBRATED_NOTE in calibrated
    assert "- Aggregate model attribution: `all_fallback`" in fallback_only
    assert FALLBACK_ONLY_NOTE in fallback_only
    assert CALIBRATED_NOTE not in fallback_only
