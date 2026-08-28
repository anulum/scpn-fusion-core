# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — HEAT-ML Magnetic-Shadow Benchmark Tests
"""Real-surface tests for the HEAT-ML magnetic-shadow scanner benchmark."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

from scpn_fusion.core.global_design_scanner import GlobalDesignExplorer
from scpn_fusion.core.heat_ml_shadow_surrogate import generate_shadow_dataset

pytestmark = pytest.mark.experimental


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "heat_ml_magnetic_shadow_benchmark.py"
SPEC = importlib.util.spec_from_file_location("heat_ml_magnetic_shadow_benchmark", MODULE_PATH)
assert SPEC and SPEC.loader
benchmark_cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark_cli
SPEC.loader.exec_module(benchmark_cli)


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
    out = benchmark_cli.run_campaign(
        seed=42, train_samples=1024, eval_samples=320, scan_samples=260
    )
    assert out["rmse_pct"] <= 10.0
    assert out["inference_seconds_200k"] <= 1.0
    assert out["mean_divertor_reduction_pct"] >= 8.0
    assert out["passes_thresholds"] is True


def test_campaign_does_not_mutate_global_numpy_rng_state() -> None:
    np.random.seed(31415)
    state = np.random.get_state()

    _ = benchmark_cli.run_campaign(seed=11, train_samples=512, eval_samples=192, scan_samples=160)

    observed = float(np.random.random())
    np.random.set_state(state)
    expected = float(np.random.random())
    assert observed == expected


def test_report_contract_and_markdown_are_descriptive() -> None:
    report = benchmark_cli.generate_report(
        seed=17,
        train_samples=256,
        eval_samples=96,
        scan_samples=80,
    )
    benchmark = benchmark_cli.validate_report(report)
    markdown = benchmark_cli.render_markdown(report)
    assert report["schema_version"] == 2
    assert report["report_kind"] == "heat_ml_magnetic_shadow_benchmark"
    assert benchmark["passes_thresholds"] is True
    assert "HEAT-ML Magnetic-Shadow Design-Scanner Benchmark" in markdown
    assert "Threshold pass: `YES`" in markdown


@pytest.mark.parametrize(
    "threshold_override",
    [
        {"rmse_threshold_pct": -1.0},
        {"inference_threshold_seconds": -1.0},
        {"reduction_threshold_pct": 101.0},
    ],
)
def test_report_threshold_failures_are_explicit(
    threshold_override: dict[str, float],
) -> None:
    report = benchmark_cli.generate_report(
        seed=19,
        train_samples=128,
        eval_samples=64,
        scan_samples=64,
        **threshold_override,
    )
    assert benchmark_cli.validate_report(report)["passes_thresholds"] is False
    assert "Threshold pass: `NO`" in benchmark_cli.render_markdown(report)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": 1}, "unsupported report schema_version"),
        ({"report_kind": "obsolete"}, "unsupported report_kind"),
        ({"generated_at_utc": ""}, "generated_at_utc must be"),
        ({"generated_at_utc": None}, "generated_at_utc must be"),
        ({"heat_ml_magnetic_shadow_benchmark": []}, "must be an object"),
    ],
)
def test_report_contract_rejects_invalid_current_payloads(
    change: dict[str, Any], message: str
) -> None:
    report = benchmark_cli.generate_report(
        seed=23,
        train_samples=128,
        eval_samples=64,
        scan_samples=64,
    )
    report.update(change)
    with pytest.raises(ValueError, match=message):
        benchmark_cli.validate_report(report)


def test_report_contract_rejects_obsolete_coded_payload_exactly() -> None:
    stale_report = {
        "generated_at_utc": "2026-08-26T00:00:00+00:00",
        "gai_03": {"passes_thresholds": True},
    }
    with pytest.raises(ValueError, match="current descriptive contract"):
        benchmark_cli.validate_report(stale_report)


def test_cli_writes_current_contract_and_enforces_strict_thresholds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_json = tmp_path / "benchmark.json"
    output_md = tmp_path / "benchmark.md"
    common_args = [
        "--seed",
        "29",
        "--train-samples",
        "128",
        "--eval-samples",
        "64",
        "--scan-samples",
        "64",
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
    ]

    assert benchmark_cli.main(common_args) == 0
    assert benchmark_cli.main([*common_args, "--strict"]) == 0
    written = json.loads(output_json.read_text(encoding="utf-8"))
    benchmark_cli.validate_report(written)
    assert output_md.read_text(encoding="utf-8").startswith(
        "# HEAT-ML Magnetic-Shadow Design-Scanner Benchmark"
    )
    assert "benchmark complete" in capsys.readouterr().out

    assert (
        benchmark_cli.main(
            [
                *common_args,
                "--strict",
                "--rmse-threshold-pct",
                "-1",
            ]
        )
        == 2
    )
