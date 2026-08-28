# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — GyroSwin-Like Turbulence Surrogate Benchmark Tests
"""Real-surface tests for the GyroSwin-like turbulence surrogate benchmark."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

from scpn_fusion.core.gyro_swin_surrogate import (
    GyroSwinLikeSurrogate,
    benchmark_speedup,
    gene_proxy_predict,
    generate_synthetic_gyrokinetic_dataset,
    rmse_percent,
    synthetic_core_turbulence_target,
)

pytestmark = pytest.mark.experimental


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "gyro_swin_turbulence_surrogate_benchmark.py"
SPEC = importlib.util.spec_from_file_location(
    "gyro_swin_turbulence_surrogate_benchmark", MODULE_PATH
)
assert SPEC and SPEC.loader
benchmark_cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark_cli
SPEC.loader.exec_module(benchmark_cli)


def test_dataset_is_deterministic_for_seed() -> None:
    a = generate_synthetic_gyrokinetic_dataset(seed=123, samples=32)
    b = generate_synthetic_gyrokinetic_dataset(seed=123, samples=32)
    np.testing.assert_allclose(a.features, b.features, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(a.chi_i, b.chi_i, rtol=0.0, atol=0.0)


def test_dataset_and_feature_contract_reject_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="samples must be >= 8"):
        generate_synthetic_gyrokinetic_dataset(seed=1, samples=7)
    with pytest.raises(ValueError, match=r"shape \(N, 10\)"):
        synthetic_core_turbulence_target(np.ones((2, 9), dtype=np.float64))


def test_surrogate_public_fit_predict_contract() -> None:
    dataset = generate_synthetic_gyrokinetic_dataset(seed=17, samples=48)
    surrogate = GyroSwinLikeSurrogate(hidden_dim=8, ridge=-1.0, seed=17)
    assert surrogate.ridge == pytest.approx(1e-10)

    with pytest.raises(RuntimeError, match="not fit"):
        surrogate.predict(dataset.features[:1])
    with pytest.raises(ValueError, match="feature/target rows"):
        surrogate.fit(dataset.features, dataset.chi_i[:-1])

    surrogate.fit(dataset.features, dataset.chi_i)
    prediction = surrogate.predict(dataset.features[0])
    assert prediction.shape == (1,)
    assert np.all(prediction > 0.0)

    with pytest.raises(ValueError, match="hidden_dim must be >= 8"):
        GyroSwinLikeSurrogate(hidden_dim=7)


def test_reference_proxy_and_error_metric_contracts() -> None:
    dataset = generate_synthetic_gyrokinetic_dataset(seed=23, samples=8)
    no_iteration = gene_proxy_predict(dataset.features, iterations=0)
    one_iteration = gene_proxy_predict(dataset.features, iterations=1)
    assert no_iteration.shape == one_iteration.shape == (8,)
    assert np.all(no_iteration > 0.0)
    assert np.all(one_iteration > 0.0)

    assert rmse_percent(dataset.chi_i, dataset.chi_i) == pytest.approx(0.0)
    assert rmse_percent(np.zeros(2), np.ones(2)) == pytest.approx(1e11)
    with pytest.raises(ValueError, match="non-empty and same shape"):
        rmse_percent(np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64))
    with pytest.raises(ValueError, match="non-empty and same shape"):
        rmse_percent(np.ones(2), np.ones(3))


def test_speed_benchmark_public_surface() -> None:
    dataset = generate_synthetic_gyrokinetic_dataset(seed=29, samples=32)
    surrogate = GyroSwinLikeSurrogate(hidden_dim=8, seed=29)
    surrogate.fit(dataset.features, dataset.chi_i)
    result = benchmark_speedup(
        dataset.features,
        surrogate,
        min_baseline_s=0.0,
        min_surrogate_s=0.0,
    )
    assert result.gene_proxy_s_per_sample > 0.0
    assert result.surrogate_s_per_sample > 0.0
    assert result.speedup > 0.0

    loop_limited = benchmark_speedup(
        dataset.features[:1],
        surrogate,
        min_baseline_s=float("inf"),
        min_surrogate_s=float("inf"),
    )
    assert loop_limited.gene_proxy_s_per_sample > 0.0
    assert loop_limited.surrogate_s_per_sample > 0.0


def test_campaign_meets_thresholds_smoke() -> None:
    out = benchmark_cli.run_campaign(
        seed=42,
        train_samples=1024,
        eval_samples=256,
        benchmark_samples=64,
        speedup_threshold=50.0,
    )
    assert out["rmse_pct"] <= 10.0
    # Speedup is hardware-dependent; smoke test uses small benchmark_samples
    # so require only 50x (the campaign default threshold of 1000x targets
    # larger sample sizes on production hardware).
    assert out["speedup_vs_gene_proxy"] >= 50.0


def test_render_markdown_contains_key_sections() -> None:
    report = benchmark_cli.generate_report(
        seed=11,
        train_samples=512,
        eval_samples=192,
        benchmark_samples=64,
        speedup_threshold=0.0,
    )
    text = benchmark_cli.render_markdown(report)
    assert report["schema_version"] == benchmark_cli.REPORT_SCHEMA_VERSION
    assert report["report_kind"] == benchmark_cli.REPORT_KIND
    assert "turbulence_surrogate_benchmark" in report
    assert "GyroSwin-Like Turbulence Surrogate Benchmark" in text
    assert "RMSE (% of mean target)" in text
    assert "Speedup vs GENE-like proxy" in text
    assert "Threshold pass: `YES`" in text


def test_render_markdown_shows_failed_threshold() -> None:
    report = benchmark_cli.generate_report(
        seed=13,
        train_samples=256,
        eval_samples=96,
        benchmark_samples=32,
        rmse_threshold_pct=-1.0,
    )
    assert "Threshold pass: `NO`" in benchmark_cli.render_markdown(report)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": 1}, "unsupported report schema_version"),
        ({"report_kind": "obsolete"}, "unsupported report_kind"),
        ({"generated_at_utc": ""}, "generated_at_utc must be"),
        ({"turbulence_surrogate_benchmark": []}, "must be an object"),
    ],
)
def test_report_contract_rejects_invalid_current_payloads(
    change: dict[str, Any], message: str
) -> None:
    report = benchmark_cli.generate_report(
        seed=31,
        train_samples=128,
        eval_samples=64,
        benchmark_samples=32,
        speedup_threshold=0.0,
    )
    report.update(change)
    with pytest.raises(ValueError, match=message):
        benchmark_cli.validate_report(report)


def test_report_contract_rejects_obsolete_coded_payload_exactly() -> None:
    stale_report = {
        "generated_at_utc": "2026-08-26T00:00:00+00:00",
        "gai_01": {"passes_thresholds": True},
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
        "37",
        "--train-samples",
        "128",
        "--eval-samples",
        "64",
        "--benchmark-samples",
        "32",
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
        "--strict",
    ]

    assert (
        benchmark_cli.main(
            [
                *common_args,
                "--rmse-threshold-pct",
                "100",
                "--speedup-threshold",
                "0",
            ]
        )
        == 0
    )
    written = benchmark_cli.json.loads(output_json.read_text(encoding="utf-8"))
    benchmark_cli.validate_report(written)
    assert output_md.read_text(encoding="utf-8").startswith(
        "# GyroSwin-Like Turbulence Surrogate Benchmark"
    )
    assert "benchmark complete" in capsys.readouterr().out

    assert benchmark_cli.main([*common_args, "--rmse-threshold-pct", "-1"]) == 2


def test_campaign_does_not_mutate_global_numpy_rng_state() -> None:
    np.random.seed(6060)
    state = np.random.get_state()

    _ = benchmark_cli.run_campaign(
        seed=5, train_samples=512, eval_samples=160, benchmark_samples=48
    )

    observed = float(np.random.random())
    np.random.set_state(state)
    expected = float(np.random.random())
    assert observed == expected
