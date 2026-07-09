# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Benchmark Regression Guard Tests

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "benchmark_regression_guard.py"
SPEC = importlib.util.spec_from_file_location("tools.benchmark_regression_guard", MODULE_PATH)
assert SPEC and SPEC.loader
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _thresholds(
    report: Path,
    *,
    metrics: list[dict[str, Any]] | None = None,
    **report_overrides: Any,
) -> dict[str, Any]:
    report_config: dict[str, Any] = {
        "id": "bench",
        "path": str(report),
        "metrics": metrics or [{"path": "gate.passes", "equals": True}],
    }
    report_config.update(report_overrides)
    return {
        "schema": "benchmark-regression-thresholds.v2",
        "reports": [report_config],
    }


def test_evaluate_passes_for_configured_metrics(tmp_path: Path) -> None:
    report = tmp_path / "bench.json"
    _write_json(report, {"gate": {"passes": True, "p95_ms": 1.5}})

    summary = guard.evaluate(
        {
            "schema": "benchmark-regression-thresholds.v2",
            "reports": [
                {
                    "id": "bench",
                    "path": str(report),
                    "metrics": [
                        {"path": "gate.passes", "equals": True},
                        {"path": "gate.p95_ms", "max": 2.0},
                    ],
                }
            ],
        }
    )

    assert summary["overall_pass"] is True
    assert summary["failed_metric_count"] == 0


def test_evaluate_fails_closed_on_regression(tmp_path: Path) -> None:
    report = tmp_path / "bench.json"
    _write_json(report, {"gate": {"passes": True, "p95_ms": 3.5}})

    summary = guard.evaluate(
        {
            "schema": "benchmark-regression-thresholds.v2",
            "reports": [
                {
                    "id": "bench",
                    "path": str(report),
                    "metrics": [{"path": "gate.p95_ms", "max": 2.0}],
                }
            ],
        }
    )

    assert summary["overall_pass"] is False
    assert summary["failed_metric_count"] == 1
    assert "max" in summary["reports"][0]["metrics"][0]["failure_reason"]


def test_evaluate_fails_closed_on_missing_metric(tmp_path: Path) -> None:
    report = tmp_path / "bench.json"
    _write_json(report, {"gate": {"passes": True}})

    with pytest.raises(KeyError):
        guard.evaluate(
            {
                "schema": "benchmark-regression-thresholds.v2",
                "reports": [
                    {
                        "id": "bench",
                        "path": str(report),
                        "metrics": [{"path": "gate.p95_ms", "max": 2.0}],
                    }
                ],
            }
        )


def test_main_writes_summary_and_returns_nonzero(tmp_path: Path) -> None:
    report = tmp_path / "bench.json"
    thresholds = tmp_path / "thresholds.json"
    summary_path = tmp_path / "summary.json"
    _write_json(report, {"gate": {"passes": False}})
    _write_json(
        thresholds,
        {
            "schema": "benchmark-regression-thresholds.v2",
            "reports": [
                {
                    "id": "bench",
                    "path": str(report),
                    "metrics": [{"path": "gate.passes", "equals": True}],
                }
            ],
        },
    )

    rc = guard.main(
        [
            "--thresholds",
            str(thresholds),
            "--summary-json",
            str(summary_path),
        ]
    )

    assert rc == 1
    assert json.loads(summary_path.read_text(encoding="utf-8"))["overall_pass"] is False


def test_main_writes_summary_and_returns_zero(tmp_path: Path) -> None:
    report = tmp_path / "bench.json"
    thresholds = tmp_path / "thresholds.json"
    summary_path = tmp_path / "summary.json"
    _write_json(report, {"gate": {"passes": True}})
    _write_json(thresholds, _thresholds(report))

    rc = guard.main(
        [
            "--thresholds",
            str(thresholds),
            "--summary-json",
            str(summary_path),
        ]
    )

    assert rc == 0
    assert json.loads(summary_path.read_text(encoding="utf-8"))["overall_pass"] is True


def test_evaluate_rejects_schema_mismatch(tmp_path: Path) -> None:
    report = tmp_path / "bench.json"
    _write_json(report, {"gate": {"passes": True}})

    with pytest.raises(ValueError, match="schema mismatch"):
        guard.evaluate(
            {
                "schema": "benchmark-regression-thresholds.v1",
                "reports": [
                    {
                        "id": "bench",
                        "path": str(report),
                        "metrics": [{"path": "gate.passes", "equals": True}],
                    }
                ],
            }
        )


def test_evaluate_rejects_duplicate_metric_paths(tmp_path: Path) -> None:
    report = tmp_path / "bench.json"
    _write_json(report, {"gate": {"p95_ms": 1.5}})

    with pytest.raises(ValueError, match="duplicate metric path"):
        guard.evaluate(
            {
                "schema": "benchmark-regression-thresholds.v2",
                "reports": [
                    {
                        "id": "bench",
                        "path": str(report),
                        "metrics": [
                            {"path": "gate.p95_ms", "max": 2.0},
                            {"path": "gate.p95_ms", "max": 3.0},
                        ],
                    }
                ],
            }
        )


def test_evaluate_rejects_equals_with_numeric_bounds(tmp_path: Path) -> None:
    report = tmp_path / "bench.json"
    _write_json(report, {"gate": {"passes": True}})

    with pytest.raises(ValueError, match="equals cannot be combined"):
        guard.evaluate(
            {
                "schema": "benchmark-regression-thresholds.v2",
                "reports": [
                    {
                        "id": "bench",
                        "path": str(report),
                        "metrics": [{"path": "gate.passes", "equals": True, "max": 1.0}],
                    }
                ],
            }
        )


def test_evaluate_fails_closed_on_stale_report(tmp_path: Path) -> None:
    report = tmp_path / "bench.json"
    _write_json(report, {"gate": {"passes": True}})
    os.utime(report, (1, 1))

    summary = guard.evaluate(
        {
            "schema": "benchmark-regression-thresholds.v2",
            "reports": [
                {
                    "id": "bench",
                    "path": str(report),
                    "max_age_seconds": 1.0,
                    "metrics": [{"path": "gate.passes", "equals": True}],
                }
            ],
        }
    )

    assert summary["overall_pass"] is False
    assert summary["reports"][0]["metrics"][-1]["path"] == "__report_age_seconds__"


def test_evaluate_fails_closed_on_schema_identity_mismatch(tmp_path: Path) -> None:
    report = tmp_path / "bench.json"
    _write_json(report, {"schema": "wrong.v1", "gate": {"passes": True}})

    summary = guard.evaluate(
        {
            "schema": "benchmark-regression-thresholds.v2",
            "reports": [
                {
                    "id": "bench",
                    "path": str(report),
                    "expected_schema": "right.v1",
                    "metrics": [{"path": "gate.passes", "equals": True}],
                }
            ],
        }
    )

    assert summary["overall_pass"] is False
    assert summary["reports"][0]["metrics"][0]["path"] == "__report_schema__"


def test_evaluate_fails_closed_on_benchmark_id_mismatch(tmp_path: Path) -> None:
    report = tmp_path / "bench.json"
    _write_json(report, {"benchmark_id": "other", "gate": {"passes": True}})

    summary = guard.evaluate(
        {
            "schema": "benchmark-regression-thresholds.v2",
            "reports": [
                {
                    "id": "bench",
                    "path": str(report),
                    "expected_benchmark_id": "expected",
                    "metrics": [{"path": "gate.passes", "equals": True}],
                }
            ],
        }
    )

    assert summary["overall_pass"] is False
    assert summary["reports"][0]["metrics"][0]["path"] == "__benchmark_id__"


def test_main_resolves_repo_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "reports" / "bench.json"
    thresholds = tmp_path / "thresholds.json"
    _write_json(report, {"gate": {"passes": True}})
    _write_json(thresholds, _thresholds(Path("reports/bench.json")))
    monkeypatch.setattr(guard, "REPO_ROOT", tmp_path)

    rc = guard.main(["--thresholds", "thresholds.json", "--summary-json", "summary.json"])

    assert rc == 0
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))[
        "reports"
    ][0]["path"] == "reports/bench.json"


def test_evaluate_uses_list_indices_and_min_bounds(tmp_path: Path) -> None:
    report = tmp_path / "bench.json"
    _write_json(report, {"samples": [{"latency_ms": 1.0}, {"latency_ms": 1.5}]})

    summary = guard.evaluate(
        _thresholds(
            report,
            metrics=[
                {"path": "samples.0.latency_ms", "min": 0.5, "max": 2.0},
                {"path": "samples.1.latency_ms", "min": 2.0},
            ],
        )
    )

    assert summary["overall_pass"] is False
    assert summary["failed_metric_count"] == 1
    assert "min" in summary["reports"][0]["metrics"][1]["failure_reason"]


def test_evaluate_rejects_missing_and_invalid_list_indices(tmp_path: Path) -> None:
    report = tmp_path / "bench.json"
    _write_json(report, {"samples": [{"latency_ms": 1.0}]})

    with pytest.raises(KeyError):
        guard.evaluate(_thresholds(report, metrics=[{"path": "samples.2.latency_ms", "max": 2.0}]))
    with pytest.raises(KeyError):
        guard.evaluate(_thresholds(report, metrics=[{"path": "samples.first.latency_ms", "max": 2.0}]))
    with pytest.raises(KeyError):
        guard.evaluate(
            _thresholds(report, metrics=[{"path": "samples.0.latency_ms.value", "max": 2.0}])
        )


@pytest.mark.parametrize("bad_value", [True, "fast"])
def test_evaluate_rejects_non_numeric_metric_values(tmp_path: Path, bad_value: object) -> None:
    report = tmp_path / "bench.json"
    _write_json(report, {"gate": {"p95_ms": bad_value}})

    with pytest.raises(TypeError, match="expected numeric metric"):
        guard.evaluate(_thresholds(report, metrics=[{"path": "gate.p95_ms", "max": 2.0}]))


def test_evaluate_rejects_non_finite_metric_values(tmp_path: Path) -> None:
    report = tmp_path / "bench.json"
    _write_json(report, {"gate": {"p95_ms": float("inf")}})

    with pytest.raises(ValueError, match="metric is not finite"):
        guard.evaluate(_thresholds(report, metrics=[{"path": "gate.p95_ms", "max": 2.0}]))


@pytest.mark.parametrize(
    ("thresholds", "match"),
    [
        ({"schema": "benchmark-regression-thresholds.v2", "reports": []}, "at least one report"),
        (
            {"schema": "benchmark-regression-thresholds.v2", "reports": ["bench"]},
            "reports\\[0\\]: expected object",
        ),
        (
            {
                "schema": "benchmark-regression-thresholds.v2",
                "reports": [
                    {"id": "bench", "path": "bench.json", "metrics": []},
                ],
            },
            "requires at least one metric",
        ),
        (
            {
                "schema": "benchmark-regression-thresholds.v2",
                "reports": [
                    {"id": "bench", "path": "bench.json", "metrics": ["gate"]},
                ],
            },
            "expected object",
        ),
        (
            {
                "schema": "benchmark-regression-thresholds.v2",
                "reports": [
                    {"id": "", "path": "bench.json", "metrics": [{"path": "x", "max": 1.0}]},
                ],
            },
            "expected non-empty string",
        ),
        (
            {
                "schema": "benchmark-regression-thresholds.v2",
                "reports": [
                    {
                        "id": "bench",
                        "path": "bench.json",
                        "metrics": [{"path": "x"}],
                    },
                ],
            },
            "requires equals, min, or max",
        ),
        (
            {
                "schema": "benchmark-regression-thresholds.v2",
                "reports": [
                    {
                        "id": "bench",
                        "path": "bench.json",
                        "metrics": [{"path": "x", "min": 2.0, "max": 1.0}],
                    },
                ],
            },
            "min cannot be greater than max",
        ),
        (
            {
                "schema": "benchmark-regression-thresholds.v2",
                "reports": [
                    {
                        "id": "bench",
                        "path": "bench.json",
                        "metrics": [{"path": "x", "max": 1.0}],
                        "max_age_seconds": 0.0,
                    },
                ],
            },
            "max_age_seconds must be > 0",
        ),
        (
            {
                "schema": "benchmark-regression-thresholds.v2",
                "reports": [
                    {"id": "bench", "path": "a.json", "metrics": [{"path": "x", "max": 1.0}]},
                    {"id": "bench", "path": "b.json", "metrics": [{"path": "x", "max": 1.0}]},
                ],
            },
            "duplicate benchmark report id",
        ),
    ],
)
def test_evaluate_rejects_invalid_threshold_configuration(
    thresholds: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        guard.evaluate(thresholds)


def test_evaluate_rejects_missing_report_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="required benchmark artefact is missing"):
        guard.evaluate(_thresholds(tmp_path / "missing.json"))


def test_evaluate_rejects_non_object_report_json(tmp_path: Path) -> None:
    report = tmp_path / "bench.json"
    report.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="expected JSON object payload"):
        guard.evaluate(_thresholds(report))


def test_main_returns_failure_for_invalid_threshold_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    summary_path = tmp_path / "summary.json"

    rc = guard.main(
        [
            "--thresholds",
            str(tmp_path / "missing-thresholds.json"),
            "--summary-json",
            str(summary_path),
        ]
    )

    assert rc == 1
    assert not summary_path.exists()
    assert "Benchmark regression guard failed:" in capsys.readouterr().out
