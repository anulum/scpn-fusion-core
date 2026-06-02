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

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "benchmark_regression_guard.py"
SPEC = importlib.util.spec_from_file_location("benchmark_regression_guard", MODULE_PATH)
assert SPEC and SPEC.loader
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
