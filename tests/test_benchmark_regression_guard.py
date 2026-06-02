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
            "reports": [
                {
                    "id": "bench",
                    "path": str(report),
                    "metrics": [
                        {"path": "gate.passes", "equals": True},
                        {"path": "gate.p95_ms", "max": 2.0},
                    ],
                }
            ]
        }
    )

    assert summary["overall_pass"] is True
    assert summary["failed_metric_count"] == 0


def test_evaluate_fails_closed_on_regression(tmp_path: Path) -> None:
    report = tmp_path / "bench.json"
    _write_json(report, {"gate": {"passes": True, "p95_ms": 3.5}})

    summary = guard.evaluate(
        {
            "reports": [
                {
                    "id": "bench",
                    "path": str(report),
                    "metrics": [{"path": "gate.p95_ms", "max": 2.0}],
                }
            ]
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
                "reports": [
                    {
                        "id": "bench",
                        "path": str(report),
                        "metrics": [{"path": "gate.p95_ms", "max": 2.0}],
                    }
                ]
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
            "reports": [
                {
                    "id": "bench",
                    "path": str(report),
                    "metrics": [{"path": "gate.passes", "equals": True}],
                }
            ]
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
