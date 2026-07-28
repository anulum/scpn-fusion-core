# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Regression tests for tools/real_data_roadmap_progress.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "real_data_roadmap_progress.py"
SPEC = importlib.util.spec_from_file_location("real_data_roadmap_progress", MODULE_PATH)
assert SPEC and SPEC.loader
real_data_roadmap_progress = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(real_data_roadmap_progress)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _cli_args(*, report: Path, targets: Path, output_json: Path, output_md: Path) -> list[str]:
    return [
        "--report",
        str(report),
        "--targets",
        str(targets),
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
    ]


def test_progress_summary_contains_expected_metrics() -> None:
    report = {
        "equilibrium": {
            "n_files": 18,
            "results": [
                {"machine": "SPARC", "file": "sparc_1300.eqdsk"},
                {"machine": "JET", "file": "jet_dt_3p5MA.geqdsk"},
                {"machine": "DIII-D", "file": "diiid_hmode.geqdsk"},
            ],
        },
        "transport": {
            "n_shots": 53,
            "shots": [
                {"machine": "JET"},
                {"machine": "DIII-D"},
                {"machine": "SPARC"},
                {"machine": "ITER"},
            ],
        },
        "disruption": {
            "n_shots": 16,
            "calibration": {"source": "diiid-disruption-risk-calibration-v1"},
            "data_source": {
                "source_types": ["synthetic_diiid_like"],
                "raw_ingestion_ready": False,
            },
        },
    }
    targets = {
        "roadmap_version": "v4.0",
        "targets": {
            "equilibrium_files_total": 20,
            "sparc_equilibria": 20,
            "transport_shots_total": 100,
            "transport_machines_total": 30,
            "disruption_shots_total": 16,
            "jet_dt_equilibria": 5,
        },
    }
    summary = real_data_roadmap_progress.evaluate_progress(report=report, targets=targets)
    assert summary["roadmap_version"] == "v4.0"
    assert summary["overall_pass"] is False
    assert summary["overall_progress_ratio"] < 1.0
    assert summary["d3d_raw_ingestion_ready"] is False
    assert summary["d3d_raw_source_type_present"] is False
    metrics = {row["metric"]: row for row in summary["metrics"]}
    assert metrics["disruption_shots_total"]["passes"] is True
    assert metrics["transport_shots_total"]["passes"] is False
    assert metrics["transport_shots_total"]["remaining_to_target"] == 47


def test_progress_uses_explicit_disruption_source_contract_for_raw_readiness() -> None:
    report = {
        "equilibrium": {"n_files": 0, "results": []},
        "transport": {"n_shots": 0, "shots": []},
        "disruption": {
            "n_shots": 1,
            "calibration": {"source": "diiid-disruption-risk-calibration-v1"},
            "data_source": {
                "source_types": ["raw_diiid_mdsplus"],
                "raw_ingestion_ready": True,
            },
        },
    }
    targets = {"targets": {"disruption_shots_total": 1}}
    summary = real_data_roadmap_progress.evaluate_progress(report=report, targets=targets)
    assert summary["d3d_raw_ingestion_ready"] is True
    assert summary["d3d_raw_source_type_present"] is True
    assert summary["d3d_disruption_source_types"] == ["raw_diiid_mdsplus"]


def test_progress_rejects_raw_ready_without_raw_source_type_contract() -> None:
    report = {
        "equilibrium": {"n_files": 0, "results": []},
        "transport": {"n_shots": 0, "shots": []},
        "disruption": {
            "n_shots": 1,
            "calibration": {"source": "diiid-disruption-risk-calibration-v1"},
            "data_source": {
                "source_types": ["synthetic_diiid_like"],
                "raw_ingestion_ready": True,
            },
        },
    }
    targets = {"targets": {"disruption_shots_total": 1}}
    summary = real_data_roadmap_progress.evaluate_progress(report=report, targets=targets)
    assert summary["d3d_raw_source_type_present"] is False
    assert summary["d3d_raw_ingestion_ready"] is False


def test_json_path_and_ratio_contracts(tmp_path: Path) -> None:
    """Exercise repository-relative paths, object-only JSON, and bounded ratios."""
    relative = Path("artifacts/example.json")
    assert real_data_roadmap_progress._resolve(relative.as_posix()) == ROOT / relative  # noqa: SLF001

    object_path = tmp_path / "object.json"
    _write_json(object_path, {"value": 1})
    assert real_data_roadmap_progress._load_json(object_path) == {"value": 1}  # noqa: SLF001

    array_path = tmp_path / "array.json"
    _write_json(array_path, [1, 2])
    with pytest.raises(ValueError, match="expected JSON object payload"):
        real_data_roadmap_progress._load_json(array_path)  # noqa: SLF001

    assert real_data_roadmap_progress._safe_ratio(5, 0) == 1.0  # noqa: SLF001
    assert real_data_roadmap_progress._safe_ratio(-1, 10) == 0.0  # noqa: SLF001
    assert real_data_roadmap_progress._safe_ratio(11, 10) == 1.0  # noqa: SLF001


def test_progress_rejects_nonobject_target_mapping() -> None:
    """Fail closed when the roadmap target mapping has the wrong JSON type."""
    with pytest.raises(ValueError, match="missing 'targets' object"):
        real_data_roadmap_progress.evaluate_progress(report={}, targets={"targets": []})


def test_progress_handles_empty_rows_and_historical_raw_source() -> None:
    """Preserve empty-row defaults and the historical raw-calibration fallback."""
    empty_summary = real_data_roadmap_progress.evaluate_progress(
        report={
            "equilibrium": {"n_files": 0, "results": {}},
            "transport": {"n_shots": 0, "shots": {}},
            "disruption": {
                "n_shots": 0,
                "calibration": [],
                "data_source": {"source_types": "raw", "raw_ingestion_ready": True},
            },
        },
        targets={"targets": {}},
    )
    assert empty_summary["overall_pass"] is True
    assert empty_summary["overall_progress_ratio"] == 1.0
    assert empty_summary["d3d_raw_ingestion_ready"] is False
    assert empty_summary["d3d_disruption_source_types"] == []

    historical_summary = real_data_roadmap_progress.evaluate_progress(
        report={
            "equilibrium": {"n_files": 0, "results": []},
            "transport": {"n_shots": 0, "shots": []},
            "disruption": {
                "n_shots": 0,
                "calibration": {"source": "DIII-D raw MDSplus archive"},
                "data_source": [],
            },
        },
        targets={"targets": {"unknown_metric": 0}},
    )
    assert historical_summary["d3d_raw_ingestion_ready"] is True
    assert historical_summary["d3d_raw_source_type_present"] is True
    assert historical_summary["metrics"][0]["current"] == 0
    assert historical_summary["metrics"][0]["progress_ratio"] == 1.0


def test_render_markdown_covers_rich_and_minimal_summaries() -> None:
    """Render metric, provenance, source-type, and machine sections honestly."""
    summary: dict[str, Any] = {
        "generated_at_utc": "2026-07-29T00:00:00+00:00",
        "roadmap_version": "v4.0",
        "overall_pass": False,
        "overall_progress_ratio": 0.5,
        "d3d_raw_ingestion_ready": False,
        "d3d_raw_source_type_present": True,
        "d3d_calibration_source": "DIII-D archive",
        "d3d_disruption_source_types": ["raw_mdsplus"],
        "transport_machines": ["DIII-D", "JET"],
        "metrics": [
            {
                "metric": "transport_shots_total",
                "current": 5,
                "target": 10,
                "progress_ratio": 0.5,
                "remaining_to_target": 5,
                "passes": False,
            }
        ],
    }
    markdown = real_data_roadmap_progress.render_markdown(summary)
    assert "Overall target pass: `NO`" in markdown
    assert "DIII-D disruption source types: `raw_mdsplus`" in markdown
    assert "| `transport_shots_total` | 5 | 10 | 5 | 50.0% | NO |" in markdown
    assert "## Transport Machine Coverage" in markdown

    minimal = real_data_roadmap_progress.render_markdown(
        {
            "generated_at_utc": "2026-07-29T00:00:00+00:00",
            "roadmap_version": "v4.0",
            "overall_pass": True,
            "d3d_raw_ingestion_ready": True,
        }
    )
    assert "Overall target pass: `YES`" in minimal
    assert "DIII-D raw-ingestion readiness: `YES`" in minimal
    assert "Transport Machine Coverage" not in minimal


def test_main_writes_artifacts_and_enforces_strict_gate(tmp_path: Path) -> None:
    """Write real JSON/Markdown outputs and fail strict mode on unmet targets."""
    report = tmp_path / "inputs" / "report.json"
    targets = tmp_path / "inputs" / "targets.json"
    output_json = tmp_path / "outputs" / "progress.json"
    output_md = tmp_path / "outputs" / "progress.md"
    _write_json(
        report,
        {
            "equilibrium": {"n_files": 0, "results": []},
            "transport": {"n_shots": 0, "shots": []},
            "disruption": {"n_shots": 0},
        },
    )
    _write_json(targets, {"targets": {"equilibrium_files_total": 1}})
    args = _cli_args(
        report=report,
        targets=targets,
        output_json=output_json,
        output_md=output_md,
    )

    assert real_data_roadmap_progress.main(args) == 0
    assert json.loads(output_json.read_text(encoding="utf-8"))["overall_pass"] is False
    assert "Overall target pass: `NO`" in output_md.read_text(encoding="utf-8")
    assert real_data_roadmap_progress.main([*args, "--strict"]) == 1

    _write_json(targets, {"targets": {"equilibrium_files_total": 0}})
    assert real_data_roadmap_progress.main([*args, "--strict"]) == 0
