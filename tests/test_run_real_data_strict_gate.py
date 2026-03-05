"""Tests for tools/run_real_data_strict_gate.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "run_real_data_strict_gate.py"
SPEC = importlib.util.spec_from_file_location("run_real_data_strict_gate", MODULE_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_main_fails_when_raw_ingestion_not_ready(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report_json = tmp_path / "real_shot.json"
    report_md = tmp_path / "real_shot.md"
    guard_summary = tmp_path / "guard_summary.json"
    progress_json = tmp_path / "progress.json"
    progress_md = tmp_path / "progress.md"
    non_reg_summary = tmp_path / "non_regression.json"
    thresholds = tmp_path / "thresholds.json"
    targets = tmp_path / "targets.json"
    baseline = tmp_path / "baseline.json"

    report_json.write_text("{}", encoding="utf-8")
    report_md.write_text("# report\n", encoding="utf-8")
    _write_json(thresholds, {})
    _write_json(targets, {"targets": {}})
    _write_json(baseline, {"metrics": {}})

    calls: list[str] = []

    def fake_run_step(name: str, cmd: list[str], *, timeout_seconds: float) -> None:
        del cmd, timeout_seconds
        calls.append(name)
        if name == "real-shot guard":
            _write_json(guard_summary, {"overall_pass": True})
        elif name == "real-data roadmap progress":
            _write_json(progress_json, {"d3d_raw_ingestion_ready": False})
            progress_md.write_text("# progress\n", encoding="utf-8")
        elif name == "real-data roadmap non-regression guard":
            _write_json(non_reg_summary, {"overall_pass": True})

    monkeypatch.setattr(runner, "_run_step", fake_run_step)
    rc = runner.main(
        [
            "--skip-validation",
            "--report-json",
            str(report_json),
            "--report-md",
            str(report_md),
            "--guard-summary-json",
            str(guard_summary),
            "--progress-json",
            str(progress_json),
            "--progress-md",
            str(progress_md),
            "--non-regression-summary-json",
            str(non_reg_summary),
            "--thresholds",
            str(thresholds),
            "--targets",
            str(targets),
            "--baseline-json",
            str(baseline),
        ]
    )
    assert rc == 1
    assert calls == ["real-shot guard", "real-data roadmap progress"]


def test_main_allows_missing_raw_ingestion_with_opt_in_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report_json = tmp_path / "real_shot.json"
    report_md = tmp_path / "real_shot.md"
    guard_summary = tmp_path / "guard_summary.json"
    progress_json = tmp_path / "progress.json"
    progress_md = tmp_path / "progress.md"
    non_reg_summary = tmp_path / "non_regression.json"
    thresholds = tmp_path / "thresholds.json"
    targets = tmp_path / "targets.json"
    baseline = tmp_path / "baseline.json"

    report_json.write_text("{}", encoding="utf-8")
    report_md.write_text("# report\n", encoding="utf-8")
    _write_json(thresholds, {})
    _write_json(targets, {"targets": {}})
    _write_json(baseline, {"metrics": {}})

    calls: list[str] = []

    def fake_run_step(name: str, cmd: list[str], *, timeout_seconds: float) -> None:
        del cmd, timeout_seconds
        calls.append(name)
        if name == "real-shot guard":
            _write_json(guard_summary, {"overall_pass": True})
        elif name == "real-data roadmap progress":
            _write_json(progress_json, {"d3d_raw_ingestion_ready": False})
            progress_md.write_text("# progress\n", encoding="utf-8")
        elif name == "real-data roadmap non-regression guard":
            _write_json(non_reg_summary, {"overall_pass": True})

    monkeypatch.setattr(runner, "_run_step", fake_run_step)
    rc = runner.main(
        [
            "--skip-validation",
            "--allow-missing-raw-ingestion",
            "--report-json",
            str(report_json),
            "--report-md",
            str(report_md),
            "--guard-summary-json",
            str(guard_summary),
            "--progress-json",
            str(progress_json),
            "--progress-md",
            str(progress_md),
            "--non-regression-summary-json",
            str(non_reg_summary),
            "--thresholds",
            str(thresholds),
            "--targets",
            str(targets),
            "--baseline-json",
            str(baseline),
        ]
    )
    assert rc == 0
    assert calls == [
        "real-shot guard",
        "real-data roadmap progress",
        "real-data roadmap non-regression guard",
    ]


def test_allow_missing_raw_ingestion_overrides_guard_threshold_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report_json = tmp_path / "real_shot.json"
    report_md = tmp_path / "real_shot.md"
    guard_summary = tmp_path / "guard_summary.json"
    progress_json = tmp_path / "progress.json"
    progress_md = tmp_path / "progress.md"
    non_reg_summary = tmp_path / "non_regression.json"
    thresholds = tmp_path / "thresholds.json"
    targets = tmp_path / "targets.json"
    baseline = tmp_path / "baseline.json"

    report_json.write_text("{}", encoding="utf-8")
    report_md.write_text("# report\n", encoding="utf-8")
    _write_json(
        thresholds,
        {
            "require_disruption_source_contract": True,
            "require_disruption_raw_ingestion_ready": True,
        },
    )
    _write_json(targets, {"targets": {}})
    _write_json(baseline, {"metrics": {}})

    guard_threshold_path: str | None = None

    def fake_run_step(name: str, cmd: list[str], *, timeout_seconds: float) -> None:
        del timeout_seconds
        nonlocal guard_threshold_path
        if name == "real-shot guard":
            idx = cmd.index("--thresholds")
            guard_threshold_path = cmd[idx + 1]
            _write_json(guard_summary, {"overall_pass": True})
        elif name == "real-data roadmap progress":
            _write_json(progress_json, {"d3d_raw_ingestion_ready": False})
            progress_md.write_text("# progress\n", encoding="utf-8")
        elif name == "real-data roadmap non-regression guard":
            _write_json(non_reg_summary, {"overall_pass": True})

    monkeypatch.setattr(runner, "_run_step", fake_run_step)
    rc = runner.main(
        [
            "--skip-validation",
            "--allow-missing-raw-ingestion",
            "--report-json",
            str(report_json),
            "--report-md",
            str(report_md),
            "--guard-summary-json",
            str(guard_summary),
            "--progress-json",
            str(progress_json),
            "--progress-md",
            str(progress_md),
            "--non-regression-summary-json",
            str(non_reg_summary),
            "--thresholds",
            str(thresholds),
            "--targets",
            str(targets),
            "--baseline-json",
            str(baseline),
        ]
    )
    assert rc == 0
    assert guard_threshold_path is not None
    assert guard_threshold_path.endswith("_tmp_real_shot_validation_thresholds_dry_run.json")
    payload = json.loads(Path(guard_threshold_path).read_text(encoding="utf-8"))
    assert payload["require_disruption_raw_ingestion_ready"] is False


def test_main_requires_report_when_validation_is_skipped(tmp_path: Path) -> None:
    rc: int | None = None
    try:
        rc = runner.main(
            [
                "--skip-validation",
                "--report-json",
                str(tmp_path / "missing.json"),
            ]
        )
    except FileNotFoundError:
        return
    raise AssertionError(f"expected FileNotFoundError, got rc={rc}")


def test_main_fails_when_raw_ready_missing_raw_source_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report_json = tmp_path / "real_shot.json"
    report_md = tmp_path / "real_shot.md"
    guard_summary = tmp_path / "guard_summary.json"
    progress_json = tmp_path / "progress.json"
    progress_md = tmp_path / "progress.md"
    non_reg_summary = tmp_path / "non_regression.json"
    thresholds = tmp_path / "thresholds.json"
    targets = tmp_path / "targets.json"
    baseline = tmp_path / "baseline.json"

    report_json.write_text("{}", encoding="utf-8")
    report_md.write_text("# report\n", encoding="utf-8")
    _write_json(thresholds, {})
    _write_json(targets, {"targets": {}})
    _write_json(baseline, {"metrics": {}})

    calls: list[str] = []

    def fake_run_step(name: str, cmd: list[str], *, timeout_seconds: float) -> None:
        del cmd, timeout_seconds
        calls.append(name)
        if name == "real-shot guard":
            _write_json(guard_summary, {"overall_pass": True})
        elif name == "real-data roadmap progress":
            _write_json(
                progress_json,
                {
                    "d3d_raw_ingestion_ready": True,
                    "d3d_disruption_source_types": ["synthetic_diiid_like"],
                },
            )
            progress_md.write_text("# progress\n", encoding="utf-8")
        elif name == "real-data roadmap non-regression guard":
            _write_json(non_reg_summary, {"overall_pass": True})

    monkeypatch.setattr(runner, "_run_step", fake_run_step)
    rc = runner.main(
        [
            "--skip-validation",
            "--report-json",
            str(report_json),
            "--report-md",
            str(report_md),
            "--guard-summary-json",
            str(guard_summary),
            "--progress-json",
            str(progress_json),
            "--progress-md",
            str(progress_md),
            "--non-regression-summary-json",
            str(non_reg_summary),
            "--thresholds",
            str(thresholds),
            "--targets",
            str(targets),
            "--baseline-json",
            str(baseline),
        ]
    )
    assert rc == 1
    assert calls == ["real-shot guard", "real-data roadmap progress"]
