# ----------------------------------------------------------------------
# SCPN Fusion Core -- Disruption Risk Calibration Tests
# ----------------------------------------------------------------------
"""Tests for tools/generate_disruption_risk_calibration.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "generate_disruption_risk_calibration.py"
SPEC = importlib.util.spec_from_file_location("generate_disruption_risk_calibration", MODULE_PATH)
assert SPEC and SPEC.loader
risk_calibration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = risk_calibration
SPEC.loader.exec_module(risk_calibration)


def _write_shot(path: Path, *, disruptive: bool, amplitude: float) -> None:
    n = 256
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    if disruptive:
        signal = 0.25 + amplitude * np.exp(4.0 * (t - 0.75))
        is_disruption = np.array(True)
        disruption_idx = np.array(210)
    else:
        signal = 0.25 + 0.005 * np.sin(2.0 * np.pi * 4.0 * t)
        is_disruption = np.array(False)
        disruption_idx = np.array(-1)
    n1 = signal
    n2 = 0.3 * signal
    np.savez(
        path,
        n1_amp=n1,
        n2_amp=n2,
        time_s=t,
        is_disruption=is_disruption,
        disruption_time_idx=disruption_idx,
    )


def test_repo_calibration_check_passes() -> None:
    rc = risk_calibration.main(["--check"])
    assert rc == 0


def test_calibration_check_detects_stale_output(tmp_path: Path) -> None:
    shot_dir = tmp_path / "shots"
    shot_dir.mkdir(parents=True, exist_ok=True)
    _write_shot(shot_dir / "shot_100001_disruptive.npz", disruptive=True, amplitude=0.8)
    _write_shot(shot_dir / "shot_100002_safe.npz", disruptive=False, amplitude=0.0)
    _write_shot(shot_dir / "shot_100003_disruptive.npz", disruptive=True, amplitude=1.0)

    manifest = {
        "shots": [
            {"file": "shot_100001_disruptive.npz", "shot": 100001},
            {"file": "shot_100002_safe.npz", "shot": 100002},
            {"file": "shot_100003_disruptive.npz", "shot": 100003},
        ]
    }
    split = {
        "train": [100001],
        "val": [100002],
        "test": [100003],
    }
    manifest_path = tmp_path / "manifest.json"
    splits_path = tmp_path / "splits.json"
    calibration_path = tmp_path / "calibration.json"
    report_path = tmp_path / "calibration.md"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    splits_path.write_text(json.dumps(split), encoding="utf-8")

    rc_write = risk_calibration.main(
        [
            "--shot-dir",
            str(shot_dir),
            "--manifest",
            str(manifest_path),
            "--splits",
            str(splits_path),
            "--calibration",
            str(calibration_path),
            "--report-md",
            str(report_path),
            "--skip-gates",
        ]
    )
    assert rc_write == 0

    calibration_path.write_text('{"stale": true}\n', encoding="utf-8")
    rc_check = risk_calibration.main(
        [
            "--shot-dir",
            str(shot_dir),
            "--manifest",
            str(manifest_path),
            "--splits",
            str(splits_path),
            "--calibration",
            str(calibration_path),
            "--report-md",
            str(report_path),
            "--check",
            "--skip-gates",
        ]
    )
    assert rc_check == 1


def test_calibration_rejects_invalid_targets() -> None:
    with pytest.raises(ValueError, match="target_recall"):
        risk_calibration.main(["--target-recall", "1.1", "--skip-gates"])


def test_load_json_rejects_oversized_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload_path = tmp_path / "oversized.json"
    payload_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(risk_calibration, "_MAX_JSON_BYTES", 1)
    with pytest.raises(ValueError, match="exceeds max JSON size"):
        risk_calibration._load_json(payload_path)


def test_load_samples_rejects_oversized_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shot_dir = tmp_path / "shots"
    shot_dir.mkdir(parents=True, exist_ok=True)
    _write_shot(shot_dir / "shot_100001_disruptive.npz", disruptive=True, amplitude=1.0)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"shots": [{"file": "shot_100001_disruptive.npz", "shot": 100001}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(risk_calibration, "_MAX_SIGNAL_SAMPLES_PER_SHOT", 16)
    with pytest.raises(ValueError, match="signal length"):
        risk_calibration._load_samples(
            shot_dir=shot_dir,
            manifest_path=manifest_path,
            split_map={100001: "train"},
            window_size=8,
        )
