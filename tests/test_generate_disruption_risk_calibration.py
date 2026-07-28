# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# ----------------------------------------------------------------------
# SCPN Fusion Core -- Disruption Risk Calibration Tests
# ----------------------------------------------------------------------
"""Tests for tools/generate_disruption_risk_calibration.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

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
    """Write one bounded production-format disruption-shot fixture."""
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
    """Exercise the real repository calibration drift-check boundary."""
    rc = risk_calibration.main(["--check"])
    assert rc == 0


def test_calibration_check_detects_stale_output(tmp_path: Path) -> None:
    """Reject a stale JSON artifact through the public CLI boundary."""
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
    """Reject an out-of-domain recall target before calibration."""
    with pytest.raises(ValueError, match="target_recall"):
        risk_calibration.main(["--target-recall", "1.1", "--skip-gates"])


def test_load_json_rejects_oversized_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject JSON inputs above the configured byte ceiling."""
    payload_path = tmp_path / "oversized.json"
    payload_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(risk_calibration, "_MAX_JSON_BYTES", 1)
    with pytest.raises(ValueError, match="exceeds max JSON size"):
        risk_calibration._load_json(payload_path)


def test_load_samples_rejects_oversized_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject shot signals above the configured sample ceiling."""
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


def _write_json(path: Path, value: Any) -> None:
    """Write a compact JSON fixture."""
    path.write_text(json.dumps(value), encoding="utf-8")


def _sample_payload(*, size: int = 6, n2_present: bool = True) -> dict[str, Any]:
    """Build one loader-compatible signal payload."""
    signal = np.linspace(0.1, 0.6, size, dtype=np.float64)
    return {
        "signal": signal,
        "n1_amp": signal,
        "n2_amp": signal * 0.5 if n2_present else None,
        "is_disruption": True,
        "disruption_time_idx": size - 1,
    }


def test_json_path_and_split_contracts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover JSON shape, path display, split value, size, and overlap guards."""
    list_path = tmp_path / "list.json"
    _write_json(list_path, [1, 2])
    with pytest.raises(ValueError, match="top-level object"):
        risk_calibration._load_json(list_path)

    relative = risk_calibration._resolve_repo_path("relative.json")
    assert relative == risk_calibration.REPO_ROOT / "relative.json"
    assert risk_calibration._resolve_repo_path(str(list_path)) == list_path
    assert risk_calibration._display_path(relative) == "relative.json"
    assert risk_calibration._display_path(list_path) == list_path.as_posix()

    for value in (None, [], [True], ["1"], [0], [1, 1]):
        with pytest.raises(ValueError):
            risk_calibration._parse_split_ids("train", value)
    monkeypatch.setattr(risk_calibration, "_MAX_SPLIT_IDS_PER_SET", 1)
    with pytest.raises(ValueError, match="exceeding max"):
        risk_calibration._parse_split_ids("train", [1, 2])

    split_path = tmp_path / "splits.json"
    _write_json(split_path, {"train": [1], "val": [1], "test": [3]})
    with pytest.raises(ValueError, match="Split overlap"):
        risk_calibration._load_split_map(split_path)


def test_payload_loader_contract_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail closed when the dynamically loaded production module is unavailable."""
    monkeypatch.setattr(
        risk_calibration.importlib.util,
        "spec_from_file_location",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(RuntimeError, match="Failed to load"):
        risk_calibration._load_payload_loader()

    class Loader:
        """Minimal import loader that leaves the target module empty."""

        def exec_module(self, module: ModuleType) -> None:
            """Leave the dynamically created module without the required API."""

    spec = SimpleNamespace(loader=Loader())
    monkeypatch.setattr(
        risk_calibration.importlib.util,
        "spec_from_file_location",
        lambda *_args, **_kwargs: spec,
    )
    monkeypatch.setattr(
        risk_calibration.importlib.util,
        "module_from_spec",
        lambda _spec: ModuleType("empty_validate_real_shots"),
    )
    with pytest.raises(RuntimeError, match="missing load_disruption_shot_payload"):
        risk_calibration._load_payload_loader()


@pytest.mark.parametrize(
    ("shots", "match"),
    [
        (None, "non-empty 'shots' list"),
        ([], "non-empty 'shots' list"),
        (["not-an-object"], "entries must be objects"),
        ([{"file": "bad.txt", "shot": 1}], "missing valid 'file'"),
        ([{"file": "shot.npz", "shot": True}], "invalid positive integer"),
        (
            [{"file": "shot.npz", "shot": 1}, {"file": "shot.npz", "shot": 1}],
            "duplicate file entry",
        ),
        ([{"file": "shot.npz", "shot": 2}], "missing in split definitions"),
    ],
)
def test_load_samples_rejects_malformed_manifest_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shots: Any,
    match: str,
) -> None:
    """Reject malformed manifest containers and entries deterministically."""
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, {"shots": shots} if shots is not None else {})
    monkeypatch.setattr(risk_calibration, "_load_payload_loader", lambda: None)
    if match == "duplicate file entry":
        (tmp_path / "shot.npz").touch()
        payload = _sample_payload()
        monkeypatch.setattr(
            risk_calibration,
            "_load_payload_loader",
            lambda: lambda _path: payload,
        )
    with pytest.raises(ValueError, match=match):
        risk_calibration._load_samples(
            shot_dir=tmp_path,
            manifest_path=manifest_path,
            split_map={1: "train"},
            window_size=2,
        )


def test_load_samples_bounds_and_optional_n2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enforce manifest/signal bounds and accept the documented absent-n2 fallback."""
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, {"shots": [{"file": "shot.npz", "shot": 1}]})
    monkeypatch.setattr(risk_calibration, "_MAX_MANIFEST_SHOTS", 0)
    with pytest.raises(ValueError, match="Manifest includes"):
        risk_calibration._load_samples(
            shot_dir=tmp_path,
            manifest_path=manifest_path,
            split_map={1: "train"},
            window_size=2,
        )
    monkeypatch.setattr(risk_calibration, "_MAX_MANIFEST_SHOTS", 1)
    with pytest.raises(FileNotFoundError, match="Shot file missing"):
        risk_calibration._load_samples(
            shot_dir=tmp_path,
            manifest_path=manifest_path,
            split_map={1: "train"},
            window_size=2,
        )

    (tmp_path / "shot.npz").touch()
    payload = _sample_payload(size=6, n2_present=False)
    monkeypatch.setattr(risk_calibration, "_load_payload_loader", lambda: lambda _path: payload)
    samples = risk_calibration._load_samples(
        shot_dir=tmp_path,
        manifest_path=manifest_path,
        split_map={1: "train"},
        window_size=2,
    )
    assert samples[0]["base_logits"].shape == (4,)

    empty_payload = _sample_payload(size=0)
    monkeypatch.setattr(
        risk_calibration,
        "_load_payload_loader",
        lambda: lambda _path: empty_payload,
    )
    with pytest.raises(ValueError, match="must not be empty"):
        risk_calibration._load_samples(
            shot_dir=tmp_path,
            manifest_path=manifest_path,
            split_map={1: "train"},
            window_size=2,
        )


def test_evaluation_and_selection_branches() -> None:
    """Exercise invalid metrics inputs, all confusion cells, and fallback selection."""
    with pytest.raises(ValueError, match="risk_threshold"):
        risk_calibration._evaluate_subset([], risk_threshold=1.0, bias_delta=0.0)
    with pytest.raises(ValueError, match="bias_delta"):
        risk_calibration._evaluate_subset([], risk_threshold=0.5, bias_delta=float("nan"))

    samples = [
        {"base_logits": np.array([2.0]), "is_disruption": True, "disruption_time_idx": 1},
        {"base_logits": np.array([-2.0]), "is_disruption": True, "disruption_time_idx": 1},
        {"base_logits": np.array([2.0]), "is_disruption": False, "disruption_time_idx": -1},
        {"base_logits": np.array([-2.0]), "is_disruption": False, "disruption_time_idx": -1},
        {"base_logits": np.array([2.0]), "is_disruption": True, "disruption_time_idx": 0},
    ]
    metrics = risk_calibration._evaluate_subset(samples, risk_threshold=0.5, bias_delta=0.0)
    assert metrics["true_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["true_negatives"] == 1

    with pytest.raises(ValueError, match="No calibration candidates"):
        risk_calibration._select_calibration(
            train_val_samples=samples,
            target_recall=1.0,
            target_fpr=0.0,
            threshold_values=np.array([], dtype=np.float64),
            bias_values=np.array([], dtype=np.float64),
        )
    selected = risk_calibration._select_calibration(
        train_val_samples=samples,
        target_recall=1.0,
        target_fpr=0.0,
        threshold_values=np.array([0.5], dtype=np.float64),
        bias_values=np.array([0.0], dtype=np.float64),
    )
    assert selected["selection_mode"] == "pareto_fallback"


def test_generate_requires_calibration_and_holdout_sets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reject datasets missing either calibration or holdout membership."""
    common = {
        "shot_dir": tmp_path,
        "manifest_path": tmp_path / "manifest.json",
        "splits_path": tmp_path / "splits.json",
        "target_recall": 0.8,
        "target_fpr": 0.3,
        "threshold_values": np.array([0.5], dtype=np.float64),
        "bias_values": np.array([0.0], dtype=np.float64),
        "window_size": 2,
    }
    monkeypatch.setattr(risk_calibration, "_load_split_map", lambda _path: {})
    monkeypatch.setattr(
        risk_calibration,
        "_load_samples",
        lambda **_kwargs: [{"split": "test", "is_disruption": False}],
    )
    with pytest.raises(ValueError, match="No train/val"):
        risk_calibration._generate(**common)
    monkeypatch.setattr(
        risk_calibration,
        "_load_samples",
        lambda **_kwargs: [{"split": "train", "is_disruption": False}],
    )
    with pytest.raises(ValueError, match="No holdout"):
        risk_calibration._generate(**common)


def test_output_check_boundaries(tmp_path: Path) -> None:
    """Detect missing and stale output files before accepting exact content."""
    calibration_path = tmp_path / "calibration.json"
    report_path = tmp_path / "report.md"
    data = {
        "version": "v",
        "selection": {
            "mode": "fallback",
            "risk_threshold": 0.5,
            "bias_delta": 0.0,
            "effective_bias": 0.0,
        },
        "metrics": {
            name: {"recall": 0.0, "false_positive_rate": 1.0}
            for name in (
                "selected_train_val",
                "selected_holdout_test",
                "baseline_train_val",
                "baseline_holdout_test",
            )
        },
        "gates": {"train_val_pass": False, "holdout_test_pass": False, "overall_pass": False},
        "targets": {"recall_min": 0.8, "false_positive_rate_max": 0.3},
    }
    assert risk_calibration._check_outputs(calibration_path, report_path, data) == 1
    calibration_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert risk_calibration._check_outputs(calibration_path, report_path, data) == 1
    report_path.write_text("stale\n", encoding="utf-8")
    assert risk_calibration._check_outputs(calibration_path, report_path, data) == 1
    report_path.write_text(risk_calibration._render_markdown(data), encoding="utf-8")
    assert risk_calibration._check_outputs(calibration_path, report_path, data) == 0


@pytest.mark.parametrize(
    ("args", "match"),
    [
        (["--window-size", "1"], "window_size"),
        (["--target-fpr", "nan"], "target_fpr"),
        (["--threshold-step", "0"], "threshold_step"),
        (["--bias-step", "nan"], "bias_step"),
        (["--threshold-min", "0", "--threshold-max", "0"], "Threshold sweep"),
        (["--bias-min", "1", "--bias-max", "0"], "Bias sweep"),
    ],
)
def test_main_rejects_invalid_cli_domains(args: list[str], match: str) -> None:
    """Reject invalid CLI scalar and sweep domains before loading datasets."""
    with pytest.raises(ValueError, match=match):
        risk_calibration.main([*args, "--skip-gates"])


def test_main_gate_failure_and_check_short_circuit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Return failure for a failed scientific gate or stale check result."""
    failed = {
        "version": "v",
        "selection": {
            "mode": "fallback",
            "risk_threshold": 0.5,
            "bias_delta": 0.0,
            "effective_bias": 0.0,
        },
        "metrics": {
            name: {"recall": 0.0, "false_positive_rate": 1.0}
            for name in (
                "selected_train_val",
                "selected_holdout_test",
                "baseline_train_val",
                "baseline_holdout_test",
            )
        },
        "gates": {
            "overall_pass": False,
            "train_val_pass": False,
            "holdout_test_pass": False,
        },
        "targets": {"recall_min": 0.8, "false_positive_rate_max": 0.3},
    }
    monkeypatch.setattr(risk_calibration, "_generate", lambda **_kwargs: failed)
    assert (
        risk_calibration.main(
            [
                "--calibration",
                str(tmp_path / "calibration.json"),
                "--report-md",
                str(tmp_path / "report.md"),
            ]
        )
        == 1
    )
    monkeypatch.setattr(risk_calibration, "_check_outputs", lambda *_args: 1)
    assert risk_calibration.main(["--check", "--skip-gates"]) == 1
