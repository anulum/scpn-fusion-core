# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# ----------------------------------------------------------------------
# SCPN Fusion Core -- CI RMSE Gate Tests
# ----------------------------------------------------------------------
"""Tests for tools/ci_rmse_gate.py bounded artifact loading and gate behavior."""

from __future__ import annotations

import importlib.util
import json
import runpy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "ci_rmse_gate.py"
SPEC = importlib.util.spec_from_file_location("tools.ci_rmse_gate", MODULE_PATH)
assert SPEC and SPEC.loader
ci_rmse_gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ci_rmse_gate
SPEC.loader.exec_module(ci_rmse_gate)


def _write_json(path: Path, payload: object) -> None:
    """Write a JSON payload to ``path``."""
    path.write_text(json.dumps(payload), encoding="utf-8")


def _artifact_dir(tmp_path: Path) -> Path:
    """Create and return the test artifact directory."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return artifacts


def test_main_reports_missing_rmse_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Main fails when the RMSE dashboard artifact is absent."""
    monkeypatch.chdir(tmp_path)

    assert ci_rmse_gate.main() == 1


def test_main_returns_error_for_oversized_rmse_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Main fails when the RMSE dashboard artifact exceeds the size limit."""
    artifacts = _artifact_dir(tmp_path)
    (artifacts / "rmse_dashboard_ci.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ci_rmse_gate, "_MAX_ARTIFACT_JSON_BYTES", 1)

    assert ci_rmse_gate.main() == 1


def test_main_returns_error_for_non_object_rmse_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Main fails when the RMSE dashboard artifact is not a JSON object."""
    artifacts = _artifact_dir(tmp_path)
    _write_json(artifacts / "rmse_dashboard_ci.json", [])
    monkeypatch.chdir(tmp_path)

    assert ci_rmse_gate.main() == 1


def test_main_returns_error_for_too_many_rmse_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Main fails when the RMSE dashboard artifact exceeds key-count limits."""
    artifacts = _artifact_dir(tmp_path)
    _write_json(artifacts / "rmse_dashboard_ci.json", {"a": {}, "b": {}})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ci_rmse_gate, "_MAX_TOP_LEVEL_KEYS", 1)

    assert ci_rmse_gate.main() == 1


def test_main_passes_with_valid_guard_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Main passes when all tracked RMSE and real-shot gates are within limits."""
    artifacts = _artifact_dir(tmp_path)
    _write_json(
        artifacts / "rmse_dashboard_ci.json",
        {
            "confinement_itpa": {"tau_rmse_s": 0.10},
            "sparc_axis": {"axis_rmse_m": 1.0},
            "beta_iter_sparc": {"beta_n_rmse": 0.05},
        },
    )
    _write_json(
        artifacts / "real_shot_validation.json",
        {
            "disruption": {"false_positive_rate": 0.10},
            "blanket": {"tbr_corrected": 1.10},
            "burn": {"Q_peak": 10.0},
        },
    )
    monkeypatch.chdir(tmp_path)

    assert ci_rmse_gate.main() == 0


def test_main_passes_when_optional_sections_are_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Main passes when optional metric sections and real-shot artifacts are absent."""
    artifacts = _artifact_dir(tmp_path)
    _write_json(artifacts / "rmse_dashboard_ci.json", {})
    monkeypatch.chdir(tmp_path)

    assert ci_rmse_gate.main() == 0


def test_main_reports_rmse_and_disruption_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Main aggregates RMSE, disruption, low-TBR, and Q-peak failures."""
    artifacts = _artifact_dir(tmp_path)
    _write_json(
        artifacts / "rmse_dashboard_ci.json",
        {
            "confinement_itpa": {"tau_rmse_s": 0.30},
            "sparc_axis": {"axis_rmse_m": 3.0},
            "beta_iter_sparc": {"beta_n_rmse": 0.30},
        },
    )
    _write_json(
        artifacts / "real_shot_validation.json",
        {
            "disruption": {"false_positive_rate": 0.30},
            "blanket": {"tbr_corrected": 0.90},
            "burn": {"Q_peak": 20.0},
        },
    )
    monkeypatch.chdir(tmp_path)

    assert ci_rmse_gate.main() == 1


def test_main_reports_high_tbr_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Main rejects unrealistically high TBR values."""
    artifacts = _artifact_dir(tmp_path)
    _write_json(artifacts / "rmse_dashboard_ci.json", {})
    _write_json(artifacts / "real_shot_validation.json", {"blanket": {"tbr_corrected": 1.50}})
    monkeypatch.chdir(tmp_path)

    assert ci_rmse_gate.main() == 1


def test_main_returns_error_for_invalid_real_shot_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Main fails when the optional real-shot artifact is malformed."""
    artifacts = _artifact_dir(tmp_path)
    _write_json(artifacts / "rmse_dashboard_ci.json", {})
    _write_json(artifacts / "real_shot_validation.json", [])
    monkeypatch.chdir(tmp_path)

    assert ci_rmse_gate.main() == 1


def test_script_entrypoint_exits_with_main_return_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The executable entrypoint delegates through ``main`` and exits with its code."""
    artifacts = _artifact_dir(tmp_path)
    _write_json(artifacts / "rmse_dashboard_ci.json", {})
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(MODULE_PATH), run_name="__main__")

    assert exc_info.value.code == 0
