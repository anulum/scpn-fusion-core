# ----------------------------------------------------------------------
# SCPN Fusion Core -- CI RMSE Gate Tests
# ----------------------------------------------------------------------
"""Tests for tools/ci_rmse_gate.py bounded artifact loading and gate behavior."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "ci_rmse_gate.py"
SPEC = importlib.util.spec_from_file_location("ci_rmse_gate", MODULE_PATH)
assert SPEC and SPEC.loader
ci_rmse_gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ci_rmse_gate
SPEC.loader.exec_module(ci_rmse_gate)


def test_load_json_object_rejects_oversized_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = tmp_path / "payload.json"
    payload.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ci_rmse_gate, "_MAX_ARTIFACT_JSON_BYTES", 1)
    with pytest.raises(ValueError, match="exceeds max JSON size"):
        ci_rmse_gate._load_json_object(payload)


def test_main_returns_error_for_oversized_rmse_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "rmse_dashboard_ci.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ci_rmse_gate, "_MAX_ARTIFACT_JSON_BYTES", 1)
    assert ci_rmse_gate.main() == 1


def test_main_passes_with_valid_guard_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "rmse_dashboard_ci.json").write_text(
        json.dumps(
            {
                "confinement_itpa": {"tau_rmse_s": 0.10},
                "sparc_axis": {"axis_rmse_m": 1.0},
                "beta_iter_sparc": {"beta_n_rmse": 0.05},
            }
        ),
        encoding="utf-8",
    )
    (artifacts / "real_shot_validation.json").write_text(
        json.dumps(
            {
                "disruption": {"false_positive_rate": 0.10},
                "blanket": {"tbr_corrected": 1.10},
                "burn": {"Q_peak": 10.0},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    assert ci_rmse_gate.main() == 0
