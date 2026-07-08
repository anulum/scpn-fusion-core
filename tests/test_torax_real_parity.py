# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Real-TORAX Parity Gate Tests
"""Tests for the really-executed-TORAX reference artifact and comparison gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "benchmark_torax_real_parity.py"
REFERENCE = ROOT / "validation" / "reference_data" / "torax" / "torax_basic_config_profiles.json"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("benchmark_torax_real_parity", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_artifact_provenance_is_complete() -> None:
    """The committed TORAX artifact carries full auditable provenance."""
    payload = json.loads(REFERENCE.read_text(encoding="utf-8"))
    provenance = payload["provenance"]
    assert provenance["code"] == "TORAX"
    assert provenance["licence"] == "Apache-2.0"
    assert provenance["torax_version"]
    assert len(provenance["config_sha256"]) == 64
    assert provenance["sim_error"] == "SimError.NO_ERROR"
    profiles = payload["profiles"]
    for key in ("rho_norm", "T_e_keV", "T_i_keV", "n_e_m3"):
        values = np.asarray(profiles[key], dtype=np.float64)
        assert values.size > 8
        assert bool(np.all(np.isfinite(values)))
    # Physical sanity of the acquired reference: monotone rho, hot core.
    rho = np.asarray(profiles["rho_norm"], dtype=np.float64)
    assert bool(np.all(np.diff(rho) > 0.0))
    assert profiles["T_e_keV"][0] > profiles["T_e_keV"][-1] > 0.0


def test_reference_integrity_check_rejects_missing_provenance(tmp_path: Path) -> None:
    """The gate fails closed on an artifact without provenance fields."""
    module = _load_module()
    broken = json.loads(REFERENCE.read_text(encoding="utf-8"))
    del broken["provenance"]["config_sha256"]
    target = tmp_path / "broken.json"
    target.write_text(json.dumps(broken), encoding="utf-8")
    module.REFERENCE = target
    with pytest.raises(ValueError, match="provenance incomplete"):
        module._load_reference()


def test_profile_checksum_is_stable_and_order_independent() -> None:
    """The checksum canonicalises key order, so it is reproducible."""
    module = _load_module()
    profiles = {"b": [1.0, 2.0], "a": [3.0]}
    reordered = {"a": [3.0], "b": [1.0, 2.0]}
    assert module._profile_checksum(profiles) == module._profile_checksum(reordered)


def test_limit_cycle_detector_flags_alternation() -> None:
    """The period-2 detector flags a crash-rebuild alternating tail."""
    module = _load_module()

    def _fake_trajectory(dt: float, steps: int) -> dict[str, Any]:
        alternating = np.array([8.7, 1.0, 8.7, 1.0, 8.7, 1.0, 8.7, 1.0])
        tail = alternating
        swings = np.abs(np.diff(tail))
        period2 = np.abs(tail[2:] - tail[:-2])
        return {
            "limit_cycle": bool(np.max(swings) > 2.0 and np.median(period2) < 0.5),
        }

    assert _fake_trajectory(0.5, 8)["limit_cycle"] is True


def test_build_report_records_divergence_and_findings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate reports acquired-reference status with recorded divergence."""
    module = _load_module()

    def _fake_evolve(dt: float, steps: int) -> dict[str, Any]:
        rho = np.linspace(0.0, 1.0, 12)
        te = 1.0 - 0.9 * rho**2
        return {
            "dt_s": dt,
            "steps": steps,
            "final_core_ti_kev": 1.63 if dt < 0.2 else 1.62,
            "peak_core_ti_kev": 2.1,
            "limit_cycle_detected": False,
            "finite": True,
            "te_kev": te,
            "rho_norm": rho,
        }

    monkeypatch.setattr(module, "_evolve_trajectory", _fake_evolve)
    report = cast(dict[str, Any], module.build_report())

    assert report["schema"] == "scpn-fusion-core.torax-real-parity.v1"
    assert report["passes_thresholds"] is True
    assert report["status"] == "real_torax_reference_acquired_divergence_documented"
    assert report["physics_equivalence_claimed"] is False
    assert report["reference"]["provenance"]["code"] == "TORAX"
    assert len(report["reference"]["profile_checksum_sha256"]) == 64
    assert "core_te_ratio_fine_over_torax" in report["divergence_metrics"]
    findings = report["solver_stability_findings"]
    assert findings["limit_cycle_at_coarse_dt"] is False
    assert findings["steady_state_dt_consistent"] is True
    assert findings["steady_state_core_ratio_coarse_over_fine"] == pytest.approx(
        1.62 / 1.63, rel=1e-9
    )
    assert findings["sawtooth_model_present_in_lane"] is False
    assert "resolved 2026-07-07" in findings["disposition"]


def test_check_report_detects_current_missing_and_stale_reports(tmp_path: Path) -> None:
    """The report drift checker accepts current payloads and rejects stale files."""
    module = _load_module()
    report = cast(dict[str, Any], module.build_report())
    current = tmp_path / "torax_real_parity.json"
    stale = tmp_path / "stale.json"
    missing = tmp_path / "missing.json"
    current.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stale.write_text(json.dumps({"schema": report["schema"]}) + "\n", encoding="utf-8")

    assert module.check_report(current) == []
    assert module.check_report(stale) == ["tracked TORAX real-parity report is stale"]
    assert module.check_report(missing) == [f"missing TORAX real-parity report: {missing}"]


def test_main_check_mode_reports_drift(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The CLI check mode returns non-zero for stale reports without rewriting."""
    module = _load_module()
    report = cast(dict[str, Any], module.build_report())
    current = tmp_path / "current.json"
    stale = tmp_path / "stale.json"
    current.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stale.write_text(json.dumps({"schema": report["schema"]}) + "\n", encoding="utf-8")

    monkeypatch.setattr(module, "build_report", lambda: report)

    assert module.main(["--output", str(current), "--check"]) == 0
    assert module.main(["--output", str(stale), "--check"]) == 1
