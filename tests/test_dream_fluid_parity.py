# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Real-DREAM Parity Gate Tests
"""Tests for the really-executed-DREAM reference artifact and comparison gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "benchmark_dream_fluid_parity.py"
REFERENCE = ROOT / "validation" / "reference_data" / "dream" / "dream_fluid_runaway_reference.json"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("benchmark_dream_fluid_parity", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_artifact_provenance_is_complete() -> None:
    """The committed DREAM artifact carries full auditable provenance."""
    payload = json.loads(REFERENCE.read_text(encoding="utf-8"))
    provenance = payload["provenance"]
    assert provenance["code"] == "DREAM"
    assert provenance["paper_doi"] == "10.1016/j.cpc.2021.108098"
    assert len(provenance["dream_git_sha"]) == 40
    assert len(provenance["settings_sha256"]) == 64
    series = payload["series"]
    n_re = np.asarray(series["n_re_m3"], dtype=np.float64)
    time_s = np.asarray(series["time_s"], dtype=np.float64)
    assert n_re.size > 50 and time_s.size == n_re.size
    assert bool(np.all(np.isfinite(n_re))) and bool(np.all(np.isfinite(time_s)))
    # Physical sanity of the acquired reference: strictly advancing time and
    # monotonically growing runaway density under constant E >> E_c.
    assert bool(np.all(np.diff(time_s) > 0.0))
    assert bool(np.all(np.diff(n_re) >= 0.0))
    for key in ("gammaDreicer", "GammaAva"):
        rates = np.asarray(series["other_fluid"][key], dtype=np.float64)
        assert rates.size > 0 and bool(np.all(np.isfinite(rates)))


def test_reference_integrity_check_rejects_missing_provenance(tmp_path: Path) -> None:
    """The gate fails closed on an artifact without provenance fields."""
    module = _load_module()
    broken = json.loads(REFERENCE.read_text(encoding="utf-8"))
    del broken["provenance"]["dream_git_sha"]
    target = tmp_path / "broken.json"
    target.write_text(json.dumps(broken), encoding="utf-8")
    module.REFERENCE = target
    with pytest.raises(ValueError, match="dream_git_sha"):
        module._load_reference()


def test_reference_integrity_check_rejects_foreign_code(tmp_path: Path) -> None:
    """The gate fails closed when the artifact is not a DREAM export."""
    module = _load_module()
    broken = json.loads(REFERENCE.read_text(encoding="utf-8"))
    broken["provenance"]["code"] = "OTHER"
    target = tmp_path / "foreign.json"
    target.write_text(json.dumps(broken), encoding="utf-8")
    module.REFERENCE = target
    with pytest.raises(ValueError, match="not a DREAM export"):
        module._load_reference()


def test_series_checksum_is_stable_and_order_independent() -> None:
    """The series checksum is canonical over key order."""
    module = _load_module()
    series_a = {"time_s": [0.0, 1.0], "n_re_m3": [1.0, 2.0]}
    series_b = {"n_re_m3": [1.0, 2.0], "time_s": [0.0, 1.0]}
    assert module._series_checksum(series_a) == module._series_checksum(series_b)
    assert module._series_checksum(series_a) != module._series_checksum({"time_s": [0.0]})


def test_build_report_records_ratios_and_findings() -> None:
    """The gate records finite cross-code ratios and the two rate findings."""
    module = _load_module()
    report = module.build_report()

    assert report["all_checks_passed"] is True
    rates = report["rates"]
    for key in (
        "ours_dreicer_m3_s",
        "dream_gamma_dreicer_m3_s",
        "dreicer_ratio_ours_over_dream",
        "ours_avalanche_exponential_s",
        "dream_gamma_ava_s",
        "avalanche_ratio_ours_over_dream",
    ):
        assert np.isfinite(rates[key]), key

    # The measured discrepancies are findings, not tolerances: the gate must
    # keep recording them until the rate models are reconciled at source.
    assert "avalanche_overestimate" in report["rate_model_findings"]
    assert "dreicer_underestimate" in report["rate_model_findings"]
    assert "NOT claimed" in report["claim_boundary"]


def test_gate_asserts_no_equivalence_thresholds() -> None:
    """No check in the gate asserts a physics-equivalence tolerance."""
    module = _load_module()
    report = module.build_report()
    for name in report["checks"]:
        assert "equivalence" not in name
        assert "tolerance" not in name
