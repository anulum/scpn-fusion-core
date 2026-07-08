# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TORAX IMAS Interchange Tests
"""Tests for the TORAX-to-IMAS interchange fixture generator."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "validation" / "reference_data" / "torax" / "torax_basic_config_profiles.json"


def _load_module() -> Any:
    """Load the validation script as an importable module."""
    return importlib.reload(importlib.import_module("validation.torax_imas_interchange"))


def test_torax_reference_to_state_preserves_profiles_and_units() -> None:
    """TORAX profiles map to the internal state with explicit density scaling."""
    module = _load_module()
    reference = module.load_torax_reference(REFERENCE)

    state = module.torax_reference_to_state(reference)

    profiles = reference["profiles"]
    assert state["rho_norm"] == profiles["rho_norm"]
    assert state["electron_temp_keV"] == profiles["T_e_keV"]
    assert np.asarray(state["electron_density_1e20_m3"])[0] == pytest.approx(
        profiles["n_e_m3"][0] / 1.0e20
    )


def test_build_core_profiles_ids_writes_imas_units() -> None:
    """The IMAS fixture uses eV and m^-3 as expected by core_profiles."""
    module = _load_module()
    reference = module.load_torax_reference(REFERENCE)

    ids = module.build_core_profiles_ids(reference)

    profile = ids["profiles_1d"][0]
    profiles = reference["profiles"]
    assert profile["time"] == pytest.approx(reference["final_time_s"])
    assert profile["grid"]["rho_tor_norm"] == profiles["rho_norm"]
    assert profile["electrons"]["temperature"][0] == pytest.approx(profiles["T_e_keV"][0] * 1.0e3)
    assert profile["electrons"]["density"][-1] == pytest.approx(profiles["n_e_m3"][-1])
    provenance = ids["ids_properties"]["provenance"]
    assert provenance["source_code"] == "TORAX"
    assert len(provenance["source_config_sha256"]) == 64


def test_write_and_check_artifacts_roundtrip(tmp_path: Path) -> None:
    """Generated IMAS and report artifacts pass their own drift check."""
    module = _load_module()
    ids_output = tmp_path / "torax_core_profiles_ids.json"
    report_json = tmp_path / "torax_imas_interchange.json"
    report_md = tmp_path / "torax_imas_interchange.md"

    report = module.write_artifacts(
        reference_path=REFERENCE,
        ids_output=ids_output,
        report_json=report_json,
        report_md=report_md,
    )

    assert report["schema"] == "scpn-fusion-core.torax-imas-interchange.v1"
    assert report["passes_thresholds"] is True
    assert report["physics_equivalence_claimed"] is False
    assert (
        module.check_artifacts(
            reference_path=REFERENCE,
            ids_output=ids_output,
            report_json=report_json,
            report_md=report_md,
        )
        == []
    )
    assert "# TORAX IMAS Interchange" in report_md.read_text(encoding="utf-8")


def test_check_artifacts_detects_stale_fixture(tmp_path: Path) -> None:
    """The drift check catches hand-edited IMAS fixture output."""
    module = _load_module()
    ids_output = tmp_path / "torax_core_profiles_ids.json"
    report_json = tmp_path / "torax_imas_interchange.json"
    report_md = tmp_path / "torax_imas_interchange.md"
    module.write_artifacts(
        reference_path=REFERENCE,
        ids_output=ids_output,
        report_json=report_json,
        report_md=report_md,
    )
    stale = cast(dict[str, Any], json.loads(ids_output.read_text(encoding="utf-8")))
    stale["profiles_1d"][0]["electrons"]["temperature"][0] = -1.0
    ids_output.write_text(json.dumps(stale), encoding="utf-8")

    errors = module.check_artifacts(
        reference_path=REFERENCE,
        ids_output=ids_output,
        report_json=report_json,
        report_md=report_md,
    )

    assert "tracked IMAS fixture is stale" in errors


def test_load_torax_reference_rejects_misaligned_profiles(tmp_path: Path) -> None:
    """Reference loading fails closed when TORAX vectors are not aligned."""
    module = _load_module()
    broken = cast(dict[str, Any], json.loads(REFERENCE.read_text(encoding="utf-8")))
    broken["profiles"]["n_e_m3"] = broken["profiles"]["n_e_m3"][:-1]
    path = tmp_path / "broken_torax.json"
    path.write_text(json.dumps(broken), encoding="utf-8")

    with pytest.raises(ValueError, match="misaligned"):
        module.load_torax_reference(path)


@pytest.mark.parametrize(
    ("edit_key", "message"),
    [
        ("schema", "unexpected schema"),
        ("provenance", "TORAX provenance"),
        ("profiles", "must contain profiles"),
        ("missing_profile_key", "missing keys"),
        ("non_increasing_rho", "strictly increasing"),
        ("string_vector", "must be a sequence"),
        ("short_vector", "at least 2"),
        ("nan_vector", "finite values"),
    ],
)
def test_load_torax_reference_rejects_invalid_contracts(
    tmp_path: Path,
    edit_key: str,
    message: str,
) -> None:
    """Reference loading rejects malformed profile and provenance contracts."""
    module = _load_module()
    broken = cast(dict[str, Any], json.loads(REFERENCE.read_text(encoding="utf-8")))
    if edit_key == "schema":
        broken["schema"] = "wrong"
    elif edit_key == "provenance":
        broken["provenance"] = {"code": "not-torax"}
    elif edit_key == "profiles":
        broken["profiles"] = []
    elif edit_key == "missing_profile_key":
        del broken["profiles"]["T_e_keV"]
    elif edit_key == "non_increasing_rho":
        broken["profiles"]["rho_norm"][1] = broken["profiles"]["rho_norm"][0]
    elif edit_key == "string_vector":
        broken["profiles"]["rho_norm"] = "0,1"
    elif edit_key == "short_vector":
        broken["profiles"]["rho_norm"] = [0.0]
    elif edit_key == "nan_vector":
        broken["profiles"]["rho_norm"][1] = float("nan")
    path = tmp_path / f"broken_{edit_key}.json"
    path.write_text(json.dumps(broken), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        module.load_torax_reference(path)


@pytest.mark.parametrize(
    ("edit_key", "message"),
    [
        ("profiles", "exactly one"),
        ("rho", "rho_tor_norm"),
        ("temperature", "temperature"),
        ("density", "density"),
    ],
)
def test_validate_ids_against_reference_rejects_mismatch(
    edit_key: str,
    message: str,
) -> None:
    """The IMAS fixture validator catches array-level drift."""
    module = _load_module()
    reference = module.load_torax_reference(REFERENCE)
    ids = module.build_core_profiles_ids(reference)
    if edit_key == "profiles":
        ids["profiles_1d"] = []
    else:
        profile = ids["profiles_1d"][0]
        if edit_key == "rho":
            profile["grid"]["rho_tor_norm"][0] = 0.1
        elif edit_key == "temperature":
            profile["electrons"]["temperature"][0] = -1.0
        elif edit_key == "density":
            profile["electrons"]["density"][0] = -1.0

    with pytest.raises(ValueError, match=message):
        module._validate_ids_against_reference(ids, reference)


def test_omas_roundtrip_status_executes_when_dependency_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optional OMAS status path records a successful runtime roundtrip."""
    module = _load_module()
    reference = module.load_torax_reference(REFERENCE)
    ids = module.build_core_profiles_ids(reference)
    monkeypatch.setattr(module, "HAS_OMAS", True)
    monkeypatch.setattr(module, "ids_to_omas_core_profiles", lambda payload: {"ids": payload})
    monkeypatch.setattr(module, "omas_core_profiles_to_ids", lambda ods: ods["ids"])

    status = module._omas_roundtrip_status(ids)

    assert status["available"] is True
    assert status["roundtrip_executed"] is True
    assert status["status"] == "roundtrip_passed"
    assert len(status["roundtrip_checksum_sha256"]) == 64


def test_check_artifacts_reports_missing_and_stale_reports(tmp_path: Path) -> None:
    """The drift checker reports missing and hand-edited report artifacts."""
    module = _load_module()
    ids_output = tmp_path / "torax_core_profiles_ids.json"
    report_json = tmp_path / "torax_imas_interchange.json"
    report_md = tmp_path / "torax_imas_interchange.md"

    missing_errors = module.check_artifacts(
        reference_path=REFERENCE,
        ids_output=ids_output,
        report_json=report_json,
        report_md=report_md,
    )
    assert any("missing IMAS fixture" in error for error in missing_errors)
    assert any("missing interchange report" in error for error in missing_errors)
    assert any("missing interchange Markdown" in error for error in missing_errors)

    module.write_artifacts(
        reference_path=REFERENCE,
        ids_output=ids_output,
        report_json=report_json,
        report_md=report_md,
    )
    report_json.write_text('{"stale": true}', encoding="utf-8")
    report_md.write_text("# stale\n", encoding="utf-8")

    stale_errors = module.check_artifacts(
        reference_path=REFERENCE,
        ids_output=ids_output,
        report_json=report_json,
        report_md=report_md,
    )
    assert "tracked interchange JSON report is stale" in stale_errors
    assert "tracked interchange Markdown report is stale" in stale_errors


def test_main_writes_and_checks_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI write and check paths return success for fresh artifacts."""
    module = _load_module()
    ids_output = tmp_path / "torax_core_profiles_ids.json"
    report_json = tmp_path / "torax_imas_interchange.json"
    report_md = tmp_path / "torax_imas_interchange.md"
    args = [
        "--reference",
        str(REFERENCE),
        "--ids-output",
        str(ids_output),
        "--report-json",
        str(report_json),
        "--report-md",
        str(report_md),
    ]

    assert module.main(args) == 0
    assert "torax_core_profiles_imas_fixture_ready" in capsys.readouterr().out
    assert module.main([*args, "--check"]) == 0
    assert "up to date" in capsys.readouterr().out


def test_main_check_returns_failure_for_drift(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI check path returns non-zero when artifacts are missing."""
    module = _load_module()
    rc = module.main(
        [
            "--reference",
            str(REFERENCE),
            "--ids-output",
            str(tmp_path / "missing_ids.json"),
            "--report-json",
            str(tmp_path / "missing_report.json"),
            "--report-md",
            str(tmp_path / "missing_report.md"),
            "--check",
        ]
    )

    assert rc == 1
    assert "TORAX IMAS INTERCHANGE ERROR" in capsys.readouterr().err
