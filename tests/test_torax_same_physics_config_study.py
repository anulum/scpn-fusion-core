# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TORAX Same-Physics Config Study Tests
"""Tests for the TORAX/native same-physics configuration study."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "validation" / "reference_data" / "torax" / "torax_basic_config_profiles.json"
CONFIG = ROOT / "validation" / "iter_config.json"


def _load_module() -> Any:
    """Load the same-physics study module through the production import path."""
    return importlib.reload(importlib.import_module("validation.torax_same_physics_config_study"))


def test_reference_profiles_convert_density_to_native_units() -> None:
    """TORAX SI density converts to the native 1e19 m^-3 transport unit."""
    module = _load_module()
    reference = module.load_torax_reference(REFERENCE)

    profiles = module.reference_profiles(reference)

    raw = reference["profiles"]
    np.testing.assert_allclose(profiles["rho_norm"], raw["rho_norm"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(profiles["T_e_keV"], raw["T_e_keV"], rtol=0.0, atol=0.0)
    assert profiles["n_e_1e19_m3"][0] == pytest.approx(raw["n_e_m3"][0] / 1.0e19)


def test_initialise_native_solver_from_torax_sets_real_solver_state() -> None:
    """The native solver is initialized from the tracked TORAX profiles."""
    module = _load_module()
    reference = module.load_torax_reference(REFERENCE)

    solver = module.initialise_native_solver_from_torax(reference, config_path=CONFIG)

    profiles = module.reference_profiles(reference)
    assert solver.nr == profiles["rho_norm"].size
    np.testing.assert_allclose(solver.rho, profiles["rho_norm"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(solver.Te, profiles["T_e_keV"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(solver.Ti, profiles["T_i_keV"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(solver.ne, profiles["n_e_1e19_m3"], rtol=0.0, atol=0.0)
    assert np.all(solver.chi_e == 1.0)
    assert np.all(solver.n_impurity == 0.0)


def test_build_report_is_fail_closed_for_threshold_tightening() -> None:
    """The report records initialized-profile readiness without equivalence claims."""
    module = _load_module()

    report = module.build_report(reference_path=REFERENCE, config_path=CONFIG)

    assert report["schema"] == "scpn-fusion-core.torax-same-physics-config-study.v1"
    assert report["status"] == "same_initial_profile_config_ready_thresholds_blocked"
    assert report["passes_thresholds"] is True
    assert report["same_physics_ready"] is False
    assert report["threshold_tightening_ready"] is False
    assert report["physics_equivalence_claimed"] is False
    native = report["native_solver"]
    assert native["finite_initial_state"] is True
    assert native["finite_after_probe"] is True
    assert native["transport_model_after_update"] == "reduced_multichannel_analytic"
    assert native["one_step_probe"]["dt_s"] == pytest.approx(1.0e-3)
    blockers = set(report["threshold_blockers"])
    assert "native_transport_model" in blockers
    assert "sources_and_boundary_conditions" in blockers
    assert "time_integration_contract" in blockers


def test_write_and_check_report_roundtrip(tmp_path: Path) -> None:
    """Fresh same-physics reports pass the drift checker."""
    module = _load_module()
    report_json = tmp_path / "same_physics.json"
    report_md = tmp_path / "same_physics.md"

    report = module.write_report(
        reference_path=REFERENCE,
        config_path=CONFIG,
        report_json=report_json,
        report_md=report_md,
    )

    assert report_json.exists()
    assert report_md.exists()
    assert report["threshold_tightening_ready"] is False
    assert (
        module.check_report(
            reference_path=REFERENCE,
            config_path=CONFIG,
            report_json=report_json,
            report_md=report_md,
        )
        == []
    )
    assert "# TORAX Same-Physics Configuration Study" in report_md.read_text(encoding="utf-8")


def test_check_report_detects_missing_and_stale_outputs(tmp_path: Path) -> None:
    """The drift checker reports missing and edited tracked outputs."""
    module = _load_module()
    report_json = tmp_path / "same_physics.json"
    report_md = tmp_path / "same_physics.md"

    missing = module.check_report(
        reference_path=REFERENCE,
        config_path=CONFIG,
        report_json=report_json,
        report_md=report_md,
    )
    assert any("missing same-physics JSON" in error for error in missing)
    assert any("missing same-physics Markdown" in error for error in missing)

    module.write_report(
        reference_path=REFERENCE,
        config_path=CONFIG,
        report_json=report_json,
        report_md=report_md,
    )
    report_json.write_text('{"stale": true}', encoding="utf-8")
    report_md.write_text("# stale\n", encoding="utf-8")

    stale = module.check_report(
        reference_path=REFERENCE,
        config_path=CONFIG,
        report_json=report_json,
        report_md=report_md,
    )
    assert "tracked same-physics JSON report is stale" in stale
    assert "tracked same-physics Markdown report is stale" in stale


def test_main_writes_and_checks_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI write and check modes succeed for fresh reports."""
    module = _load_module()
    report_json = tmp_path / "same_physics.json"
    report_md = tmp_path / "same_physics.md"
    args = [
        "--reference",
        str(REFERENCE),
        "--config",
        str(CONFIG),
        "--report-json",
        str(report_json),
        "--report-md",
        str(report_md),
    ]

    assert module.main(args) == 0
    assert "same_initial_profile_config_ready_thresholds_blocked" in capsys.readouterr().out
    assert module.main([*args, "--check"]) == 0
    assert "up to date" in capsys.readouterr().out


def test_main_check_returns_failure_for_missing_reports(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI check mode fails when tracked reports are missing."""
    module = _load_module()
    rc = module.main(
        [
            "--reference",
            str(REFERENCE),
            "--config",
            str(CONFIG),
            "--report-json",
            str(tmp_path / "missing.json"),
            "--report-md",
            str(tmp_path / "missing.md"),
            "--check",
        ]
    )

    assert rc == 1
    assert "TORAX SAME-PHYSICS CONFIG ERROR" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("edit_key", "message"),
    [
        ("schema", "unexpected schema"),
        ("provenance", "TORAX provenance"),
        ("profiles", "profile data"),
        ("missing", "missing keys"),
        ("misaligned", "misaligned"),
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
    """Reference loading fails closed for malformed same-physics inputs."""
    module = _load_module()
    broken = cast(dict[str, Any], json.loads(REFERENCE.read_text(encoding="utf-8")))
    if edit_key == "schema":
        broken["schema"] = "wrong"
    elif edit_key == "provenance":
        broken["provenance"] = {"code": "not-torax"}
    elif edit_key == "profiles":
        broken["profiles"] = []
    elif edit_key == "missing":
        del broken["profiles"]["T_i_keV"]
    elif edit_key == "misaligned":
        broken["profiles"]["T_i_keV"] = broken["profiles"]["T_i_keV"][:-1]
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
