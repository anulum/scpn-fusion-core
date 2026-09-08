# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — cfspopcon operating-map tests
"""Exercise the real installed upstream and separate SCPN reference process."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from validation.cfspopcon_operating_map import (
    SOURCE_COMMIT,
    run_operating_map,
    validate_request,
    verify_upstream,
)

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(os.environ.get("SCPN_REFERENCE_PYTHON", str(ROOT / ".venv/bin/python")))
FIXTURE = ROOT / "validation/reference_data/cfspopcon_uniform_dt.json"


@pytest.fixture()
def request_data() -> dict[str, Any]:
    """Load the committed uniform-DT physical case through its JSON interface."""
    return cast(dict[str, Any], json.loads(FIXTURE.read_text()))


def test_real_grid_preserves_power_balance_and_all_coordinates(
    request_data: dict[str, Any],
) -> None:
    """Run the actual25-point map and compare power accounting and rate units."""
    result = run_operating_map(request_data, reference_python=REFERENCE)
    assert result["source"]["commit"] == SOURCE_COMMIT
    assert len(result["points"]) == 25
    assert not any(result[key] for key in ("actionable", "evidence_claimed", "federated"))
    assert len(result["local_reference"]["source_sha256"]) == 64
    accepted = [p for p in result["points"] if p["status"] == "within_declared_limits"]
    assert accepted
    reasons = {reason for p in result["points"] for reason in p["reasons"]}
    assert {
        "negative_auxiliary_requirement",
        "auxiliary_limit_exceeded",
        "greenwald_limit_exceeded",
    } <= reasons
    for point in result["points"]:
        v = point["values"]
        assert v["reactivity_relative_difference"] < 1e-12
        assert v["stored_energy_mj"] / v["tau_e_s"] == pytest.approx(v["loss_power_mw"])
        assert v["loss_power_mw"] == pytest.approx(
            v["alpha_power_mw"] + 1.0 + 0.9 * v["auxiliary_launched_mw"]
        )
        if point["reasons"]:
            assert point["q_launched"] is None
        else:
            assert point["q_launched"] == pytest.approx(
                v["fusion_power_mw"] / (1.0 + v["auxiliary_launched_mw"])
            )
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema", "unknown"),
        ("extra", 1),
        ("major_radius_m", True),
        ("major_radius_m", "1.85"),
        ("h98", float("nan")),
        ("h98", float("inf")),
        ("h98", 0),
        ("ohmic_power_mw", -1),
        ("minor_radius_m", 2),
        ("coupling_fraction", 1.1),
        ("density_m3", []),
        ("density_m3", "bad"),
        ("density_m3", [0]),
        ("density_m3", [True]),
        ("density_m3", [1e20, 1e20]),
        ("temperature_kev", [0.1]),
        ("temperature_kev", [101]),
        ("temperature_kev", list(range(1, 258))),
        ("density_m3", list(range(1, 100))),
    ],
)
def test_invalid_requests_refuse_before_runtime(
    request_data: dict[str, Any], key: str, value: Any
) -> None:
    """Refuse malformed physical inputs before any upstream call or subprocess."""
    request_data[key] = value
    with pytest.raises(ValueError):
        run_operating_map(request_data, reference_python=REFERENCE)


def test_missing_field_is_refused(request_data: dict[str, Any]) -> None:
    """Require every declared physical parameter rather than guessing defaults."""
    request_data.pop("ohmic_power_mw")
    with pytest.raises(ValueError):
        validate_request(request_data)


def test_temperature_endpoints_and_zero_ohmic_are_valid(request_data: dict[str, Any]) -> None:
    """Exercise actual upstream formula endpoints without silently clipping them."""
    request_data.update(temperature_kev=[0.2, 100], density_m3=[1e20], ohmic_power_mw=0.0)
    result = run_operating_map(request_data, reference_python=REFERENCE)
    assert len(result["points"]) == 2
    assert all(p["values"]["reactivity_m3_s"] > 0 for p in result["points"])


def test_source_custody_rejects_missing_and_changed_package(tmp_path: Path) -> None:
    """Reject absent or changed source through the public package verifier."""
    with pytest.raises(ValueError, match="source"):
        verify_upstream(tmp_path)
    root = Path(cast(str, importlib.import_module("cfspopcon").__file__)).parent
    copy = tmp_path / "package"
    shutil.copytree(root, copy, ignore=shutil.ignore_patterns("__pycache__"))
    verify_upstream(copy)
    with (copy / "__init__.py").open("a") as stream:
        stream.write("\n# changed source\n")
    with pytest.raises(ValueError, match="source"):
        verify_upstream(copy)


def test_reference_process_failure_is_not_substituted(request_data: dict[str, Any]) -> None:
    """Propagate a real reference-process failure without synthetic fallback."""
    with pytest.raises(subprocess.CalledProcessError):
        run_operating_map(request_data, reference_python=Path("/bin/false"))


def test_cli_writes_finite_design_artifact(tmp_path: Path) -> None:
    """Exercise the command-line request-to-artifact boundary in real processes."""
    output = tmp_path / "map.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "validation.cfspopcon_operating_map",
            str(FIXTURE),
            str(output),
            "--reference-python",
            str(REFERENCE),
        ],
        cwd=ROOT,
        check=True,
        timeout=45,
    )
    result = json.loads(output.read_text())
    assert len(result["points"]) == 25
    assert result["request"] == json.loads(FIXTURE.read_text())


def test_overflow_is_retained_as_failed_point(request_data: dict[str, Any]) -> None:
    """Retain a numerically overflowing state instead of dropping its coordinate."""
    request_data.update(density_m3=[1e308], temperature_kev=[10.0])
    result = run_operating_map(request_data, reference_python=REFERENCE)
    assert len(result["points"]) == 1
    point = result["points"][0]
    assert point["status"] in ("calculation_failed", "outside_declared_limits")
    assert point["q_launched"] is None
    assert point["reasons"]
    json.dumps(result, allow_nan=False)


def test_changed_reference_reactivity_is_detected(
    request_data: dict[str, Any], tmp_path: Path
) -> None:
    """Detect an actual changed fit coefficient in an isolated SCPN checkout."""
    shutil.copytree(ROOT / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__"))
    target = tmp_path / "validation/reference_data/itpa"
    target.mkdir(parents=True)
    shutil.copyfile(
        ROOT / "validation/reference_data/itpa/ipb98y2_coefficients.json",
        target / "ipb98y2_coefficients.json",
    )
    shutil.copyfile(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    source = tmp_path / "src/scpn_fusion/core/uncertainty.py"
    original = source.read_text()
    assert "_C1 = 1.17302e-9" in original
    source.write_text(original.replace("_C1 = 1.17302e-9", "_C1 = 1.27302e-9"))
    request_data.update(density_m3=[2e20], temperature_kev=[10.0])
    result = run_operating_map(request_data, reference_python=REFERENCE, reference_root=tmp_path)
    point = result["points"][0]
    assert "reactivity_code_parity_failed" in point["reasons"]
    assert point["q_launched"] is None


def test_arithmetic_failure_keeps_coordinate(request_data: dict[str, Any]) -> None:
    """Return an explicit failed point when finite geometry overflows its volume."""
    request_data.update(
        major_radius_m=1e201, minor_radius_m=1e200, density_m3=[1e20], temperature_kev=[10.0]
    )
    result = run_operating_map(request_data, reference_python=REFERENCE)
    assert result["points"][0]["status"] == "calculation_failed"
    assert result["points"][0]["reasons"] == ["OverflowError"]
