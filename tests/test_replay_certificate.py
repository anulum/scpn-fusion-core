# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Replay Certificate Tests
"""Tests for the deterministic replay certificate and verifier contract."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import runpy
import sys
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "replay_certificate.py"
CERTIFICATE = ROOT / "validation" / "reference_data" / "replay" / "replay_certificate.json"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("replay_certificate", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _committed_payload() -> dict[str, Any]:
    payload = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    return cast(dict[str, Any], payload)


def _set_nested(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target: dict[str, Any] = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def test_committed_certificate_structure() -> None:
    """The committed certificate carries hashes, claims, and the manifest."""
    payload = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    assert payload["schema"] == "scpn-fusion-core.replay-certificate.v1"
    for section in ("numpy_floor", "fastest_tier"):
        hashes = payload[section]["component_hashes"]
        assert set(hashes) == {
            "equilibrium_multigrid",
            "phase_control_upde",
            "disruption_indicator",
        }
        assert all(len(h) == 64 for h in hashes.values())
        assert len(payload[section]["combined_hash"]) == 64
        assert payload[section]["claim"]
    env = payload["environment"]
    assert env["numpy"] and env["python"] and env["machine"]
    assert set(env["fastest_tier_selection"]) == {
        "multigrid_solve",
        "upde_run",
        "simulate_tearing_mode",
    }


def test_numpy_floor_episode_is_bit_identical_across_runs() -> None:
    """Two same-process NumPy-floor episodes hash identically."""
    module = _load_module()
    assert module.run_episode("numpy") == module.run_episode("numpy")


def test_fastest_tier_episode_is_bit_identical_across_runs() -> None:
    """Two same-process fastest-tier episodes hash identically."""
    module = _load_module()
    assert module.run_episode("fastest") == module.run_episode("fastest")


def test_committed_hashes_reproduce_per_the_environment_conditional_claim() -> None:
    """Cross-machine claim as revised by the first two-machine comparison.

    On the generating machine class (environment matches the certificate)
    the full NumPy-floor hash map must reproduce. On any other machine the
    asserted invariant is the transcendental-free ``disruption_indicator``
    component — the first CI comparison (run 28841804121) proved that
    components exercising vectorised np.exp/np.sin round differently
    across CPU microarchitectures even with an identical numpy wheel.
    """
    module = _load_module()
    result = module.verify_certificate(CERTIFICATE)
    if result["environment_matches"]:
        assert result["numpy_floor_bit_identical"] is True
    else:
        assert result["numpy_component_matches"]["disruption_indicator"] is True


def test_verify_reports_fastest_tier_without_asserting_cross_machine() -> None:
    """verify_certificate records the fastest-tier comparison as evidence.

    Cross-machine bit-identity of the accelerated tier is deliberately NOT
    asserted (platform libm may differ); the fields must exist either way.
    """
    module = _load_module()
    result = module.verify_certificate(CERTIFICATE)
    assert "numpy_floor_bit_identical" in result
    assert "fastest_tier_bit_identical" in result
    assert "environment_matches" in result
    assert result["verifier_environment"]["numpy"]


def test_run_episode_rejects_unknown_tier() -> None:
    """The tier argument is a closed vocabulary."""
    module = _load_module()
    with pytest.raises(ValueError, match="tier must be"):
        module.run_episode("cuda")


def test_verify_rejects_foreign_schema(tmp_path: Path) -> None:
    """Verification fails closed on an artifact with the wrong schema."""
    module = _load_module()
    broken = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    broken["schema"] = "other.schema"
    target = tmp_path / "broken.json"
    target.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected certificate schema"):
        module.verify_certificate(target)


def test_validate_certificate_accepts_committed_contract() -> None:
    module = _load_module()
    payload = _committed_payload()

    assert module.validate_certificate(payload) is payload


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("schema",), "other.schema", "unexpected certificate schema"),
        (("episode_seed",), True, "episode_seed must equal"),
        (("episode_seed",), 2025, "episode_seed must equal"),
        (("numpy_floor",), [], "numpy_floor must be an object"),
        (
            ("numpy_floor", "component_hashes"),
            [],
            "component_hashes has unexpected components",
        ),
        (
            ("numpy_floor", "component_hashes", "equilibrium_multigrid"),
            7,
            "64-character lowercase",
        ),
        (
            ("numpy_floor", "component_hashes", "equilibrium_multigrid"),
            "a" * 63,
            "64-character lowercase",
        ),
        (
            ("numpy_floor", "component_hashes", "equilibrium_multigrid"),
            "A" * 64,
            "64-character lowercase",
        ),
        (("numpy_floor", "combined_hash"), "0" * 64, "does not match"),
        (("numpy_floor", "claim"), " ", "claim must be a non-empty string"),
        (("environment",), [], "environment keys do not match"),
        (("environment", "python"), "", "environment.python must be"),
        (
            ("environment", "fastest_tier_selection"),
            [],
            "fastest_tier_selection has unexpected components",
        ),
        (
            ("environment", "fastest_tier_selection", "multigrid_solve"),
            "",
            "values must be non-empty strings",
        ),
    ],
)
def test_validate_certificate_rejects_invalid_values(
    path: tuple[str, ...], value: Any, message: str
) -> None:
    module = _load_module()
    payload = copy.deepcopy(_committed_payload())
    _set_nested(payload, path, value)

    with pytest.raises(ValueError, match=message):
        module.validate_certificate(payload)


def test_validate_certificate_rejects_non_object_and_shape_drift() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="certificate must be an object"):
        module.validate_certificate([])

    payload = _committed_payload()
    payload["extra"] = True
    with pytest.raises(ValueError, match="certificate keys"):
        module.validate_certificate(payload)

    payload = _committed_payload()
    payload["numpy_floor"]["extra"] = True
    with pytest.raises(ValueError, match="numpy_floor keys"):
        module.validate_certificate(payload)

    payload = _committed_payload()
    payload["numpy_floor"]["component_hashes"].pop("phase_control_upde")
    with pytest.raises(ValueError, match="unexpected components"):
        module.validate_certificate(payload)

    payload = _committed_payload()
    payload["environment"].pop("machine")
    with pytest.raises(ValueError, match="environment keys"):
        module.validate_certificate(payload)

    payload = _committed_payload()
    payload["environment"]["fastest_tier_selection"].pop("upde_run")
    with pytest.raises(ValueError, match="unexpected components"):
        module.validate_certificate(payload)


def test_verify_rejects_tampered_combined_hash_before_replay(tmp_path: Path) -> None:
    module = _load_module()
    payload = _committed_payload()
    payload["numpy_floor"]["combined_hash"] = "0" * 64
    target = tmp_path / "tampered.json"
    target.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match component hashes"):
        module.verify_certificate(target)


def test_build_certificate_validates_double_run_and_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    numpy_hashes = {name: "0" * 64 for name in module._COMPONENTS}
    fastest_hashes = {name: "1" * 64 for name in module._COMPONENTS}
    results = [numpy_hashes, numpy_hashes, fastest_hashes, fastest_hashes]
    monkeypatch.setattr(module, "run_episode", lambda _tier: results.pop(0))
    monkeypatch.setattr(
        module, "_environment_manifest", lambda: _committed_payload()["environment"]
    )

    certificate = module.build_certificate()

    assert module.validate_certificate(certificate) == certificate
    assert certificate["numpy_floor"]["component_hashes"] == numpy_hashes
    assert certificate["fastest_tier"]["component_hashes"] == fastest_hashes


@pytest.mark.parametrize(
    ("results", "message"),
    [
        ([{"x": "0"}, {"x": "1"}], "NumPy floor failed"),
        (
            [{"x": "0"}, {"x": "0"}, {"x": "1"}, {"x": "2"}],
            "fastest tier failed",
        ),
    ],
)
def test_build_certificate_rejects_nondeterministic_runs(
    monkeypatch: pytest.MonkeyPatch,
    results: list[dict[str, str]],
    message: str,
) -> None:
    module = _load_module()
    pending = list(results)
    monkeypatch.setattr(module, "run_episode", lambda _tier: pending.pop(0))

    with pytest.raises(RuntimeError, match=message):
        module.build_certificate()


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            {
                "environment_matches": True,
                "numpy_floor_bit_identical": True,
                "numpy_component_matches": {"disruption_indicator": False},
            },
            0,
        ),
        (
            {
                "environment_matches": True,
                "numpy_floor_bit_identical": False,
                "numpy_component_matches": {"disruption_indicator": True},
            },
            1,
        ),
        (
            {
                "environment_matches": False,
                "numpy_floor_bit_identical": False,
                "numpy_component_matches": {"disruption_indicator": True},
            },
            0,
        ),
        (
            {
                "environment_matches": False,
                "numpy_floor_bit_identical": True,
                "numpy_component_matches": {"disruption_indicator": False},
            },
            1,
        ),
    ],
)
def test_main_applies_environment_conditional_verification_policy(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    result: dict[str, Any],
    expected: int,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "verify_certificate", lambda _path: result)

    assert module.main(["--verify", "--output", str(CERTIFICATE)]) == expected
    assert "environment_matches" in capsys.readouterr().out


def test_main_writes_generated_certificate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    payload = _committed_payload()
    monkeypatch.setattr(module, "build_certificate", lambda: payload)
    target = tmp_path / "nested" / "certificate.json"

    assert module.main(["--output", str(target)]) == 0
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert f"wrote {target}" in capsys.readouterr().out


def test_equilibrium_episode_fails_closed_when_solver_does_not_converge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    from scpn_fusion.core import _multi_compat_providers as providers

    def failed_solve(*_args: Any, **_kwargs: Any) -> tuple[list[float], float, int, bool]:
        return [0.0], 1.0, 200, False

    monkeypatch.setattr(providers, "_numpy_multigrid_solve", failed_solve)
    with pytest.raises(RuntimeError, match="did not converge"):
        module._episode_equilibrium("numpy")


def test_script_entry_point_verifies_committed_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [str(MODULE_PATH), "--verify", "--output", str(CERTIFICATE)],
    )

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(MODULE_PATH), run_name="__main__")

    assert exc_info.value.code == 0
