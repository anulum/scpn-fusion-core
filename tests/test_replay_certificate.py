# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Replay Certificate Tests
"""Tests for the deterministic replay certificate lane (M-2)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

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


def test_committed_numpy_floor_hashes_reproduce_here() -> None:
    """The ASSERTED cross-machine contract: the NumPy floor replays the
    committed certificate bit-identically for the pinned numpy wheel.

    When this test runs on a different machine class (e.g. the CI runner)
    a pass IS the cross-machine bit-identical evidence for the floor.
    """
    module = _load_module()
    committed = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    assert module.run_episode("numpy") == committed["numpy_floor"]["component_hashes"]


def test_verify_reports_fastest_tier_without_asserting_cross_machine() -> None:
    """verify_certificate records the fastest-tier comparison as evidence.

    Cross-machine bit-identity of the accelerated tier is deliberately NOT
    asserted (platform libm may differ); the field must exist either way.
    """
    module = _load_module()
    result = module.verify_certificate(CERTIFICATE)
    assert result["numpy_floor_bit_identical"] is True
    assert "fastest_tier_bit_identical" in result
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
