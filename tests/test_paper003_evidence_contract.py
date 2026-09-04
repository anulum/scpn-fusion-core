# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — submission 003 exact-evidence contract tests.
"""Verify submission 003 against immutable evidence and production surfaces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from scpn_fusion.control.fueling_mode import build_ice_pellet_fueling_controller


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = (
    ROOT / "papers" / "submissions" / "003_stochastic_petri_net_tokamak_control_conference_abstract"
)
EVIDENCE = SUBMISSION / "evidence"


def _load_json(path: Path) -> dict[str, Any]:
    """Load one test evidence object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 digest of exact bytes."""
    return hashlib.sha256(payload).hexdigest()


def test_companion_evidence_matches_immutable_git_objects() -> None:
    """Prove that every packaged result is an exact historical Git object."""
    manifest = _load_json(EVIDENCE / "evidence_manifest.json")
    for name, custody in manifest["source_custody"].items():
        object_spec = f"{custody['source_revision']}:{custody['source_path']}"
        source_bytes = subprocess.run(
            ["git", "show", object_spec],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        blob_oid = subprocess.run(
            ["git", "rev-parse", object_spec],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        assert blob_oid == custody["git_blob_oid_sha1"]
        assert _sha256_bytes(source_bytes) == custody["source_sha256"]
        assert (EVIDENCE / name).read_bytes() == source_bytes
        assert custody["packaging_transform"] == "exact byte copy"


def test_claimed_implementation_matches_the_bound_revision() -> None:
    """Bind every described implementation surface to the package revision."""
    manifest = _load_json(EVIDENCE / "evidence_manifest.json")
    revision = manifest["repository_revision"]
    for relative_path, custody in manifest["implementation_sources"].items():
        object_spec = f"{revision}:{relative_path}"
        source_bytes = subprocess.run(
            ["git", "show", object_spec],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        blob_oid = subprocess.run(
            ["git", "rev-parse", object_spec],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        assert blob_oid == custody["git_blob_oid_sha1"]
        assert _sha256_bytes(source_bytes) == custody["sha256"]
        assert (ROOT / relative_path).read_bytes() == source_bytes


def test_manuscript_reports_exact_results_and_non_claims() -> None:
    """Keep every quantitative result coupled to its scientific boundary."""
    manuscript = (SUBMISSION / "manuscript.tex").read_text(encoding="utf-8")
    vertical = _load_json(EVIDENCE / "vertical_stability_reduced_order.json")
    diiid = _load_json(EVIDENCE / "real_diiid_145419_validation.json")
    sparc = _load_json(EVIDENCE / "sparc_geqdsk_rmse_benchmark.json")

    assert vertical["controllers"]["PID"]["settling_ms"] == 25.0
    assert vertical["controllers"]["LQR"]["settling_ms"] == 39.0
    assert vertical["controllers"]["SNN"]["outcome"] == "diverges"
    assert vertical["controllers"]["SNN"]["settling_ms"] is None
    assert vertical["controllers"]["SNN"]["peak_abs_displacement_mm"] == 159.68272099657466
    assert "159.683 mm" in manuscript
    assert "does not execute a compiled Petri artifact" in manuscript

    assert diiid["full_domain_reproduction"]["deep_rms_rel_span"] == 0.019084943379848895
    assert diiid["full_domain_cold_start"]["deep_rms_rel_span"] == 1.2678812266175519
    assert "1.908\\%" in manuscript
    assert "126.788\\%" in manuscript
    assert "not blind prediction" in diiid["disclosure"]

    gated = [case for case in sparc["cases"] if case["gated"] is True]
    assert len(gated) == 16
    assert all(case["passes"] is True for case in gated)
    assert sum(case["geqdsk_adapted_source_contract_pass"] is True for case in gated) == 8
    assert "16/16" in manuscript
    assert "8/16" in manuscript
    assert "no claim of experimental plasma control" in manuscript


def test_production_petri_controller_preserves_runtime_bounds() -> None:
    """Exercise the public Petri-to-controller path without widening paper claims."""
    controller = build_ice_pellet_fueling_controller()

    assert controller.artifact.meta.name == "ice_pellet_fueling_controller"
    assert controller.artifact.nT == 4
    assert len(controller.artifact.topology.transitions) == controller.artifact.nT
    assert controller.artifact.readout.abs_max == [5000.0, 5000.0]

    action = controller.step({"R_axis_m": 5.0, "Z_axis_m": 1.0}, 0)
    assert set(action) == {"dI_PF3_A", "dI_PF_topbot_A"}
    assert abs(float(action["dI_PF3_A"])) <= 5000.0
    assert abs(float(action["dI_PF_topbot_A"])) <= 5000.0
    assert all(0.0 <= value <= 1.0 for value in controller.marking)


def test_evidence_generator_is_byte_deterministic() -> None:
    """Require regeneration to leave all exact evidence bytes unchanged."""
    tracked_paths = [
        EVIDENCE / "vertical_stability_reduced_order.json",
        EVIDENCE / "real_diiid_145419_validation.json",
        EVIDENCE / "sparc_geqdsk_rmse_benchmark.json",
        EVIDENCE / "evidence_manifest.json",
    ]
    before = {path: path.read_bytes() for path in tracked_paths}

    subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), str(SUBMISSION / "generate_evidence_manifest.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert {path: path.read_bytes() for path in tracked_paths} == before
