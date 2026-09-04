# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — Paper 002 evidence-evolution tests.
"""Protect the public historical, wiring and fresh-cohort evidence boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = REPO_ROOT / "papers/submissions/002_neuromorphic_vertical_stability_control/evidence"
HISTORICAL_REVISION = "476908debdd886d3a35bf0ae85216e684727adce"
HISTORICAL_PATH = "validation/reports/stress_test_campaign.json"


def _load(path: Path) -> dict[str, Any]:
    """Load one paper evidence JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 digest of exact evidence bytes."""
    return hashlib.sha256(payload).hexdigest()


def test_historical_latency_is_exact_git_evidence_not_reconstructed_output() -> None:
    """The paper-local legacy record must match the immutable tracked Git object."""
    historical = _load(EVIDENCE / "historical_controller_latency.json")
    source = subprocess.run(
        ["git", "show", f"{HISTORICAL_REVISION}:{HISTORICAL_PATH}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    blob = (
        subprocess.run(
            ["git", "hash-object", "--stdin"],
            cwd=REPO_ROOT,
            check=True,
            input=source,
            capture_output=True,
            text=False,
        )
        .stdout.decode("ascii")
        .strip()
    )

    assert historical["source_custody"]["artifact_sha256"] == _sha256_bytes(source)
    assert historical["source_custody"]["git_blob_oid_sha1"] == blob
    assert historical["methodology"]["exact_invocation_episode_count"] is None
    assert historical["methodology"]["master_seed"] is None

    source_results = json.loads(source)
    for lane, result in historical["results"].items():
        assert result["p50_latency_us"] == source_results[lane]["p50_latency_us"]
        assert result["p95_latency_us"] == source_results[lane]["p95_latency_us"]
        assert result["p99_latency_us"] == source_results[lane]["p99_latency_us"]
        assert result["disruption_rate"] == source_results[lane]["disruption_rate"]


def test_schema_v3_wiring_assessment_matches_live_report_and_stays_ineligible() -> None:
    """The transition summary must match the exact report and remain wiring-only."""
    assessment = _load(EVIDENCE / "schema_v3_wiring_assessment.json")
    report_path = REPO_ROOT / assessment["source_custody"]["path"]
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)

    assert assessment["source_custody"]["artifact_sha256"] == _sha256_bytes(report_bytes)
    assert assessment["evidence_status"] == "wiring_only_not_promotion_eligible"
    assert assessment["result"]["campaign_complete"] is report["campaign_complete"] is False
    assert assessment["result"]["promotion_eligible"] is report["promotion_eligible"] is False
    assert assessment["result"]["common_completed_lane_outcome"]["t_disruption_s"] == 0.0
    assert assessment["result"]["lanes"]["NMPC-JAX"]["status"] == "unavailable"

    for lane, summary in assessment["result"]["lanes"].items():
        source = report["controllers"][lane]
        assert summary["status"] == source["status"]
        assert summary["p50_policy_latency_us"] == source["p50_control_policy_latency_us"]
        assert summary["p95_policy_latency_us"] == source["p95_control_policy_latency_us"]


def test_fresh_cohort_is_fully_preregistered_and_fail_closed() -> None:
    """The fresh request must enumerate every lane and every current blocker."""
    protocol = _load(EVIDENCE / "fresh_cohort_protocol.json")
    request = protocol["preregistered_request"]
    blockers = {gate["gate"] for gate in protocol["pre_run_gates"] if gate["status"] == "blocked"}

    assert protocol["status"] == "prepared_blocked_not_executable"
    assert request["episodes_per_lane"] == 200
    assert request["master_seed"] == 42
    assert request["surrogate"] is False
    assert request["controllers_in_order"] == [
        "PID",
        "H-infinity",
        "LQR",
        "MPC",
        "NMPC-JAX",
        "LIF-NEF-SNN",
        "Rust-PID",
    ]
    assert blockers == {
        "admissible_real_kernel_initial_state",
        "calibrated_held_out_nmpc",
        "complete_controller_registry",
    }
    assert "--surrogate" not in " ".join(protocol["execution_template"])
    assert "--controllers PID,H-infinity,LQR,MPC,NMPC-JAX,LIF-NEF-SNN,Rust-PID" in " ".join(
        protocol["execution_template"]
    )


def test_manuscript_keeps_evidence_layers_explicitly_separate() -> None:
    """Public prose must not collapse legacy, wiring and future evidence."""
    manuscript = (EVIDENCE.parent / "manuscript.tex").read_text(encoding="utf-8")

    assert "Historical methodology and preserved output" in manuscript
    assert "Schema-v3 transition run" in manuscript
    assert "Pre-registered fresh cohort" in manuscript
    assert "exact invocation count that produced" in manuscript
    assert "prepared but intentionally blocked" in manuscript
    assert "controller_latency_workstation.json" not in manuscript
