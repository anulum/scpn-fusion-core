#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — neuromorphic-paper evidence manifest generator.
"""Bind historical, transition and fresh-cohort evidence to exact sources."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_REVISION = "8df066b60bf713d8b79af1714109fdff6a2ca355"
EXPECTED_HISTORICAL_ARTIFACT_SHA256 = (
    "394be6c32a564f6ae1e93e9c39f33ab1ace3d68c591db91dfd89e81fcaa3f66d"
)
EXPECTED_HISTORICAL_BLOB = "5616f192a7b41f2114777da76a16433a11818aef"
EXPECTED_WIRING_REPORT_SHA256 = "aab54de6f5c751670df6b630d48c7face75b04ed2a834de043b3f950f4638371"
EXPECTED_CONTROLLERS = [
    "PID",
    "H-infinity",
    "LQR",
    "MPC",
    "NMPC-JAX",
    "LIF-NEF-SNN",
    "Rust-PID",
]


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object or reject a non-object evidence payload."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one exact file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_latency(record: dict[str, Any], lane: str) -> None:
    """Reject missing, non-finite or inverted p50/p95 latency evidence."""
    p50 = float(record["p50_latency_us"])
    p95 = float(record["p95_latency_us"])
    if not math.isfinite(p50) or not math.isfinite(p95) or p50 <= 0.0 or p95 < p50:
        raise ValueError(f"Invalid latency evidence for {lane}: {record}")


def _validate_historical(historical: dict[str, Any]) -> None:
    """Validate custody and fail-closed semantics of the legacy evidence."""
    if historical.get("schema") != "scpn-fusion-core.paper002-historical-controller-latency.v1":
        raise ValueError("Historical evidence schema is unsupported.")
    if historical.get("evidence_status") != "historical_non_comparable":
        raise ValueError("Historical evidence must remain explicitly non-comparable.")
    custody = historical.get("source_custody", {})
    if custody.get("artifact_sha256") != EXPECTED_HISTORICAL_ARTIFACT_SHA256:
        raise ValueError("Historical evidence lost its exact source-artifact SHA-256.")
    if custody.get("git_blob_oid_sha1") != EXPECTED_HISTORICAL_BLOB:
        raise ValueError("Historical evidence lost its exact Git blob identity.")
    methodology = historical.get("methodology", {})
    if methodology.get("exact_invocation_episode_count") is not None:
        raise ValueError("Unknown historical invocation count must not be invented.")
    if methodology.get("master_seed") is not None:
        raise ValueError("Unknown historical master seed must not be invented.")
    results = historical.get("results", {})
    expected_lanes = {"PID", "Rust-PID", "H-infinity", "NMPC-JAX", "Nengo-SNN"}
    if set(results) != expected_lanes:
        raise ValueError("Historical result lane inventory drifted.")
    for lane, record in results.items():
        _require_latency(record, lane)
    if results["H-infinity"].get("status") != "invalidated_stale_scalar_plant_calibration":
        raise ValueError("The invalidated historical H-infinity lane cannot be rehabilitated.")


def _validate_wiring(
    wiring: dict[str, Any], live_report: dict[str, Any], live_report_sha256: str
) -> None:
    """Prove that the paper-local wiring summary matches the exact live report."""
    if wiring.get("schema") != "scpn-fusion-core.paper002-schema-v3-wiring-assessment.v1":
        raise ValueError("Schema-v3 wiring-assessment schema is unsupported.")
    if wiring.get("evidence_status") != "wiring_only_not_promotion_eligible":
        raise ValueError("The wiring assessment cannot be presented as promotion evidence.")
    custody = wiring["source_custody"]
    if custody.get("repository_revision") != EXPECTED_REVISION:
        raise ValueError("Wiring evidence revision drifted.")
    if custody.get("artifact_sha256") != live_report_sha256:
        raise ValueError("Wiring evidence does not match the exact live report bytes.")
    if live_report_sha256 != EXPECTED_WIRING_REPORT_SHA256:
        raise ValueError("The witnessed schema-v3 wiring report changed.")
    if live_report.get("schema") != "scpn-fusion-core.stress-campaign-results.v3":
        raise ValueError("The live wiring report uses an unsupported result schema.")
    result = wiring["result"]
    for field in ("campaign_complete", "promotion_eligible", "promotion_ineligibility_reason"):
        if result[field] != live_report[field]:
            raise ValueError(f"Wiring assessment differs from the live report field {field}.")
    live_lanes = live_report["controllers"]
    if set(result["lanes"]) != set(EXPECTED_CONTROLLERS):
        raise ValueError("Wiring assessment lane inventory drifted.")
    for lane, summary in result["lanes"].items():
        live = live_lanes[lane]
        if summary["status"] != live["status"] or summary.get("reason") != live.get("reason"):
            raise ValueError(f"Wiring status differs from the live report for {lane}.")
        if summary["p50_policy_latency_us"] != live["p50_control_policy_latency_us"]:
            raise ValueError(f"Wiring p50 differs from the live report for {lane}.")
        if summary["p95_policy_latency_us"] != live["p95_control_policy_latency_us"]:
            raise ValueError(f"Wiring p95 differs from the live report for {lane}.")
    common = result["common_completed_lane_outcome"]
    for lane, live in live_lanes.items():
        if live["status"] != "complete":
            continue
        if live["mean_abs_r_error_m"] != common["mean_abs_r_error_m"]:
            raise ValueError(f"Completed lane {lane} differs in radial wiring outcome.")
        if live["mean_abs_z_error_m"] != common["mean_abs_z_error_m"]:
            raise ValueError(f"Completed lane {lane} differs in vertical wiring outcome.")
        if live["disruption_rate"] != common["disruption_rate"]:
            raise ValueError(f"Completed lane {lane} differs in disruption wiring outcome.")
        if live["episodes"][0]["t_disruption_s"] != common["t_disruption_s"]:
            raise ValueError(f"Completed lane {lane} differs in time-zero disruption evidence.")


def _validate_fresh_protocol(protocol: dict[str, Any], repository: Path) -> None:
    """Validate that the preregistration stays bound to code and blockers."""
    if protocol.get("schema") != "scpn-fusion-core.paper002-fresh-cohort-protocol.v1":
        raise ValueError("Fresh-cohort protocol schema is unsupported.")
    if protocol.get("status") != "prepared_blocked_not_executable":
        raise ValueError("Fresh cohort must remain blocked until its gates pass.")
    if protocol.get("protocol_revision") != EXPECTED_REVISION:
        raise ValueError("Fresh-cohort protocol revision drifted.")
    inputs = protocol["inputs"]
    source_path = repository / inputs["campaign_source_path"]
    config_path = repository / inputs["config_path"]
    if _sha256(source_path) != inputs["campaign_source_sha256"]:
        raise ValueError("Fresh-cohort campaign source hash drifted.")
    if _sha256(config_path) != inputs["config_sha256"]:
        raise ValueError("Fresh-cohort config hash drifted.")
    request = protocol["preregistered_request"]
    if request.get("controllers_in_order") != EXPECTED_CONTROLLERS:
        raise ValueError("Fresh-cohort controller order drifted.")
    if (
        request.get("surrogate") is not False
        or request.get("required_scope") != "controller_comparison"
    ):
        raise ValueError("Fresh cohort must use the real comparison scope.")
    blockers = {gate["gate"] for gate in protocol["pre_run_gates"] if gate["status"] == "blocked"}
    expected_blockers = {
        "admissible_real_kernel_initial_state",
        "calibrated_held_out_nmpc",
        "complete_controller_registry",
    }
    if blockers != expected_blockers:
        raise ValueError("Fresh-cohort blocker inventory drifted.")


def main() -> None:
    """Validate all evidence layers and write their deterministic manifest."""
    submission = Path(__file__).resolve().parent.parent
    evidence = submission / "evidence"
    repository = submission.parents[2]
    metadata = _load_json(submission / "submission_metadata.json")
    revision = str(metadata["repository_revision"])
    if revision != EXPECTED_REVISION:
        raise ValueError("Submission metadata does not bind the schema-v3 contract revision.")

    historical = _load_json(evidence / "historical_controller_latency.json")
    wiring = _load_json(evidence / "schema_v3_wiring_assessment.json")
    protocol = _load_json(evidence / "fresh_cohort_protocol.json")
    live_report_path = repository / "validation/reports/stress_test_campaign.json"
    live_report = _load_json(live_report_path)
    live_report_sha256 = _sha256(live_report_path)
    _validate_historical(historical)
    _validate_wiring(wiring, live_report, live_report_sha256)
    _validate_fresh_protocol(protocol, repository)

    roles = {
        "historical_controller_latency.json": (
            "exact legacy outputs plus Git-object custody and non-comparability boundary"
        ),
        "schema_v3_wiring_assessment.json": (
            "exact summary of the incomplete one-episode shared-contract wiring run"
        ),
        "fresh_cohort_protocol.json": (
            "preregistered comparison request, blockers, endpoints and execution template"
        ),
        "controller_latency_table.tex": "generated historical non-comparable latency table",
        "schema_v3_wiring_table.tex": "generated wiring-only policy-latency table",
        "vertical_stability_reduced_order.json": (
            "generated reduced-order fast-VDE controller-outcome record"
        ),
    }
    files: dict[str, dict[str, str]] = {}
    for name, role in roles.items():
        path = evidence / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[name] = {"sha256": _sha256(path), "role": role}

    generator_paths = (
        repository / "validation/stress_test_campaign.py",
        submission / "figures/generate_latency_table.py",
        submission / "figures/fig_latency_comparison.py",
        submission / "figures/fig_vertical_stability.py",
        submission / "figures/generate_evidence_manifest.py",
    )
    generators = {
        path.relative_to(repository).as_posix(): {"sha256": _sha256(path)}
        for path in generator_paths
    }
    manifest = {
        "SPDX-License-Identifier": "AGPL-3.0-or-later",
        "commercialLicense": "available",
        "conceptsCopyright": "1996-2026 Miroslav Sotek. All rights reserved.",
        "codeCopyright": "2020-2026 Miroslav Sotek. All rights reserved.",
        "orcid": "0009-0009-3560-0851",
        "contact": "www.anulum.li | protoscience@anulum.li",
        "projectDescription": "SCPN-FUSION-CORE neuromorphic-paper evidence manifest.",
        "schema_version": "2.0",
        "repository_revision": revision,
        "evidence_layers": [
            "historical_non_comparable",
            "schema_v3_wiring_only",
            "fresh_cohort_preregistration",
        ],
        "source_git_objects": historical["source_custody"],
        "generators": generators,
        "files": files,
    }
    destination = evidence / "evidence_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("  [OK] evidence_manifest (historical, wiring and fresh-cohort custody)")


if __name__ == "__main__":
    main()
