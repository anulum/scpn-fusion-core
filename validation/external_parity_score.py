#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — External Parity Score Report
"""Aggregate TORAX and FreeGS/FreeGSNKE parity evidence into scored lanes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "validation" / "reports"
TORAX_REAL_PARITY = REPORT_DIR / "torax_real_parity.json"
TORAX_SAME_PHYSICS = REPORT_DIR / "torax_same_physics_config_study.json"
TORAX_IMAS_INTERCHANGE = REPORT_DIR / "torax_imas_interchange.json"
FREE_BOUNDARY_STRICT_PARITY = REPORT_DIR / "free_boundary_strict_parity_benchmark.json"
FREEGS_PUBLIC_RECONSTRUCTION = REPORT_DIR / "freegs_public_example_reconstruction.json"
FREE_BOUNDARY_MACHINE_METADATA = (
    REPORT_DIR / "free_boundary_public_machine_metadata_inventory.json"
)
DEFAULT_JSON_REPORT = REPORT_DIR / "external_parity_score.json"
DEFAULT_MD_REPORT = REPORT_DIR / "external_parity_score.md"
SCHEMA = "scpn-fusion-core.external-parity-score.v1"

JsonObject = dict[str, Any]


def _rel(path: Path) -> str:
    """Return a repository-relative report path."""
    resolved = path if path.is_absolute() else ROOT / path
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for an input artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 digest for a JSON mapping."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_report(path: Path, *, expected_schema: str) -> JsonObject:
    """Load a report JSON object and validate its schema."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{_rel(path)} must contain a JSON object")
    report = cast(JsonObject, payload)
    if report.get("schema") != expected_schema:
        raise ValueError(
            f"{_rel(path)} schema mismatch: expected {expected_schema}, got {report.get('schema')}"
        )
    return report


def _input_record(report_id: str, path: Path, payload: Mapping[str, Any]) -> JsonObject:
    """Build provenance and checksum metadata for one source report."""
    return {
        "report_id": report_id,
        "path": _rel(path),
        "schema": payload.get("schema"),
        "status": payload.get("status"),
        "file_sha256": _sha256_file(path),
        "payload_sha256": _sha256_json(payload),
    }


def _component(component_id: str, ready: bool, evidence: str) -> JsonObject:
    """Build one scored component row."""
    return {"component": component_id, "ready": bool(ready), "evidence": evidence}


def _score(components: Sequence[Mapping[str, Any]]) -> float:
    """Return the unweighted readiness fraction for scored components."""
    if not components:
        return 0.0
    ready_count = sum(1 for component in components if component.get("ready") is True)
    return round(ready_count / len(components), 6)


def _matrix_has_status(
    matrix: Sequence[Mapping[str, Any]],
    component: str,
    status: str,
) -> bool:
    """Return whether a same-physics matrix component has the expected status."""
    return any(row.get("component") == component and row.get("status") == status for row in matrix)


def _torax_lane(
    *,
    real_parity: Mapping[str, Any],
    same_physics: Mapping[str, Any],
    imas_interchange: Mapping[str, Any],
) -> JsonObject:
    """Build the scored TORAX transport parity lane."""
    same_matrix = cast(Sequence[Mapping[str, Any]], same_physics.get("same_physics_matrix", []))
    threshold_blockers = list(cast(Sequence[str], same_physics.get("threshold_blockers", [])))
    divergence_metrics = cast(Mapping[str, Any], real_parity.get("divergence_metrics", {}))
    reproducibility_components = [
        _component(
            "torax_reference_provenance",
            isinstance(cast(Mapping[str, Any], real_parity.get("reference", {})).get("provenance"), Mapping),
            "TORAX real-reference report carries provenance.",
        ),
        _component(
            "torax_real_metrics_recorded",
            bool(real_parity.get("passes_thresholds") is True and divergence_metrics),
            "Real-reference divergence metrics are recorded.",
        ),
        _component(
            "torax_imas_interchange",
            bool(imas_interchange.get("passes_thresholds") is True),
            "TORAX profiles round through the tracked IMAS fixture.",
        ),
        _component(
            "torax_radial_grid_shared",
            _matrix_has_status(same_matrix, "radial_grid", "shared_ready"),
            "Same-physics study marks radial grid as shared.",
        ),
        _component(
            "torax_initial_profiles_shared",
            _matrix_has_status(same_matrix, "initial_profiles", "shared_ready"),
            "Same-physics study marks initial profiles as shared.",
        ),
    ]
    parity_components = [
        _component(
            "torax_physics_equivalence_claimed",
            bool(real_parity.get("physics_equivalence_claimed") is True),
            "TORAX real-reference report does not yet claim physics equivalence.",
        ),
        _component(
            "torax_same_physics_ready",
            bool(same_physics.get("same_physics_ready") is True),
            "Same-physics model and controls must be ready before parity thresholds tighten.",
        ),
        _component(
            "torax_threshold_tightening_ready",
            bool(same_physics.get("threshold_tightening_ready") is True),
            "TORAX/native quantitative thresholds remain blocked.",
        ),
        _component(
            "torax_no_threshold_blockers",
            not threshold_blockers,
            "Threshold blockers must be empty.",
        ),
        _component(
            "torax_divergence_reference_passes",
            bool(real_parity.get("passes_thresholds") is True),
            "Current gate records finite divergence against a real TORAX artifact.",
        ),
    ]
    parity_score = _score(parity_components)
    return {
        "lane": "torax_transport",
        "external_code": "TORAX",
        "status": "blocked_same_physics_thresholds"
        if parity_score < 1.0
        else "accepted_external_parity",
        "reproducibility_score": _score(reproducibility_components),
        "parity_score": parity_score,
        "reproducibility_components": reproducibility_components,
        "parity_components": parity_components,
        "blocked_requirements": threshold_blockers
        or [
            "native_transport_model",
            "sources_and_boundary_conditions",
            "time_integration_contract",
        ],
    }


def _free_boundary_lane(
    *,
    strict_parity: Mapping[str, Any],
    public_reconstruction: Mapping[str, Any],
    machine_metadata: Mapping[str, Any],
) -> JsonObject:
    """Build the scored FreeGS/FreeGSNKE free-boundary parity lane."""
    checks = cast(Mapping[str, Any], strict_parity.get("checks", {}))
    reproducibility_components = [
        _component(
            "freegs_public_reconstruction_report",
            bool(public_reconstruction.get("reference_output_ready") is True),
            "Public FreeGS reconstruction report links same-case output.",
        ),
        _component(
            "freegsnke_machine_metadata",
            bool(machine_metadata.get("machine_metadata_ready") is True),
            "FreeGSNKE machine metadata inventory is ready.",
        ),
        _component(
            "freegsnke_commit_recorded",
            bool(machine_metadata.get("freegsnke_commit")),
            "FreeGSNKE source commit is recorded.",
        ),
        _component(
            "strict_parity_source_checksums",
            isinstance(strict_parity.get("source_checksums"), Mapping),
            "Strict parity report carries source report checksums.",
        ),
        _component(
            "strict_parity_grid_convergence",
            bool(checks.get("grid_convergence_ready") is True),
            "Strict parity report records grid-convergence readiness.",
        ),
    ]
    parity_components = [
        _component(
            "free_boundary_accepted_full_fidelity",
            bool(strict_parity.get("accepted_full_fidelity") is True),
            "Strict parity gate accepts the current free-boundary lane.",
        ),
        _component(
            "external_nonlinear_output",
            bool(checks.get("external_nonlinear_output_ready") is True),
            "Same-case external nonlinear output is ready.",
        ),
        _component(
            "native_same_case_profile_source",
            bool(checks.get("native_same_case_profile_source_ready") is True),
            "Native same-case profile-source comparison is ready.",
        ),
        _component(
            "strict_threshold_acceptance",
            bool(checks.get("strict_threshold_acceptance_ready") is True),
            "Strict threshold acceptance is ready.",
        ),
        _component(
            "public_machine_metadata",
            bool(checks.get("machine_metadata_ready") is True),
            "Public machine metadata is ready.",
        ),
    ]
    parity_score = _score(parity_components)
    return {
        "lane": "freegsnke_free_boundary",
        "external_code": "FreeGS/FreeGSNKE",
        "status": "accepted_external_parity"
        if parity_score == 1.0
        else "blocked_free_boundary_external_parity",
        "reproducibility_score": _score(reproducibility_components),
        "parity_score": parity_score,
        "reproducibility_components": reproducibility_components,
        "parity_components": parity_components,
        "blocked_requirements": list(cast(Sequence[str], strict_parity.get("blockers", []))),
    }


def build_report(
    *,
    torax_real_parity_path: Path = TORAX_REAL_PARITY,
    torax_same_physics_path: Path = TORAX_SAME_PHYSICS,
    torax_imas_interchange_path: Path = TORAX_IMAS_INTERCHANGE,
    free_boundary_strict_parity_path: Path = FREE_BOUNDARY_STRICT_PARITY,
    freegs_public_reconstruction_path: Path = FREEGS_PUBLIC_RECONSTRUCTION,
    free_boundary_machine_metadata_path: Path = FREE_BOUNDARY_MACHINE_METADATA,
) -> JsonObject:
    """Build the integrated external parity score report."""
    torax_real = _load_report(
        torax_real_parity_path,
        expected_schema="scpn-fusion-core.torax-real-parity.v1",
    )
    torax_same = _load_report(
        torax_same_physics_path,
        expected_schema="scpn-fusion-core.torax-same-physics-config-study.v1",
    )
    torax_imas = _load_report(
        torax_imas_interchange_path,
        expected_schema="scpn-fusion-core.torax-imas-interchange.v1",
    )
    strict_parity = _load_report(
        free_boundary_strict_parity_path,
        expected_schema="free-boundary-strict-parity-benchmark.v1",
    )
    public_reconstruction = _load_report(
        freegs_public_reconstruction_path,
        expected_schema="freegs-public-example-reconstruction-report.v1",
    )
    machine_metadata = _load_report(
        free_boundary_machine_metadata_path,
        expected_schema="free-boundary-public-machine-metadata-inventory-report.v1",
    )
    source_reports = [
        _input_record("torax_real_parity", torax_real_parity_path, torax_real),
        _input_record("torax_same_physics_config_study", torax_same_physics_path, torax_same),
        _input_record("torax_imas_interchange", torax_imas_interchange_path, torax_imas),
        _input_record("free_boundary_strict_parity", free_boundary_strict_parity_path, strict_parity),
        _input_record(
            "freegs_public_example_reconstruction",
            freegs_public_reconstruction_path,
            public_reconstruction,
        ),
        _input_record(
            "free_boundary_public_machine_metadata_inventory",
            free_boundary_machine_metadata_path,
            machine_metadata,
        ),
    ]
    lanes = [
        _torax_lane(
            real_parity=torax_real,
            same_physics=torax_same,
            imas_interchange=torax_imas,
        ),
        _free_boundary_lane(
            strict_parity=strict_parity,
            public_reconstruction=public_reconstruction,
            machine_metadata=machine_metadata,
        ),
    ]
    reproducibility_score = round(
        sum(float(lane["reproducibility_score"]) for lane in lanes) / len(lanes), 6
    )
    parity_score = round(sum(float(lane["parity_score"]) for lane in lanes) / len(lanes), 6)
    accepted = all(float(lane["parity_score"]) == 1.0 for lane in lanes)
    return {
        "schema": SCHEMA,
        "status": "accepted_external_parity_score" if accepted else "blocked_external_parity_score",
        "acceptance_passed": accepted,
        "reproducibility_score": reproducibility_score,
        "parity_score": parity_score,
        "scoring": {
            "scale": "0.0_to_1.0",
            "lane_weighting": "equal_lane_average",
            "component_weighting": "equal_component_average",
            "claim_boundary": (
                "Scores summarize current report readiness. They do not promote a lane whose "
                "underlying fail-closed report remains blocked."
            ),
        },
        "source_reports": source_reports,
        "lanes": lanes,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the external parity score report as Markdown."""
    lines = [
        "# External Parity Score",
        "",
        f"- Schema: `{report['schema']}`",
        f"- Status: `{report['status']}`",
        f"- Acceptance passed: `{report['acceptance_passed']}`",
        f"- Reproducibility score: `{report['reproducibility_score']}`",
        f"- Parity score: `{report['parity_score']}`",
        "",
        str(cast(Mapping[str, Any], report["scoring"])["claim_boundary"]),
        "",
        "## Lanes",
        "",
        "| Lane | External code | Status | Reproducibility score | Parity score | Blocked requirements |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for lane in cast(Sequence[Mapping[str, Any]], report["lanes"]):
        blockers = "<br>".join(str(value) for value in lane["blocked_requirements"]) or "none"
        lines.append(
            "| {lane} | {code} | `{status}` | `{repro}` | `{parity}` | {blockers} |".format(
                lane=lane["lane"],
                code=lane["external_code"],
                status=lane["status"],
                repro=lane["reproducibility_score"],
                parity=lane["parity_score"],
                blockers=blockers,
            )
        )
    lines.extend(
        [
            "",
            "## Source Reports",
            "",
            "| Report | Status | File SHA-256 | Payload SHA-256 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for source in cast(Sequence[Mapping[str, Any]], report["source_reports"]):
        lines.append(
            "| `{path}` | `{status}` | `{file}` | `{payload}` |".format(
                path=source["path"],
                status=source["status"],
                file=source["file_sha256"],
                payload=source["payload_sha256"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_report(
    *,
    json_report_path: Path = DEFAULT_JSON_REPORT,
    md_report_path: Path = DEFAULT_MD_REPORT,
) -> JsonObject:
    """Write the integrated external parity score JSON and Markdown reports."""
    report = build_report()
    json_report_path.parent.mkdir(parents=True, exist_ok=True)
    json_report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_report_path.write_text(render_markdown(report), encoding="utf-8")
    return report


def check_report(
    *,
    json_report_path: Path = DEFAULT_JSON_REPORT,
    md_report_path: Path = DEFAULT_MD_REPORT,
) -> list[str]:
    """Return drift errors for the tracked external parity score reports."""
    expected = build_report()
    expected_md = render_markdown(expected)
    errors: list[str] = []
    if not json_report_path.exists():
        errors.append(f"missing external parity score JSON report: {json_report_path}")
    else:
        observed = cast(JsonObject, json.loads(json_report_path.read_text(encoding="utf-8")))
        if _sha256_json(observed) != _sha256_json(expected):
            errors.append("tracked external parity score JSON report is stale")
    if not md_report_path.exists():
        errors.append(f"missing external parity score Markdown report: {md_report_path}")
    elif md_report_path.read_text(encoding="utf-8") != expected_md:
        errors.append("tracked external parity score Markdown report is stale")
    return errors


def main(argv: list[str] | None = None) -> int:
    """Run or check the integrated external parity score report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_REPORT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero unless every scored external parity lane is accepted.",
    )
    args = parser.parse_args(argv)

    if args.check:
        errors = check_report(json_report_path=args.report_json, md_report_path=args.report_md)
        for error in errors:
            print(f"EXTERNAL PARITY SCORE ERROR: {error}", file=sys.stderr)
        if errors:
            return 1
        print(f"External parity score reports are up to date: {args.report_json}")
        return 0

    report = write_report(json_report_path=args.report_json, md_report_path=args.report_md)
    print(json.dumps({"status": report["status"], "parity_score": report["parity_score"]}))
    return 0 if (not args.strict or report["acceptance_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
