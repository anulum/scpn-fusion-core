# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — full reproduction evidence command
"""One-command reproduction evidence for the full-fidelity validation chain."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

from validation import full_fidelity_end_to_end_campaign as campaign


JsonObject = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON_REPORT = REPO_ROOT / "validation" / "reports" / "full_reproduction_evidence.json"
DEFAULT_MARKDOWN_REPORT = REPO_ROOT / "validation" / "reports" / "full_reproduction_evidence.md"


def _display_path(path: Path) -> str:
    """Return a stable path label for report output."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: JsonObject) -> str:
    """Return a SHA-256 digest for a canonical JSON payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _current_commit() -> str:
    """Return the current repository commit, or ``unknown`` when Git is unavailable."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if result.returncode != 0 or not commit:
        return "unknown"
    return commit


def _json_metadata(path: Path) -> tuple[str | None, str | None]:
    """Return schema and status metadata for a JSON report."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(payload, dict):
        return None, "non_object_json"
    schema = payload.get("schema")
    status = payload.get("status")
    return (
        schema if isinstance(schema, str) else None,
        status if isinstance(status, str) else None,
    )


def _artifact_record(path: Path) -> JsonObject:
    """Return path, existence, checksum, and JSON metadata for one evidence artifact."""
    exists = path.is_file()
    record: JsonObject = {
        "path": _display_path(path),
        "exists": exists,
        "sha256": _sha256_file(path) if exists else None,
        "bytes": path.stat().st_size if exists else 0,
        "schema": None,
        "status": None,
    }
    if exists and path.suffix == ".json":
        schema, status = _json_metadata(path)
        record["schema"] = schema
        record["status"] = status
    return record


def _source_artifacts(ledger: JsonObject) -> list[JsonObject]:
    """Return source report artifacts plus the generated ledger sidecars."""
    source_reports = cast(list[JsonObject], ledger["source_reports"])
    artifact_paths = [
        REPO_ROOT / str(report["path"])
        for report in source_reports
        if isinstance(report.get("path"), str)
    ]
    artifact_paths.extend(
        (
            campaign.LEDGER_JSON_REPORT,
            campaign.LEDGER_MD_REPORT,
            campaign.MD_REPORT,
        )
    )
    return [_artifact_record(path) for path in dict.fromkeys(artifact_paths)]


def build_reproduction_report() -> JsonObject:
    """Run the full-fidelity campaign and return checksummed reproduction evidence."""
    campaign_report = campaign.run_campaign()
    campaign.write_reports(campaign_report)
    ledger = campaign.build_public_ledger(campaign_report)
    artifacts = _source_artifacts(ledger)
    missing_artifacts = [artifact["path"] for artifact in artifacts if not artifact["exists"]]
    blocked_lanes = [
        lane["lane"]
        for lane in cast(list[JsonObject], ledger["lanes"])
        if bool(lane["blocked_for_public_full_fidelity"])
    ]
    report: JsonObject = {
        "schema": "scpn-fusion-full-reproduction-evidence.v1",
        "producer": "scpn_fusion.repro.build_reproduction_report",
        "command": "scpn-fusion repro --full",
        "source_commit": _current_commit(),
        "status": str(campaign_report["status"]),
        "acceptance_passed": bool(campaign_report["acceptance_passed"]),
        "full_fidelity_ready": bool(ledger["ledger_publication_ready"]),
        "blocked_lane_count": int(ledger["blocked_lane_count"]),
        "accepted_full_fidelity_lane_count": int(ledger["accepted_full_fidelity_lane_count"]),
        "missing_artifact_count": len(missing_artifacts),
        "missing_artifacts": missing_artifacts,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "campaign_report": str(ledger["campaign_report"]),
        "campaign_schema": str(ledger["campaign_schema"]),
        "ledger_schema": str(ledger["schema"]),
        "lanes": ledger["lanes"],
        "blocked_lanes": blocked_lanes,
        "claim_boundary": (
            "The command reproduces the local fail-closed evidence chain. It does not mark "
            "blocked full-fidelity lanes ready until their external parity artifacts exist and "
            "the ledger publication gate passes."
        ),
    }
    report["evidence_payload_sha256"] = _canonical_json_sha256(report)
    return report


def render_reproduction_markdown(report: JsonObject) -> str:
    """Render a concise Markdown companion for the reproduction evidence report."""
    lines = [
        "# Full Reproduction Evidence",
        "",
        "Generated by `scpn-fusion repro --full`.",
        "",
        f"- Schema: `{report['schema']}`",
        f"- Status: `{report['status']}`",
        f"- Acceptance passed: `{report['acceptance_passed']}`",
        f"- Full-fidelity ready: `{report['full_fidelity_ready']}`",
        f"- Blocked lane count: `{report['blocked_lane_count']}`",
        f"- Accepted full-fidelity lane count: `{report['accepted_full_fidelity_lane_count']}`",
        f"- Evidence payload SHA-256: `{report['evidence_payload_sha256']}`",
        "",
        "## Artifacts",
        "",
        "| Artifact | Exists | SHA-256 | Schema | Status |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for artifact in cast(list[JsonObject], report["artifacts"]):
        lines.append(
            "| `{path}` | `{exists}` | `{sha}` | `{schema}` | `{status}` |".format(
                path=artifact["path"],
                exists=artifact["exists"],
                sha=artifact["sha256"] or "missing",
                schema=artifact["schema"] or "n/a",
                status=artifact["status"] or "n/a",
            )
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            str(report["claim_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)


def run_full_reproduction(
    *,
    json_output: Path = DEFAULT_JSON_REPORT,
    markdown_output: Path = DEFAULT_MARKDOWN_REPORT,
) -> JsonObject:
    """Run the full reproduction command and persist JSON plus Markdown evidence.

    Args:
        json_output: Destination for the machine-readable evidence report.
        markdown_output: Destination for the human-readable evidence report.

    Returns:
        The JSON-serializable evidence payload that was written to ``json_output``.
    """
    report = build_reproduction_report()
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_output.write_text(render_reproduction_markdown(report), encoding="utf-8")
    return report
