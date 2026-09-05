# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Strict parity report rendering
"""Render non-admitting strict parity diagnostics without evaluating evidence."""

from __future__ import annotations

from typing import Any


def render_markdown(report: dict[str, Any]) -> str:
    """Render strict parity diagnostics without modifying or qualifying evidence.

    Parameters
    ----------
    report : dict
        Version-2 output from the strict parity evaluator.

    Returns
    -------
    str
        Markdown with admission boundaries, provenance and diagnostic failures.
    """
    lines = [
        "# Free-boundary Strict Parity Benchmark",
        "",
        "No positive scientific evidence contract is admitted by this classifier.",
        "These diagnostics check stored summaries and source bytes; they do not",
        "establish raw field/q-profile custody or independent predictive validation.",
        "",
        f"- Schema: `{report['schema']}`",
        f"- Status: `{report['status']}`",
        f"- Evidence classification: `{report['evidence_classification']}`",
        f"- Accepted full fidelity: `{report['accepted_full_fidelity']}`",
        f"- Case count: `{report['case_count']}`",
        f"- Failed threshold checks: `{report['failed_threshold_check_count']}`",
        "",
        "## Checks",
        "",
        "| Check | Ready |",
        "| --- | ---: |",
    ]
    for key, value in report["checks"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Acceptance matrix",
            "",
            "| Requirement | Ready |",
            "| --- | ---: |",
        ]
    )
    for key, value in report["acceptance_matrix"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Blockers", ""])
    if report["blockers"]:
        for blocker in report["blockers"]:
            lines.append(f"- `{blocker}`")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Threshold cases",
            "",
            "| Case | External output | Native comparison | Thresholds ready | Failed checks |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for case in report["threshold_cases"]:
        lines.append(
            "| {case_id} | `{external}` | `{native}` | `{threshold}` | {failed} |".format(
                case_id=case["case_id"],
                external=case["external_nonlinear_output_ready"],
                native=case["native_same_case_profile_source_ready"],
                threshold=case["strict_threshold_acceptance_ready"],
                failed=case["failed_threshold_check_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Grid-convergence cases",
            "",
            "| Case | Machine | Observed | Required | Missing | Ready | Blocker |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for case in report["grid_convergence"]["cases"]:
        lines.append(
            "| {case_id} | {machine} | {observed} | {required} | {missing} | `{ready}` | {reason} |".format(
                case_id=case["case_id"],
                machine=case["machine_class"],
                observed=case["observed_resolution_count"],
                required=case["required_resolution_count"],
                missing=case["missing_resolution_count"],
                ready=case["grid_convergence_case_ready"],
                reason=case["blocking_reason"],
            )
        )
    lines.extend(["", "## Machine metadata", ""])
    metadata = report["machine_metadata"]
    lines.append(f"- Schema: `{metadata['schema']}`")
    lines.append(f"- Status: `{metadata['status']}`")
    lines.append(f"- Machine config count: `{metadata['machine_config_count']}`")
    lines.append(f"- Machines: `{', '.join(str(v) for v in metadata['machines'])}`")
    lines.extend(["", "## Provenance and checksums", ""])
    provenance = report["provenance"]
    lines.append(f"- Generator: `{provenance['generator']}`")
    lines.append(f"- Source commit: `{provenance['source_commit']}`")
    lines.append(f"- Python version: `{provenance['python_version']}`")
    lines.extend(["", "| Input report | Payload SHA-256 | File SHA-256 |", "| --- | --- | --- |"])
    for source in provenance["input_reports"]:
        lines.append(
            "| {path} | `{payload}` | `{file}` |".format(
                path=source["path"],
                payload=source["payload_sha256"],
                file=source["file_sha256"],
            )
        )
    lines.extend(["", "| Evidence section | SHA-256 |", "| --- | --- |"])
    for key, value in report["evidence_checksums"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Input byte binding", ""])
    for source in provenance["input_reports"]:
        lines.append(
            f"- `{source['path']}`: payload matches file `{source['payload_matches_file']}`"
        )
    lines.extend(["", "## Source example files", ""])
    for source in report["source_example_files"]:
        lines.append(
            f"- `{source['case_id']}`: `{source['path']}`; "
            f"SHA-256 `{source['file_sha256']}`; "
            f"declared hash matches `{source['declared_hash_matches_file']}`"
        )
    lines.extend(["", "## Case identity errors", ""])
    lines.extend(f"- `{error}`" for error in report["case_identity_errors"])
    if not report["case_identity_errors"]:
        lines.append("- No reported identity discrepancies; this is not field custody.")
    lines.extend(["", "## Threshold diagnostics", ""])
    for case in report["threshold_cases"]:
        lines.append(f"- `{case['case_id']}`:")
        for check in case["failed_threshold_checks"]:
            lines.append(
                f"  - Failed `{check['metric']}`: `{check['value']}` "
                f"against `{check['comparator']} {check['limit']}`."
            )
        for metric in case["contradictory_threshold_metrics"]:
            lines.append(f"  - Contradictory reported verdict: `{metric}`.")
        for metric in case["unverified_threshold_metrics"]:
            lines.append(f"  - Unverified: `{metric}`.")
    lines.extend(["", "## Grid diagnostic discrepancies", ""])
    for case in report["grid_convergence"]["cases"]:
        lines.append(f"- `{case['case_id']}`:")
        for field in case["contradictory_grid_fields"]:
            lines.append(f"  - Contradictory grid field: `{field}`.")
        for row in case["resolution_diagnostics"]:
            resolution = row["resolution"]
            for field in row["inconsistent_metric_values"]:
                lines.append(f"  - Grid `{resolution}` inconsistent metric: `{field}`.")
            for check in row["failed_threshold_checks"]:
                lines.append(
                    f"  - Grid `{resolution}` failed `{check['metric']}`: `{check['value']}`."
                )
            for field in row["contradictory_threshold_metrics"]:
                lines.append(f"  - Grid `{resolution}` contradictory threshold: `{field}`.")
            if not row["reported_failed_count_matches"]:
                lines.append(f"  - Grid `{resolution}` reported failure count disagrees.")
    return "\n".join(lines) + "\n"
