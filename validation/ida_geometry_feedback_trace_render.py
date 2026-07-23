# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Markdown rendering for IDA geometry-feedback trace evidence."""

from __future__ import annotations

from typing import Any


def render_markdown(report: dict[str, Any]) -> str:
    """Render a concise checkpoint table after contract validation."""
    from validation.ida_geometry_feedback_trace_contract import validate_report

    validate_report(report)
    rows = []
    for run_name in ("cold", "warm"):
        for row in report["runs"][run_name]["checkpoints"]:
            rows.append(
                f"| {run_name} | {row['iteration_index']} | {row['phase']} | "
                f"{row['separatrix_refinement']:.3f} | "
                f"{row['production_current']['total_variation_distance']:.9g} | "
                f"{row['reference_boundary_counterfactual']['total_variation_distance']:.9g} | "
                f"{row['fixed_point']['residual_relative_l2']:.9g} | "
                f"{str(row['terminal']).lower()} |"
            )
    routing = report["routing"]
    return "\n".join(
        [
            "# IDA geometry/source feedback trace",
            "",
            f"- Status: `{report['status']}`",
            f"- Payload SHA-256: `{report['payload_sha256']}`",
            f"- Same-case terminal parity: `{str(routing['trace_matches_same_case_candidate']).lower()}`",
            "- Facility/control/PCS/safety/scientific admission: `false`",
            "",
            "| Run | Iteration | Phase | Refinement | Production TV | Ref-boundary TV | FP rel L2 | Terminal |",
            "|---|---:|---|---:|---:|---:|---:|---|",
            *rows,
            "",
            (
                "- Largest sparse TV increase: "
                f"`{routing['largest_sparse_tv_increase']['delta']:.9g}` into "
                f"`{routing['largest_sparse_tv_increase']['phase']}`"
            ),
            f"- Next ratcheting target: `{routing['next_ratcheting_target']}`",
            "",
            "This trace routes one engineering correction; it is not physical validation.",
            "",
        ]
    )
