#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Subsystem Fault Hardening Campaign
"""Reduced-order REBCO, DEC, and structural fault-hardening campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpn_fusion.core.direct_energy_conversion import (  # noqa: E402
    evaluate_direct_energy_conversion_fault,
)
from scpn_fusion.core.disruption_structural_response import (  # noqa: E402
    evaluate_disruption_structural_response,
)
from scpn_fusion.core.hts_quench import evaluate_rebco_quench  # noqa: E402


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _lane(
    scenario_id: str,
    *,
    evidence_source: str,
    response_time_s: float | None,
    report: dict[str, Any],
    summary: str,
) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "status": "measured_reduced_order",
        "evidence_source": evidence_source,
        "response_time_s": response_time_s,
        "passes_thresholds": bool(report["passes_thresholds"]),
        "summary": summary,
        "claim_boundary": report["claim_boundary"],
    }


def run_campaign() -> dict[str, Any]:
    """Run the three reduced-order subsystem hardening lanes.

    Returns
    -------
    dict[str, Any]
        Deterministic subsystem evidence matrix and diagnostic payloads.
    """
    rebco = evaluate_rebco_quench().to_dict()
    dec = evaluate_direct_energy_conversion_fault().to_dict()
    structural = evaluate_disruption_structural_response().to_dict()
    lanes = [
        _lane(
            "rebco_quench_fault",
            evidence_source="scpn_fusion.core.hts_quench.evaluate_rebco_quench",
            response_time_s=float(rebco["detection_time_s"]),
            report=rebco,
            summary=(
                f"hotspot {rebco['hotspot_temperature_k']:.2f} K; "
                f"terminal {rebco['peak_terminal_voltage_v']:.1f} V; "
                f"detection {rebco['detection_voltage_v']:.4f} V"
            ),
        ),
        _lane(
            "direct_energy_conversion_fault",
            evidence_source=(
                "scpn_fusion.core.direct_energy_conversion.evaluate_direct_energy_conversion_fault"
            ),
            response_time_s=float(dec["fail_closed_time_ms"]) * 1.0e-3,
            report=dec,
            summary=(
                f"fail-closed {dec['fail_closed_time_ms']:.2f} ms; "
                f"isolated energy {dec['isolated_energy_mj']:.4f} MJ; "
                f"bus overvoltage {dec['bus_overvoltage_fraction']:.4f}"
            ),
        ),
        _lane(
            "disruption_structural_shock_strain",
            evidence_source=(
                "scpn_fusion.core.disruption_structural_response."
                "evaluate_disruption_structural_response"
            ),
            response_time_s=None,
            report=structural,
            summary=(
                f"equivalent stress {structural['equivalent_stress_mpa']:.2f} MPa; "
                f"stress margin {structural['stress_margin']:.2f}; "
                f"strain margin {structural['strain_margin']:.2f}"
            ),
        ),
    ]
    payload = {
        "schema": "scpn-fusion-core.subsystem_fault_hardening.v1",
        "campaign_status": "reduced_order_software_evidence_no_hardware_claim",
        "passes_available_evidence": all(bool(row["passes_thresholds"]) for row in lanes),
        "full_fidelity_claim_ready": False,
        "claim_boundary": (
            "Reduced-order software campaign only. This is not certified quench "
            "protection, validated direct-energy conversion, finite-element "
            "analysis, hardware-in-the-loop evidence, or plant qualification."
        ),
        "scenario_rows": lanes,
        "measured_lane_count": len(lanes),
        "diagnostics": {
            "rebco_quench": rebco,
            "direct_energy_conversion": dec,
            "disruption_structural_response": structural,
        },
    }
    payload["trace_checksum"] = _canonical_hash(
        {
            "scenario_rows": lanes,
            "diagnostics": payload["diagnostics"],
        }
    )
    return payload


def generate_report() -> dict[str, Any]:
    """Create a timestamped subsystem fault-hardening report."""
    report = run_campaign()
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return report


def render_markdown(report: dict[str, Any]) -> str:
    """Render the subsystem campaign report as Markdown."""
    lines = [
        "# Subsystem Fault Hardening Campaign",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Schema: `{report['schema']}`",
        f"- Status: `{report['campaign_status']}`",
        f"- Available evidence pass: `{'YES' if report['passes_available_evidence'] else 'NO'}`",
        f"- Full-fidelity claim ready: `{'YES' if report['full_fidelity_claim_ready'] else 'NO'}`",
        f"- Trace checksum: `{report['trace_checksum']}`",
        "",
        report["claim_boundary"],
        "",
        "## Scenario Matrix",
        "",
        "| Scenario | Status | Evidence | Response time [s] | Pass | Summary |",
        "|----------|--------|----------|-------------------|------|---------|",
    ]
    for row in report["scenario_rows"]:
        response = "n/a" if row["response_time_s"] is None else f"{row['response_time_s']:.4f}"
        lines.append(
            f"| {row['scenario_id']} | {row['status']} | {row['evidence_source']} | "
            f"{response} | {'YES' if row['passes_thresholds'] else 'NO'} | {row['summary']} |"
        )

    diagnostics = report["diagnostics"]
    rebco = diagnostics["rebco_quench"]
    dec = diagnostics["direct_energy_conversion"]
    structural = diagnostics["disruption_structural_response"]
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            f"- REBCO current-sharing margin: `{rebco['current_sharing_margin_k']:.2f} K`",
            f"- REBCO hotspot temperature: `{rebco['hotspot_temperature_k']:.2f} K`",
            f"- DEC isolated energy: `{dec['isolated_energy_mj']:.4f} MJ`",
            f"- DEC peak dump power: `{dec['peak_dump_power_mw']:.2f} MW`",
            f"- Structural equivalent stress: `{structural['equivalent_stress_mpa']:.2f} MPa`",
            f"- Structural displacement: `{structural['displacement_mm']:.3f} mm`",
            "",
            "## Boundaries",
            "",
        ]
    )
    for row in report["scenario_rows"]:
        lines.append(f"- `{row['scenario_id']}`: {row['claim_boundary']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the campaign and write JSON/Markdown artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "subsystem_fault_hardening_campaign.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "subsystem_fault_hardening_campaign.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    report = generate_report()
    report["runtime_seconds"] = float(time.perf_counter() - started)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    print("Subsystem fault-hardening campaign complete.")
    print(
        "Summary -> "
        f"available_evidence_pass={report['passes_available_evidence']}, "
        f"measured_lanes={report['measured_lane_count']}, "
        f"trace_checksum={report['trace_checksum'][:16]}"
    )
    if args.strict and not report["passes_available_evidence"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
