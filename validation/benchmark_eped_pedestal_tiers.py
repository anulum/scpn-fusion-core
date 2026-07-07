#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — EPED Pedestal Tier Benchmark vs Digitised EPED1 References
"""Benchmark both pedestal tiers against digitised EPED1 DIII-D references.

Runs the width-height-scaling fast tier and the peeling-ballooning + KBM
constraint-loop tier on the DIII-D Ip-scan cases digitised from the public
EPED presentation (see ``validation/reference_data/eped/``), and documents
the divergence of each tier from the published EPED1 predictions and DIII-D
measurements. This is a documentation benchmark, fail-closed against parity
claims: no quantitative EPED parity is asserted, and the named blockers for
such a claim are recorded in the report.

Known honest outcomes encoded here:

- The fast tier under-predicts the published DIII-D-class pedestal heights
  and widths (its width-height calibration was never EPED-validated).
- The PB-KBM tier collapses to its scan floor for DIII-D-class inputs:
  the s-alpha first-stability ballooning boundary lacks the shaped-geometry
  second-stability access that real pedestals exploit. Shaped (Miller)
  ballooning is the recorded blocker.
- The slide states only Bt, kappa, delta, and Ip; R0, a, and n_ped are
  assumptions declared in the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from scpn_fusion.core.eped_pb_kbm import PBKBMPedestalModel
from scpn_fusion.core.eped_pedestal import EpedPedestalModel

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (
    ROOT / "validation" / "reference_data" / "eped" / "eped1_snyder_apsdpp_diiid_ip_scan.json"
)
DEFAULT_REPORT = ROOT / "validation" / "reports" / "eped_pedestal_tiers_benchmark.json"
REFERENCE_SCHEMA = "scpn-fusion-core.eped1-digitised-reference.v1"
REPORT_SCHEMA = "scpn-fusion-core.eped-pedestal-tiers-benchmark.v1"

ASSUMED_GEOMETRY = {"R0_m": 1.67, "a_m": 0.67}
DENSITY_SCAN_1E19 = (4.0, 6.0, 8.0)
BLOCKERS = (
    "shaped_geometry_miller_ballooning_required_for_second_stability_access",
    "n_ped_and_geometry_not_published_on_reference_slide",
)


def load_reference(path: Path) -> dict[str, Any]:
    """Load and validate the digitised EPED1 reference artifact.

    Parameters
    ----------
    path : Path
        Filesystem path to the digitised reference JSON.

    Returns
    -------
    dict
        Parsed reference payload.

    Raises
    ------
    ValueError
        If the schema is foreign or the Ip-scan cases are missing.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != REFERENCE_SCHEMA:
        raise ValueError("unexpected digitised EPED1 reference schema")
    cases = payload.get("diiid_ip_scan", {}).get("cases")
    if not isinstance(cases, list) or len(cases) == 0:
        raise ValueError("digitised reference must carry non-empty diiid_ip_scan.cases")
    return payload


def _best_over_density(predictions: Sequence[Mapping[str, float]], key: str) -> Mapping[str, float]:
    return max(predictions, key=lambda item: item[key])


def run_case(case: Mapping[str, Any], machine: Mapping[str, Any]) -> dict[str, Any]:
    """Run both pedestal tiers for one digitised Ip-scan case.

    Parameters
    ----------
    case : Mapping
        Digitised reference row (Ip, EPED1 prediction, measurement).
    machine : Mapping
        Slide-stated machine inputs (Bt, kappa, delta).

    Returns
    -------
    dict
        Per-case comparison row with best-over-density tier predictions,
        ratios against EPED1 and measurement, and per-tier verdicts.
    """
    ip_ma = float(case["Ip_MA"])
    b0 = float(machine["B0_T"])
    kappa = float(machine["kappa"])
    delta = float(machine["delta"])
    r0 = ASSUMED_GEOMETRY["R0_m"]
    a_minor = ASSUMED_GEOMETRY["a_m"]

    fast_predictions: list[dict[str, float]] = []
    pbkbm_predictions: list[dict[str, float]] = []
    for n_ped in DENSITY_SCAN_1E19:
        fast_model = EpedPedestalModel(R0=r0, a=a_minor, B0=b0, Ip_MA=ip_ma, kappa=kappa)
        fast = fast_model.predict(n_ped, domain_mode="ignore")
        fast_predictions.append(
            {
                "n_ped_1e19": n_ped,
                "p_ped_kPa": fast.p_ped_kPa,
                "T_ped_keV": fast.T_ped_keV,
                "Delta_psiN": fast.Delta_ped,
            }
        )
        pbkbm_model = PBKBMPedestalModel(
            R0=r0, a=a_minor, B0=b0, Ip_MA=ip_ma, kappa=kappa, delta=delta
        )
        pbkbm = pbkbm_model.predict(n_ped, coarse_points=15, refine_iterations=8)
        pbkbm_predictions.append(
            {
                "n_ped_1e19": n_ped,
                "p_ped_kPa": pbkbm.p_ped_kPa,
                "T_ped_keV": pbkbm.T_ped_keV,
                "Delta_psiN": pbkbm.Delta_ped,
                "converged": float(pbkbm.converged),
                "pb_boundary_radius": pbkbm.pb_boundary_radius,
                "alpha_crit": pbkbm.alpha_crit,
            }
        )

    eped1_p = float(case["eped1_p_ped_kPa"])
    measured_p = float(case["measured_p_ped_kPa"])
    fast_best = _best_over_density(fast_predictions, "p_ped_kPa")
    pbkbm_best = _best_over_density(pbkbm_predictions, "p_ped_kPa")
    pbkbm_collapsed = all(item["converged"] == 0.0 for item in pbkbm_predictions)

    return {
        "Ip_MA": ip_ma,
        "reference": {
            "eped1_p_ped_kPa": eped1_p,
            "eped1_Delta_psiN": float(case["eped1_Delta_psiN"]),
            "measured_p_ped_kPa": measured_p,
            "measured_Delta_psiN": float(case["measured_Delta_psiN"]),
        },
        "fast_tier": {
            "predictions": fast_predictions,
            "best_p_ped_kPa": fast_best["p_ped_kPa"],
            "best_Delta_psiN": fast_best["Delta_psiN"],
            "height_ratio_to_eped1": fast_best["p_ped_kPa"] / eped1_p,
            "height_ratio_to_measured": fast_best["p_ped_kPa"] / measured_p,
            "verdict": (
                "within_digitisation_band"
                if abs(fast_best["p_ped_kPa"] - eped1_p) <= 1.0
                else "divergent"
            ),
        },
        "pb_kbm_tier": {
            "predictions": pbkbm_predictions,
            "best_p_ped_kPa": pbkbm_best["p_ped_kPa"],
            "best_Delta_psiN": pbkbm_best["Delta_psiN"],
            "height_ratio_to_eped1": pbkbm_best["p_ped_kPa"] / eped1_p,
            "height_ratio_to_measured": pbkbm_best["p_ped_kPa"] / measured_p,
            "collapsed_to_scan_floor": pbkbm_collapsed,
            "verdict": ("collapsed_salpha_first_stability" if pbkbm_collapsed else "divergent"),
        },
    }


def build_report(reference: dict[str, Any]) -> dict[str, Any]:
    """Build the full tier-benchmark report payload.

    Parameters
    ----------
    reference : dict
        Validated digitised EPED1 reference payload.

    Returns
    -------
    dict
        Schema-versioned report with per-case rows, declared assumptions,
        named blockers, and the fail-closed status string.
    """
    machine = reference["diiid_ip_scan"]["machine_inputs"]
    rows = [run_case(case, machine) for case in reference["diiid_ip_scan"]["cases"]]
    fast_ratios = [row["fast_tier"]["height_ratio_to_eped1"] for row in rows]
    return {
        "schema": REPORT_SCHEMA,
        "status": "documented_divergence_no_parity_claim",
        "reference_source_url": reference["source"]["url"],
        "reference_citations": reference["source"]["citations"],
        "digitisation_uncertainty": reference["digitisation_uncertainty"],
        "assumed_geometry": dict(ASSUMED_GEOMETRY),
        "assumption_note": reference["diiid_ip_scan"]["machine_inputs"]["note"],
        "density_scan_1e19": list(DENSITY_SCAN_1E19),
        "blockers": list(BLOCKERS),
        "eped1_validation_statistics": reference["validation_statistics"],
        "rows": rows,
        "summary": {
            "cases": len(rows),
            "fast_tier_height_ratio_min": min(fast_ratios),
            "fast_tier_height_ratio_max": max(fast_ratios),
            "fast_tier_underpredicts_all_cases": all(r < 1.0 for r in fast_ratios),
            "pb_kbm_collapsed_all_cases": all(
                row["pb_kbm_tier"]["collapsed_to_scan_floor"] for row in rows
            ),
        },
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "",
        "# EPED Pedestal Tier Benchmark (documented divergence)",
        "",
        f"- Status: `{report['status']}`",
        f"- Reference: digitised EPED1 DIII-D Ip scan ({report['reference_source_url']})",
        f"- Assumed geometry: R0={report['assumed_geometry']['R0_m']} m, "
        f"a={report['assumed_geometry']['a_m']} m (not stated on the slide)",
        f"- Density scan: {report['density_scan_1e19']} ×1e19 m⁻³",
        "",
        "| Ip [MA] | EPED1 p_ped [kPa] | measured [kPa] | fast tier best [kPa] | ratio | PB-KBM best [kPa] | PB-KBM verdict |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['Ip_MA']} | {row['reference']['eped1_p_ped_kPa']} | "
            f"{row['reference']['measured_p_ped_kPa']} | "
            f"{row['fast_tier']['best_p_ped_kPa']:.1f} | "
            f"{row['fast_tier']['height_ratio_to_eped1']:.2f} | "
            f"{row['pb_kbm_tier']['best_p_ped_kPa']:.1f} | "
            f"`{row['pb_kbm_tier']['verdict']}` |"
        )
    lines += [
        "",
        "Named blockers for any quantitative EPED parity claim:",
        "",
    ]
    lines += [f"- `{blocker}`" for blocker in report["blockers"]]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the tier benchmark and write the tracked report.

    Parameters
    ----------
    argv : Sequence[str], optional
        CLI argument list; defaults to process arguments.

    Returns
    -------
    int
        ``0`` when the report is written with its documentation invariants
        intact, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=str, default=str(REFERENCE))
    parser.add_argument("--output-json", type=str, default=str(DEFAULT_REPORT))
    args = parser.parse_args(argv)

    reference = load_reference(Path(args.reference))
    report = build_report(reference)

    # Documentation invariants (fail closed): every row must carry ratios,
    # the status string must not claim parity, and blockers must be present.
    if report["status"] != "documented_divergence_no_parity_claim":
        print("unexpected status", file=sys.stderr)
        return 1
    if not report["blockers"]:
        print("blockers missing", file=sys.stderr)
        return 1
    for row in report["rows"]:
        for tier in ("fast_tier", "pb_kbm_tier"):
            if not (row[tier]["height_ratio_to_eped1"] > 0.0):
                print(f"invalid ratio in {tier} for Ip={row['Ip_MA']}", file=sys.stderr)
                return 1

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(_render_markdown(report), encoding="utf-8")
    print(f"Wrote EPED pedestal tier benchmark: {output}")
    summary = report["summary"]
    print(
        f"fast tier height ratio {summary['fast_tier_height_ratio_min']:.2f}–"
        f"{summary['fast_tier_height_ratio_max']:.2f}; "
        f"pb_kbm collapsed on all cases: {summary['pb_kbm_collapsed_all_cases']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
