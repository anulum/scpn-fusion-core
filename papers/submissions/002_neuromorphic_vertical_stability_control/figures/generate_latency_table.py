#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — historical and wiring evidence table generator.
"""Generate historical and schema-v3 tables from paper-local evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path


HISTORICAL_LANES = ("PID", "Rust-PID", "H-infinity", "NMPC-JAX", "Nengo-SNN")
WIRING_LANES = (
    "PID",
    "Rust-PID",
    "H-infinity",
    "LQR",
    "MPC",
    "LIF-NEF-SNN",
    "NMPC-JAX",
)


def _header(description: str) -> list[str]:
    """Return the mandatory header for one generated LaTeX evidence file."""
    return [
        "% SPDX-License-Identifier: AGPL-3.0-or-later",
        "% Commercial license available",
        "% © Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "% © Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "% ORCID: 0009-0009-3560-0851",
        "% Contact: www.anulum.li | protoscience@anulum.li",
        f"% SCPN-FUSION-CORE — {description}.",
    ]


def _validated_latency(record: dict[str, object], lane: str) -> tuple[float, float]:
    """Return finite ordered p50/p95 values from one evidence record."""
    p50 = float(record["p50_latency_us"])
    p95 = float(record["p95_latency_us"])
    if not math.isfinite(p50) or not math.isfinite(p95) or p50 <= 0.0 or p95 < p50:
        raise ValueError(f"Invalid latency record for {lane}: {record}")
    return p50, p95


def main() -> None:
    """Write deterministic tables for the historical and wiring-only records."""
    submission = Path(__file__).resolve().parent.parent
    evidence_dir = submission / "evidence"
    historical_source = evidence_dir / "historical_controller_latency.json"
    historical = json.loads(historical_source.read_text(encoding="utf-8"))["results"]
    historical_rows = []
    for lane in HISTORICAL_LANES:
        record = historical[lane]
        p50, p95 = _validated_latency(record, lane)
        label = f"{lane}$^{{\\dagger}}$" if lane == "H-infinity" else lane
        historical_rows.append(f"{label} & {p50:.3f} & {p95:.3f} \\\\")
    historical_table = "\n".join(
        [
            *_header("generated historical-latency table"),
            "\\begin{table}[ht]",
            "\\centering",
            "\\caption{Historical legacy-harness latency output. The exact invocation "
            "episode count was not serialized; lanes are not accuracy-equivalent.}",
            "\\label{tab:historical-latency}",
            "\\begin{tabular}{lrr}",
            "\\toprule",
            "Legacy lane & p50 ($\\mu$s) & p95 ($\\mu$s) \\\\",
            "\\midrule",
            *historical_rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\vspace{2pt}",
            "\\parbox{0.92\\linewidth}{\\footnotesize $^{\\dagger}$Invalidated stale "
            "scalar-plant calibration. All other rows remain historical orientation only.}",
            "\\end{table}",
            "",
        ]
    )
    (evidence_dir / "controller_latency_table.tex").write_text(historical_table, encoding="utf-8")

    wiring_source = evidence_dir / "schema_v3_wiring_assessment.json"
    wiring = json.loads(wiring_source.read_text(encoding="utf-8"))["result"]["lanes"]
    wiring_rows = []
    for lane in WIRING_LANES:
        record = wiring[lane]
        status = str(record["status"]).replace("_", "\\_")
        p50_value = record["p50_policy_latency_us"]
        p95_value = record["p95_policy_latency_us"]
        if p50_value is None or p95_value is None:
            p50_text = p95_text = "---"
        else:
            p50, p95 = _validated_latency(
                {"p50_latency_us": p50_value, "p95_latency_us": p95_value}, lane
            )
            p50_text = f"{p50:.3f}"
            p95_text = f"{p95:.3f}"
        wiring_rows.append(f"{lane} & {status} & {p50_text} & {p95_text} \\\\")
    wiring_table = "\n".join(
        [
            *_header("generated schema-v3 wiring table"),
            "\\begin{table}[ht]",
            "\\centering",
            "\\caption{One-episode schema-v3 wiring assessment. Policy-only timing is "
            "recorded, but the run is incomplete and promotion-ineligible.}",
            "\\label{tab:wiring-latency}",
            "\\begin{tabular}{llrr}",
            "\\toprule",
            "Lane & Status & p50 ($\\mu$s) & p95 ($\\mu$s) \\\\",
            "\\midrule",
            *wiring_rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    (evidence_dir / "schema_v3_wiring_table.tex").write_text(wiring_table, encoding="utf-8")
    print("  [OK] controller_latency_table (historical non-comparable lanes)")
    print("  [OK] schema_v3_wiring_table (wiring-only lanes)")


if __name__ == "__main__":
    main()
