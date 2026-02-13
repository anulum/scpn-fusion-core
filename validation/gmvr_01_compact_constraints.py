# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GMVR-01 Compact-Constraint Validation
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""GMVR-01: compact MVR constraint scan with HTS/divertor/Zeff caps."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scpn_fusion.core.global_design_scanner import GlobalDesignExplorer


def run_campaign(
    *,
    seed: int = 42,
    scan_samples: int = 2600,
) -> dict[str, Any]:
    t0 = time.perf_counter()

    explorer = GlobalDesignExplorer(
        "dummy",
        divertor_flux_cap_mw_m2=45.0,
        zeff_cap=0.4,
        hts_peak_cap_t=21.0,
    )
    df = explorer.run_compact_scan(n_samples=scan_samples, seed=seed)

    feasible = df[
        (df["Constraint_OK"])
        & (df["R"] >= 1.2)
        & (df["R"] <= 1.5)
        & (df["Q"] > 5.0)
    ]

    feasible_count = int(len(feasible))
    best = feasible.loc[feasible["Cost"].idxmin()] if feasible_count > 0 else None

    result = {
        "seed": int(seed),
        "scan_samples": int(scan_samples),
        "feasible_count": feasible_count,
        "target_constraints": {
            "radius_min_m": 1.2,
            "radius_max_m": 1.5,
            "q_min": 5.0,
            "divertor_flux_cap_mw_m2": 45.0,
            "zeff_cap": 0.4,
            "hts_peak_cap_t": 21.0,
        },
        "runtime_seconds": float(time.perf_counter() - t0),
    }

    if best is not None:
        result["best_design"] = {
            "R_m": float(best["R"]),
            "B_t": float(best["B"]),
            "Ip_MA": float(best["Ip"]),
            "Q": float(best["Q"]),
            "Div_Load_Optimized_MW_m2": float(best["Div_Load_Optimized"]),
            "Zeff_Est": float(best["Zeff_Est"]),
            "B_peak_HTS_T": float(best["B_peak_HTS_T"]),
        }
        result["passes_thresholds"] = True
    else:
        result["best_design"] = None
        result["passes_thresholds"] = False

    return result


def generate_report(**kwargs: Any) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gmvr_01": run_campaign(**kwargs),
    }


def render_markdown(report: dict[str, Any]) -> str:
    g = report["gmvr_01"]
    lines = [
        "# GMVR-01 Compact Constraint Validation",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{g['runtime_seconds']:.3f} s`",
        f"- Feasible designs in target window: `{g['feasible_count']}`",
        "",
        "## Target Caps",
        "",
        f"- Radius window: `{g['target_constraints']['radius_min_m']:.1f}..{g['target_constraints']['radius_max_m']:.1f} m`",
        f"- Q target: `>{g['target_constraints']['q_min']:.1f}`",
        f"- Divertor flux cap: `<= {g['target_constraints']['divertor_flux_cap_mw_m2']:.1f} MW/m2`",
        f"- Zeff cap: `<= {g['target_constraints']['zeff_cap']:.2f}`",
        f"- HTS peak cap: `<= {g['target_constraints']['hts_peak_cap_t']:.1f} T`",
        "",
    ]

    best = g["best_design"]
    if best:
        lines.extend(
            [
                "## Best Feasible Design",
                "",
                f"- R: `{best['R_m']:.3f} m`",
                f"- B: `{best['B_t']:.3f} T`",
                f"- Ip: `{best['Ip_MA']:.3f} MA`",
                f"- Q: `{best['Q']:.3f}`",
                f"- Divertor (optimized): `{best['Div_Load_Optimized_MW_m2']:.3f} MW/m2`",
                f"- Zeff estimate: `{best['Zeff_Est']:.3f}`",
                f"- HTS peak field: `{best['B_peak_HTS_T']:.3f} T`",
                "",
            ]
        )
    else:
        lines.extend(["## Best Feasible Design", "", "- None found in current scan.", ""])

    lines.append(f"- Threshold pass: `{'YES' if g['passes_thresholds'] else 'NO'}`")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scan-samples", type=int, default=2600)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "gmvr_01_compact_constraints.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "gmvr_01_compact_constraints.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(seed=args.seed, scan_samples=args.scan_samples)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = report["gmvr_01"]
    print("GMVR-01 compact-constraint validation complete.")
    print(
        f"feasible_count={g['feasible_count']}, "
        f"passes_thresholds={g['passes_thresholds']}"
    )

    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
