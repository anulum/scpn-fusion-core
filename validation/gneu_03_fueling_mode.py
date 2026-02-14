# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GNEU-03 Fueling Validation
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Validate ice-pellet fueling mode on reduced ITER-like density dynamics."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

from scpn_fusion.control.fueling_mode import simulate_iter_density_control


def generate_report(
    *,
    target_density: float = 1.0,
    initial_density: float = 0.82,
    steps: int = 3000,
    dt_s: float = 1e-3,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    result = simulate_iter_density_control(
        target_density=target_density,
        initial_density=initial_density,
        steps=steps,
        dt_s=dt_s,
    )
    elapsed = time.perf_counter() - t0

    threshold = 1e-3
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": elapsed,
        "gneu_03": {
            "target_density": target_density,
            "initial_density": initial_density,
            "steps": result.steps,
            "dt_s": result.dt_s,
            "final_density": result.final_density,
            "final_abs_error": result.final_abs_error,
            "rmse": result.rmse,
            "threshold_final_abs_error": threshold,
            "passes_thresholds": bool(result.final_abs_error <= threshold),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    g = report["gneu_03"]
    lines = [
        "# GNEU-03 Fueling Validation",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{report['runtime_seconds']:.3f} s`",
        "",
        "## Results",
        "",
        f"- Final density: `{g['final_density']:.6f}`",
        f"- Final absolute error: `{g['final_abs_error']:.6e}`",
        f"- RMSE: `{g['rmse']:.6e}`",
        f"- Threshold (`<=1e-3`) pass: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-density", type=float, default=1.0)
    parser.add_argument("--initial-density", type=float, default=0.82)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--dt-s", type=float, default=1e-3)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "gneu_03_fueling.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "gneu_03_fueling.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(
        target_density=args.target_density,
        initial_density=args.initial_density,
        steps=args.steps,
        dt_s=args.dt_s,
    )
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = report["gneu_03"]
    print("GNEU-03 fueling validation complete.")
    print(
        f"final_abs_error={g['final_abs_error']:.6e}, "
        f"threshold={g['threshold_final_abs_error']:.6e}, "
        f"passes_thresholds={g['passes_thresholds']}"
    )

    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
