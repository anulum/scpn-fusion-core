#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Transport Power-Balance Benchmark
# Verifies MW -> keV/s source normalization in integrated transport.
# ──────────────────────────────────────────────────────────────────────
"""Benchmark auxiliary-heating source power-balance reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from scpn_fusion.core.integrated_transport_solver import TransportSolver


def _run_case(
    config_path: str,
    *,
    p_aux_mw: float,
    multi_ion: bool,
    electron_fraction: float,
) -> dict[str, Any]:
    solver = TransportSolver(config_path, multi_ion=multi_ion)
    solver.aux_heating_electron_fraction = electron_fraction
    solver._compute_aux_heating_sources(p_aux_mw)
    bal = solver._last_aux_heating_balance

    target = float(bal["target_total_MW"])
    reconstructed = float(bal["reconstructed_total_MW"])
    rel_err = abs(reconstructed - target) / max(abs(target), 1e-12)

    return {
        "mode": "multi_ion" if multi_ion else "single_ion",
        "p_aux_mw": float(p_aux_mw),
        "target_total_mw": target,
        "reconstructed_total_mw": reconstructed,
        "target_ion_mw": float(bal["target_ion_MW"]),
        "target_electron_mw": float(bal["target_electron_MW"]),
        "reconstructed_ion_mw": float(bal["reconstructed_ion_MW"]),
        "reconstructed_electron_mw": float(bal["reconstructed_electron_MW"]),
        "relative_error": float(rel_err),
    }


def run_benchmark(
    *,
    config_path: str | None = None,
    powers_mw: list[float] | None = None,
    electron_fraction: float = 0.5,
) -> dict[str, Any]:
    cfg = config_path or str(ROOT / "iter_config.json")
    powers = powers_mw or [10.0, 30.0, 50.0, 100.0]
    threshold = 1e-6

    cases: list[dict[str, Any]] = []
    for p in powers:
        cases.append(
            _run_case(
                cfg,
                p_aux_mw=float(p),
                multi_ion=False,
                electron_fraction=0.0,
            )
        )
        cases.append(
            _run_case(
                cfg,
                p_aux_mw=float(p),
                multi_ion=True,
                electron_fraction=electron_fraction,
            )
        )

    rel_errors = np.asarray([float(c["relative_error"]) for c in cases], dtype=np.float64)
    max_rel_error = float(np.max(rel_errors)) if rel_errors.size else 0.0
    mean_rel_error = float(np.mean(rel_errors)) if rel_errors.size else 0.0
    passes = bool(max_rel_error <= threshold)

    return {
        "transport_power_balance_benchmark": {
            "n_cases": int(len(cases)),
            "max_relative_error": max_rel_error,
            "mean_relative_error": mean_rel_error,
            "threshold_max_relative_error": threshold,
            "passes_thresholds": passes,
            "cases": cases,
        }
    }


def render_markdown(report: dict[str, Any]) -> str:
    g = report["transport_power_balance_benchmark"]
    lines = [
        "# Transport Power-Balance Benchmark",
        "",
        f"- Cases: `{g['n_cases']}`",
        f"- Max relative error: `{g['max_relative_error']:.3e}` "
        f"(threshold `<= {g['threshold_max_relative_error']:.1e}`)",
        f"- Mean relative error: `{g['mean_relative_error']:.3e}`",
        f"- Passes thresholds: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        "",
        "| Mode | P_aux [MW] | Target total [MW] | Reconstructed total [MW] | Rel error |",
        "|------|------------|-------------------|--------------------------|-----------|",
    ]
    for c in g["cases"]:
        lines.append(
            f"| {c['mode']} | {c['p_aux_mw']:.1f} | {c['target_total_mw']:.6f} | "
            f"{c['reconstructed_total_mw']:.6f} | {c['relative_error']:.3e} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run transport power-balance benchmark.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "iter_config.json"),
        help="Path to transport config JSON.",
    )
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "transport_power_balance_benchmark.json"),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "transport_power_balance_benchmark.md"),
        help="Output Markdown report path.",
    )
    args = parser.parse_args()

    report = run_benchmark(config_path=args.config)

    json_path = Path(args.output_json)
    md_path = Path(args.output_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    g = report["transport_power_balance_benchmark"]
    print("Transport power-balance benchmark complete.")
    print(
        "cases={cases}, max_rel_error={max_err:.3e}, passes={passes}".format(
            cases=g["n_cases"],
            max_err=g["max_relative_error"],
            passes=g["passes_thresholds"],
        )
    )
    return 0 if g["passes_thresholds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
