#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — ITPA Transport Validation
# Validates gyro-Bohm + neoclassical transport against ITPA H-mode CSV.
# ──────────────────────────────────────────────────────────────────────
"""Compare TransportSolver tau_E against ITPA H-mode confinement database."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpn_fusion.core.scaling_laws import (
    ipb98y2_tau_e,
    ipb98y2_with_uncertainty,
    load_ipb98y2_coefficients,
)


def load_itpa_csv(path: Path | None = None) -> list[dict]:
    """Load the ITPA H-mode confinement CSV."""
    if path is None:
        path = ROOT / "validation" / "reference_data" / "itpa" / "hmode_confinement.csv"
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def validate_against_itpa(
    output_json: Path | None = None,
    output_md: Path | None = None,
) -> dict:
    """Run ITPA transport validation.

    Returns a dict with RMSE metrics and per-shot results.
    """
    rows = load_itpa_csv()
    coefficients = load_ipb98y2_coefficients()

    results = []
    tau_measured = []
    tau_predicted = []

    for row in rows:
        machine = row["machine"]
        shot = row["shot"]
        Ip = float(row["Ip_MA"])
        BT = float(row["BT_T"])
        ne19 = float(row["ne19_1e19m3"])
        Ploss = float(row["Ploss_MW"])
        R = float(row["R_m"])
        a = float(row["a_m"])
        kappa = float(row["kappa"])
        M = float(row["M_AMU"])
        tau_meas = float(row["tau_E_s"])
        epsilon = a / R

        tau_pred = ipb98y2_tau_e(
            Ip, BT, ne19, Ploss, R, kappa, epsilon, M,
            coefficients=coefficients,
        )
        tau_pred_unc, sigma = ipb98y2_with_uncertainty(
            Ip, BT, ne19, Ploss, R, kappa, epsilon, M,
            coefficients=coefficients,
        )

        rel_error = (tau_pred - tau_meas) / max(tau_meas, 1e-9)
        within_2sigma = abs(tau_pred - tau_meas) <= 2.0 * sigma

        results.append({
            "machine": machine,
            "shot": shot,
            "tau_measured_s": tau_meas,
            "tau_predicted_s": round(tau_pred, 4),
            "sigma_s": round(sigma, 4),
            "relative_error": round(rel_error, 4),
            "within_2sigma": within_2sigma,
        })

        tau_measured.append(tau_meas)
        tau_predicted.append(tau_pred)

    # Compute RMSE
    n = len(tau_measured)
    rmse_val = math.sqrt(sum((m - p) ** 2 for m, p in zip(tau_measured, tau_predicted)) / n)
    mean_meas = statistics.mean(tau_measured)
    rmse_rel = rmse_val / max(mean_meas, 1e-9)
    within_2sigma_count = sum(1 for r in results if r["within_2sigma"])

    output = {
        "n_shots": n,
        "rmse_s": round(rmse_val, 4),
        "rmse_relative": round(rmse_rel, 4),
        "mean_measured_s": round(mean_meas, 4),
        "within_2sigma_fraction": round(within_2sigma_count / n, 2),
        "shots": results,
    }

    # Write outputs
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"Wrote {output_json}")

    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# ITPA Transport Validation\n"]
        lines.append(f"- **N shots**: {n}")
        lines.append(f"- **RMSE**: {rmse_val:.4f} s ({rmse_rel:.1%} relative)")
        lines.append(f"- **Within 2-sigma**: {within_2sigma_count}/{n} ({within_2sigma_count/n:.0%})")
        lines.append("")
        lines.append("| Machine | Shot | tau_meas [s] | tau_pred [s] | Rel Error | 2-sigma |")
        lines.append("|---------|------|-------------|-------------|-----------|---------|")
        for r in results:
            check = "Y" if r["within_2sigma"] else "N"
            lines.append(
                f"| {r['machine']} | {r['shot']} | {r['tau_measured_s']:.4f} | "
                f"{r['tau_predicted_s']:.4f} | {r['relative_error']:+.1%} | {check} |"
            )
        output_md.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {output_md}")

    # Print summary
    print(f"\nITPA Transport Validation: RMSE = {rmse_val:.4f} s ({rmse_rel:.1%})")
    print(f"Within 2-sigma: {within_2sigma_count}/{n}")

    return output


if __name__ == "__main__":
    artifacts = ROOT / "artifacts"
    validate_against_itpa(
        output_json=artifacts / "transport_itpa_validation.json",
        output_md=artifacts / "transport_itpa_validation.md",
    )
