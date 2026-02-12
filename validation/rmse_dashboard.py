# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — RMSE Dashboard
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Generate Phase-1 validation RMSE metrics for ITER/SPARC and ITPA data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scpn_fusion.core.eqdsk import GEqdsk, read_geqdsk
from scpn_fusion.core.fusion_ignition_sim import FusionBurnPhysics


def ipb98_tau_e(
    ip_ma: float,
    b_t: float,
    n_e19: float,
    p_loss_mw: float,
    r_m: float,
    kappa: float,
    epsilon: float,
    a_eff_amu: float = 2.5,
) -> float:
    """IPB98(y,2) confinement scaling law [s]."""
    return (
        0.0562
        * (ip_ma**0.93)
        * (b_t**0.15)
        * (n_e19**0.41)
        * (p_loss_mw**-0.69)
        * (r_m**1.97)
        * (kappa**0.78)
        * (epsilon**0.58)
        * (a_eff_amu**0.19)
    )


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("RMSE requires non-empty lists of equal length.")
    return math.sqrt(
        statistics.mean((float(t) - float(p)) ** 2 for t, p in zip(y_true, y_pred))
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_eq_axis(eq: GEqdsk) -> float:
    """Return axis localization error norm in meters from GEQDSK fields."""
    if eq.simag < eq.sibry:
        idx = int(np.argmin(eq.psirz))
    else:
        idx = int(np.argmax(eq.psirz))
    iz, ir = np.unravel_index(idx, eq.psirz.shape)
    r_psi = eq.r[ir]
    z_psi = eq.z[iz]
    return float(math.hypot(r_psi - eq.rmaxis, z_psi - eq.zmaxis))


def confinement_rmse_itpa(csv_path: Path) -> dict[str, Any]:
    tau_true: list[float] = []
    tau_pred: list[float] = []
    h98_true: list[float] = []
    h98_pred: list[float] = []
    rows: list[dict[str, Any]] = []

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tau_m = float(row["tau_E_s"])
            tau_p = ipb98_tau_e(
                ip_ma=float(row["Ip_MA"]),
                b_t=float(row["BT_T"]),
                n_e19=float(row["ne19_1e19m3"]),
                p_loss_mw=float(row["Ploss_MW"]),
                r_m=float(row["R_m"]),
                kappa=float(row["kappa"]),
                epsilon=float(row["a_m"]) / float(row["R_m"]),
                a_eff_amu=float(row["M_AMU"]),
            )
            h98_m = float(row["H98y2"])
            h98_p = tau_m / tau_p if tau_p > 0 else 0.0

            tau_true.append(tau_m)
            tau_pred.append(tau_p)
            h98_true.append(h98_m)
            h98_pred.append(h98_p)
            rows.append(
                {
                    "machine": row["machine"],
                    "shot": row["shot"],
                    "tau_measured_s": tau_m,
                    "tau_pred_s": tau_p,
                    "h98_measured": h98_m,
                    "h98_pred": h98_p,
                }
            )

    tau_rel_abs_pct = [
        abs(t - p) / t * 100.0 for t, p in zip(tau_true, tau_pred) if t > 0
    ]
    return {
        "count": len(tau_true),
        "tau_rmse_s": rmse(tau_true, tau_pred),
        "tau_mae_rel_pct": statistics.mean(tau_rel_abs_pct),
        "h98_rmse": rmse(h98_true, h98_pred),
        "rows": rows,
    }


def confinement_rmse_iter_sparc(reference_dir: Path) -> dict[str, Any]:
    refs = [
        load_json(reference_dir / "iter_reference.json"),
        load_json(reference_dir / "sparc_reference.json"),
    ]
    tau_true: list[float] = []
    tau_pred: list[float] = []
    rows: list[dict[str, Any]] = []

    for ref in refs:
        pred = ipb98_tau_e(
            ip_ma=float(ref["I_p_MA"]),
            b_t=float(ref["B_t_T"]),
            n_e19=float(ref["n_e_1e19"]),
            p_loss_mw=float(ref["P_loss_MW"]),
            r_m=float(ref["R_m"]),
            kappa=float(ref["kappa"]),
            epsilon=float(ref["a_m"]) / float(ref["R_m"]),
            a_eff_amu=float(ref["A_eff_amu"]),
        )
        obs = float(ref["tau_E_s"])
        tau_true.append(obs)
        tau_pred.append(pred)
        rows.append(
            {
                "scenario": ref["scenario"],
                "tau_measured_s": obs,
                "tau_pred_s": pred,
                "relative_error_pct": ((pred - obs) / obs * 100.0) if obs else 0.0,
            }
        )

    return {
        "count": len(tau_true),
        "tau_rmse_s": rmse(tau_true, tau_pred),
        "rows": rows,
    }


def estimate_beta_n_from_burn(
    reference: dict[str, Any],
    config_path: Path,
) -> tuple[float, dict[str, Any]]:
    """Estimate beta_N from burn-model thermal energy and machine geometry."""
    sim = FusionBurnPhysics(str(config_path))
    sim.solve_equilibrium()
    metrics = sim.calculate_thermodynamics(P_aux_MW=float(reference["P_aux_MW"]))

    w_thermal_j = float(metrics["W_MJ"]) * 1e6
    r_m = float(reference["R_m"])
    a_m = float(reference["a_m"])
    kappa = float(reference["kappa"])
    b_t = float(reference["B_t_T"])
    i_p = float(reference["I_p_MA"])

    volume = 2.0 * math.pi * math.pi * r_m * a_m * a_m * kappa
    p_avg = w_thermal_j / (3.0 * volume) if volume > 0 else 0.0
    mu0 = 4.0 * math.pi * 1e-7
    beta_t = (2.0 * mu0 * p_avg / (b_t * b_t)) if b_t > 0 else 0.0
    beta_n = (100.0 * beta_t) * a_m * b_t / i_p if i_p > 0 else 0.0
    return beta_n, metrics


def beta_rmse_iter_sparc(reference_dir: Path, validation_dir: Path) -> dict[str, Any]:
    pairs = [
        ("iter_reference.json", "iter_validated_config.json"),
        ("sparc_reference.json", "sparc_config.json"),
    ]
    beta_true: list[float] = []
    beta_pred: list[float] = []
    rows: list[dict[str, Any]] = []

    for ref_name, cfg_name in pairs:
        ref = load_json(reference_dir / ref_name)
        beta_obs = float(ref["beta_N"])
        beta_est, metrics = estimate_beta_n_from_burn(ref, validation_dir / cfg_name)
        beta_true.append(beta_obs)
        beta_pred.append(beta_est)
        rows.append(
            {
                "scenario": ref["scenario"],
                "beta_n_measured": beta_obs,
                "beta_n_estimated": beta_est,
                "relative_error_pct": ((beta_est - beta_obs) / beta_obs * 100.0)
                if beta_obs
                else 0.0,
                "model_q": float(metrics["Q"]),
                "model_p_fusion_mw": float(metrics["P_fusion_MW"]),
            }
        )

    return {
        "count": len(beta_true),
        "beta_n_rmse": rmse(beta_true, beta_pred),
        "rows": rows,
    }


def sparc_axis_rmse(sparc_dir: Path) -> dict[str, Any]:
    files = sorted(sparc_dir.glob("*.geqdsk")) + sorted(sparc_dir.glob("*.eqdsk"))
    errors: list[float] = []
    rows: list[dict[str, Any]] = []
    for path in files:
        eq = read_geqdsk(path)
        err = compare_eq_axis(eq)
        errors.append(err)
        rows.append({"file": path.name, "axis_error_m": err})
    return {"count": len(errors), "axis_rmse_m": rmse(errors, [0.0] * len(errors)), "rows": rows}


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# SCPN RMSE Dashboard")
    lines.append("")
    lines.append(f"- Generated: `{report['generated_at_utc']}`")
    lines.append(f"- Runtime: `{report['runtime_seconds']:.2f}s`")
    lines.append("")

    itpa = report["confinement_itpa"]
    lines.append("## Confinement RMSE (ITPA H-mode)")
    lines.append("")
    lines.append(f"- Samples: `{itpa['count']}`")
    lines.append(f"- `tau_E` RMSE: `{itpa['tau_rmse_s']:.4f} s`")
    lines.append(f"- `tau_E` mean absolute relative error: `{itpa['tau_mae_rel_pct']:.2f}%`")
    lines.append(f"- `H98(y,2)` RMSE: `{itpa['h98_rmse']:.4f}`")
    lines.append("")

    lines.append("## Confinement RMSE (ITER + SPARC references)")
    lines.append("")
    itsp = report["confinement_iter_sparc"]
    lines.append(f"- Samples: `{itsp['count']}`")
    lines.append(f"- `tau_E` RMSE: `{itsp['tau_rmse_s']:.4f} s`")
    lines.append("")

    beta = report["beta_iter_sparc"]
    lines.append("## Beta_N RMSE (ITER + SPARC references)")
    lines.append("")
    lines.append(f"- Samples: `{beta['count']}`")
    lines.append(f"- `beta_N` RMSE: `{beta['beta_n_rmse']:.4f}`")
    lines.append("")

    sparc_axis = report["sparc_axis"]
    lines.append("## SPARC GEQDSK Axis Error")
    lines.append("")
    lines.append(f"- Files: `{sparc_axis['count']}`")
    lines.append(f"- Axis RMSE: `{sparc_axis['axis_rmse_m']:.6f} m`")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- `beta_N` estimates are derived from `FusionBurnPhysics` thermal-energy output and should be treated as Phase-1 surrogate metrics."
    )
    lines.append("- Use this report for trend tracking; not as a replacement for full transport/MHD validation.")
    lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SCPN RMSE validation dashboard.")
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "rmse_dashboard.json"),
        help="Path to write JSON report.",
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "rmse_dashboard.md"),
        help="Path to write Markdown report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()

    validation_dir = ROOT / "validation"
    reference_dir = validation_dir / "reference_data"
    itpa_csv = reference_dir / "itpa" / "hmode_confinement.csv"
    sparc_eq_dir = reference_dir / "sparc"

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "confinement_itpa": confinement_rmse_itpa(itpa_csv),
        "confinement_iter_sparc": confinement_rmse_iter_sparc(reference_dir),
        "beta_iter_sparc": beta_rmse_iter_sparc(reference_dir, validation_dir),
        "sparc_axis": sparc_axis_rmse(sparc_eq_dir),
    }
    report["runtime_seconds"] = time.perf_counter() - t0

    json_path = Path(args.output_json)
    md_path = Path(args.output_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(render_markdown(report))

    print("SCPN RMSE dashboard generated.")
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")
    print(
        "Summary -> "
        f"ITPA tau_E RMSE={report['confinement_itpa']['tau_rmse_s']:.4f}s, "
        f"beta_N RMSE={report['beta_iter_sparc']['beta_n_rmse']:.4f}, "
        f"SPARC axis RMSE={report['sparc_axis']['axis_rmse_m']:.6f}m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
