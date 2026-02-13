# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GDEP-03 Blind Validation Dashboard
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""GDEP-03: deterministic blind validation on EU-DEMO/K-DEMO synthetic holdout."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BLIND_REFERENCE_DIR = ROOT / "validation" / "reference_data" / "blind"
BLIND_REFERENCE_FILES = ("eu_demo_reference.json", "k_demo_reference.json")


def _load_rmse_dashboard_module() -> Any:
    module_path = ROOT / "validation" / "rmse_dashboard.py"
    spec = importlib.util.spec_from_file_location("rmse_dashboard", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load RMSE dashboard module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_RMSE_DASHBOARD = _load_rmse_dashboard_module()
ipb98_tau_e = _RMSE_DASHBOARD.ipb98_tau_e
rmse = _RMSE_DASHBOARD.rmse


def load_blind_references(reference_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for filename in BLIND_REFERENCE_FILES:
        path = reference_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing blind reference file: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        machine = str(payload["machine"])
        for shot in payload.get("shots", []):
            row = dict(shot)
            row["machine"] = machine
            rows.append(row)
    if not rows:
        raise ValueError("No blind reference rows loaded.")
    return rows


def estimate_beta_n_proxy(row: dict[str, Any], tau_pred_s: float) -> float:
    return (
        10.0
        * 0.18
        * float(row["n_e_1e19"])
        * (tau_pred_s / 5.0)
        * (float(row["P_loss_MW"]) / 100.0) ** 0.25
        / (float(row["B_t_T"]) ** 1.35)
        * (float(row["R_m"]) / 6.0) ** 0.3
        * (float(row["I_p_MA"]) / 15.0) ** 0.2
    )


def estimate_core_edge_match_proxy(tau_pred_s: float, beta_pred: float) -> float:
    raw = (
        0.90
        + 0.04 * math.tanh((tau_pred_s - 3.5) / 2.0)
        + 0.03 * math.tanh((beta_pred - 1.6) / 0.8)
    )
    return min(0.995, max(0.82, raw))


def _mean_abs_relative_pct(y_true: list[float], y_pred: list[float]) -> float:
    rel = [
        abs(t - p) / max(abs(t), 1e-9) * 100.0
        for t, p in zip(y_true, y_pred)
    ]
    return float(statistics.mean(rel))


def _evaluate_rows(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Evaluation requires non-empty rows.")

    tau_true: list[float] = []
    tau_pred: list[float] = []
    beta_true: list[float] = []
    beta_pred: list[float] = []
    core_true: list[float] = []
    core_pred: list[float] = []
    shot_rows: list[dict[str, Any]] = []

    for row in rows:
        epsilon = float(row["a_m"]) / float(row["R_m"])
        tau_p = ipb98_tau_e(
            ip_ma=float(row["I_p_MA"]),
            b_t=float(row["B_t_T"]),
            n_e19=float(row["n_e_1e19"]),
            p_loss_mw=float(row["P_loss_MW"]),
            r_m=float(row["R_m"]),
            kappa=float(row["kappa"]),
            epsilon=epsilon,
            a_eff_amu=float(row["A_eff_amu"]),
        )
        beta_p = estimate_beta_n_proxy(row, tau_p)
        core_p = estimate_core_edge_match_proxy(tau_p, beta_p)

        tau_m = float(row["tau_E_s"])
        beta_m = float(row["beta_N"])
        core_m = float(row["core_edge_match"])

        tau_true.append(tau_m)
        tau_pred.append(tau_p)
        beta_true.append(beta_m)
        beta_pred.append(beta_p)
        core_true.append(core_m)
        core_pred.append(core_p)

        shot_rows.append(
            {
                "machine": str(row["machine"]),
                "shot": str(row["shot"]),
                "tau_measured_s": tau_m,
                "tau_pred_s": tau_p,
                "beta_n_measured": beta_m,
                "beta_n_pred": beta_p,
                "core_edge_measured": core_m,
                "core_edge_pred": core_p,
            }
        )

    tau_rmse_s = float(rmse(tau_true, tau_pred))
    beta_rmse = float(rmse(beta_true, beta_pred))
    core_edge_rmse = float(rmse(core_true, core_pred))
    tau_mae_rel_pct = _mean_abs_relative_pct(tau_true, tau_pred)
    beta_mae_rel_pct = _mean_abs_relative_pct(beta_true, beta_pred)
    core_edge_mae_pct = float(
        statistics.mean(abs(t - p) for t, p in zip(core_true, core_pred)) * 100.0
    )
    parity_pct = max(
        0.0,
        100.0 - statistics.mean([tau_mae_rel_pct, beta_mae_rel_pct, core_edge_mae_pct]),
    )

    passes = bool(
        tau_rmse_s <= thresholds["max_tau_rmse_s"]
        and beta_rmse <= thresholds["max_beta_rmse"]
        and core_edge_rmse <= thresholds["max_core_edge_rmse"]
        and parity_pct >= thresholds["min_parity_pct"]
    )

    return {
        "count": len(rows),
        "tau_rmse_s": tau_rmse_s,
        "beta_rmse": beta_rmse,
        "core_edge_rmse": core_edge_rmse,
        "tau_mae_rel_pct": tau_mae_rel_pct,
        "beta_mae_rel_pct": beta_mae_rel_pct,
        "core_edge_mae_pct": core_edge_mae_pct,
        "parity_pct": parity_pct,
        "passes_thresholds": passes,
        "rows": shot_rows,
    }


def run_campaign(*, reference_dir: Path | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    ref_dir = reference_dir or BLIND_REFERENCE_DIR
    rows = load_blind_references(ref_dir)

    thresholds = {
        "max_tau_rmse_s": 0.35,
        "max_beta_rmse": 0.15,
        "max_core_edge_rmse": 0.02,
        "min_parity_pct": 95.0,
    }

    by_machine: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_machine.setdefault(str(row["machine"]), []).append(row)

    machines: list[dict[str, Any]] = []
    for machine in sorted(by_machine):
        metrics = _evaluate_rows(by_machine[machine], thresholds)
        metrics["machine"] = machine
        machines.append(metrics)

    aggregate = _evaluate_rows(rows, thresholds)
    passes = bool(aggregate["passes_thresholds"] and all(m["passes_thresholds"] for m in machines))

    return {
        "reference_dir": str(ref_dir),
        "sample_count": len(rows),
        "thresholds": thresholds,
        "aggregate": aggregate,
        "machines": machines,
        "passes_thresholds": passes,
        "runtime_seconds": float(time.perf_counter() - t0),
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gdep_03": run_campaign(**kwargs),
    }


def render_markdown(report: dict[str, Any]) -> str:
    g = report["gdep_03"]
    th = g["thresholds"]
    agg = g["aggregate"]

    lines = [
        "# GDEP-03 Blind Validation Dashboard",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{g['runtime_seconds']:.3f} s`",
        f"- Samples: `{g['sample_count']}`",
        "",
        "## Thresholds",
        "",
        f"- `tau_E` RMSE <= `{th['max_tau_rmse_s']:.3f} s`",
        f"- `beta_N` RMSE <= `{th['max_beta_rmse']:.3f}`",
        f"- Core-edge RMSE <= `{th['max_core_edge_rmse']:.3f}`",
        f"- Aggregate parity >= `{th['min_parity_pct']:.1f}%`",
        "",
        "## Aggregate Metrics",
        "",
        f"- `tau_E` RMSE: `{agg['tau_rmse_s']:.6f} s`",
        f"- `beta_N` RMSE: `{agg['beta_rmse']:.6f}`",
        f"- Core-edge RMSE: `{agg['core_edge_rmse']:.6f}`",
        f"- Parity score: `{agg['parity_pct']:.2f}%`",
        f"- Pass: `{'YES' if agg['passes_thresholds'] else 'NO'}`",
        "",
    ]

    for machine in g["machines"]:
        lines.extend(
            [
                f"## {machine['machine']}",
                "",
                f"- Samples: `{machine['count']}`",
                f"- `tau_E` RMSE: `{machine['tau_rmse_s']:.6f} s`",
                f"- `beta_N` RMSE: `{machine['beta_rmse']:.6f}`",
                f"- Core-edge RMSE: `{machine['core_edge_rmse']:.6f}`",
                f"- Parity score: `{machine['parity_pct']:.2f}%`",
                f"- Pass: `{'YES' if machine['passes_thresholds'] else 'NO'}`",
                "",
            ]
        )

    lines.append(f"- Overall pass: `{'YES' if g['passes_thresholds'] else 'NO'}`")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-dir",
        default=str(BLIND_REFERENCE_DIR),
        help="Directory containing blind reference JSON files.",
    )
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "gdep_03_blind_validation.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "gdep_03_blind_validation.md"),
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = generate_report(reference_dir=Path(args.reference_dir))

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = report["gdep_03"]
    agg = g["aggregate"]
    print("GDEP-03 blind validation complete.")
    print(f"passes_thresholds={g['passes_thresholds']}")
    print(
        "Summary -> "
        f"tau_rmse={agg['tau_rmse_s']:.6f}s, "
        f"beta_rmse={agg['beta_rmse']:.6f}, "
        f"core_edge_rmse={agg['core_edge_rmse']:.6f}, "
        f"parity={agg['parity_pct']:.2f}%"
    )

    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
