# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — EU-DEMO/K-DEMO Synthetic Holdout Validation
"""Validate confinement proxies against bundled EU-DEMO/K-DEMO synthetic holdouts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLIND_REFERENCE_DIR = ROOT / "validation" / "reference_data" / "blind"
BLIND_REFERENCE_FILES = ("eu_demo_reference.json", "k_demo_reference.json")
REPORT_SCHEMA_VERSION = 2
REPORT_KIND = "eu_demo_k_demo_synthetic_holdout_validation"
REPORT_PAYLOAD_KEY = "eu_demo_k_demo_synthetic_holdout_validation"


class ConfinementTimeModel(Protocol):
    """Callable contract for the repository confinement-time implementation."""

    def __call__(
        self,
        *,
        ip_ma: float,
        b_t: float,
        n_e19: float,
        p_loss_mw: float,
        r_m: float,
        kappa: float,
        epsilon: float,
        a_eff_amu: float,
    ) -> float:
        """Return a predicted confinement time in seconds."""
        ...


def load_rmse_dashboard_functions(
    module_path: Path,
) -> tuple[ConfinementTimeModel, Callable[[list[float], list[float]], float]]:
    """Load typed confinement and RMSE callables from a dashboard module path."""
    spec = importlib.util.spec_from_file_location("rmse_dashboard", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load RMSE dashboard module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    confinement_model = cast(ConfinementTimeModel, module.ipb98_tau_e)
    metric = cast(Callable[[list[float], list[float]], float], module.rmse)
    return confinement_model, metric


ipb98_tau_e, rmse = load_rmse_dashboard_functions(ROOT / "validation" / "rmse_dashboard.py")


def load_blind_references(reference_dir: Path) -> list[dict[str, Any]]:
    """Load all blind reference rows from the configured JSON fixture directory."""
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
    """Estimate βN from a deterministic proxy model for blind-validation diagnostics."""
    return float(
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
    """Compute a stable core-edge match proxy from predicted confinement and βN."""
    return (
        0.90
        + 0.04 * math.tanh((tau_pred_s - 3.5) / 2.0)
        + 0.03 * math.tanh((beta_pred - 1.6) / 0.8)
    )


def _mean_abs_relative_pct(y_true: list[float], y_pred: list[float]) -> float:
    rel = [abs(t - p) / max(abs(t), 1e-9) * 100.0 for t, p in zip(y_true, y_pred)]
    return float(statistics.mean(rel))


def _evaluate_rows(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
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


def _nonnegative_finite(value: float, *, name: str) -> float:
    checked = float(value)
    if not np.isfinite(checked) or checked < 0.0:
        raise ValueError(f"{name} must be finite and >= 0.")
    return checked


def _percentage(value: float, *, name: str) -> float:
    checked = float(value)
    if not np.isfinite(checked) or checked < 0.0 or checked > 100.0:
        raise ValueError(f"{name} must be finite and in [0, 100].")
    return checked


def run_campaign(
    *,
    reference_dir: Path | None = None,
    max_tau_rmse_s: float = 0.35,
    max_beta_rmse: float = 0.15,
    max_core_edge_rmse: float = 0.02,
    min_parity_pct: float = 95.0,
) -> dict[str, Any]:
    """Evaluate bundled synthetic holdouts against configured acceptance gates.

    Parameters
    ----------
    reference_dir : pathlib.Path or None
        Directory containing the EU-DEMO and K-DEMO synthetic reference files.
    max_tau_rmse_s : float
        Maximum accepted confinement-time root mean square error in seconds.
    max_beta_rmse : float
        Maximum accepted normalised-beta root mean square error.
    max_core_edge_rmse : float
        Maximum accepted core-edge proxy root mean square error.
    min_parity_pct : float
        Minimum accepted aggregate parity score in percent.

    Returns
    -------
    dict[str, Any]
        Per-machine and aggregate errors, thresholds, provenance path and gate
        status.

    Raises
    ------
    ValueError
        If an acceptance threshold is invalid or the reference rows are empty.
    FileNotFoundError
        If either required synthetic reference file is absent.
    """
    t0 = time.perf_counter()
    ref_dir = reference_dir or BLIND_REFERENCE_DIR
    rows = load_blind_references(ref_dir)

    thresholds = {
        "max_tau_rmse_s": _nonnegative_finite(max_tau_rmse_s, name="max_tau_rmse_s"),
        "max_beta_rmse": _nonnegative_finite(max_beta_rmse, name="max_beta_rmse"),
        "max_core_edge_rmse": _nonnegative_finite(
            max_core_edge_rmse,
            name="max_core_edge_rmse",
        ),
        "min_parity_pct": _percentage(min_parity_pct, name="min_parity_pct"),
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
    """Generate the versioned EU-DEMO/K-DEMO synthetic holdout report."""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_kind": REPORT_KIND,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        REPORT_PAYLOAD_KEY: run_campaign(**kwargs),
    }


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    """Validate the current serialized report contract and return its payload."""
    expected_keys = {
        "schema_version",
        "report_kind",
        "generated_at_utc",
        REPORT_PAYLOAD_KEY,
    }
    if set(report) != expected_keys:
        raise ValueError("report keys do not match the current descriptive contract")
    if report["schema_version"] != REPORT_SCHEMA_VERSION:
        raise ValueError(f"unsupported report schema_version: {report['schema_version']!r}")
    if report["report_kind"] != REPORT_KIND:
        raise ValueError(f"unsupported report_kind: {report['report_kind']!r}")
    generated_at = report["generated_at_utc"]
    if not isinstance(generated_at, str) or not generated_at:
        raise ValueError("generated_at_utc must be a non-empty string")
    payload = report[REPORT_PAYLOAD_KEY]
    if not isinstance(payload, dict):
        raise ValueError(f"{REPORT_PAYLOAD_KEY} must be an object")
    return payload


def render_markdown(report: dict[str, Any]) -> str:
    """Render the EU-DEMO/K-DEMO synthetic holdout report as Markdown."""
    g = validate_report(report)
    th = g["thresholds"]
    agg = g["aggregate"]

    lines = [
        "# EU-DEMO/K-DEMO Synthetic Holdout Validation",
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
    """Parse command-line arguments for synthetic holdout validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-dir",
        default=str(BLIND_REFERENCE_DIR),
        help="Directory containing blind reference JSON files.",
    )
    parser.add_argument("--max-tau-rmse-s", type=float, default=0.35)
    parser.add_argument("--max-beta-rmse", type=float, default=0.15)
    parser.add_argument("--max-core-edge-rmse", type=float, default=0.02)
    parser.add_argument("--min-parity-pct", type=float, default=95.0)
    parser.add_argument(
        "--output-json",
        default=str(
            ROOT / "validation" / "reports" / "eu_demo_k_demo_synthetic_holdout_validation.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=str(
            ROOT / "validation" / "reports" / "eu_demo_k_demo_synthetic_holdout_validation.md"
        ),
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run synthetic holdout validation and write JSON and Markdown reports."""
    args = parse_args(argv)
    report = generate_report(
        reference_dir=Path(args.reference_dir),
        max_tau_rmse_s=args.max_tau_rmse_s,
        max_beta_rmse=args.max_beta_rmse,
        max_core_edge_rmse=args.max_core_edge_rmse,
        min_parity_pct=args.min_parity_pct,
    )

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = validate_report(report)
    agg = g["aggregate"]
    print("EU-DEMO/K-DEMO synthetic holdout validation complete.")
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
