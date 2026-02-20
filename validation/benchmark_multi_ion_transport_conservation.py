#!/usr/bin/env python3
# ----------------------------------------------------------------------
# SCPN Fusion Core -- Multi-Ion Transport Conservation Benchmark
# ----------------------------------------------------------------------
"""Contract benchmark for multi-ion D/T/He-ash transport conservation."""

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


CONTRACT_THRESHOLDS = {
    "max_quasineutral_residual": 1e-10,
    "max_late_energy_error_p95": 2.0,
    "min_he_ash_peak": 1e-4,
}


def _render_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def run_benchmark(
    *,
    config_path: str | Path | None = None,
    steps: int = 30,
    dt_s: float = 0.1,
    p_aux_mw: float = 30.0,
) -> dict[str, Any]:
    cfg_path = Path(config_path) if config_path is not None else ROOT / "iter_config.json"
    solver = TransportSolver(str(cfg_path), multi_ion=True)
    solver.Ti = 5.0 * (1.0 - solver.rho**2)
    solver.Te = solver.Ti.copy()
    solver.ne = 5.0 * (1.0 - solver.rho**2) ** 0.5
    solver.n_D = 0.5 * solver.ne.copy()
    solver.n_T = 0.5 * solver.ne.copy()
    solver.n_He = np.zeros(solver.nr, dtype=np.float64)

    p_aux = float(p_aux_mw)
    dt = float(dt_s)
    if steps < 4:
        raise ValueError("steps must be >= 4")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be finite and > 0")
    if not np.isfinite(p_aux) or p_aux <= 0.0:
        raise ValueError("p_aux_mw must be finite and > 0")

    solver.update_transport_model(p_aux)
    energy_errors: list[float] = []
    for _ in range(steps):
        solver.update_transport_model(p_aux)
        solver.evolve_profiles(dt=dt, P_aux=p_aux, enforce_conservation=False)
        energy_errors.append(float(solver._last_conservation_error))

    late_errors = np.asarray(energy_errors[steps // 2 :], dtype=np.float64)
    late_energy_error_p95 = float(np.percentile(late_errors, 95))
    late_energy_error_mean = float(np.mean(late_errors))
    late_energy_pass = bool(
        np.all(np.isfinite(late_errors))
        and late_energy_error_p95 <= CONTRACT_THRESHOLDS["max_late_energy_error_p95"]
    )

    z_w = 10.0
    expected_ne = (
        np.asarray(solver.n_D, dtype=np.float64)
        + np.asarray(solver.n_T, dtype=np.float64)
        + 2.0 * np.asarray(solver.n_He, dtype=np.float64)
        + z_w * np.maximum(np.asarray(solver.n_impurity, dtype=np.float64), 0.0)
    )
    expected_ne = np.maximum(expected_ne, 0.1)
    quasineutral_residual = float(
        np.max(np.abs(np.asarray(solver.ne, dtype=np.float64) - expected_ne))
    )
    quasineutral_pass = bool(
        quasineutral_residual <= CONTRACT_THRESHOLDS["max_quasineutral_residual"]
    )

    finite_pass = bool(
        np.all(np.isfinite(np.asarray(solver.Ti, dtype=np.float64)))
        and np.all(np.isfinite(np.asarray(solver.Te, dtype=np.float64)))
        and np.all(np.isfinite(np.asarray(solver.ne, dtype=np.float64)))
        and np.all(np.isfinite(np.asarray(solver.n_D, dtype=np.float64)))
        and np.all(np.isfinite(np.asarray(solver.n_T, dtype=np.float64)))
        and np.all(np.isfinite(np.asarray(solver.n_He, dtype=np.float64)))
    )
    positivity_pass = bool(
        float(np.min(np.asarray(solver.n_D, dtype=np.float64))) >= 0.0
        and float(np.min(np.asarray(solver.n_T, dtype=np.float64))) >= 0.0
        and float(np.min(np.asarray(solver.n_He, dtype=np.float64))) >= 0.0
    )
    he_ash_peak = float(np.max(np.asarray(solver.n_He, dtype=np.float64)))
    he_ash_pass = bool(he_ash_peak >= CONTRACT_THRESHOLDS["min_he_ash_peak"])

    passes = bool(
        finite_pass
        and positivity_pass
        and quasineutral_pass
        and late_energy_pass
        and he_ash_pass
    )

    return {
        "multi_ion_transport_conservation_benchmark": {
            "config_path": _render_path(cfg_path),
            "steps": int(steps),
            "dt_s": dt,
            "p_aux_mw": p_aux,
            "finite_pass": finite_pass,
            "positivity_pass": positivity_pass,
            "quasineutral_pass": quasineutral_pass,
            "late_energy_pass": late_energy_pass,
            "he_ash_pass": he_ash_pass,
            "passes_thresholds": passes,
            "thresholds": dict(CONTRACT_THRESHOLDS),
            "metrics": {
                "quasineutral_residual": quasineutral_residual,
                "late_energy_error_mean": late_energy_error_mean,
                "late_energy_error_p95": late_energy_error_p95,
                "he_ash_peak_1e19m3": he_ash_peak,
                "n_D_min": float(np.min(np.asarray(solver.n_D, dtype=np.float64))),
                "n_T_min": float(np.min(np.asarray(solver.n_T, dtype=np.float64))),
                "n_He_min": float(np.min(np.asarray(solver.n_He, dtype=np.float64))),
                "z_eff_final": float(solver._Z_eff),
            },
        }
    }


def render_markdown(report: dict[str, Any]) -> str:
    g = report["multi_ion_transport_conservation_benchmark"]
    m = g["metrics"]
    t = g["thresholds"]
    lines = [
        "# Multi-Ion Transport Conservation Benchmark",
        "",
        f"- Config: `{g['config_path']}`",
        f"- Steps: `{g['steps']}` at dt=`{g['dt_s']:.3f}` s, P_aux=`{g['p_aux_mw']:.1f}` MW",
        f"- Finite pass: `{'YES' if g['finite_pass'] else 'NO'}`",
        f"- Positivity pass: `{'YES' if g['positivity_pass'] else 'NO'}`",
        f"- Quasi-neutrality pass: `{'YES' if g['quasineutral_pass'] else 'NO'}`",
        f"- Late-energy pass: `{'YES' if g['late_energy_pass'] else 'NO'}`",
        f"- He-ash growth pass: `{'YES' if g['he_ash_pass'] else 'NO'}`",
        f"- Overall pass: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        "",
        "| Metric | Value | Threshold |",
        "|--------|-------|-----------|",
        (
            f"| quasineutral_residual | {float(m['quasineutral_residual']):.3e} | "
            f"<= {float(t['max_quasineutral_residual']):.1e} |"
        ),
        (
            f"| late_energy_error_p95 | {float(m['late_energy_error_p95']):.4f} | "
            f"<= {float(t['max_late_energy_error_p95']):.2f} |"
        ),
        (
            f"| he_ash_peak_1e19m3 | {float(m['he_ash_peak_1e19m3']):.6f} | "
            f">= {float(t['min_he_ash_peak']):.1e} |"
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "iter_config.json"))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--dt-s", type=float, default=0.1)
    parser.add_argument("--p-aux-mw", type=float, default=30.0)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "multi_ion_transport_conservation_benchmark.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "multi_ion_transport_conservation_benchmark.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report = run_benchmark(
        config_path=Path(args.config),
        steps=args.steps,
        dt_s=args.dt_s,
        p_aux_mw=args.p_aux_mw,
    )
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = report["multi_ion_transport_conservation_benchmark"]
    print("Multi-ion transport conservation benchmark complete.")
    print(
        "quasineutral={q}, late_energy={e}, he_ash={h}, pass={p}".format(
            q=g["quasineutral_pass"],
            e=g["late_energy_pass"],
            h=g["he_ash_pass"],
            p=g["passes_thresholds"],
        )
    )
    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
