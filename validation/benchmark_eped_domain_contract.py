#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — EPED Domain Contract Benchmark
# Verifies domain metadata + bounded extrapolation penalties.
# ──────────────────────────────────────────────────────────────────────
"""Benchmark domain-validity contracts for the EPED-like pedestal surrogate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from scpn_fusion.core.eped_pedestal import EpedPedestalModel


def _run_case(
    *,
    case_id: str,
    label: str,
    model_kwargs: dict[str, float],
    n_ped_1e19: float,
    t_guess_keV: float,
) -> dict[str, Any]:
    model = EpedPedestalModel(**model_kwargs)
    res = model.predict(
        n_ped_1e19=n_ped_1e19,
        T_ped_guess_keV=t_guess_keV,
        domain_mode="ignore",
    )
    return {
        "case_id": case_id,
        "label": label,
        "n_ped_1e19": float(n_ped_1e19),
        "T_ped_guess_keV": float(t_guess_keV),
        "in_domain": bool(res.in_domain),
        "extrapolation_score": float(res.extrapolation_score),
        "extrapolation_penalty": float(res.extrapolation_penalty),
        "domain_violations": list(res.domain_violations),
        "Delta_ped": float(res.Delta_ped),
        "T_ped_keV": float(res.T_ped_keV),
        "p_ped_kPa": float(res.p_ped_kPa),
    }


def run_benchmark() -> dict[str, Any]:
    base_model = {
        "R0": 6.2,
        "a": 2.0,
        "B0": 5.3,
        "Ip_MA": 15.0,
        "kappa": 1.75,
        "A_ion": 2.5,
        "Z_eff": 1.7,
    }
    compact_model = {
        "R0": 1.9,
        "a": 0.62,
        "B0": 9.1,
        "Ip_MA": 10.5,
        "kappa": 1.8,
        "A_ion": 2.2,
        "Z_eff": 1.9,
    }
    out_of_domain_machine = dict(base_model)
    out_of_domain_machine["kappa"] = 2.8

    cases = [
        _run_case(
            case_id="in_ref",
            label="ITER-like in-domain reference",
            model_kwargs=base_model,
            n_ped_1e19=8.0,
            t_guess_keV=3.0,
        ),
        _run_case(
            case_id="in_compact",
            label="compact high-field in-domain",
            model_kwargs=compact_model,
            n_ped_1e19=10.0,
            t_guess_keV=4.0,
        ),
        _run_case(
            case_id="out_density",
            label="density extrapolation",
            model_kwargs=base_model,
            n_ped_1e19=22.0,
            t_guess_keV=3.0,
        ),
        _run_case(
            case_id="out_temperature",
            label="temperature extrapolation",
            model_kwargs=base_model,
            n_ped_1e19=9.0,
            t_guess_keV=12.0,
        ),
        _run_case(
            case_id="out_shape",
            label="shape extrapolation (kappa)",
            model_kwargs=out_of_domain_machine,
            n_ped_1e19=8.0,
            t_guess_keV=3.0,
        ),
    ]

    in_cases = [case for case in cases if case["case_id"].startswith("in_")]
    out_cases = [case for case in cases if case["case_id"].startswith("out_")]

    in_domain_pass = bool(
        all(case["in_domain"] for case in in_cases)
        and all(case["extrapolation_penalty"] == 1.0 for case in in_cases)
        and all(not case["domain_violations"] for case in in_cases)
    )
    out_domain_flag_pass = bool(
        all(not case["in_domain"] for case in out_cases)
        and all(case["domain_violations"] for case in out_cases)
    )
    penalty_bounds_pass = bool(
        all(0.65 <= case["extrapolation_penalty"] <= 1.0 for case in out_cases)
        and all(case["extrapolation_penalty"] < 1.0 for case in out_cases)
    )
    passes = bool(in_domain_pass and out_domain_flag_pass and penalty_bounds_pass)

    return {
        "eped_domain_contract_benchmark": {
            "n_cases": int(len(cases)),
            "in_domain_case_count": int(len(in_cases)),
            "out_of_domain_case_count": int(len(out_cases)),
            "in_domain_pass": in_domain_pass,
            "out_domain_flag_pass": out_domain_flag_pass,
            "penalty_bounds_pass": penalty_bounds_pass,
            "passes_thresholds": passes,
            "domain_metadata": EpedPedestalModel.domain_metadata(),
            "cases": cases,
        }
    }


def render_markdown(report: dict[str, Any]) -> str:
    g = report["eped_domain_contract_benchmark"]
    lines = [
        "# EPED Domain Contract Benchmark",
        "",
        f"- Cases: `{g['n_cases']}` "
        f"(in-domain `{g['in_domain_case_count']}`, out-of-domain `{g['out_of_domain_case_count']}`)",
        f"- In-domain contract pass: `{'YES' if g['in_domain_pass'] else 'NO'}`",
        f"- Out-of-domain flag pass: `{'YES' if g['out_domain_flag_pass'] else 'NO'}`",
        f"- Penalty bounds pass: `{'YES' if g['penalty_bounds_pass'] else 'NO'}`",
        f"- Overall pass: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        "",
        "| Case | In domain | Score | Penalty | Violations | Delta_ped | T_ped [keV] |",
        "|------|-----------|-------|---------|------------|-----------|-------------|",
    ]
    for case in g["cases"]:
        violations = "; ".join(case["domain_violations"]) if case["domain_violations"] else "—"
        lines.append(
            f"| {case['case_id']} | {'YES' if case['in_domain'] else 'NO'} | "
            f"{case['extrapolation_score']:.3f} | {case['extrapolation_penalty']:.3f} | "
            f"{violations} | {case['Delta_ped']:.4f} | {case['T_ped_keV']:.4f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "eped_domain_contract_benchmark.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "eped_domain_contract_benchmark.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report = run_benchmark()
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = report["eped_domain_contract_benchmark"]
    print("EPED domain contract benchmark complete.")
    print(
        "cases={cases}, in_domain_pass={in_pass}, out_domain_flag_pass={out_pass}, "
        "penalty_bounds_pass={pen_pass}, passes={passes}".format(
            cases=g["n_cases"],
            in_pass=g["in_domain_pass"],
            out_pass=g["out_domain_flag_pass"],
            pen_pass=g["penalty_bounds_pass"],
            passes=g["passes_thresholds"],
        )
    )
    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
