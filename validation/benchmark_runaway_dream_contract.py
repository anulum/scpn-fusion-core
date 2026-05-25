"""Benchmark DREAM-style runaway-electron fluid contracts.

This validates scalar density-balance invariants compatible with DREAM fluid
runs: subcritical avalanche suppression, supercritical avalanche growth,
mitigation loss accounting, and density-cap enforcement. It does not claim
parity with DREAM's kinetic momentum-space distribution solver.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scpn_fusion.core.runaway_electrons import (  # noqa: E402
    RunawayEvolution,
    RunawayParams,
    critical_field,
    dream_fluid_density_balance,
)

REPORT_DIR = ROOT / "validation" / "reports"
JSON_REPORT = REPORT_DIR / "runaway_dream_contract_benchmark.json"
MD_REPORT = REPORT_DIR / "runaway_dream_contract_benchmark.md"


def _json_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _case_result(name: str, params: RunawayParams, n_re: float, loss_time_s: float) -> dict[str, object]:
    start = time.perf_counter()
    balance = dream_fluid_density_balance(params, n_re, loss_time_s=loss_time_s)
    wall_time_s = time.perf_counter() - start
    return {
        "case": name,
        "dreicer_source_m3_s": balance.dreicer_source,
        "avalanche_source_m3_s": balance.avalanche_source,
        "loss_source_m3_s": balance.loss_source,
        "total_source_m3_s": balance.total_source,
        "runaway_fraction": balance.runaway_fraction,
        "growth_time_s": _json_float(balance.growth_time_s),
        "wall_time_s": wall_time_s,
    }


def run_benchmark(repeats: int = 25) -> dict[str, object]:
    params_subcritical = RunawayParams(
        ne_20=1.0,
        Te_keV=0.04,
        E_par=0.5 * critical_field(1.0),
        Z_eff=2.0,
        B0=5.0,
        R0=6.0,
    )
    params_supercritical = RunawayParams(
        ne_20=1.0,
        Te_keV=0.04,
        E_par=8.0,
        Z_eff=2.0,
        B0=5.0,
        R0=6.0,
    )

    cases = [
        _case_result("subcritical_no_avalanche", params_subcritical, 1.0e12, np.inf),
        _case_result("supercritical_growth", params_supercritical, 2.0e12, np.inf),
        _case_result("mitigated_loss_accounting", params_supercritical, 2.0e12, 0.2),
    ]

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        dream_fluid_density_balance(params_supercritical, 2.0e12, loss_time_s=0.2)
        timings.append(time.perf_counter() - start)

    evo = RunawayEvolution(params_supercritical)
    capped_density = evo.step(1.0, 9.0e13, 50.0, max_runaway_fraction=1.0e-6)

    invariants = {
        "subcritical_avalanche_zero": bool(cases[0]["avalanche_source_m3_s"] == 0.0),
        "supercritical_avalanche_positive": bool(cases[1]["avalanche_source_m3_s"] > 0.0),
        "loss_accounting_exact": bool(
            np.isclose(
                cases[2]["total_source_m3_s"],
                cases[2]["dreicer_source_m3_s"]
                + cases[2]["avalanche_source_m3_s"]
                - cases[2]["loss_source_m3_s"],
            )
        ),
        "density_cap_enforced": bool(np.isclose(capped_density, 1.0e14)),
    }

    return {
        "benchmark": "runaway_dream_contract",
        "description": "DREAM-style fluid runaway density-balance contract; not kinetic DREAM parity.",
        "cases": cases,
        "timing": {
            "repeats": repeats,
            "median_balance_wall_time_s": median(timings),
            "min_balance_wall_time_s": min(timings),
            "max_balance_wall_time_s": max(timings),
        },
        "invariants": invariants,
        "passed": all(invariants.values()),
    }


def write_reports(results: dict[str, object]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    cases = results["cases"]
    timing = results["timing"]
    invariants = results["invariants"]
    lines = [
        "# Runaway DREAM-Style Fluid Contract Benchmark",
        "",
        "This benchmark validates scalar runaway-density balance contracts compatible with DREAM fluid runs.",
        "It does not claim parity with DREAM's kinetic momentum-space distribution solver.",
        "",
        "## Timing",
        "",
        f"- Repeats: {timing['repeats']}",
        f"- Median balance wall time: {timing['median_balance_wall_time_s']:.6e} s",
        f"- Minimum balance wall time: {timing['min_balance_wall_time_s']:.6e} s",
        f"- Maximum balance wall time: {timing['max_balance_wall_time_s']:.6e} s",
        "",
        "## Cases",
        "",
        "| Case | Dreicer source [m^-3 s^-1] | Avalanche source [m^-3 s^-1] | Loss source [m^-3 s^-1] | Total source [m^-3 s^-1] | Runaway fraction | Growth time [s] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in cases:
        growth_time = case["growth_time_s"]
        growth_time_text = "inf" if growth_time is None else f"{growth_time:.6e}"
        lines.append(
            f"| {case['case']} | {case['dreicer_source_m3_s']:.6e} | "
            f"{case['avalanche_source_m3_s']:.6e} | {case['loss_source_m3_s']:.6e} | "
            f"{case['total_source_m3_s']:.6e} | {case['runaway_fraction']:.6e} | "
            f"{growth_time_text} |"
        )
    lines.extend(["", "## Invariants", ""])
    for name, passed in invariants.items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend(["", f"Overall: {'PASS' if results['passed'] else 'FAIL'}", ""])
    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    results = run_benchmark()
    write_reports(results)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
