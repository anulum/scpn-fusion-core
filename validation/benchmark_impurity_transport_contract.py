"""Benchmark trace impurity transport contracts.

This validates Aurora/STRAHL-style transport invariants available in the native
trace impurity surface: positivity, edge-source particle conservation,
neoclassical inward pinch sign, and monotonic radiated power. It does not claim
collisional-operator parity with Aurora, STRAHL, or JINTRAC.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scpn_fusion.core.impurity_transport import (  # noqa: E402
    ImpuritySpecies,
    ImpurityTransportSolver,
    build_aurora_strahl_charge_state_artifact,
    neoclassical_impurity_pinch,
    total_radiated_power,
)

REPORT_DIR = ROOT / "validation" / "reports"
JSON_REPORT = REPORT_DIR / "impurity_transport_contract_benchmark.json"
MD_REPORT = REPORT_DIR / "impurity_transport_contract_benchmark.md"


def _inventory(n_z: np.ndarray, rho: np.ndarray, R0: float, a: float) -> float:
    vol_element = 4.0 * np.pi**2 * R0 * a**2 * rho
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(n_z * vol_element, rho))


def run_benchmark() -> dict[str, Any]:
    """Run impurity transport contract checks and return invariant summary."""
    rho = np.linspace(0.0, 1.0, 80)
    R0 = 6.2
    a = 2.0
    dt = 0.2
    source_rate = 1.0e16
    ne = 1.0e20 * (1.0 - 0.2 * rho**2)
    Te = 1500.0 * (1.0 - 0.3 * rho**2)
    Ti = 5000.0 * (1.0 - 0.6 * rho**2)
    q = 1.0 + rho
    eps = 0.2 + 0.2 * rho

    solver = ImpurityTransportSolver(
        rho,
        R0,
        a,
        [ImpuritySpecies("W", 74, 183.8, source_rate=source_rate)],
    )
    pinch = neoclassical_impurity_pinch(74, ne, Te, Ti, q, rho, R0, a, eps)
    result = solver.step(dt, ne, Te, Ti, D_anom=0.0, V_pinch={"W": np.zeros_like(rho)})
    n_w = np.asarray(result["W"], dtype=float)

    expected_particles = source_rate * (4.0 * np.pi**2 * R0 * a) * dt
    actual_particles = _inventory(n_w, rho, R0, a)
    conservation_error = abs(actual_particles - expected_particles) / expected_particles

    low_rad = total_radiated_power(ne, {"W": ne * 1.0e-5}, Te, rho, R0, a)
    high_rad = total_radiated_power(ne, {"W": ne * 1.0e-4}, Te, rho, R0, a)
    radius_m = rho * a
    time_s = np.array([0.0, 1.0e-5, 2.0e-5], dtype=float)
    charge_states = np.array([0, 1, 2, 3], dtype=float)
    ne_t_r = np.tile(ne, (time_s.size, 1))
    Te_t_r = np.tile(Te, (time_s.size, 1))
    density_r_z = np.zeros((rho.size, charge_states.size), dtype=float)
    density_r_z[:, 1] = 1.0e15 * (1.0 - 0.1 * rho)
    cr_artifact = build_aurora_strahl_charge_state_artifact(
        element="Ar",
        charge_states=charge_states,
        radius_m=radius_m,
        time_s=time_s,
        ne_t_r=ne_t_r,
        Te_t_r=Te_t_r,
        initial_charge_state_density_rz=density_r_z,
        major_radius_m=R0,
    )
    cr_payload = cr_artifact.to_dict()
    charge_density = np.asarray(cr_payload["observables"]["charge_state_density_r_t"])
    total_density = np.asarray(cr_payload["observables"]["total_impurity_density_r_t"])

    invariants = {
        "positivity": bool(np.all(n_w >= 0.0) and np.all(np.isfinite(n_w))),
        "edge_source_conservation": bool(conservation_error <= 2.0e-2),
        "inward_pinch_midradius": bool(pinch[len(rho) // 2] < 0.0),
        "radiation_monotonicity": bool(high_rad > low_rad > 0.0),
        "charge_state_artifact_contract": bool(
            cr_payload["schema"] == "aurora-strahl-charge-state-artifact.v1"
            and charge_density.shape == (time_s.size, rho.size, charge_states.size)
            and total_density.shape == (time_s.size, rho.size)
            and np.all(np.isfinite(charge_density))
        ),
        "charge_state_density_closure": bool(
            np.allclose(total_density, np.sum(charge_density, axis=2), rtol=1.0e-13)
        ),
        "charge_state_particle_conservation": bool(
            cr_artifact.conservation["relative_inventory_error"] <= 1.0e-12
        ),
    }

    return {
        "benchmark": "impurity_transport_contract",
        "description": "Trace impurity transport contract; not Aurora/STRAHL/JINTRAC collisional-operator parity.",
        "metrics": {
            "actual_particles": actual_particles,
            "expected_particles": expected_particles,
            "relative_conservation_error": conservation_error,
            "midradius_pinch_m_s": float(pinch[len(rho) // 2]),
            "low_radiated_power_mw": low_rad,
            "high_radiated_power_mw": high_rad,
            "edge_density_m3": float(n_w[-1]),
            "charge_state_inventory_error": cr_artifact.conservation["relative_inventory_error"],
            "charge_state_count": int(charge_states.size),
        },
        "thresholds": {
            "max_relative_conservation_error": 2.0e-2,
            "max_charge_state_inventory_error": 1.0e-12,
        },
        "artifact_contract": {
            "schema": cr_payload["schema"],
            "coordinates": list(cr_payload["coordinates"].keys()),
            "observables": list(cr_payload["observables"].keys()),
            "parity_status": cr_payload["provenance"]["parity_status"],
        },
        "invariants": invariants,
        "passed": all(invariants.values()),
    }


def write_reports(results: dict[str, Any]) -> None:
    """Write JSON and markdown artifacts for impurity transport benchmark."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metrics = results["metrics"]
    thresholds = results["thresholds"]
    invariants = results["invariants"]
    lines = [
        "# Impurity Transport Contract Benchmark",
        "",
        "This benchmark validates native trace impurity transport contracts.",
        "It does not claim Aurora/STRAHL/JINTRAC collisional-operator parity.",
        "",
        "## Metrics",
        "",
        f"- Actual particles: {metrics['actual_particles']:.6e}",
        f"- Expected particles: {metrics['expected_particles']:.6e}",
        (
            "- Relative conservation error: "
            f"{metrics['relative_conservation_error']:.6e} "
            f"(threshold {thresholds['max_relative_conservation_error']:.2e})"
        ),
        f"- Midradius pinch: {metrics['midradius_pinch_m_s']:.6e} m/s",
        f"- Low radiated power: {metrics['low_radiated_power_mw']:.6e} MW",
        f"- High radiated power: {metrics['high_radiated_power_mw']:.6e} MW",
        f"- Edge density: {metrics['edge_density_m3']:.6e} m^-3",
        f"- Charge-state count: {metrics['charge_state_count']}",
        (
            "- Charge-state inventory error: "
            f"{metrics['charge_state_inventory_error']:.6e} "
            f"(threshold {thresholds['max_charge_state_inventory_error']:.2e})"
        ),
        "",
        "## Aurora/STRAHL-style artifact contract",
        "",
        f"- Schema: `{results['artifact_contract']['schema']}`",
        f"- Coordinates: {', '.join(results['artifact_contract']['coordinates'])}",
        f"- Observables: {', '.join(results['artifact_contract']['observables'])}",
        f"- Parity status: `{results['artifact_contract']['parity_status']}`",
        "",
        "## Invariants",
        "",
    ]
    for name, passed in invariants.items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend(["", f"Overall: {'PASS' if results['passed'] else 'FAIL'}", ""])
    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Execute impurity transport contract benchmark and print JSON report."""
    results = run_benchmark()
    write_reports(results)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
