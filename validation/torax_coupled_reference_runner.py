#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Coupled TORAX Reference Runner
"""Run the frozen coupled model-intersection deck in the TORAX environment.

This runner belongs in the dedicated ``.venv-torax`` environment.  It emits a
provenance-bound reference artifact for verification by the main project
environment; the native comparison never imports TORAX or its dependencies.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import numpy.typing as npt

ROOT = Path(__file__).resolve().parents[1]
DECK_PATH = (
    ROOT
    / "validation"
    / "reference_data"
    / "torax"
    / "coupled_transport_model_intersection_deck.py"
)
DEFAULT_OUTPUT = (
    ROOT / "validation" / "reference_data" / "torax" / "torax_coupled_transport_reference.json"
)

sys.path.insert(0, str(ROOT))

from validation.reference_data.torax.coupled_transport_model_intersection_deck import (  # noqa: E402
    CONFIG,
    MODEL_INTERSECTION,
)

FloatArray = npt.NDArray[np.float64]


def _checksum(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _interpolate_profiles(
    rho: FloatArray,
    values: FloatArray,
    comparison_rho: FloatArray,
) -> list[list[float]]:
    return [
        np.interp(comparison_rho, rho, np.asarray(snapshot, dtype=np.float64)).tolist()
        for snapshot in values
    ]


def _integral(values: FloatArray, measure: FloatArray, rho: FloatArray) -> float:
    integrand = np.asarray(values, dtype=np.float64) * measure
    trapezoid = getattr(np, "trapezoid", None)
    if callable(trapezoid):
        return float(trapezoid(integrand, rho))
    return float(np.trapz(integrand, rho))


def _cell_integral(values: FloatArray, measure: FloatArray, rho: FloatArray) -> float:
    """Integrate TORAX cell-centered values with its uniform finite-volume mesh."""
    spacing = np.diff(rho)
    if spacing.size == 0 or not np.allclose(spacing, spacing[0], rtol=0.0, atol=1e-12):
        raise ValueError("TORAX source coordinate must be a uniform cell grid")
    return float(np.sum(np.asarray(values, dtype=np.float64) * measure) * spacing[0])


def _source_totals(profiles: Any) -> dict[str, list[float]]:
    geometry = cast(Mapping[str, Any], MODEL_INTERSECTION["geometry"])
    major_radius = float(geometry["major_radius_m"])
    minor_radius = float(geometry["minor_radius_m"])

    def integrate_series(name: str, *, area_measure: bool = False) -> list[float]:
        variable = profiles[name]
        radial_dimension = variable.dims[-1]
        source_rho = np.asarray(variable.coords[radial_dimension].values, dtype=np.float64)
        if area_measure:
            measure = 2.0 * np.pi * minor_radius**2 * source_rho
        else:
            measure = 4.0 * np.pi**2 * major_radius * minor_radius**2 * source_rho
        return [
            _cell_integral(snapshot, measure, source_rho)
            for snapshot in np.asarray(variable.values)
        ]

    return {
        "ion_heat_w": integrate_series("p_generic_heat_i"),
        "electron_heat_w": integrate_series("p_generic_heat_e"),
        "particles_s": integrate_series("s_generic_particle"),
        "driven_current_a": integrate_series("j_generic_current", area_measure=True),
        "ion_electron_exchange_w": integrate_series("ei_exchange"),
    }


def _state_budgets(profiles: Any, rho: FloatArray) -> list[dict[str, float]]:
    geometry = cast(Mapping[str, Any], MODEL_INTERSECTION["geometry"])
    major_radius = float(geometry["major_radius_m"])
    minor_radius = float(geometry["minor_radius_m"])
    volume_derivative = 4.0 * np.pi**2 * major_radius * minor_radius**2 * rho
    kev_j = 1.602176634e-16
    ti = np.asarray(profiles["T_i"].values, dtype=np.float64)
    te = np.asarray(profiles["T_e"].values, dtype=np.float64)
    ne = np.asarray(profiles["n_e"].values, dtype=np.float64)
    psi = np.asarray(profiles["psi"].values, dtype=np.float64)
    budgets: list[dict[str, float]] = []
    for index in range(ti.shape[0]):
        thermal_density = 1.5 * ne[index] * kev_j * (ti[index] + te[index])
        budgets.append(
            {
                "thermal_energy_j": _integral(thermal_density, volume_derivative, rho),
                "particle_inventory": _integral(ne[index], volume_derivative, rho),
                "poloidal_flux_l2": float(np.linalg.norm(psi[index])),
            }
        )
    return budgets


def _run_case(torax_module: Any, *, dt_s: float) -> tuple[dict[str, Any], float]:
    config = copy.deepcopy(CONFIG)
    config["numerics"]["fixed_dt"] = dt_s
    torax_config = torax_module.ToraxConfig.from_dict(config)
    start = time.perf_counter()
    data_tree, history = torax_module.run_simulation(torax_config, progress_bar=False)
    elapsed = time.perf_counter() - start
    profiles = data_tree.profiles
    rho = np.asarray(profiles["T_e"].coords["rho_norm"].values, dtype=np.float64)
    time_s = np.asarray(data_tree.time.values, dtype=np.float64)
    geometry = cast(Mapping[str, Any], MODEL_INTERSECTION["geometry"])
    comparison_rho = np.linspace(
        0.0,
        1.0,
        int(geometry["comparison_points"]),
        dtype=np.float64,
    )
    state_names = {
        "ion_temperature_kev": "T_i",
        "electron_temperature_kev": "T_e",
        "electron_density_m3": "n_e",
        "poloidal_flux_wb_per_rad": "psi",
    }
    raw_profiles = {
        output_name: np.asarray(profiles[torax_name].values, dtype=np.float64).tolist()
        for output_name, torax_name in state_names.items()
    }
    comparison_profiles = {
        output_name: _interpolate_profiles(
            rho,
            np.asarray(profiles[torax_name].values, dtype=np.float64),
            comparison_rho,
        )
        for output_name, torax_name in state_names.items()
    }
    return (
        {
            "dt_s": dt_s,
            "sim_error": str(history.sim_error),
            "time_s": time_s.tolist(),
            "rho_norm": rho.tolist(),
            "comparison_rho_norm": comparison_rho.tolist(),
            "raw_profiles": raw_profiles,
            "comparison_profiles": comparison_profiles,
            "source_totals": _source_totals(profiles),
            "state_budgets": _state_budgets(profiles, rho),
        },
        elapsed,
    )


def build_reference() -> dict[str, Any]:
    """Execute primary, warm-repeat, and refined TORAX cases."""
    import torax  # type: ignore[import-not-found]

    time_config = cast(Mapping[str, Any], MODEL_INTERSECTION["time"])
    primary_dt = float(time_config["primary_dt_s"])
    refined_dt = float(time_config["refined_dt_s"])
    primary, cold_seconds = _run_case(torax, dt_s=primary_dt)
    warm_repeat, warm_seconds = _run_case(torax, dt_s=primary_dt)
    refined, refined_seconds = _run_case(torax, dt_s=refined_dt)
    primary_projection = {
        "time_s": primary["time_s"],
        "profiles": primary["comparison_profiles"],
        "sources": primary["source_totals"],
        "budgets": primary["state_budgets"],
        "sim_error": primary["sim_error"],
    }
    warm_projection = {
        "time_s": warm_repeat["time_s"],
        "profiles": warm_repeat["comparison_profiles"],
        "sources": warm_repeat["source_totals"],
        "budgets": warm_repeat["state_budgets"],
        "sim_error": warm_repeat["sim_error"],
    }
    return {
        "schema": "scpn-fusion-core.torax-coupled-transport-reference.v1",
        "provenance": {
            "code": "TORAX",
            "code_url": "https://github.com/google-deepmind/torax",
            "licence": "Apache-2.0",
            "torax_version": torax.__version__,
            "deck_path": str(DECK_PATH.relative_to(ROOT)),
            "deck_sha256": hashlib.sha256(DECK_PATH.read_bytes()).hexdigest(),
            "deck_payload_sha256": _checksum(MODEL_INTERSECTION),
            "runner_path": str(Path(__file__).resolve().relative_to(ROOT)),
            "runner_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "precision": "float64",
            "runtime_backend": "cpu",
        },
        "model_intersection": MODEL_INTERSECTION,
        "primary": primary,
        "warm_repeat": warm_repeat,
        "refined": refined,
        "determinism": {
            "primary_projection_sha256": _checksum(primary_projection),
            "warm_projection_sha256": _checksum(warm_projection),
            "byte_identical_scientific_projection": _checksum(primary_projection)
            == _checksum(warm_projection),
        },
        "runtime_seconds": {
            "cold_primary": cold_seconds,
            "warm_primary": warm_seconds,
            "refined": refined_seconds,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run TORAX and write one provenance-bound reference artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    reference = build_reference()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reference, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"deterministic={reference['determinism']['byte_identical_scientific_projection']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
