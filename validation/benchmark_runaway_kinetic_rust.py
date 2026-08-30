# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Full Runaway Kinetic Rust Benchmark
"""Measure exact-output NumPy/Rust parity and host-conditioned speedup."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.runaway_kinetic_coefficients import (
    RunawayKineticCoefficients,
)
from scpn_fusion.core.runaway_kinetic_grid import RunawayKineticGrid
from scpn_fusion.core.runaway_kinetic_operator import RunawayKineticOperator
from scpn_fusion.core.runaway_kinetic_solver import (
    RunawayKineticSolver,
    RunawayKineticTrajectory,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "validation/benchmark_runaway_kinetic_rust.py",
    "src/scpn_fusion/core/runaway_kinetic_grid.py",
    "src/scpn_fusion/core/runaway_kinetic_coefficients.py",
    "src/scpn_fusion/core/runaway_kinetic_operator.py",
    "src/scpn_fusion/core/runaway_kinetic_solver.py",
    "src/scpn_fusion/core/runaway_kinetic_diagnostics.py",
    "scpn-fusion-rs/crates/fusion-physics/src/runaway_kinetic/grid.rs",
    "scpn-fusion-rs/crates/fusion-physics/src/runaway_kinetic/coefficients.rs",
    "scpn-fusion-rs/crates/fusion-physics/src/runaway_kinetic/operator.rs",
    "scpn-fusion-rs/crates/fusion-physics/src/runaway_kinetic/solver.rs",
    "scpn-fusion-rs/crates/fusion-physics/src/runaway_kinetic/mod.rs",
    "scpn-fusion-rs/crates/fusion-python/src/bindings/runaway_kinetic.rs",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checksum(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_provenance() -> dict[str, str]:
    missing = [relative for relative in SOURCE_PATHS if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"missing runaway kinetic source paths: {missing}")
    return {relative: _sha256(ROOT / relative) for relative in SOURCE_PATHS}


def _repository_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _extension_provenance() -> dict[str, str]:
    specification = importlib.util.find_spec("scpn_fusion_rs.scpn_fusion_rs")
    if specification is None or specification.origin is None:
        raise RuntimeError("compiled scpn_fusion_rs extension is unavailable")
    extension_path = Path(specification.origin).resolve()
    if not extension_path.is_file():
        raise RuntimeError(f"compiled extension is not a file: {extension_path}")
    return {
        "artifact_name": extension_path.name,
        "artifact_sha256": _sha256(extension_path),
    }


def scientific_projection(report: dict[str, Any]) -> dict[str, Any]:
    deterministic_gates = {
        name: report["gates"][name]
        for name in (
            "all_declared_outputs_compared",
            "full_output_relative_l2_within_2e_12",
            "internal_ssprk3_steps_exact",
        )
    }
    return {
        "schema": report["schema"],
        "problem": report["problem"],
        "parity": report["parity"],
        "deterministic_gates": deterministic_gates,
        "source_provenance": report["source_provenance"],
    }


def _problem() -> tuple[
    RunawayKineticSolver,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    grid = RunawayKineticGrid(
        radius_faces_m=np.linspace(0.0, 0.2, 9),
        pitch_faces=np.linspace(-1.0, 1.0, 17),
        momentum_faces_mc=np.linspace(0.02, 20.0, 33),
    )
    radial_shape = (grid.nr + 1, grid.nxi, grid.np)
    momentum_shape = (grid.nr, grid.nxi, grid.np + 1)
    pitch_shape = (grid.nr, grid.nxi + 1, grid.np)
    radial_index = np.arange(np.prod(radial_shape), dtype=np.float64).reshape(radial_shape)
    momentum_index = np.arange(np.prod(momentum_shape), dtype=np.float64).reshape(momentum_shape)
    pitch_index = np.arange(np.prod(pitch_shape), dtype=np.float64).reshape(pitch_shape)
    cell_index = np.arange(np.prod(grid.shape), dtype=np.float64).reshape(grid.shape)
    coefficients = RunawayKineticCoefficients.checked(
        grid,
        radial_advection=1.0e-4 * (1.0 + radial_index / radial_index.size),
        momentum_electric_advection=2.0e-3 * (1.0 + momentum_index / momentum_index.size),
        momentum_collision_advection=-7.0e-4 * (1.0 + momentum_index / momentum_index.size),
        momentum_synchrotron_advection=-2.0e-4 * (1.0 + momentum_index / momentum_index.size),
        momentum_bremsstrahlung_advection=-1.0e-4 * (1.0 + momentum_index / momentum_index.size),
        pitch_electric_advection=8.0e-4 * (1.0 + pitch_index / pitch_index.size),
        pitch_synchrotron_advection=-3.0e-4 * (1.0 + pitch_index / pitch_index.size),
        radial_diffusion=2.0e-6 * (1.0 + radial_index / radial_index.size),
        momentum_diffusion=3.0e-6 * (1.0 + momentum_index / momentum_index.size),
        pitch_diffusion=4.0e-6 * (1.0 + pitch_index / pitch_index.size),
        momentum_pitch_diffusion=5.0e-7 * (1.0 + momentum_index / momentum_index.size),
        pitch_momentum_diffusion=6.0e-7 * (1.0 + pitch_index / pitch_index.size),
        avalanche_source_kernel=2.0e-30 * (1.0 + cell_index / cell_index.size),
        total_electron_density_m3=np.linspace(1.0e19, 2.0e19, grid.nr),
        total_density_avalanche_rate_s_inv=np.linspace(2.0, 4.0, grid.nr),
        total_density_external_source_m3_s=np.linspace(2.0e7, 4.0e7, grid.nr),
        external_source=3.0e5 * (1.0 + cell_index / cell_index.size),
    )
    initial = 1.0e10 * (1.0 + 0.2 * cell_index / cell_index.size)
    density = np.linspace(1.0e12, 2.0e12, grid.nr)
    times = np.array([0.0, 5.0e-8, 1.0e-7])
    return (
        RunawayKineticSolver(
            RunawayKineticOperator(grid, coefficients),
            maximum_step_s=2.5e-8,
            negativity_tolerance=1.0e-10,
        ),
        initial,
        density,
        times,
    )


def _run(
    solver: RunawayKineticSolver,
    initial: NDArray[np.float64],
    density: NDArray[np.float64],
    times: NDArray[np.float64],
    backend: Literal["numpy", "rust"],
) -> RunawayKineticTrajectory:
    return solver.solve(
        initial,
        times,
        initial_runaway_density_m3=density,
        backend=backend,
    )


def _relative_l2(actual: NDArray[np.float64], expected: NDArray[np.float64]) -> float:
    denominator = max(float(np.linalg.norm(expected.ravel())), 1.0)
    return float(np.linalg.norm((actual - expected).ravel()) / denominator)


def benchmark(repeats: int) -> dict[str, Any]:
    if repeats < 3:
        raise ValueError("repeats must be at least three")
    solver, initial, density, times = _problem()
    numpy_result = _run(solver, initial, density, times, "numpy")
    rust_result = _run(solver, initial, density, times, "rust")
    for _ in range(2):
        _run(solver, initial, density, times, "numpy")
        _run(solver, initial, density, times, "rust")

    samples: dict[str, list[float]] = {"numpy": [], "rust": []}
    for _ in range(repeats):
        for backend in ("numpy", "rust"):
            start = perf_counter()
            _run(solver, initial, density, times, backend)
            samples[backend].append(perf_counter() - start)
    medians = {backend: statistics.median(values) for backend, values in samples.items()}
    direct_parity_fields = (
        "times_s",
        "distribution",
        "radial_transport",
        "electric_acceleration",
        "collisional_drag_diffusion",
        "pitch_scattering",
        "cross_diffusion",
        "synchrotron_loss",
        "bremsstrahlung_loss",
        "avalanche_generation",
        "external_source",
        "total_tendency",
        "runaway_density_m3",
        "runaway_density_radial_transport_m3_s",
        "runaway_density_avalanche_generation_m3_s",
        "runaway_density_external_source_m3_s",
        "runaway_density_tendency_m3_s",
    )
    parity_arrays = {
        name: (
            np.asarray(getattr(rust_result, name)),
            np.asarray(getattr(numpy_result, name)),
        )
        for name in direct_parity_fields
    }
    parity_arrays.update(
        {
            f"moments.{name}": (
                np.asarray(getattr(rust_result.moments, name)),
                np.asarray(getattr(numpy_result.moments, name)),
            )
            for name in (
                "density_m3",
                "current_density_a_m2",
                "kinetic_energy_density_j_m3",
            )
        }
    )
    parity_arrays["minimum_distribution"] = (
        np.asarray([rust_result.minimum_distribution]),
        np.asarray([numpy_result.minimum_distribution]),
    )
    expected_parity_fields = set(direct_parity_fields) | {
        "moments.density_m3",
        "moments.current_density_a_m2",
        "moments.kinetic_energy_density_j_m3",
        "minimum_distribution",
    }
    parity = {
        name: {
            "maximum_absolute_error": float(np.max(np.abs(rust_values - numpy_values))),
            "relative_l2_error": _relative_l2(
                rust_values,
                numpy_values,
            ),
        }
        for name, (rust_values, numpy_values) in parity_arrays.items()
    }
    internal_steps_match = rust_result.internal_steps == numpy_result.internal_steps
    array_parity_passed = all(values["relative_l2_error"] <= 2.0e-12 for values in parity.values())
    parity_passed = internal_steps_match and array_parity_passed
    speedup = medians["numpy"] / medians["rust"]
    rust_selected = parity_passed and speedup > 1.0
    gates = {
        "all_declared_outputs_compared": set(parity) == expected_parity_fields,
        "full_output_relative_l2_within_2e_12": array_parity_passed,
        "internal_ssprk3_steps_exact": internal_steps_match,
        "rust_measured_faster_on_this_host_problem": speedup > 1.0,
    }
    all_pass = all(gates.values())
    report: dict[str, Any] = {
        "schema": "runaway-kinetic-rust-benchmark.v1",
        "schema_version": 1,
        "status": (
            "accepted_full_kinetic_rust_backend_parity"
            if all_pass
            else "blocked_full_kinetic_rust_backend_parity"
        ),
        "all_pass": all_pass,
        "gates": gates,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "load_average_1m_5m_15m": list(os.getloadavg()),
            "conditioning": (
                "Local shared host; a concurrent pinned DREAM reference run was active. "
                "Timings are evidence for this host and problem only."
            ),
        },
        "problem": {
            "grid_shape_radius_pitch_momentum": list(solver.operator.grid.shape),
            "requested_output_times": int(times.size),
            "internal_ssprk3_steps": numpy_result.internal_steps,
            "all_declared_tendencies_returned": True,
        },
        "timing_seconds": {
            "repeats": repeats,
            "numpy_samples": samples["numpy"],
            "rust_samples": samples["rust"],
            "numpy_median": medians["numpy"],
            "rust_median": medians["rust"],
            "rust_speedup_over_numpy": speedup,
        },
        "parity": parity,
        "language_selection": {
            "production_explicit_kernel": "rust" if rust_selected else "numpy",
            "python_role": "orchestration, DREAM execution, HDF5 and validation",
            "julia_gate": (
                "Evaluate when a stiff implicit, sparse nonlinear, adjoint or "
                "differentiable kinetic branch is required; require measured advantage."
            ),
            "go_role": "service and orchestration surfaces, not this tensor kernel",
        },
    }
    report["source_revision"] = _repository_head()
    report["source_provenance"] = _source_provenance()
    report["runtime_provenance"] = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "extension": _extension_provenance(),
    }
    report["scientific_projection_sha256"] = _checksum(scientific_projection(report))
    return report


def _markdown(report: dict[str, Any]) -> str:
    timing = report["timing_seconds"]
    parity = report["parity"]
    maximum_relative = max(float(values["relative_l2_error"]) for values in parity.values())
    return f"""# Full runaway kinetic Rust benchmark

- Grid: `{report["problem"]["grid_shape_radius_pitch_momentum"]}` in radius-pitch-momentum order
- Requested outputs: `{report["problem"]["requested_output_times"]}`
- Internal SSPRK3 steps: `{report["problem"]["internal_ssprk3_steps"]}`
- NumPy median: `{timing["numpy_median"]:.9f} s`
- Rust median: `{timing["rust_median"]:.9f} s`
- Rust speedup on this host/problem: `{timing["rust_speedup_over_numpy"]:.3f}x`
- Maximum component relative L2 error: `{maximum_relative:.3e}`
- Host load averages: `{report["host"]["load_average_1m_5m_15m"]}`
- Source revision: `{report["source_revision"]}`
- Scientific projection SHA-256: `{report["scientific_projection_sha256"]}`
- Compiled extension SHA-256: `{report["runtime_provenance"]["extension"]["artifact_sha256"]}`

The timing is host-conditioned: a pinned DREAM reference run was active concurrently.
Rust is selected for the explicit production kernel because it preserves all outputs
and has a measured advantage here. Python remains the orchestration/DREAM/HDF5 tier.
Julia is gated on a future measured need for stiff implicit, sparse nonlinear, adjoint,
or differentiable solves. Go remains appropriate for service/orchestration surfaces,
not this tensor kernel.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()
    report = benchmark(args.repeats)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json is None:
        print(rendered, end="")
    else:
        args.output_json.write_text(rendered, encoding="utf-8")
    if args.output_markdown is not None:
        args.output_markdown.write_text(_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
