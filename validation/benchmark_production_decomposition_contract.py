#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Fail-closed production-scale decomposition contract benchmark."""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from importlib import metadata, util
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scpn_fusion.core.gk_domain_decomposition import (  # noqa: E402
    GKDomainDecompositionPlan,
    build_radial_toroidal_decomposition,
    local_decomposed_phase_execution,
    local_multiprocess_rank_tile_execution,
    rank_tile_communication_contract,
    serial_halo_exchange,
)

REPORT_DIR = ROOT / "validation" / "reports"
JSON_REPORT = REPORT_DIR / "production_decomposition_contract.json"
MD_REPORT = REPORT_DIR / "production_decomposition_contract.md"
RELATIVE_REDUCTION_TOLERANCE = 1.0e-12
RECONSTRUCTION_LINF_TOLERANCE = 0.0
FLOAT64_BYTES = 8
DISTRIBUTED_RUNS_ENV = "SCPN_PRODUCTION_DECOMPOSITION_DISTRIBUTED_RUNS_JSON"
REQUIRED_DISTRIBUTED_RANK_COUNTS = [1, 2, 4, 8, 16, 32]
MINIMUM_PARALLEL_EFFICIENCY = 0.70
MINIMUM_WEAK_SCALING_EFFICIENCY = 0.80
REQUIRED_DISTRIBUTED_RUN_FIELDS = [
    "rank_count",
    "wall_time_s",
    "parallel_efficiency",
    "weak_scaling_efficiency",
    "owned_phase_cells_per_rank",
    "halo_exchange_bytes_per_step",
    "decomposition_invariant_pass",
    "hardware_metadata",
    "command",
    "artifact_sha256",
]
MPI_RUNTIME_RANK_COUNT = 4
OPTIONAL_RUNTIME_DEPENDENCIES = {
    "cupy": "cupy-cuda12x>=13.6,<14.0",
    "mpi4py": "mpi4py>=4.1",
    "nvidia.cuda_nvrtc": "nvidia-cuda-nvrtc-cu12>=12.0,<13.0",
    "numpy": "numpy>=1.24,<2.0",
}
FACE_OPPOSITES = {
    "radial_lower": "radial_upper",
    "radial_upper": "radial_lower",
    "toroidal_lower": "toroidal_upper",
    "toroidal_upper": "toroidal_lower",
}


def _plan_row(case_id: str, plan: GKDomainDecompositionPlan) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "halo": plan.halo,
        "halo_overhead_ratio": plan.halo_overhead_ratio,
        "max_owned_cells": plan.max_owned_cells,
        "min_owned_cells": plan.min_owned_cells,
        "n_mu": plan.n_mu,
        "n_radial": plan.n_radial,
        "n_theta": plan.n_theta,
        "n_toroidal": plan.n_toroidal,
        "n_vpar": plan.n_vpar,
        "owned_cell_imbalance": plan.owned_cell_imbalance,
        "radial_parts": plan.radial_parts,
        "toroidal_parts": plan.toroidal_parts,
        "total_halo_phase_cells": plan.total_halo_phase_cells,
        "total_owned_phase_cells": plan.total_owned_phase_cells,
        "total_ranks": plan.total_ranks,
    }


def _cpu_benchmark_row(case_id: str, plan: GKDomainDecompositionPlan) -> dict[str, Any]:
    cell_count = plan.total_owned_phase_cells
    state = np.linspace(0.0, 1.0, num=cell_count, dtype=np.float64).reshape(
        plan.n_radial,
        plan.n_toroidal,
        plan.n_theta,
        plan.n_vpar,
        plan.n_mu,
    )
    started = time.perf_counter()
    metrics = local_decomposed_phase_execution(plan, state)
    elapsed = max(time.perf_counter() - started, 1.0e-12)
    return {
        "case_id": case_id,
        "cells_per_second": cell_count / elapsed,
        "elapsed_s": elapsed,
        "free_energy_relative_error": metrics.free_energy_relative_error,
        "global_free_energy": metrics.global_free_energy,
        "global_inventory": metrics.global_inventory,
        "global_parallel_moment": metrics.global_parallel_moment,
        "global_shape": list(metrics.global_shape),
        "halo_exchange_pass": metrics.halo_exchange_pass,
        "inventory_relative_error": metrics.inventory_relative_error,
        "local_decomposed_execution_pass": metrics.decomposition_invariant_pass,
        "local_free_energy": metrics.local_free_energy,
        "local_inventory": metrics.local_inventory,
        "local_parallel_moment": metrics.local_parallel_moment,
        "owned_phase_cells": cell_count,
        "parallel_moment_relative_error": metrics.parallel_moment_relative_error,
        "rank_count": metrics.rank_count,
        "reconstruction_linf_error": metrics.reconstruction_linf_error,
    }


def _shape_convergence_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return same-physics convergence evidence across decomposition shapes."""
    if not rows:
        return {
            "max_free_energy_relative_deviation": float("inf"),
            "max_inventory_relative_deviation": float("inf"),
            "max_reconstruction_linf_error": float("inf"),
            "reconstruction_linf_tolerance": RECONSTRUCTION_LINF_TOLERANCE,
            "relative_reduction_tolerance": RELATIVE_REDUCTION_TOLERANCE,
            "schema": "production-decomposition-shape-convergence.v1",
            "shape_convergence_pass": False,
            "shape_count": 0,
            "shape_rows": [],
            "status": "blocked_no_local_shape_rows",
        }

    reference_inventory = float(rows[0]["local_inventory"])
    reference_free_energy = float(rows[0]["local_free_energy"])
    reference_parallel_moment = float(rows[0]["local_parallel_moment"])
    shape_rows: list[dict[str, Any]] = []
    for row in rows:
        inventory_deviation = abs(float(row["local_inventory"]) - reference_inventory) / max(
            abs(reference_inventory), 1.0e-30
        )
        free_energy_deviation = abs(float(row["local_free_energy"]) - reference_free_energy) / max(
            abs(reference_free_energy), 1.0e-30
        )
        parallel_moment_deviation = abs(
            float(row["local_parallel_moment"]) - reference_parallel_moment
        ) / max(abs(reference_parallel_moment), 1.0e-30)
        reconstruction_error = float(row["reconstruction_linf_error"])
        shape_pass = bool(
            row["local_decomposed_execution_pass"]
            and inventory_deviation <= RELATIVE_REDUCTION_TOLERANCE
            and free_energy_deviation <= RELATIVE_REDUCTION_TOLERANCE
            and parallel_moment_deviation <= RELATIVE_REDUCTION_TOLERANCE
            and reconstruction_error <= RECONSTRUCTION_LINF_TOLERANCE
        )
        shape_rows.append(
            {
                "case_id": row["case_id"],
                "cells_per_second": row["cells_per_second"],
                "free_energy_relative_deviation_from_reference": free_energy_deviation,
                "inventory_relative_deviation_from_reference": inventory_deviation,
                "owned_phase_cells": row["owned_phase_cells"],
                "parallel_moment_relative_deviation_from_reference": (parallel_moment_deviation),
                "rank_count": row["rank_count"],
                "reconstruction_linf_error": reconstruction_error,
                "shape_convergence_pass": shape_pass,
            }
        )

    max_inventory_deviation = max(
        float(row["inventory_relative_deviation_from_reference"]) for row in shape_rows
    )
    max_free_energy_deviation = max(
        float(row["free_energy_relative_deviation_from_reference"]) for row in shape_rows
    )
    max_parallel_moment_deviation = max(
        float(row["parallel_moment_relative_deviation_from_reference"]) for row in shape_rows
    )
    max_reconstruction_error = max(float(row["reconstruction_linf_error"]) for row in shape_rows)
    shape_convergence_pass = bool(
        len(shape_rows) >= 3
        and all(bool(row["shape_convergence_pass"]) for row in shape_rows)
        and max_inventory_deviation <= RELATIVE_REDUCTION_TOLERANCE
        and max_free_energy_deviation <= RELATIVE_REDUCTION_TOLERANCE
        and max_parallel_moment_deviation <= RELATIVE_REDUCTION_TOLERANCE
        and max_reconstruction_error <= RECONSTRUCTION_LINF_TOLERANCE
    )
    return {
        "max_free_energy_relative_deviation": max_free_energy_deviation,
        "max_inventory_relative_deviation": max_inventory_deviation,
        "max_parallel_moment_relative_deviation": max_parallel_moment_deviation,
        "max_reconstruction_linf_error": max_reconstruction_error,
        "reconstruction_linf_tolerance": RECONSTRUCTION_LINF_TOLERANCE,
        "reference_case_id": rows[0]["case_id"],
        "relative_reduction_tolerance": RELATIVE_REDUCTION_TOLERANCE,
        "schema": "production-decomposition-shape-convergence.v1",
        "shape_convergence_pass": shape_convergence_pass,
        "shape_count": len(shape_rows),
        "shape_rows": shape_rows,
        "status": "accepted_local_same_physics_shape_convergence"
        if shape_convergence_pass
        else "blocked_local_same_physics_shape_convergence_failed",
    }


def _large_grid_cpu_evidence() -> dict[str, Any]:
    """Return executable larger-grid CPU timing evidence without scaling claims."""
    plan = build_radial_toroidal_decomposition(
        n_radial=96,
        n_toroidal=48,
        n_theta=16,
        n_vpar=16,
        n_mu=8,
        radial_parts=6,
        toroidal_parts=4,
        halo=1,
    )
    row = _cpu_benchmark_row("large_cpu_96x48_6x4", plan)
    invariant_pass = bool(
        row["local_decomposed_execution_pass"]
        and row["halo_exchange_pass"]
        and row["reconstruction_linf_error"] <= RECONSTRUCTION_LINF_TOLERANCE
        and row["inventory_relative_error"] <= RELATIVE_REDUCTION_TOLERANCE
        and row["free_energy_relative_error"] <= RELATIVE_REDUCTION_TOLERANCE
        and row["parallel_moment_relative_error"] <= RELATIVE_REDUCTION_TOLERANCE
    )
    return {
        "blocking_reason": (
            "This is single-process CPU evidence only. It does not satisfy the "
            "distributed MPI or multi-GPU scaling requirement."
        ),
        "large_grid_cpu_benchmark_ready": invariant_pass,
        "max_parallel_moment_relative_error": row["parallel_moment_relative_error"],
        "max_reconstruction_linf_error": row["reconstruction_linf_error"],
        "rows": [row],
        "schema": "production-decomposition-large-grid-cpu-evidence.v1",
        "status": "accepted_local_large_grid_cpu_evidence"
        if invariant_pass
        else "blocked_large_grid_cpu_invariants_failed",
    }


def _local_multiprocess_cpu_evidence() -> dict[str, Any]:
    """Return process-isolated local CPU rank-tile execution evidence."""
    plan = build_radial_toroidal_decomposition(
        n_radial=32,
        n_toroidal=16,
        n_theta=8,
        n_vpar=8,
        n_mu=4,
        radial_parts=4,
        toroidal_parts=2,
        halo=1,
    )
    state = np.cos(np.arange(plan.total_owned_phase_cells, dtype=np.float64) / 23.0).reshape(
        plan.n_radial,
        plan.n_toroidal,
        plan.n_theta,
        plan.n_vpar,
        plan.n_mu,
    )
    started = time.perf_counter()
    metrics = local_multiprocess_rank_tile_execution(plan, state, max_workers=4)
    elapsed = max(time.perf_counter() - started, 1.0e-12)
    execution_ready = bool(
        metrics.decomposition_invariant_pass
        and metrics.halo_exchange_pass
        and metrics.reconstruction_linf_error <= RECONSTRUCTION_LINF_TOLERANCE
        and metrics.inventory_relative_error <= RELATIVE_REDUCTION_TOLERANCE
        and metrics.free_energy_relative_error <= RELATIVE_REDUCTION_TOLERANCE
        and metrics.parallel_moment_relative_error <= RELATIVE_REDUCTION_TOLERANCE
        and metrics.halo_checksum_relative_error <= RELATIVE_REDUCTION_TOLERANCE
    )
    return {
        "blocking_reason": (
            "This is local CPU process isolation over rank tiles. It is not MPI "
            "or multi-GPU execution and does not satisfy the cluster scaling gate."
        ),
        "case_id": "local_multiprocess_cpu_32x16_4x2",
        "cells_per_second": plan.total_owned_phase_cells / elapsed,
        "elapsed_s": elapsed,
        "free_energy_relative_error": metrics.free_energy_relative_error,
        "global_shape": list(metrics.global_shape),
        "halo_checksum_relative_error": metrics.halo_checksum_relative_error,
        "halo_exchange_pass": metrics.halo_exchange_pass,
        "inventory_relative_error": metrics.inventory_relative_error,
        "local_multiprocess_cpu_execution_ready": execution_ready,
        "owned_phase_cells": plan.total_owned_phase_cells,
        "parallel_moment_relative_error": metrics.parallel_moment_relative_error,
        "rank_count": metrics.rank_count,
        "rank_rows": list(metrics.rank_rows),
        "reconstruction_linf_error": metrics.reconstruction_linf_error,
        "schema": "production-decomposition-local-multiprocess-cpu-evidence.v1",
        "status": "accepted_local_multiprocess_cpu_rank_execution"
        if execution_ready
        else "blocked_local_multiprocess_cpu_rank_execution_failed",
        "unique_worker_process_count": metrics.unique_worker_process_count,
        "worker_count": metrics.worker_count,
    }


def _python_for_optional_runtime() -> str:
    """Return the Python executable with optional local runtime deps if present."""
    venv_python = ROOT / ".venv" / "bin" / "python"
    return str(venv_python) if venv_python.exists() else sys.executable


def _runtime_dependency_evidence() -> dict[str, Any]:
    """Return import and version evidence for optional decomposition runtimes."""
    module_to_distribution = {
        "cupy": "cupy-cuda12x",
        "mpi4py": "mpi4py",
        "nvidia.cuda_nvrtc": "nvidia-cuda-nvrtc-cu12",
        "numpy": "numpy",
    }
    rows: list[dict[str, Any]] = []
    for module_name, requirement in OPTIONAL_RUNTIME_DEPENDENCIES.items():
        distribution = module_to_distribution[module_name]
        spec = util.find_spec(module_name)
        version = None
        if spec is not None:
            try:
                version = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                version = None
        rows.append(
            {
                "distribution": distribution,
                "importable": spec is not None,
                "module": module_name,
                "required_specifier": requirement,
                "version": version,
            }
        )
    numpy_row = next(row for row in rows if row["module"] == "numpy")
    numpy_version = str(numpy_row["version"] or "")
    numpy_contract_pass = bool(
        numpy_row["importable"] and numpy_version and not numpy_version.startswith("2.")
    )
    optional_runtime_dependency_ready = bool(
        numpy_contract_pass
        and all(row["importable"] for row in rows if row["module"] in {"cupy", "mpi4py"})
    )
    return {
        "numpy_contract_pass": numpy_contract_pass,
        "optional_runtime_dependency_ready": optional_runtime_dependency_ready,
        "python_executable": _python_for_optional_runtime(),
        "rows": rows,
        "schema": "production-decomposition-runtime-dependencies.v1",
        "status": "accepted_optional_runtime_dependencies"
        if optional_runtime_dependency_ready
        else "blocked_optional_runtime_dependency_contract",
    }


def _mpi_runtime_evidence() -> dict[str, Any]:
    """Run a real MPI rank-tile execution probe when MPI is installed."""
    mpiexec = shutil.which("mpiexec") or shutil.which("mpirun")
    if mpiexec is None:
        return {
            "blocking_reason": "mpiexec_or_mpirun_not_found",
            "mpi_runtime_execution_ready": False,
            "rank_count": MPI_RUNTIME_RANK_COUNT,
            "schema": "production-decomposition-mpi-runtime-evidence.v1",
            "status": "blocked_mpi_launcher_missing",
        }
    python_executable = _python_for_optional_runtime()
    runner = ROOT / "validation" / "mpi_production_decomposition_runner.py"
    with tempfile.TemporaryDirectory(prefix="scpn_mpi_decomposition_") as tmp_dir:
        output = Path(tmp_dir) / "mpi_result.json"
        command = [
            mpiexec,
            "-n",
            str(MPI_RUNTIME_RANK_COUNT),
            python_executable,
            str(runner),
            "--output",
            str(output),
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            check=False,
            env={**os.environ, "OMPI_MCA_rmaps_base_oversubscribe": "1"},
            capture_output=True,
            text=True,
            timeout=30.0,
        )
        elapsed = max(time.perf_counter() - started, 1.0e-12)
        if completed.returncode != 0 or not output.exists():
            return {
                "blocking_reason": completed.stderr.strip() or completed.stdout.strip(),
                "command": command,
                "elapsed_s": elapsed,
                "mpi_runtime_execution_ready": False,
                "rank_count": MPI_RUNTIME_RANK_COUNT,
                "returncode": completed.returncode,
                "schema": "production-decomposition-mpi-runtime-evidence.v1",
                "status": "blocked_mpi_runtime_execution_failed",
            }
        payload = json.loads(output.read_text(encoding="utf-8"))
    ready = bool(
        payload["decomposition_invariant_pass"]
        and payload["halo_exchange_pass"]
        and payload.get("topology") == "radial_toroidal_2d"
        and payload.get("radial_parts") == 2
        and payload.get("toroidal_parts") == 2
        and payload.get("min_halo_verified_fraction", 0.0) == 1.0
        and payload["reconstruction_linf_error"] <= RECONSTRUCTION_LINF_TOLERANCE
        and payload["inventory_relative_error"] <= RELATIVE_REDUCTION_TOLERANCE
        and payload["free_energy_relative_error"] <= RELATIVE_REDUCTION_TOLERANCE
        and payload["parallel_moment_relative_error"] <= RELATIVE_REDUCTION_TOLERANCE
    )
    return {
        **payload,
        "blocking_reason": (
            "MPI rank-tile execution passed locally. Cluster scaling and multi-GPU "
            "runtime evidence are still required for production-scale readiness."
        ),
        "command": command,
        "elapsed_s": elapsed,
        "mpi_runtime_execution_ready": ready,
        "returncode": 0,
        "schema": "production-decomposition-mpi-runtime-evidence.v1",
        "status": "accepted_local_mpi_rank_tile_execution"
        if ready
        else "blocked_mpi_runtime_invariants_failed",
    }


def _preload_cuda_nvrtc() -> str | None:
    """Preload venv-packaged CUDA NVRTC for CuPy wheels when needed."""
    venv_lib = ROOT / ".venv" / "lib"
    candidates = list(venv_lib.glob("python*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12"))
    if not candidates:
        return None
    lib_path = candidates[0]
    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
    return str(lib_path)


def _gpu_rank_tile_evidence() -> dict[str, Any]:
    """Return CUDA rank-tile reduction evidence when CuPy and a GPU are available."""
    try:
        preloaded_nvrtc = _preload_cuda_nvrtc()
        import cupy as cp
    except Exception as exc:
        return {
            "blocking_reason": f"cupy_import_failed: {exc}",
            "gpu_rank_tile_execution_ready": False,
            "multi_gpu_runtime_ready": False,
            "schema": "production-decomposition-gpu-rank-tile-evidence.v1",
            "status": "blocked_cupy_runtime_missing",
        }
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        return {
            "blocking_reason": f"cuda_device_query_failed: {exc}",
            "gpu_rank_tile_execution_ready": False,
            "multi_gpu_runtime_ready": False,
            "schema": "production-decomposition-gpu-rank-tile-evidence.v1",
            "status": "blocked_cuda_device_query_failed",
        }
    if device_count <= 0:
        return {
            "blocking_reason": "no_cuda_devices_visible",
            "device_count": device_count,
            "gpu_rank_tile_execution_ready": False,
            "multi_gpu_runtime_ready": False,
            "schema": "production-decomposition-gpu-rank-tile-evidence.v1",
            "status": "blocked_no_cuda_devices_visible",
        }
    plan = build_radial_toroidal_decomposition(
        n_radial=24,
        n_toroidal=12,
        n_theta=8,
        n_vpar=6,
        n_mu=4,
        radial_parts=4,
        toroidal_parts=2,
        halo=1,
    )
    state = np.sin(np.arange(plan.total_owned_phase_cells, dtype=np.float64) / 31.0).reshape(
        plan.n_radial,
        plan.n_toroidal,
        plan.n_theta,
        plan.n_vpar,
        plan.n_mu,
    )
    weights = np.linspace(-1.0, 1.0, num=plan.n_vpar, dtype=np.float64).reshape(
        1, 1, 1, plan.n_vpar, 1
    )
    started = time.perf_counter()
    rank_rows: list[dict[str, Any]] = []
    local_inventory = 0.0
    local_free_energy = 0.0
    local_parallel_moment = 0.0
    for tile in plan.tiles:
        device_id = tile.rank % device_count
        owned = state[
            tile.radial.start : tile.radial.stop,
            tile.toroidal.start : tile.toroidal.stop,
            :,
            :,
            :,
        ]
        with cp.cuda.Device(device_id):
            owned_gpu = cp.asarray(owned)
            inventory = float(cp.sum(owned_gpu).get())
            free_energy = float(cp.sum(owned_gpu * owned_gpu).get())
            parallel_moment = float(cp.sum(owned_gpu * cp.asarray(weights)).get())
        local_inventory += inventory
        local_free_energy += free_energy
        local_parallel_moment += parallel_moment
        rank_rows.append(
            {
                "device_id": device_id,
                "owned_shape": [int(axis) for axis in owned.shape],
                "rank": tile.rank,
            }
        )
    elapsed = max(time.perf_counter() - started, 1.0e-12)
    global_inventory = float(np.sum(state))
    global_free_energy = float(np.sum(state * state))
    global_parallel_moment = float(np.sum(state * weights))
    inventory_relative_error = abs(local_inventory - global_inventory) / max(
        abs(global_inventory), 1.0e-30
    )
    free_energy_relative_error = abs(local_free_energy - global_free_energy) / max(
        abs(global_free_energy), 1.0e-30
    )
    parallel_moment_relative_error = abs(local_parallel_moment - global_parallel_moment) / max(
        abs(global_parallel_moment), 1.0e-30
    )
    ready = bool(
        inventory_relative_error <= RELATIVE_REDUCTION_TOLERANCE
        and free_energy_relative_error <= RELATIVE_REDUCTION_TOLERANCE
        and parallel_moment_relative_error <= RELATIVE_REDUCTION_TOLERANCE
    )
    return {
        "blocking_reason": (
            "Single-GPU rank-tile reductions passed locally; multi-GPU readiness "
            "requires at least two visible CUDA devices and scaling rows."
            if device_count < 2
            else "GPU rank-tile reductions passed locally; multi-GPU scaling rows remain required."
        ),
        "cells_per_second": plan.total_owned_phase_cells / elapsed,
        "device_count": device_count,
        "elapsed_s": elapsed,
        "free_energy_relative_error": free_energy_relative_error,
        "gpu_rank_tile_execution_ready": ready,
        "inventory_relative_error": inventory_relative_error,
        "multi_gpu_runtime_ready": bool(device_count >= 2 and ready),
        "owned_phase_cells": plan.total_owned_phase_cells,
        "parallel_moment_relative_error": parallel_moment_relative_error,
        "rank_count": plan.total_ranks,
        "rank_rows": rank_rows,
        "preloaded_nvrtc": preloaded_nvrtc,
        "schema": "production-decomposition-gpu-rank-tile-evidence.v1",
        "status": "accepted_local_gpu_rank_tile_execution"
        if ready
        else "blocked_gpu_rank_tile_invariants_failed",
    }


def _halo_face_integrity_evidence(case_id: str, plan: GKDomainDecompositionPlan) -> dict[str, Any]:
    """Return serial halo-face integrity evidence for future distributed exchange."""
    state: NDArray[np.float64] = np.arange(plan.total_owned_phase_cells, dtype=np.float64).reshape(
        plan.n_radial,
        plan.n_toroidal,
        plan.n_theta,
        plan.n_vpar,
        plan.n_mu,
    )
    local_tiles = serial_halo_exchange(plan, state)
    communication_by_rank = {
        int(row["rank"]): row for row in rank_tile_communication_contract(plan)
    }
    face_rows: list[dict[str, Any]] = []
    for local in local_tiles:
        tile = plan.tiles[local.rank]
        radial_offset = tile.radial.start - tile.radial_with_halo.start
        toroidal_offset = tile.toroidal.start - tile.toroidal_with_halo.start
        face_slices = {
            "radial_lower": (
                slice(0, radial_offset),
                slice(toroidal_offset, toroidal_offset + tile.toroidal.size),
                slice(tile.radial_with_halo.start, tile.radial.start),
                slice(tile.toroidal.start, tile.toroidal.stop),
            ),
            "radial_upper": (
                slice(tile.radial.stop - tile.radial_with_halo.start, None),
                slice(toroidal_offset, toroidal_offset + tile.toroidal.size),
                slice(tile.radial.stop, tile.radial_with_halo.stop),
                slice(tile.toroidal.start, tile.toroidal.stop),
            ),
            "toroidal_lower": (
                slice(radial_offset, radial_offset + tile.radial.size),
                slice(0, toroidal_offset),
                slice(tile.radial.start, tile.radial.stop),
                slice(tile.toroidal_with_halo.start, tile.toroidal.start),
            ),
            "toroidal_upper": (
                slice(radial_offset, radial_offset + tile.radial.size),
                slice(tile.toroidal.stop - tile.toroidal_with_halo.start, None),
                slice(tile.radial.start, tile.radial.stop),
                slice(tile.toroidal.stop, tile.toroidal_with_halo.stop),
            ),
        }
        neighbours = communication_by_rank[local.rank]["neighbour_ranks"]
        for face, (local_r, local_t, global_r, global_t) in face_slices.items():
            if neighbours[face] is None:
                continue
            actual = local.with_halo[local_r, local_t, :, :, :]
            expected = state[global_r, global_t, :, :, :]
            linf_error = float(np.max(np.abs(actual - expected)))
            face_rows.append(
                {
                    "case_id": case_id,
                    "face": face,
                    "face_integrity_pass": bool(linf_error == 0.0),
                    "face_payload_shape": [int(axis) for axis in actual.shape],
                    "linf_error": linf_error,
                    "neighbour_rank": neighbours[face],
                    "rank": local.rank,
                }
            )
    max_linf_error = max(float(row["linf_error"]) for row in face_rows) if face_rows else np.inf
    halo_face_integrity_pass = bool(
        face_rows and all(bool(row["face_integrity_pass"]) for row in face_rows)
    )
    return {
        "case_id": case_id,
        "checked_face_count": len(face_rows),
        "distributed_runtime_halo_exchange_ready": False,
        "face_rows": face_rows,
        "halo_face_integrity_pass": halo_face_integrity_pass,
        "max_halo_face_linf_error": max_linf_error,
        "schema": "production-decomposition-halo-face-integrity.v1",
        "status": "accepted_local_serial_halo_face_integrity"
        if halo_face_integrity_pass
        else "blocked_local_serial_halo_face_integrity_failed",
    }


def _communication_volume_evidence(
    plan: GKDomainDecompositionPlan, communication_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return fail-closed per-step halo communication volume evidence."""
    rank_rows: list[dict[str, Any]] = []
    total_bytes = 0
    communicating_face_count = 0
    for row in communication_rows:
        face_rows: list[dict[str, Any]] = []
        rank_bytes = 0
        halo_shapes = row["halo_face_payload_shapes"]
        if not isinstance(halo_shapes, dict):
            raise TypeError("halo_face_payload_shapes must be a dictionary")
        for face, shape in halo_shapes.items():
            if shape is None:
                continue
            shape_list = [int(axis) for axis in shape]
            value_count = int(np.prod(shape_list, dtype=np.int64))
            face_bytes = value_count * FLOAT64_BYTES
            communicating_face_count += 1
            rank_bytes += face_bytes
            face_rows.append(
                {
                    "face": face,
                    "payload_bytes": face_bytes,
                    "payload_shape": shape_list,
                    "payload_values": value_count,
                }
            )
        total_bytes += rank_bytes
        rank_rows.append(
            {
                "face_rows": face_rows,
                "rank": int(row["rank"]),
                "rank_halo_exchange_bytes_per_step": rank_bytes,
            }
        )
    max_rank_bytes = (
        max(int(row["rank_halo_exchange_bytes_per_step"]) for row in rank_rows) if rank_rows else 0
    )
    return {
        "bytes_per_float64": FLOAT64_BYTES,
        "communicating_face_count": communicating_face_count,
        "distributed_runtime_ready": False,
        "halo_dtype": "float64",
        "max_rank_halo_exchange_bytes_per_step": max_rank_bytes,
        "rank_count": plan.total_ranks,
        "rank_rows": rank_rows,
        "schema": "production-decomposition-communication-volume.v1",
        "status": "blocked_missing_distributed_runtime_execution",
        "total_halo_exchange_bytes_per_step": total_bytes,
    }


def _shape_list(shape: Any) -> list[int] | None:
    """Return a normalised payload shape list for a halo face."""
    if shape is None:
        return None
    if not isinstance(shape, list):
        raise TypeError("halo payload shape must be a list or None")
    return [int(axis) for axis in shape]


def _payload_bytes(shape: list[int] | None) -> int | None:
    """Return payload bytes for a normalised float64 halo-face shape."""
    if shape is None:
        return None
    return int(np.prod(shape, dtype=np.int64)) * FLOAT64_BYTES


def _reciprocal_neighbour_graph_evidence(
    plan: GKDomainDecompositionPlan, communication_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return reciprocal rank-neighbour and payload-symmetry evidence."""
    rows_by_rank = {int(row["rank"]): row for row in communication_rows}
    link_rows: list[dict[str, Any]] = []
    for row in communication_rows:
        rank = int(row["rank"])
        neighbours = row["neighbour_ranks"]
        halo_shapes = row["halo_face_payload_shapes"]
        if not isinstance(neighbours, dict):
            raise TypeError("neighbour_ranks must be a dictionary")
        if not isinstance(halo_shapes, dict):
            raise TypeError("halo_face_payload_shapes must be a dictionary")
        for face, neighbour_rank in neighbours.items():
            if neighbour_rank is None:
                continue
            opposite_face = FACE_OPPOSITES[str(face)]
            neighbour_row = rows_by_rank.get(int(neighbour_rank))
            reciprocal_neighbour = None
            reciprocal_shape: list[int] | None = None
            if neighbour_row is not None:
                neighbour_neighbours = neighbour_row["neighbour_ranks"]
                neighbour_shapes = neighbour_row["halo_face_payload_shapes"]
                if not isinstance(neighbour_neighbours, dict):
                    raise TypeError("neighbour_ranks must be a dictionary")
                if not isinstance(neighbour_shapes, dict):
                    raise TypeError("halo_face_payload_shapes must be a dictionary")
                reciprocal_neighbour = neighbour_neighbours.get(opposite_face)
                reciprocal_shape = _shape_list(neighbour_shapes.get(opposite_face))
            payload_shape = _shape_list(halo_shapes.get(face))
            payload_bytes = _payload_bytes(payload_shape)
            reciprocal_payload_bytes = _payload_bytes(reciprocal_shape)
            reciprocal_rank_match = reciprocal_neighbour == rank
            payload_shape_match = payload_shape == reciprocal_shape
            payload_byte_match = payload_bytes == reciprocal_payload_bytes
            payload_byte_asymmetry = (
                None
                if payload_bytes is None or reciprocal_payload_bytes is None
                else abs(payload_bytes - reciprocal_payload_bytes)
            )
            link_pass = bool(reciprocal_rank_match and payload_shape_match and payload_byte_match)
            link_rows.append(
                {
                    "face": face,
                    "link_pass": link_pass,
                    "neighbour_rank": int(neighbour_rank),
                    "opposite_face": opposite_face,
                    "payload_byte_asymmetry": payload_byte_asymmetry,
                    "payload_byte_match": payload_byte_match,
                    "payload_bytes": payload_bytes,
                    "payload_shape": payload_shape,
                    "payload_shape_match": payload_shape_match,
                    "rank": rank,
                    "reciprocal_neighbour_rank": reciprocal_neighbour,
                    "reciprocal_payload_bytes": reciprocal_payload_bytes,
                    "reciprocal_payload_shape": reciprocal_shape,
                    "reciprocal_rank_match": reciprocal_rank_match,
                }
            )
    mismatched_link_count = sum(1 for row in link_rows if not bool(row["link_pass"]))
    max_payload_byte_asymmetry = (
        max(int(row["payload_byte_asymmetry"] or 0) for row in link_rows) if link_rows else 0
    )
    reciprocal_neighbour_graph_pass = bool(
        link_rows and len(link_rows) % 2 == 0 and mismatched_link_count == 0
    )
    return {
        "directed_link_count": len(link_rows),
        "link_rows": link_rows,
        "max_payload_byte_asymmetry": max_payload_byte_asymmetry,
        "mismatched_link_count": mismatched_link_count,
        "rank_count": plan.total_ranks,
        "reciprocal_neighbour_graph_pass": reciprocal_neighbour_graph_pass,
        "schema": "production-decomposition-reciprocal-neighbour-graph.v1",
        "status": "accepted_local_reciprocal_neighbour_graph"
        if reciprocal_neighbour_graph_pass
        else "blocked_local_reciprocal_neighbour_graph_failed",
        "undirected_link_count": len(link_rows) // 2,
    }


def _distributed_scaling_gate_evidence(
    communication_volume_evidence: dict[str, Any],
    measured_runs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return fail-closed distributed runtime scaling acceptance evidence."""
    distributed_runs = [] if measured_runs is None else measured_runs
    estimated_halo_bytes = int(communication_volume_evidence["total_halo_exchange_bytes_per_step"])
    return {
        "blocking_reason": (
            "MPI or multi-GPU distributed runtime measurements are required "
            "before production-scale decomposition can be accepted."
        ),
        "distributed_scaling_ready": False,
        "estimated_halo_bytes_per_step": estimated_halo_bytes,
        "measured_run_count": len(distributed_runs),
        "measured_runs": distributed_runs,
        "minimum_parallel_efficiency_threshold": MINIMUM_PARALLEL_EFFICIENCY,
        "minimum_weak_scaling_efficiency_threshold": MINIMUM_WEAK_SCALING_EFFICIENCY,
        "required_measurements": [
            "wall_time_s by rank count for the same physics deck",
            "parallel efficiency relative to the one-rank baseline",
            "weak-scaling efficiency at fixed owned phase cells per rank",
            "hardware metadata for CPU, accelerator, interconnect, and driver stack",
            "decomposition-invariant physics checks for every distributed run",
        ],
        "required_rank_counts": REQUIRED_DISTRIBUTED_RANK_COUNTS,
        "schema": "production-decomposition-distributed-scaling-gate.v1",
        "status": "blocked_missing_distributed_scaling_measurements",
    }


def _load_distributed_measurement_rows() -> list[dict[str, Any]]:
    """Load optional distributed runtime measurement rows from a JSON sidecar."""
    sidecar = os.environ.get(DISTRIBUTED_RUNS_ENV)
    if not sidecar:
        return []
    path = Path(sidecar)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"{DISTRIBUTED_RUNS_ENV} must point to a JSON list")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise TypeError(f"distributed measurement row {index} must be a dictionary")
        rows.append(dict(row))
    return rows


def _distributed_run_acceptance_manifest(
    scaling_gate: dict[str, Any],
) -> dict[str, Any]:
    """Return fail-closed acceptance requirements for future distributed runs."""
    required_rank_counts = [int(rank) for rank in scaling_gate["required_rank_counts"]]
    measured_runs = scaling_gate.get("measured_runs", [])
    if not isinstance(measured_runs, list):
        raise TypeError("measured_runs must be a list")
    accepted_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for run in measured_runs:
        if not isinstance(run, dict):
            raise TypeError("distributed run rows must be dictionaries")
        missing_fields = [field for field in REQUIRED_DISTRIBUTED_RUN_FIELDS if field not in run]
        rank_count = int(run.get("rank_count", -1))
        parallel_efficiency = float(run.get("parallel_efficiency", 0.0))
        weak_scaling_efficiency = float(run.get("weak_scaling_efficiency", 0.0))
        invariant_pass = bool(run.get("decomposition_invariant_pass", False))
        row_pass = bool(
            not missing_fields
            and rank_count in required_rank_counts
            and float(run["wall_time_s"]) > 0.0
            and parallel_efficiency >= MINIMUM_PARALLEL_EFFICIENCY
            and weak_scaling_efficiency >= MINIMUM_WEAK_SCALING_EFFICIENCY
            and invariant_pass
        )
        row = {
            "missing_fields": missing_fields,
            "rank_count": rank_count,
            "run_acceptance_pass": row_pass,
        }
        candidate_rows.append(row)
        if row_pass:
            accepted_rows.append(row)
    accepted_rank_counts = sorted({int(row["rank_count"]) for row in accepted_rows})
    missing_rank_counts = [
        rank for rank in required_rank_counts if rank not in accepted_rank_counts
    ]
    acceptance_ready = bool(
        not missing_rank_counts and len(accepted_rank_counts) == len(required_rank_counts)
    )
    return {
        "accepted_rank_counts": accepted_rank_counts,
        "accepted_run_count": len(accepted_rows),
        "candidate_run_count": len(candidate_rows),
        "candidate_rows": candidate_rows,
        "distributed_run_acceptance_ready": acceptance_ready,
        "estimated_halo_bytes_per_step": int(scaling_gate["estimated_halo_bytes_per_step"]),
        "minimum_parallel_efficiency_threshold": MINIMUM_PARALLEL_EFFICIENCY,
        "minimum_weak_scaling_efficiency_threshold": MINIMUM_WEAK_SCALING_EFFICIENCY,
        "missing_rank_counts": missing_rank_counts,
        "required_fields": REQUIRED_DISTRIBUTED_RUN_FIELDS,
        "required_rank_counts": required_rank_counts,
        "schema": "production-decomposition-distributed-run-acceptance.v1",
        "status": "accepted_distributed_run_measurements"
        if acceptance_ready
        else "blocked_no_distributed_measurement_rows"
        if not candidate_rows
        else "blocked_incomplete_distributed_measurement_rows",
    }


def _hardware_metadata() -> dict[str, Any]:
    return {
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }


def run_benchmark() -> dict[str, Any]:
    """Return decomposition-contract evidence while keeping scaling fail-closed."""
    medium_plan = build_radial_toroidal_decomposition(
        n_radial=64,
        n_toroidal=32,
        n_theta=16,
        n_vpar=16,
        n_mu=8,
        radial_parts=4,
        toroidal_parts=2,
        halo=1,
    )
    production_plan = build_radial_toroidal_decomposition(
        n_radial=256,
        n_toroidal=128,
        n_theta=32,
        n_vpar=32,
        n_mu=16,
        radial_parts=8,
        toroidal_parts=4,
        halo=1,
    )
    local_cpu_plan = build_radial_toroidal_decomposition(
        n_radial=64,
        n_toroidal=32,
        n_theta=8,
        n_vpar=8,
        n_mu=4,
        radial_parts=4,
        toroidal_parts=2,
        halo=1,
    )
    cases = [
        _plan_row("medium_64x32_4x2", medium_plan),
        _plan_row("production_256x128_8x4", production_plan),
    ]
    communication_contract_rows = rank_tile_communication_contract(production_plan)
    communication_volume_evidence = _communication_volume_evidence(
        production_plan, communication_contract_rows
    )
    distributed_measurement_rows = _load_distributed_measurement_rows()
    reciprocal_neighbour_graph_evidence = _reciprocal_neighbour_graph_evidence(
        production_plan, communication_contract_rows
    )
    distributed_scaling_gate_evidence = _distributed_scaling_gate_evidence(
        communication_volume_evidence,
        distributed_measurement_rows,
    )
    distributed_run_acceptance_manifest = _distributed_run_acceptance_manifest(
        distributed_scaling_gate_evidence
    )
    local_cpu_shape_variant_plan = build_radial_toroidal_decomposition(
        n_radial=64,
        n_toroidal=32,
        n_theta=8,
        n_vpar=8,
        n_mu=4,
        radial_parts=8,
        toroidal_parts=1,
        halo=1,
    )
    local_cpu_toroidal_variant_plan = build_radial_toroidal_decomposition(
        n_radial=64,
        n_toroidal=32,
        n_theta=8,
        n_vpar=8,
        n_mu=4,
        radial_parts=2,
        toroidal_parts=4,
        halo=1,
    )
    cpu_benchmark_rows = [
        _cpu_benchmark_row("local_cpu_64x32_4x2", local_cpu_plan),
        _cpu_benchmark_row("local_cpu_64x32_8x1", local_cpu_shape_variant_plan),
        _cpu_benchmark_row("local_cpu_64x32_2x4", local_cpu_toroidal_variant_plan),
    ]
    halo_face_integrity_evidence = _halo_face_integrity_evidence(
        "local_cpu_64x32_4x2", local_cpu_plan
    )
    same_physics_shape_convergence = _shape_convergence_evidence(cpu_benchmark_rows)
    large_grid_cpu_evidence = _large_grid_cpu_evidence()
    local_multiprocess_cpu_evidence = _local_multiprocess_cpu_evidence()
    runtime_dependency_evidence = _runtime_dependency_evidence()
    mpi_runtime_evidence = _mpi_runtime_evidence()
    gpu_rank_tile_evidence = _gpu_rank_tile_evidence()
    coverage_pass = all(case["min_owned_cells"] > 0 for case in cases)
    imbalance_pass = all(case["owned_cell_imbalance"] <= 1.05 for case in cases)
    communication_contract_ready = all(
        bool(row["communication_contract_ready"]) for row in communication_contract_rows
    )
    halo_exchange_pass = all(bool(row["halo_exchange_pass"]) for row in cpu_benchmark_rows)
    local_decomposed_execution_pass = all(
        bool(row["local_decomposed_execution_pass"]) for row in cpu_benchmark_rows
    )
    decomposition_invariant_pass = all(
        row["reconstruction_linf_error"] == 0.0
        and row["inventory_relative_error"] <= RELATIVE_REDUCTION_TOLERANCE
        and row["free_energy_relative_error"] <= RELATIVE_REDUCTION_TOLERANCE
        and row["parallel_moment_relative_error"] <= RELATIVE_REDUCTION_TOLERANCE
        for row in cpu_benchmark_rows
    )
    parallel_moment_invariant_pass = all(
        row["parallel_moment_relative_error"] <= RELATIVE_REDUCTION_TOLERANCE
        for row in cpu_benchmark_rows
    )
    same_physics_decomposition_shape_pass = bool(
        same_physics_shape_convergence["shape_convergence_pass"]
    )
    reciprocal_neighbour_graph_pass = bool(
        reciprocal_neighbour_graph_evidence["reciprocal_neighbour_graph_pass"]
    )
    contract_pass = bool(
        coverage_pass
        and imbalance_pass
        and communication_contract_ready
        and reciprocal_neighbour_graph_pass
        and halo_face_integrity_evidence["halo_face_integrity_pass"]
        and local_decomposed_execution_pass
        and halo_exchange_pass
        and decomposition_invariant_pass
        and parallel_moment_invariant_pass
        and same_physics_decomposition_shape_pass
        and local_multiprocess_cpu_evidence["local_multiprocess_cpu_execution_ready"]
    )
    return {
        "benchmark": "production_decomposition_contract",
        "schema": "production-decomposition-contract.v1",
        "description": (
            "Deterministic radial/toroidal decomposition contract for production-scale "
            "5D nonlinear GK scheduling with executable local rank-tile evidence. "
            "This is not distributed MPI or multi-GPU scaling evidence."
        ),
        "cases": cases,
        "communication_contract_ready": communication_contract_ready,
        "communication_contract_rows": communication_contract_rows,
        "contract_pass": contract_pass,
        "coverage_pass": coverage_pass,
        "cpu_benchmark_rows": cpu_benchmark_rows,
        "decomposition_invariant_pass": decomposition_invariant_pass,
        "distributed_communication_volume_evidence": communication_volume_evidence,
        "distributed_run_acceptance_manifest": distributed_run_acceptance_manifest,
        "distributed_scaling_gate_evidence": distributed_scaling_gate_evidence,
        "halo_face_integrity_evidence": halo_face_integrity_evidence,
        "halo_exchange_pass": halo_exchange_pass,
        "hardware_metadata": _hardware_metadata(),
        "imbalance_pass": imbalance_pass,
        "large_grid_cpu_evidence": large_grid_cpu_evidence,
        "local_decomposed_execution_pass": local_decomposed_execution_pass,
        "local_multiprocess_cpu_evidence": local_multiprocess_cpu_evidence,
        "local_multiprocess_cpu_execution_pass": bool(
            local_multiprocess_cpu_evidence["local_multiprocess_cpu_execution_ready"]
        ),
        "gpu_rank_tile_evidence": gpu_rank_tile_evidence,
        "gpu_rank_tile_execution_pass": bool(
            gpu_rank_tile_evidence["gpu_rank_tile_execution_ready"]
        ),
        "mpi_runtime_evidence": mpi_runtime_evidence,
        "mpi_runtime_execution_pass": bool(mpi_runtime_evidence["mpi_runtime_execution_ready"]),
        "parallel_moment_invariant_pass": parallel_moment_invariant_pass,
        "production_scale_ready": False,
        "reciprocal_neighbour_graph_evidence": reciprocal_neighbour_graph_evidence,
        "reciprocal_neighbour_graph_pass": reciprocal_neighbour_graph_pass,
        "reproducible_commands": [
            "python validation/benchmark_production_decomposition_contract.py",
            "python -m pytest tests/test_gk_domain_decomposition.py -q",
        ],
        "runtime_dependency_evidence": runtime_dependency_evidence,
        "same_physics_decomposition_shape_pass": same_physics_decomposition_shape_pass,
        "same_physics_shape_convergence_evidence": same_physics_shape_convergence,
        "status": "blocked_local_decomposition_ready_missing_distributed_runtime_scaling",
        "missing_requirements": [
            "cluster MPI scaling report over the declared rank tiles",
            "multi-GPU distributed execution path over the declared rank tiles",
            "large-grid cluster/GPU wall-time scaling report",
            "same-physics convergence evidence across distributed MPI/multi-GPU decomposition shapes",
            "hardware-specific multi-rank throughput and efficiency thresholds",
            "accepted distributed scaling gate over required rank counts",
            "accepted distributed run manifests with reproducibility fields and checksums",
        ],
    }


def write_reports(report: dict[str, Any]) -> None:
    """Write JSON and Markdown decomposition reports."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Production Decomposition Contract",
        "",
        report["description"],
        "",
        f"- Schema: `{report['schema']}`",
        f"- Status: `{report['status']}`",
        f"- Contract pass: `{report['contract_pass']}`",
        f"- Communication contract ready: `{report['communication_contract_ready']}`",
        (
            "- Halo-face integrity pass: "
            f"`{report['halo_face_integrity_evidence']['halo_face_integrity_pass']}`"
        ),
        f"- Local decomposed execution pass: `{report['local_decomposed_execution_pass']}`",
        (
            "- Local multiprocess CPU execution pass: "
            f"`{report['local_multiprocess_cpu_execution_pass']}`"
        ),
        f"- MPI runtime execution pass: `{report['mpi_runtime_execution_pass']}`",
        f"- GPU rank-tile execution pass: `{report['gpu_rank_tile_execution_pass']}`",
        f"- Halo exchange pass: `{report['halo_exchange_pass']}`",
        f"- Decomposition invariant pass: `{report['decomposition_invariant_pass']}`",
        f"- Parallel-moment invariant pass: `{report['parallel_moment_invariant_pass']}`",
        f"- Reciprocal neighbour graph pass: `{report['reciprocal_neighbour_graph_pass']}`",
        f"- Same-physics decomposition shape pass: `{report['same_physics_decomposition_shape_pass']}`",
        f"- Production-scale ready: `{report['production_scale_ready']}`",
        f"- Python: `{report['hardware_metadata']['python_version']}`",
        f"- CPU count: `{report['hardware_metadata']['cpu_count']}`",
        "",
        "| Case | R x T | Parts | Ranks | Imbalance | Halo overhead |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        lines.append(
            "| {case_id} | {nr} x {nt} | {rp} x {tp} | {ranks} | {imb:.6f} | {halo:.6f} |".format(
                case_id=case["case_id"],
                nr=case["n_radial"],
                nt=case["n_toroidal"],
                rp=case["radial_parts"],
                tp=case["toroidal_parts"],
                ranks=case["total_ranks"],
                imb=case["owned_cell_imbalance"],
                halo=case["halo_overhead_ratio"],
            )
        )
    dependency_evidence = report["runtime_dependency_evidence"]
    lines.extend(
        [
            "",
            "## Runtime dependency evidence",
            "",
            f"- Schema: `{dependency_evidence['schema']}`",
            f"- Status: `{dependency_evidence['status']}`",
            (
                "- Optional runtime dependency ready: "
                f"`{dependency_evidence['optional_runtime_dependency_ready']}`"
            ),
            f"- NumPy contract pass: `{dependency_evidence['numpy_contract_pass']}`",
            f"- Python executable: `{dependency_evidence['python_executable']}`",
            "",
            "| Module | Distribution | Required specifier | Importable | Version |",
            "|---|---|---|:---:|---|",
        ]
    )
    for row in dependency_evidence["rows"]:
        lines.append(
            "| {module} | `{distribution}` | `{specifier}` | `{importable}` | `{version}` |".format(
                distribution=row["distribution"],
                importable=row["importable"],
                module=row["module"],
                specifier=row["required_specifier"],
                version=row["version"],
            )
        )
    halo_evidence = report["halo_face_integrity_evidence"]
    lines.extend(
        [
            "",
            "## Local serial halo-face integrity",
            "",
            f"- Schema: `{halo_evidence['schema']}`",
            f"- Status: `{halo_evidence['status']}`",
            f"- Case: `{halo_evidence['case_id']}`",
            f"- Checked faces: `{halo_evidence['checked_face_count']}`",
            f"- Max halo-face L_inf error: `{halo_evidence['max_halo_face_linf_error']:.6e}`",
            (
                "- Distributed runtime halo exchange ready: "
                f"`{halo_evidence['distributed_runtime_halo_exchange_ready']}`"
            ),
            "",
            "| Rank | Face | Neighbour | Shape | L_inf | Pass |",
            "|---:|---|---:|---|---:|:---:|",
        ]
    )
    for row in halo_evidence["face_rows"]:
        lines.append(
            "| {rank} | {face} | {neighbour} | `{shape}` | {linf:.6e} | `{passes}` |".format(
                face=row["face"],
                linf=row["linf_error"],
                neighbour=row["neighbour_rank"],
                passes=row["face_integrity_pass"],
                rank=row["rank"],
                shape=json.dumps(row["face_payload_shape"]),
            )
        )
    lines.extend(
        [
            "",
            "## Rank communication contract",
            "",
            "| Rank | Neighbours | Halo face shapes ready |",
            "|---:|---|:---:|",
        ]
    )
    for row in report["communication_contract_rows"]:
        neighbours = ", ".join(f"{face}={rank}" for face, rank in row["neighbour_ranks"].items())
        lines.append(
            "| {rank} | {neighbours} | `{ready}` |".format(
                neighbours=neighbours,
                rank=row["rank"],
                ready=row["communication_contract_ready"],
            )
        )
    volume = report["distributed_communication_volume_evidence"]
    lines.extend(
        [
            "",
            "## Distributed communication volume evidence",
            "",
            f"- Schema: `{volume['schema']}`",
            f"- Status: `{volume['status']}`",
            f"- Distributed runtime ready: `{volume['distributed_runtime_ready']}`",
            f"- Halo dtype: `{volume['halo_dtype']}`",
            f"- Communicating faces: `{volume['communicating_face_count']}`",
            (
                "- Total halo-exchange bytes per step: "
                f"`{volume['total_halo_exchange_bytes_per_step']}`"
            ),
            (
                "- Max rank halo-exchange bytes per step: "
                f"`{volume['max_rank_halo_exchange_bytes_per_step']}`"
            ),
            "",
            "| Rank | Bytes/step | Communicating faces |",
            "|---:|---:|---:|",
        ]
    )
    for row in volume["rank_rows"]:
        lines.append(
            "| {rank} | {bytes_per_step} | {faces} |".format(
                bytes_per_step=row["rank_halo_exchange_bytes_per_step"],
                faces=len(row["face_rows"]),
                rank=row["rank"],
            )
        )
    reciprocal = report["reciprocal_neighbour_graph_evidence"]
    lines.extend(
        [
            "",
            "## Reciprocal neighbour graph evidence",
            "",
            f"- Schema: `{reciprocal['schema']}`",
            f"- Status: `{reciprocal['status']}`",
            (
                "- Reciprocal neighbour graph pass: "
                f"`{reciprocal['reciprocal_neighbour_graph_pass']}`"
            ),
            f"- Directed links: `{reciprocal['directed_link_count']}`",
            f"- Undirected links: `{reciprocal['undirected_link_count']}`",
            f"- Mismatched links: `{reciprocal['mismatched_link_count']}`",
            f"- Max payload byte asymmetry: `{reciprocal['max_payload_byte_asymmetry']}`",
            "",
            "| Rank | Face | Neighbour | Opposite face | Payload bytes | Reciprocal bytes | Pass |",
            "|---:|---|---:|---|---:|---:|:---:|",
        ]
    )
    for row in reciprocal["link_rows"]:
        lines.append(
            "| {rank} | {face} | {neighbour} | {opposite} | {payload} | {reciprocal_payload} | `{passes}` |".format(
                face=row["face"],
                neighbour=row["neighbour_rank"],
                opposite=row["opposite_face"],
                passes=row["link_pass"],
                payload=row["payload_bytes"],
                rank=row["rank"],
                reciprocal_payload=row["reciprocal_payload_bytes"],
            )
        )
    scaling = report["distributed_scaling_gate_evidence"]
    lines.extend(
        [
            "",
            "## Distributed scaling gate",
            "",
            f"- Schema: `{scaling['schema']}`",
            f"- Status: `{scaling['status']}`",
            f"- Distributed scaling ready: `{scaling['distributed_scaling_ready']}`",
            f"- Measured run count: `{scaling['measured_run_count']}`",
            (f"- Required rank counts: `{json.dumps(scaling['required_rank_counts'])}`"),
            (
                "- Minimum parallel efficiency threshold: "
                f"`{scaling['minimum_parallel_efficiency_threshold']:.2f}`"
            ),
            (
                "- Minimum weak-scaling efficiency threshold: "
                f"`{scaling['minimum_weak_scaling_efficiency_threshold']:.2f}`"
            ),
            (f"- Estimated halo bytes per step: `{scaling['estimated_halo_bytes_per_step']}`"),
            f"- Blocking reason: {scaling['blocking_reason']}",
            "",
            "Required measurements:",
        ]
    )
    lines.extend(f"- {item}" for item in scaling["required_measurements"])
    manifest = report["distributed_run_acceptance_manifest"]
    lines.extend(
        [
            "",
            "## Distributed run acceptance manifest",
            "",
            f"- Schema: `{manifest['schema']}`",
            f"- Status: `{manifest['status']}`",
            (
                "- Distributed run acceptance ready: "
                f"`{manifest['distributed_run_acceptance_ready']}`"
            ),
            f"- Candidate run count: `{manifest['candidate_run_count']}`",
            f"- Accepted run count: `{manifest['accepted_run_count']}`",
            (f"- Required rank counts: `{json.dumps(manifest['required_rank_counts'])}`"),
            (f"- Missing rank counts: `{json.dumps(manifest['missing_rank_counts'])}`"),
            (f"- Estimated halo bytes per step: `{manifest['estimated_halo_bytes_per_step']}`"),
            "",
            "Required distributed-run fields:",
        ]
    )
    lines.extend(f"- `{item}`" for item in manifest["required_fields"])
    lines.extend(
        [
            "",
            "## Local CPU halo/invariant benchmark",
            "",
            "| Case | Ranks | Owned phase cells | Elapsed s | Cells/s | Local execution | Halo | Reconstruction L_inf | Inventory rel | Free-energy rel |",
            "|---|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|",
        ]
    )
    for row in report["cpu_benchmark_rows"]:
        lines.append(
            "| {case_id} | {ranks} | {cells} | {elapsed:.6e} | {rate:.6e} | `{local}` | `{halo}` | {recon:.6e} | {inventory:.6e} | {energy:.6e} |".format(
                case_id=row["case_id"],
                cells=row["owned_phase_cells"],
                elapsed=row["elapsed_s"],
                energy=row["free_energy_relative_error"],
                halo=row["halo_exchange_pass"],
                inventory=row["inventory_relative_error"],
                local=row["local_decomposed_execution_pass"],
                ranks=row["rank_count"],
                rate=row["cells_per_second"],
                recon=row["reconstruction_linf_error"],
            )
        )
    multiprocess = report["local_multiprocess_cpu_evidence"]
    lines.extend(
        [
            "",
            "## Local multiprocess CPU rank execution",
            "",
            f"- Schema: `{multiprocess['schema']}`",
            f"- Status: `{multiprocess['status']}`",
            f"- Case: `{multiprocess['case_id']}`",
            (
                "- Local multiprocess CPU execution ready: "
                f"`{multiprocess['local_multiprocess_cpu_execution_ready']}`"
            ),
            f"- Worker count: `{multiprocess['worker_count']}`",
            f"- Unique worker process count: `{multiprocess['unique_worker_process_count']}`",
            f"- Rank count: `{multiprocess['rank_count']}`",
            f"- Owned phase cells: `{multiprocess['owned_phase_cells']}`",
            f"- Elapsed s: `{multiprocess['elapsed_s']:.6e}`",
            f"- Cells/s: `{multiprocess['cells_per_second']:.6e}`",
            (f"- Reconstruction L_inf error: `{multiprocess['reconstruction_linf_error']:.6e}`"),
            (f"- Inventory relative error: `{multiprocess['inventory_relative_error']:.6e}`"),
            (f"- Free-energy relative error: `{multiprocess['free_energy_relative_error']:.6e}`"),
            (
                "- Parallel-moment relative error: "
                f"`{multiprocess['parallel_moment_relative_error']:.6e}`"
            ),
            (
                "- Halo-checksum relative error: "
                f"`{multiprocess['halo_checksum_relative_error']:.6e}`"
            ),
            f"- Blocking reason: {multiprocess['blocking_reason']}",
            "",
            "| Rank | PID | Owned shape | Halo shape |",
            "|---:|---:|---|---|",
        ]
    )
    for row in multiprocess["rank_rows"]:
        lines.append(
            "| {rank} | {pid} | `{owned}` | `{halo}` |".format(
                halo=json.dumps(row["halo_shape"]),
                owned=json.dumps(row["owned_shape"]),
                pid=row["pid"],
                rank=row["rank"],
            )
        )
    mpi_runtime = report["mpi_runtime_evidence"]
    lines.extend(
        [
            "",
            "## MPI runtime rank execution",
            "",
            f"- Schema: `{mpi_runtime['schema']}`",
            f"- Status: `{mpi_runtime['status']}`",
            f"- MPI runtime execution ready: `{mpi_runtime['mpi_runtime_execution_ready']}`",
            f"- Rank count: `{mpi_runtime['rank_count']}`",
            f"- Blocking reason: {mpi_runtime['blocking_reason']}",
        ]
    )
    if mpi_runtime.get("mpi_runtime_execution_ready"):
        lines.extend(
            [
                f"- Elapsed s: `{mpi_runtime['elapsed_s']:.6e}`",
                f"- Topology: `{mpi_runtime['topology']}`",
                f"- Radial parts: `{mpi_runtime['radial_parts']}`",
                f"- Toroidal parts: `{mpi_runtime['toroidal_parts']}`",
                (
                    "- Minimum halo verified fraction: "
                    f"`{mpi_runtime['min_halo_verified_fraction']:.6e}`"
                ),
                f"- Reconstruction L_inf error: `{mpi_runtime['reconstruction_linf_error']:.6e}`",
                f"- Inventory relative error: `{mpi_runtime['inventory_relative_error']:.6e}`",
                f"- Free-energy relative error: `{mpi_runtime['free_energy_relative_error']:.6e}`",
                (
                    "- Parallel-moment relative error: "
                    f"`{mpi_runtime['parallel_moment_relative_error']:.6e}`"
                ),
                "",
                "| Rank | Owned shape | Face neighbours | Corner neighbours | Halo verified fraction | Halo L_inf |",
                "|---:|---|---|---|---:|---:|",
            ]
        )
        for row in mpi_runtime["rank_rows"]:
            lines.append(
                "| {rank} | `{shape}` | `{neighbours}` | `{corners}` | {fraction:.6e} | {halo:.6e} |".format(
                    corners=json.dumps(row["corner_neighbour_ranks"], sort_keys=True),
                    halo=row["halo_linf_error"],
                    fraction=row["halo_verified_fraction"],
                    neighbours=json.dumps(row["neighbour_ranks"], sort_keys=True),
                    rank=row["rank"],
                    shape=json.dumps(row["owned_shape"]),
                )
            )
    gpu_evidence = report["gpu_rank_tile_evidence"]
    lines.extend(
        [
            "",
            "## GPU rank-tile execution",
            "",
            f"- Schema: `{gpu_evidence['schema']}`",
            f"- Status: `{gpu_evidence['status']}`",
            (f"- GPU rank-tile execution ready: `{gpu_evidence['gpu_rank_tile_execution_ready']}`"),
            f"- Multi-GPU runtime ready: `{gpu_evidence['multi_gpu_runtime_ready']}`",
            f"- Blocking reason: {gpu_evidence['blocking_reason']}",
        ]
    )
    if gpu_evidence.get("gpu_rank_tile_execution_ready"):
        lines.extend(
            [
                f"- Device count: `{gpu_evidence['device_count']}`",
                f"- Rank count: `{gpu_evidence['rank_count']}`",
                f"- Owned phase cells: `{gpu_evidence['owned_phase_cells']}`",
                f"- Elapsed s: `{gpu_evidence['elapsed_s']:.6e}`",
                f"- Cells/s: `{gpu_evidence['cells_per_second']:.6e}`",
                f"- Inventory relative error: `{gpu_evidence['inventory_relative_error']:.6e}`",
                f"- Free-energy relative error: `{gpu_evidence['free_energy_relative_error']:.6e}`",
                (
                    "- Parallel-moment relative error: "
                    f"`{gpu_evidence['parallel_moment_relative_error']:.6e}`"
                ),
                "",
                "| Rank | Device | Owned shape |",
                "|---:|---:|---|",
            ]
        )
        for row in gpu_evidence["rank_rows"]:
            lines.append(
                "| {rank} | {device} | `{shape}` |".format(
                    device=row["device_id"],
                    rank=row["rank"],
                    shape=json.dumps(row["owned_shape"]),
                )
            )
    large_grid = report["large_grid_cpu_evidence"]
    lines.extend(
        [
            "",
            "## Large-grid CPU decomposition evidence",
            "",
            f"- Schema: `{large_grid['schema']}`",
            f"- Status: `{large_grid['status']}`",
            (f"- Large-grid CPU benchmark ready: `{large_grid['large_grid_cpu_benchmark_ready']}`"),
            (
                "- Max reconstruction L_inf error: "
                f"`{large_grid['max_reconstruction_linf_error']:.6e}`"
            ),
            (
                "- Max parallel-moment relative error: "
                f"`{large_grid['max_parallel_moment_relative_error']:.6e}`"
            ),
            f"- Blocking reason: {large_grid['blocking_reason']}",
            "",
            "| Case | Ranks | Owned phase cells | Elapsed s | Cells/s | Local execution | Halo | Reconstruction L_inf | Inventory rel | Free-energy rel |",
            "|---|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|",
        ]
    )
    for row in large_grid["rows"]:
        lines.append(
            "| {case_id} | {ranks} | {cells} | {elapsed:.6e} | {rate:.6e} | `{local}` | `{halo}` | {recon:.6e} | {inventory:.6e} | {energy:.6e} |".format(
                case_id=row["case_id"],
                cells=row["owned_phase_cells"],
                elapsed=row["elapsed_s"],
                energy=row["free_energy_relative_error"],
                halo=row["halo_exchange_pass"],
                inventory=row["inventory_relative_error"],
                local=row["local_decomposed_execution_pass"],
                ranks=row["rank_count"],
                rate=row["cells_per_second"],
                recon=row["reconstruction_linf_error"],
            )
        )
    shape_evidence = report["same_physics_shape_convergence_evidence"]
    lines.extend(
        [
            "",
            "## Same-physics decomposition-shape convergence",
            "",
            f"- Schema: `{shape_evidence['schema']}`",
            f"- Status: `{shape_evidence['status']}`",
            f"- Shape convergence pass: `{shape_evidence['shape_convergence_pass']}`",
            f"- Reference case: `{shape_evidence['reference_case_id']}`",
            (
                "- Max inventory relative deviation: "
                f"`{shape_evidence['max_inventory_relative_deviation']:.6e}`"
            ),
            (
                "- Max free-energy relative deviation: "
                f"`{shape_evidence['max_free_energy_relative_deviation']:.6e}`"
            ),
            (
                "- Max parallel-moment relative deviation: "
                f"`{shape_evidence['max_parallel_moment_relative_deviation']:.6e}`"
            ),
            (
                "- Relative reduction tolerance: "
                f"`{shape_evidence['relative_reduction_tolerance']:.6e}`"
            ),
            "",
            "| Case | Ranks | Owned phase cells | Cells/s | Inventory rel dev | Free-energy rel dev | Parallel-moment rel dev | Reconstruction L_inf | Pass |",
            "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in shape_evidence["shape_rows"]:
        lines.append(
            "| {case_id} | {ranks} | {cells} | {rate:.6e} | {inventory:.6e} | {energy:.6e} | {moment:.6e} | {recon:.6e} | `{passes}` |".format(
                case_id=row["case_id"],
                cells=row["owned_phase_cells"],
                energy=row["free_energy_relative_deviation_from_reference"],
                inventory=row["inventory_relative_deviation_from_reference"],
                moment=row["parallel_moment_relative_deviation_from_reference"],
                passes=row["shape_convergence_pass"],
                ranks=row["rank_count"],
                rate=row["cells_per_second"],
                recon=row["reconstruction_linf_error"],
            )
        )
    lines.extend(["", "## Reproducible commands", ""])
    lines.extend(f"- `{item}`" for item in report["reproducible_commands"])
    lines.extend(["", "## Missing requirements", ""])
    lines.extend(f"- {item}" for item in report["missing_requirements"])
    lines.append("")
    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    report = run_benchmark()
    write_reports(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
