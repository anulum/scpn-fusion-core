#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Fail-closed production-scale decomposition contract benchmark."""

from __future__ import annotations

import json
import os
import platform
import sys
import time
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
                "parallel_moment_relative_deviation_from_reference": (
                    parallel_moment_deviation
                ),
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
    max_rank_bytes = max(
        int(row["rank_halo_exchange_bytes_per_step"]) for row in rank_rows
    ) if rank_rows else 0
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
            link_pass = bool(
                reciprocal_rank_match and payload_shape_match and payload_byte_match
            )
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
    max_payload_byte_asymmetry = max(
        int(row["payload_byte_asymmetry"] or 0) for row in link_rows
    ) if link_rows else 0
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
    estimated_halo_bytes = int(
        communication_volume_evidence["total_halo_exchange_bytes_per_step"]
    )
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
        "parallel_moment_invariant_pass": parallel_moment_invariant_pass,
        "production_scale_ready": False,
        "reciprocal_neighbour_graph_evidence": reciprocal_neighbour_graph_evidence,
        "reciprocal_neighbour_graph_pass": reciprocal_neighbour_graph_pass,
        "reproducible_commands": [
            "python validation/benchmark_production_decomposition_contract.py",
            "python -m pytest tests/test_gk_domain_decomposition.py -q",
        ],
        "same_physics_decomposition_shape_pass": same_physics_decomposition_shape_pass,
        "same_physics_shape_convergence_evidence": same_physics_shape_convergence,
        "status": "blocked_local_decomposition_ready_missing_distributed_runtime_scaling",
        "missing_requirements": [
            "MPI or multi-GPU distributed execution path over the declared rank tiles",
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
            (
                "- Required rank counts: "
                f"`{json.dumps(scaling['required_rank_counts'])}`"
            ),
            (
                "- Minimum parallel efficiency threshold: "
                f"`{scaling['minimum_parallel_efficiency_threshold']:.2f}`"
            ),
            (
                "- Minimum weak-scaling efficiency threshold: "
                f"`{scaling['minimum_weak_scaling_efficiency_threshold']:.2f}`"
            ),
            (
                "- Estimated halo bytes per step: "
                f"`{scaling['estimated_halo_bytes_per_step']}`"
            ),
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
            (
                "- Required rank counts: "
                f"`{json.dumps(manifest['required_rank_counts'])}`"
            ),
            (
                "- Missing rank counts: "
                f"`{json.dumps(manifest['missing_rank_counts'])}`"
            ),
            (
                "- Estimated halo bytes per step: "
                f"`{manifest['estimated_halo_bytes_per_step']}`"
            ),
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
    large_grid = report["large_grid_cpu_evidence"]
    lines.extend(
        [
            "",
            "## Large-grid CPU decomposition evidence",
            "",
            f"- Schema: `{large_grid['schema']}`",
            f"- Status: `{large_grid['status']}`",
            (
                "- Large-grid CPU benchmark ready: "
                f"`{large_grid['large_grid_cpu_benchmark_ready']}`"
            ),
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
