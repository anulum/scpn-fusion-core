#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for radial/toroidal nonlinear GK decomposition contracts."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_fusion.core.gk_domain_decomposition import (
    build_radial_toroidal_decomposition,
    decomposition_invariant_metrics,
    local_decomposed_phase_execution,
    rank_tile_communication_contract,
    reconstruct_owned_phase_state,
    serial_halo_exchange,
)
from validation.benchmark_production_decomposition_contract import run_benchmark, write_reports


def test_radial_toroidal_decomposition_covers_domain_once() -> None:
    plan = build_radial_toroidal_decomposition(
        n_radial=17,
        n_toroidal=9,
        n_theta=8,
        n_vpar=6,
        n_mu=4,
        radial_parts=4,
        toroidal_parts=3,
        halo=1,
    )

    assert plan.total_ranks == 12
    assert plan.total_owned_phase_cells == 17 * 9 * 8 * 6 * 4
    assert plan.owned_cell_imbalance <= 1.6
    assert plan.halo_overhead_ratio > 1.0
    plan.validate()


def test_decomposition_rejects_invalid_partition() -> None:
    with pytest.raises(ValueError, match="parts must not exceed axis size"):
        build_radial_toroidal_decomposition(
            n_radial=4,
            n_toroidal=4,
            n_theta=8,
            n_vpar=6,
            n_mu=4,
            radial_parts=5,
            toroidal_parts=1,
        )


def test_serial_halo_exchange_reconstructs_owned_5d_state_and_invariants() -> None:
    plan = build_radial_toroidal_decomposition(
        n_radial=17,
        n_toroidal=9,
        n_theta=4,
        n_vpar=3,
        n_mu=2,
        radial_parts=4,
        toroidal_parts=3,
        halo=1,
    )
    state: NDArray[np.float64] = np.arange(plan.total_owned_phase_cells, dtype=np.float64).reshape(
        plan.n_radial,
        plan.n_toroidal,
        plan.n_theta,
        plan.n_vpar,
        plan.n_mu,
    )

    local_tiles = serial_halo_exchange(plan, state)
    reconstructed = reconstruct_owned_phase_state(plan, local_tiles)
    metrics = decomposition_invariant_metrics(plan, state)

    assert len(local_tiles) == plan.total_ranks
    for local in local_tiles:
        tile = plan.tiles[local.rank]
        assert local.owned.shape[:2] == (tile.radial.size, tile.toroidal.size)
        assert local.with_halo.shape[:2] == (
            tile.radial_with_halo.size,
            tile.toroidal_with_halo.size,
        )
        np.testing.assert_allclose(
            local.with_halo,
            state[
                tile.radial_with_halo.start : tile.radial_with_halo.stop,
                tile.toroidal_with_halo.start : tile.toroidal_with_halo.stop,
            ],
        )
    np.testing.assert_allclose(reconstructed, state)
    assert metrics.halo_exchange_pass
    assert metrics.reconstruction_linf_error == 0.0
    assert metrics.inventory_relative_error == 0.0
    assert metrics.free_energy_relative_error == 0.0
    assert metrics.parallel_moment_relative_error == 0.0


def test_rank_tile_communication_contract_declares_neighbour_faces() -> None:
    plan = build_radial_toroidal_decomposition(
        n_radial=8,
        n_toroidal=6,
        n_theta=4,
        n_vpar=3,
        n_mu=2,
        radial_parts=2,
        toroidal_parts=3,
        halo=1,
    )

    rows = rank_tile_communication_contract(plan)

    assert len(rows) == plan.total_ranks
    centre = next(row for row in rows if row["rank"] == 1)
    assert centre["neighbour_ranks"] == {
        "radial_lower": None,
        "radial_upper": 4,
        "toroidal_lower": 0,
        "toroidal_upper": 2,
    }
    assert centre["halo_face_payload_shapes"] == {
        "radial_lower": None,
        "radial_upper": [plan.halo, 2, 4, 3, 2],
        "toroidal_lower": [4, plan.halo, 4, 3, 2],
        "toroidal_upper": [4, plan.halo, 4, 3, 2],
    }
    assert all(row["communication_contract_ready"] for row in rows)


def test_local_decomposed_phase_execution_matches_monolithic_reductions() -> None:
    plan = build_radial_toroidal_decomposition(
        n_radial=12,
        n_toroidal=10,
        n_theta=4,
        n_vpar=3,
        n_mu=2,
        radial_parts=3,
        toroidal_parts=2,
        halo=1,
    )
    state: NDArray[np.float64] = np.sin(
        np.arange(plan.total_owned_phase_cells, dtype=np.float64) / 17.0
    ).reshape(
        plan.n_radial,
        plan.n_toroidal,
        plan.n_theta,
        plan.n_vpar,
        plan.n_mu,
    )

    result = local_decomposed_phase_execution(plan, state)

    assert result.rank_count == plan.total_ranks
    assert result.global_shape == state.shape
    assert result.halo_exchange_pass
    assert result.decomposition_invariant_pass
    assert result.reconstruction_linf_error == 0.0
    assert result.inventory_relative_error <= 1.0e-12
    assert result.free_energy_relative_error <= 1.0e-12
    assert result.parallel_moment_relative_error <= 1.0e-12
    np.testing.assert_allclose(result.local_inventory, float(np.sum(state)))
    np.testing.assert_allclose(result.local_free_energy, float(np.sum(state * state)))


def test_production_decomposition_contract_is_fail_closed() -> None:
    report = run_benchmark()
    write_reports(report)

    assert report["schema"] == "production-decomposition-contract.v1"
    assert report["contract_pass"] is True
    assert report["communication_contract_ready"] is True
    assert report["local_decomposed_execution_pass"] is True
    assert report["halo_exchange_pass"] is True
    assert report["decomposition_invariant_pass"] is True
    assert report["parallel_moment_invariant_pass"] is True
    assert report["same_physics_decomposition_shape_pass"] is True
    halo_evidence = report["halo_face_integrity_evidence"]
    assert halo_evidence["schema"] == "production-decomposition-halo-face-integrity.v1"
    assert halo_evidence["status"] == "accepted_local_serial_halo_face_integrity"
    assert halo_evidence["case_id"] == "local_cpu_64x32_4x2"
    assert halo_evidence["checked_face_count"] == 20
    assert halo_evidence["max_halo_face_linf_error"] == 0.0
    assert halo_evidence["halo_face_integrity_pass"] is True
    assert halo_evidence["distributed_runtime_halo_exchange_ready"] is False
    assert all(row["face_integrity_pass"] for row in halo_evidence["face_rows"])
    communication_volume = report["distributed_communication_volume_evidence"]
    assert communication_volume["schema"] == "production-decomposition-communication-volume.v1"
    assert communication_volume["status"] == "blocked_missing_distributed_runtime_execution"
    assert communication_volume["distributed_runtime_ready"] is False
    assert communication_volume["halo_dtype"] == "float64"
    assert communication_volume["bytes_per_float64"] == 8
    assert communication_volume["rank_count"] == 32
    assert communication_volume["communicating_face_count"] > 0
    assert communication_volume["total_halo_exchange_bytes_per_step"] > 0
    assert communication_volume["max_rank_halo_exchange_bytes_per_step"] > 0
    assert all(row["rank_halo_exchange_bytes_per_step"] >= 0 for row in communication_volume["rank_rows"])
    reciprocal = report["reciprocal_neighbour_graph_evidence"]
    assert reciprocal["schema"] == "production-decomposition-reciprocal-neighbour-graph.v1"
    assert reciprocal["status"] == "accepted_local_reciprocal_neighbour_graph"
    assert reciprocal["reciprocal_neighbour_graph_pass"] is True
    assert reciprocal["rank_count"] == 32
    assert reciprocal["directed_link_count"] == communication_volume["communicating_face_count"]
    assert reciprocal["undirected_link_count"] * 2 == reciprocal["directed_link_count"]
    assert reciprocal["mismatched_link_count"] == 0
    assert reciprocal["max_payload_byte_asymmetry"] == 0
    assert all(row["link_pass"] for row in reciprocal["link_rows"])
    scaling = report["distributed_scaling_gate_evidence"]
    assert scaling["schema"] == "production-decomposition-distributed-scaling-gate.v1"
    assert scaling["status"] == "blocked_missing_distributed_scaling_measurements"
    assert scaling["distributed_scaling_ready"] is False
    assert scaling["measured_run_count"] == 0
    assert scaling["required_rank_counts"] == [1, 2, 4, 8, 16, 32]
    assert scaling["minimum_parallel_efficiency_threshold"] == 0.7
    assert scaling["minimum_weak_scaling_efficiency_threshold"] == 0.8
    assert scaling["estimated_halo_bytes_per_step"] == communication_volume[
        "total_halo_exchange_bytes_per_step"
    ]
    assert "MPI" in scaling["blocking_reason"]
    assert "multi-GPU" in scaling["blocking_reason"]
    shape_evidence = report["same_physics_shape_convergence_evidence"]
    assert shape_evidence["schema"] == "production-decomposition-shape-convergence.v1"
    assert shape_evidence["shape_convergence_pass"] is True
    assert shape_evidence["shape_count"] >= 3
    assert (
        shape_evidence["max_inventory_relative_deviation"]
        <= shape_evidence["relative_reduction_tolerance"]
    )
    assert (
        shape_evidence["max_free_energy_relative_deviation"]
        <= shape_evidence["relative_reduction_tolerance"]
    )
    assert (
        shape_evidence["max_parallel_moment_relative_deviation"]
        <= shape_evidence["relative_reduction_tolerance"]
    )
    assert (
        shape_evidence["max_reconstruction_linf_error"]
        <= shape_evidence["reconstruction_linf_tolerance"]
    )
    assert all(row["shape_convergence_pass"] for row in shape_evidence["shape_rows"])
    assert report["cpu_benchmark_rows"]
    assert report["hardware_metadata"]["python_version"]
    assert report["reproducible_commands"]
    assert report["production_scale_ready"] is False
    assert report["status"].startswith("blocked_")
    assert report["missing_requirements"]
