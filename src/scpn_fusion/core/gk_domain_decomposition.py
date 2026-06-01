#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Radial/toroidal decomposition contracts for production nonlinear GK runs."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

_REDUCTION_RELATIVE_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class AxisBlock:
    """One half-open partition on a decomposed axis."""

    start: int
    stop: int

    @property
    def size(self) -> int:
        """Number of owned cells in this block."""
        return self.stop - self.start


@dataclass(frozen=True)
class RankTile:
    """One radial x toroidal rank tile with halo-extended local extents."""

    rank: int
    radial: AxisBlock
    toroidal: AxisBlock
    radial_with_halo: AxisBlock
    toroidal_with_halo: AxisBlock

    @property
    def owned_cells(self) -> int:
        """Owned radial x toroidal cells on this rank."""
        return self.radial.size * self.toroidal.size

    @property
    def halo_cells(self) -> int:
        """Halo-extended radial x toroidal cells on this rank."""
        return self.radial_with_halo.size * self.toroidal_with_halo.size


@dataclass(frozen=True)
class RankPhaseTile:
    """One decomposed local 5D phase-space payload with owned and halo views."""

    rank: int
    owned: NDArray[np.float64]
    with_halo: NDArray[np.float64]


@dataclass(frozen=True)
class DecompositionInvariantMetrics:
    """Scalar invariants comparing monolithic and decomposed phase-space state."""

    halo_exchange_pass: bool
    reconstruction_linf_error: float
    inventory_relative_error: float
    free_energy_relative_error: float
    parallel_moment_relative_error: float
    decomposition_invariant_pass: bool


@dataclass(frozen=True)
class LocalDecomposedExecutionResult:
    """Local executable multi-rank decomposition result for one 5D phase state."""

    rank_count: int
    global_shape: tuple[int, int, int, int, int]
    local_inventory: float
    global_inventory: float
    inventory_relative_error: float
    local_free_energy: float
    global_free_energy: float
    free_energy_relative_error: float
    local_parallel_moment: float
    global_parallel_moment: float
    parallel_moment_relative_error: float
    reconstruction_linf_error: float
    halo_exchange_pass: bool
    decomposition_invariant_pass: bool


@dataclass(frozen=True)
class LocalMultiprocessExecutionResult:
    """Process-isolated local rank-tile execution evidence for one 5D state."""

    rank_count: int
    worker_count: int
    unique_worker_process_count: int
    global_shape: tuple[int, int, int, int, int]
    local_inventory: float
    global_inventory: float
    inventory_relative_error: float
    local_free_energy: float
    global_free_energy: float
    free_energy_relative_error: float
    local_parallel_moment: float
    global_parallel_moment: float
    parallel_moment_relative_error: float
    halo_checksum_relative_error: float
    reconstruction_linf_error: float
    halo_exchange_pass: bool
    decomposition_invariant_pass: bool
    rank_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class GKDomainDecompositionPlan:
    """Validated radial/toroidal decomposition plan for 5D nonlinear GK state."""

    n_radial: int
    n_toroidal: int
    n_theta: int
    n_vpar: int
    n_mu: int
    radial_parts: int
    toroidal_parts: int
    halo: int
    tiles: tuple[RankTile, ...]

    @property
    def total_ranks(self) -> int:
        """Total MPI/GPU ranks implied by this plan."""
        return self.radial_parts * self.toroidal_parts

    @property
    def max_owned_cells(self) -> int:
        """Maximum owned radial x toroidal cells on any rank."""
        return max(tile.owned_cells for tile in self.tiles)

    @property
    def min_owned_cells(self) -> int:
        """Minimum owned radial x toroidal cells on any rank."""
        return min(tile.owned_cells for tile in self.tiles)

    @property
    def owned_cell_imbalance(self) -> float:
        """Load imbalance ratio over owned radial x toroidal cells."""
        return self.max_owned_cells / max(self.min_owned_cells, 1)

    @property
    def total_owned_phase_cells(self) -> int:
        """Total owned 5D phase-space cells over all ranks."""
        return self.n_radial * self.n_toroidal * self.n_theta * self.n_vpar * self.n_mu

    @property
    def total_halo_phase_cells(self) -> int:
        """Total halo-extended 5D phase-space cells over all ranks."""
        velocity_parallel = self.n_theta * self.n_vpar * self.n_mu
        return sum(tile.halo_cells * velocity_parallel for tile in self.tiles)

    @property
    def halo_overhead_ratio(self) -> float:
        """Halo-to-owned phase-space storage overhead ratio."""
        return self.total_halo_phase_cells / max(self.total_owned_phase_cells, 1)

    def validate(self) -> None:
        """Validate full coverage, non-overlap, and rank count invariants."""
        if len(self.tiles) != self.total_ranks:
            raise ValueError("tile count must match radial_parts * toroidal_parts")
        covered = set()
        for tile in self.tiles:
            if tile.radial.size <= 0 or tile.toroidal.size <= 0:
                raise ValueError("rank tiles must own non-empty radial and toroidal ranges")
            for i in range(tile.radial.start, tile.radial.stop):
                for j in range(tile.toroidal.start, tile.toroidal.stop):
                    cell = (i, j)
                    if cell in covered:
                        raise ValueError("rank tiles must not overlap owned cells")
                    covered.add(cell)
        expected = self.n_radial * self.n_toroidal
        if len(covered) != expected:
            raise ValueError("rank tiles must cover every radial/toroidal cell exactly once")


def _balanced_blocks(size: int, parts: int) -> tuple[AxisBlock, ...]:
    if size <= 0:
        raise ValueError("axis size must be positive")
    if parts <= 0:
        raise ValueError("parts must be positive")
    if parts > size:
        raise ValueError("parts must not exceed axis size")
    base = size // parts
    remainder = size % parts
    blocks = []
    start = 0
    for idx in range(parts):
        width = base + (1 if idx < remainder else 0)
        stop = start + width
        blocks.append(AxisBlock(start=start, stop=stop))
        start = stop
    return tuple(blocks)


def _with_halo(block: AxisBlock, size: int, halo: int) -> AxisBlock:
    return AxisBlock(start=max(0, block.start - halo), stop=min(size, block.stop + halo))


def _validate_phase_state(
    plan: GKDomainDecompositionPlan, phase_state: NDArray[np.float64]
) -> NDArray[np.float64]:
    state = np.asarray(phase_state, dtype=np.float64)
    expected_shape = (
        plan.n_radial,
        plan.n_toroidal,
        plan.n_theta,
        plan.n_vpar,
        plan.n_mu,
    )
    if state.shape != expected_shape:
        raise ValueError(f"phase_state shape must be {expected_shape}, got {state.shape}")
    if not bool(np.all(np.isfinite(state))):
        raise ValueError("phase_state must be finite")
    return state


def _normalized_vpar_weights(n_vpar: int) -> NDArray[np.float64]:
    """Return dimensionless parallel-velocity weights for reduction contracts."""
    if n_vpar == 1:
        return np.zeros(1, dtype=np.float64)
    return np.linspace(-1.0, 1.0, num=n_vpar, dtype=np.float64)


def _parallel_moment(state: NDArray[np.float64], n_vpar: int) -> float:
    """Return the normalized-vpar first moment of a 5D phase-space payload."""
    weights = _normalized_vpar_weights(n_vpar).reshape(1, 1, 1, n_vpar, 1)
    return float(np.sum(state * weights))


def _rank_process_reductions(
    payload: tuple[int, NDArray[np.float64], NDArray[np.float64], int]
) -> dict[str, Any]:
    """Return rank-local reductions from a worker process."""
    rank, owned, with_halo, n_vpar = payload
    return {
        "free_energy": float(np.sum(owned * owned)),
        "halo_checksum": float(np.sum(with_halo)),
        "halo_shape": [int(axis) for axis in with_halo.shape],
        "inventory": float(np.sum(owned)),
        "owned_shape": [int(axis) for axis in owned.shape],
        "parallel_moment": _parallel_moment(owned, n_vpar),
        "pid": os.getpid(),
        "rank": rank,
    }


def build_radial_toroidal_decomposition(
    *,
    n_radial: int,
    n_toroidal: int,
    n_theta: int,
    n_vpar: int,
    n_mu: int,
    radial_parts: int,
    toroidal_parts: int,
    halo: int = 1,
) -> GKDomainDecompositionPlan:
    """Build a deterministic radial/toroidal decomposition plan.

    The plan covers the 5D ``radial x toroidal x theta x vpar x mu`` storage
    contract used for production-scale nonlinear GK scheduling. It is a
    partition contract only; distributed MPI/GPU execution is separately gated
    by benchmark evidence.
    """
    if min(n_theta, n_vpar, n_mu) <= 0:
        raise ValueError("theta, vpar, and mu sizes must be positive")
    if halo < 0:
        raise ValueError("halo must be non-negative")
    radial_blocks = _balanced_blocks(n_radial, radial_parts)
    toroidal_blocks = _balanced_blocks(n_toroidal, toroidal_parts)
    tiles = []
    rank = 0
    for radial in radial_blocks:
        for toroidal in toroidal_blocks:
            tiles.append(
                RankTile(
                    rank=rank,
                    radial=radial,
                    toroidal=toroidal,
                    radial_with_halo=_with_halo(radial, n_radial, halo),
                    toroidal_with_halo=_with_halo(toroidal, n_toroidal, halo),
                )
            )
            rank += 1
    plan = GKDomainDecompositionPlan(
        n_radial=n_radial,
        n_toroidal=n_toroidal,
        n_theta=n_theta,
        n_vpar=n_vpar,
        n_mu=n_mu,
        radial_parts=radial_parts,
        toroidal_parts=toroidal_parts,
        halo=halo,
        tiles=tuple(tiles),
    )
    plan.validate()
    return plan


def serial_halo_exchange(
    plan: GKDomainDecompositionPlan, phase_state: NDArray[np.float64]
) -> tuple[RankPhaseTile, ...]:
    """Return a deterministic serial reference for radial/toroidal halo exchange.

    This function is the CPU reference contract for future MPI or GPU halo
    exchange. It extracts every rank's owned 5D block and the halo-extended
    block that a distributed exchange must reconstruct from neighbour-owned
    data. It does not claim distributed runtime execution.
    """
    state = _validate_phase_state(plan, phase_state)
    local_tiles: list[RankPhaseTile] = []
    for tile in plan.tiles:
        owned = state[
            tile.radial.start : tile.radial.stop,
            tile.toroidal.start : tile.toroidal.stop,
            :,
            :,
            :,
        ].copy()
        with_halo = state[
            tile.radial_with_halo.start : tile.radial_with_halo.stop,
            tile.toroidal_with_halo.start : tile.toroidal_with_halo.stop,
            :,
            :,
            :,
        ].copy()
        local_tiles.append(RankPhaseTile(rank=tile.rank, owned=owned, with_halo=with_halo))
    return tuple(local_tiles)


def reconstruct_owned_phase_state(
    plan: GKDomainDecompositionPlan, local_tiles: tuple[RankPhaseTile, ...]
) -> NDArray[np.float64]:
    """Reconstruct the global 5D state from decomposed owned rank payloads."""
    if len(local_tiles) != plan.total_ranks:
        raise ValueError("local tile count must match decomposition rank count")
    reconstructed: NDArray[np.float64] = np.zeros(
        (plan.n_radial, plan.n_toroidal, plan.n_theta, plan.n_vpar, plan.n_mu),
        dtype=np.float64,
    )
    seen: set[int] = set()
    for local in local_tiles:
        if local.rank in seen:
            raise ValueError("duplicate rank payload")
        seen.add(local.rank)
        tile = plan.tiles[local.rank]
        expected_shape = (
            tile.radial.size,
            tile.toroidal.size,
            plan.n_theta,
            plan.n_vpar,
            plan.n_mu,
        )
        if local.owned.shape != expected_shape:
            raise ValueError(f"rank {local.rank} owned shape must be {expected_shape}")
        reconstructed[
            tile.radial.start : tile.radial.stop,
            tile.toroidal.start : tile.toroidal.stop,
            :,
            :,
            :,
        ] = local.owned
    if seen != {tile.rank for tile in plan.tiles}:
        raise ValueError("rank payloads must cover every rank exactly once")
    return reconstructed


def rank_tile_communication_contract(plan: GKDomainDecompositionPlan) -> list[dict[str, object]]:
    """Return deterministic neighbour and halo-face payload contracts per rank.

    This is the fail-closed CPU contract that a future MPI or multi-GPU runtime
    must implement. It declares neighbour ranks and the exact phase-space face
    shapes needed for one-layer radial/toroidal halo exchange without claiming
    distributed execution.
    """
    by_block = {
        (tile.radial.start, tile.radial.stop, tile.toroidal.start, tile.toroidal.stop): tile
        for tile in plan.tiles
    }

    def neighbour_rank(
        tile: RankTile, *, radial_offset: int = 0, toroidal_offset: int = 0
    ) -> int | None:
        radial_start = tile.radial.start + radial_offset
        radial_stop = tile.radial.stop + radial_offset
        toroidal_start = tile.toroidal.start + toroidal_offset
        toroidal_stop = tile.toroidal.stop + toroidal_offset
        neighbour = by_block.get((radial_start, radial_stop, toroidal_start, toroidal_stop))
        return None if neighbour is None else neighbour.rank

    rows: list[dict[str, object]] = []
    velocity_shape = [plan.n_theta, plan.n_vpar, plan.n_mu]
    for tile in plan.tiles:
        radial_width = tile.radial.size
        toroidal_width = tile.toroidal.size
        neighbours = {
            "radial_lower": neighbour_rank(tile, radial_offset=-radial_width),
            "radial_upper": neighbour_rank(tile, radial_offset=radial_width),
            "toroidal_lower": neighbour_rank(tile, toroidal_offset=-toroidal_width),
            "toroidal_upper": neighbour_rank(tile, toroidal_offset=toroidal_width),
        }
        halo_shapes = {
            "radial_lower": [plan.halo, toroidal_width, *velocity_shape]
            if neighbours["radial_lower"] is not None and plan.halo > 0
            else None,
            "radial_upper": [plan.halo, toroidal_width, *velocity_shape]
            if neighbours["radial_upper"] is not None and plan.halo > 0
            else None,
            "toroidal_lower": [radial_width, plan.halo, *velocity_shape]
            if neighbours["toroidal_lower"] is not None and plan.halo > 0
            else None,
            "toroidal_upper": [radial_width, plan.halo, *velocity_shape]
            if neighbours["toroidal_upper"] is not None and plan.halo > 0
            else None,
        }
        rows.append(
            {
                "communication_contract_ready": all(
                    shape is not None or neighbours[face] is None
                    for face, shape in halo_shapes.items()
                ),
                "halo_face_payload_shapes": halo_shapes,
                "neighbour_ranks": neighbours,
                "rank": tile.rank,
            }
        )
    return rows


def decomposition_invariant_metrics(
    plan: GKDomainDecompositionPlan, phase_state: NDArray[np.float64]
) -> DecompositionInvariantMetrics:
    """Validate inventory and free-energy invariants under decomposition."""
    state = _validate_phase_state(plan, phase_state)
    local_tiles = serial_halo_exchange(plan, state)
    reconstructed = reconstruct_owned_phase_state(plan, local_tiles)
    reconstruction_error = float(np.max(np.abs(reconstructed - state)))
    global_inventory = float(np.sum(state))
    reconstructed_inventory = float(np.sum(reconstructed))
    inventory_relative_error = abs(reconstructed_inventory - global_inventory) / max(
        abs(global_inventory), 1.0e-30
    )
    global_free_energy = float(np.sum(state * state))
    reconstructed_free_energy = float(np.sum(reconstructed * reconstructed))
    free_energy_relative_error = abs(reconstructed_free_energy - global_free_energy) / max(
        abs(global_free_energy), 1.0e-30
    )
    global_parallel_moment = _parallel_moment(state, plan.n_vpar)
    reconstructed_parallel_moment = _parallel_moment(reconstructed, plan.n_vpar)
    parallel_moment_relative_error = abs(
        reconstructed_parallel_moment - global_parallel_moment
    ) / max(abs(global_parallel_moment), 1.0e-30)
    halo_exchange_pass = all(
        np.array_equal(
            local.with_halo,
            state[
                plan.tiles[local.rank].radial_with_halo.start : plan.tiles[
                    local.rank
                ].radial_with_halo.stop,
                plan.tiles[local.rank].toroidal_with_halo.start : plan.tiles[
                    local.rank
                ].toroidal_with_halo.stop,
                :,
                :,
                :,
            ],
        )
        for local in local_tiles
    )
    invariant_pass = bool(
        halo_exchange_pass
        and reconstruction_error == 0.0
        and inventory_relative_error <= _REDUCTION_RELATIVE_TOLERANCE
        and free_energy_relative_error <= _REDUCTION_RELATIVE_TOLERANCE
        and parallel_moment_relative_error <= _REDUCTION_RELATIVE_TOLERANCE
    )
    return DecompositionInvariantMetrics(
        halo_exchange_pass=halo_exchange_pass,
        reconstruction_linf_error=reconstruction_error,
        inventory_relative_error=inventory_relative_error,
        free_energy_relative_error=free_energy_relative_error,
        parallel_moment_relative_error=parallel_moment_relative_error,
        decomposition_invariant_pass=invariant_pass,
    )


def local_decomposed_phase_execution(
    plan: GKDomainDecompositionPlan, phase_state: NDArray[np.float64]
) -> LocalDecomposedExecutionResult:
    """Execute the decomposition locally over rank-owned 5D payloads.

    This is an executable CPU reference for future MPI or multi-GPU rank
    execution. It materialises per-rank owned payloads, reconstructs the global
    owned state, and checks inventory and free-energy reductions against the
    monolithic state without claiming distributed runtime scaling.
    """
    state = _validate_phase_state(plan, phase_state)
    local_tiles = serial_halo_exchange(plan, state)
    reconstructed = reconstruct_owned_phase_state(plan, local_tiles)
    reconstruction_error = float(np.max(np.abs(reconstructed - state)))
    global_inventory = float(np.sum(state))
    local_inventory = float(sum(float(np.sum(local.owned)) for local in local_tiles))
    inventory_relative_error = abs(local_inventory - global_inventory) / max(
        abs(global_inventory), 1.0e-30
    )
    global_free_energy = float(np.sum(state * state))
    local_free_energy = float(
        sum(float(np.sum(local.owned * local.owned)) for local in local_tiles)
    )
    free_energy_relative_error = abs(local_free_energy - global_free_energy) / max(
        abs(global_free_energy), 1.0e-30
    )
    global_parallel_moment = _parallel_moment(state, plan.n_vpar)
    local_parallel_moment = float(
        sum(_parallel_moment(local.owned, plan.n_vpar) for local in local_tiles)
    )
    parallel_moment_relative_error = abs(local_parallel_moment - global_parallel_moment) / max(
        abs(global_parallel_moment), 1.0e-30
    )
    halo_exchange_pass = all(
        np.array_equal(
            local.with_halo,
            state[
                plan.tiles[local.rank].radial_with_halo.start : plan.tiles[
                    local.rank
                ].radial_with_halo.stop,
                plan.tiles[local.rank].toroidal_with_halo.start : plan.tiles[
                    local.rank
                ].toroidal_with_halo.stop,
                :,
                :,
                :,
            ],
        )
        for local in local_tiles
    )
    invariant_pass = bool(
        halo_exchange_pass
        and reconstruction_error == 0.0
        and inventory_relative_error <= _REDUCTION_RELATIVE_TOLERANCE
        and free_energy_relative_error <= _REDUCTION_RELATIVE_TOLERANCE
        and parallel_moment_relative_error <= _REDUCTION_RELATIVE_TOLERANCE
    )
    return LocalDecomposedExecutionResult(
        rank_count=len(local_tiles),
        global_shape=state.shape,
        local_inventory=local_inventory,
        global_inventory=global_inventory,
        inventory_relative_error=inventory_relative_error,
        local_free_energy=local_free_energy,
        global_free_energy=global_free_energy,
        free_energy_relative_error=free_energy_relative_error,
        local_parallel_moment=local_parallel_moment,
        global_parallel_moment=global_parallel_moment,
        parallel_moment_relative_error=parallel_moment_relative_error,
        reconstruction_linf_error=reconstruction_error,
        halo_exchange_pass=halo_exchange_pass,
        decomposition_invariant_pass=invariant_pass,
    )


def local_multiprocess_rank_tile_execution(
    plan: GKDomainDecompositionPlan,
    phase_state: NDArray[np.float64],
    *,
    max_workers: int | None = None,
) -> LocalMultiprocessExecutionResult:
    """Execute rank-local reductions in separate worker processes.

    This is local CPU process isolation over the declared rank tiles. It is not
    MPI and it is not multi-GPU execution; benchmark gates must keep production
    scaling blocked until those runtime artefacts exist.
    """
    state = _validate_phase_state(plan, phase_state)
    local_tiles = serial_halo_exchange(plan, state)
    worker_count = max(1, min(plan.total_ranks, max_workers or (os.cpu_count() or 1)))
    payloads = [
        (local.rank, local.owned, local.with_halo, plan.n_vpar) for local in local_tiles
    ]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        rank_rows = tuple(sorted(executor.map(_rank_process_reductions, payloads), key=lambda row: int(row["rank"])))

    reconstructed = reconstruct_owned_phase_state(plan, local_tiles)
    reconstruction_error = float(np.max(np.abs(reconstructed - state)))
    global_inventory = float(np.sum(state))
    local_inventory = float(sum(float(row["inventory"]) for row in rank_rows))
    inventory_relative_error = abs(local_inventory - global_inventory) / max(
        abs(global_inventory), 1.0e-30
    )
    global_free_energy = float(np.sum(state * state))
    local_free_energy = float(sum(float(row["free_energy"]) for row in rank_rows))
    free_energy_relative_error = abs(local_free_energy - global_free_energy) / max(
        abs(global_free_energy), 1.0e-30
    )
    global_parallel_moment = _parallel_moment(state, plan.n_vpar)
    local_parallel_moment = float(sum(float(row["parallel_moment"]) for row in rank_rows))
    parallel_moment_relative_error = abs(local_parallel_moment - global_parallel_moment) / max(
        abs(global_parallel_moment), 1.0e-30
    )
    local_halo_checksum = float(sum(float(row["halo_checksum"]) for row in rank_rows))
    reference_halo_checksum = float(sum(float(np.sum(local.with_halo)) for local in local_tiles))
    halo_checksum_relative_error = abs(local_halo_checksum - reference_halo_checksum) / max(
        abs(reference_halo_checksum), 1.0e-30
    )
    halo_exchange_pass = all(
        list(local.with_halo.shape) == rank_rows[local.rank]["halo_shape"]
        for local in local_tiles
    )
    invariant_pass = bool(
        halo_exchange_pass
        and reconstruction_error == 0.0
        and inventory_relative_error <= _REDUCTION_RELATIVE_TOLERANCE
        and free_energy_relative_error <= _REDUCTION_RELATIVE_TOLERANCE
        and parallel_moment_relative_error <= _REDUCTION_RELATIVE_TOLERANCE
        and halo_checksum_relative_error <= _REDUCTION_RELATIVE_TOLERANCE
    )
    return LocalMultiprocessExecutionResult(
        rank_count=len(local_tiles),
        worker_count=worker_count,
        unique_worker_process_count=len({int(row["pid"]) for row in rank_rows}),
        global_shape=state.shape,
        local_inventory=local_inventory,
        global_inventory=global_inventory,
        inventory_relative_error=inventory_relative_error,
        local_free_energy=local_free_energy,
        global_free_energy=global_free_energy,
        free_energy_relative_error=free_energy_relative_error,
        local_parallel_moment=local_parallel_moment,
        global_parallel_moment=global_parallel_moment,
        parallel_moment_relative_error=parallel_moment_relative_error,
        halo_checksum_relative_error=halo_checksum_relative_error,
        reconstruction_linf_error=reconstruction_error,
        halo_exchange_pass=halo_exchange_pass,
        decomposition_invariant_pass=invariant_pass,
        rank_rows=rank_rows,
    )


__all__ = [
    "AxisBlock",
    "DecompositionInvariantMetrics",
    "GKDomainDecompositionPlan",
    "LocalDecomposedExecutionResult",
    "LocalMultiprocessExecutionResult",
    "RankPhaseTile",
    "RankTile",
    "build_radial_toroidal_decomposition",
    "decomposition_invariant_metrics",
    "local_decomposed_phase_execution",
    "local_multiprocess_rank_tile_execution",
    "rank_tile_communication_contract",
    "reconstruct_owned_phase_state",
    "serial_halo_exchange",
]
