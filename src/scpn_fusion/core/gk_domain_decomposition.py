#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Radial/toroidal decomposition contracts for production nonlinear GK runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


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
    decomposition_invariant_pass: bool


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
        and inventory_relative_error == 0.0
        and free_energy_relative_error == 0.0
    )
    return DecompositionInvariantMetrics(
        halo_exchange_pass=halo_exchange_pass,
        reconstruction_linf_error=reconstruction_error,
        inventory_relative_error=inventory_relative_error,
        free_energy_relative_error=free_energy_relative_error,
        decomposition_invariant_pass=invariant_pass,
    )


__all__ = [
    "AxisBlock",
    "DecompositionInvariantMetrics",
    "GKDomainDecompositionPlan",
    "RankPhaseTile",
    "RankTile",
    "build_radial_toroidal_decomposition",
    "decomposition_invariant_metrics",
    "reconstruct_owned_phase_state",
    "serial_halo_exchange",
]
