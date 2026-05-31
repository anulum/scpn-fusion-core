# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Radial/toroidal decomposition contracts for production nonlinear GK runs."""

from __future__ import annotations

from dataclasses import dataclass


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


__all__ = [
    "AxisBlock",
    "GKDomainDecompositionPlan",
    "RankTile",
    "build_radial_toroidal_decomposition",
]
