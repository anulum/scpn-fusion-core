#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""MPI rank-tile execution runner for production decomposition evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scpn_fusion.core.gk_domain_decomposition import build_radial_toroidal_decomposition


def _parallel_moment(state: NDArray[np.float64], n_vpar: int) -> float:
    if n_vpar == 1:
        weights = np.zeros(1, dtype=np.float64)
    else:
        weights = np.linspace(-1.0, 1.0, num=n_vpar, dtype=np.float64)
    return float(np.sum(state * weights.reshape(1, 1, 1, n_vpar, 1)))


def _relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-30)


def _build_state(shape: tuple[int, int, int, int, int]) -> NDArray[np.float64]:
    cell_count = int(np.prod(shape, dtype=np.int64))
    return np.sin(np.arange(cell_count, dtype=np.float64) / 29.0).reshape(shape)


def run(output: Path) -> int:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    radial_parts = 2
    toroidal_parts = 2
    if size != radial_parts * toroidal_parts:
        if rank == 0:
            output.write_text(
                json.dumps(
                    {
                        "blocking_reason": "runner_requires_2x2_topology_with_4_ranks",
                        "rank_count": size,
                        "schema": "production-decomposition-mpi-runtime-runner.v1",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        return 1
    plan = build_radial_toroidal_decomposition(
        n_radial=16,
        n_toroidal=8,
        n_theta=4,
        n_vpar=4,
        n_mu=3,
        radial_parts=radial_parts,
        toroidal_parts=toroidal_parts,
        halo=1,
    )
    tile = plan.tiles[rank]
    radial_index = rank // toroidal_parts
    toroidal_index = rank % toroidal_parts
    state_shape = (
        plan.n_radial,
        plan.n_toroidal,
        plan.n_theta,
        plan.n_vpar,
        plan.n_mu,
    )
    state = _build_state(state_shape)
    owned = state[
        tile.radial.start : tile.radial.stop,
        tile.toroidal.start : tile.toroidal.stop,
        :,
        :,
        :,
    ].copy()
    halo_shape = (
        tile.radial_with_halo.size,
        tile.toroidal_with_halo.size,
        plan.n_theta,
        plan.n_vpar,
        plan.n_mu,
    )
    with_halo = np.zeros(halo_shape, dtype=np.float64)
    radial_offset = tile.radial.start - tile.radial_with_halo.start
    toroidal_offset = tile.toroidal.start - tile.toroidal_with_halo.start
    with_halo[
        radial_offset : radial_offset + tile.radial.size,
        toroidal_offset : toroidal_offset + tile.toroidal.size,
        :,
        :,
        :,
    ] = owned

    radial_lower_rank = (
        (radial_index - 1) * toroidal_parts + toroidal_index
        if radial_index > 0
        else MPI.PROC_NULL
    )
    radial_upper_rank = (
        (radial_index + 1) * toroidal_parts + toroidal_index
        if radial_index < radial_parts - 1
        else MPI.PROC_NULL
    )
    toroidal_lower_rank = (
        radial_index * toroidal_parts + toroidal_index - 1
        if toroidal_index > 0
        else MPI.PROC_NULL
    )
    toroidal_upper_rank = (
        radial_index * toroidal_parts + toroidal_index + 1
        if toroidal_index < toroidal_parts - 1
        else MPI.PROC_NULL
    )
    corner_ranks = {
        "radial_lower_toroidal_lower": (radial_index - 1) * toroidal_parts + toroidal_index - 1
        if radial_index > 0 and toroidal_index > 0
        else MPI.PROC_NULL,
        "radial_lower_toroidal_upper": (radial_index - 1) * toroidal_parts + toroidal_index + 1
        if radial_index > 0 and toroidal_index < toroidal_parts - 1
        else MPI.PROC_NULL,
        "radial_upper_toroidal_lower": (radial_index + 1) * toroidal_parts + toroidal_index - 1
        if radial_index < radial_parts - 1 and toroidal_index > 0
        else MPI.PROC_NULL,
        "radial_upper_toroidal_upper": (radial_index + 1) * toroidal_parts + toroidal_index + 1
        if radial_index < radial_parts - 1 and toroidal_index < toroidal_parts - 1
        else MPI.PROC_NULL,
    }
    radial_face_shape = (
        plan.halo,
        tile.toroidal.size,
        plan.n_theta,
        plan.n_vpar,
        plan.n_mu,
    )
    toroidal_face_shape = (
        tile.radial.size,
        plan.halo,
        plan.n_theta,
        plan.n_vpar,
        plan.n_mu,
    )

    radial_lower_recv = np.empty(radial_face_shape, dtype=np.float64)
    radial_upper_recv = np.empty(radial_face_shape, dtype=np.float64)
    comm.Sendrecv(
        sendbuf=owned[: plan.halo, :, :, :, :].copy(),
        dest=radial_lower_rank,
        sendtag=11,
        recvbuf=radial_upper_recv,
        source=radial_upper_rank,
        recvtag=11,
    )
    comm.Sendrecv(
        sendbuf=owned[-plan.halo :, :, :, :].copy(),
        dest=radial_upper_rank,
        sendtag=17,
        recvbuf=radial_lower_recv,
        source=radial_lower_rank,
        recvtag=17,
    )
    toroidal_lower_recv = np.empty(toroidal_face_shape, dtype=np.float64)
    toroidal_upper_recv = np.empty(toroidal_face_shape, dtype=np.float64)
    comm.Sendrecv(
        sendbuf=owned[:, : plan.halo, :, :, :].copy(),
        dest=toroidal_lower_rank,
        sendtag=23,
        recvbuf=toroidal_upper_recv,
        source=toroidal_upper_rank,
        recvtag=23,
    )
    comm.Sendrecv(
        sendbuf=owned[:, -plan.halo :, :, :, :].copy(),
        dest=toroidal_upper_rank,
        sendtag=29,
        recvbuf=toroidal_lower_recv,
        source=toroidal_lower_rank,
        recvtag=29,
    )
    if radial_lower_rank != MPI.PROC_NULL:
        with_halo[
            :radial_offset,
            toroidal_offset : toroidal_offset + tile.toroidal.size,
            :,
            :,
            :,
        ] = radial_lower_recv
    if radial_upper_rank != MPI.PROC_NULL:
        with_halo[
            radial_offset + tile.radial.size :,
            toroidal_offset : toroidal_offset + tile.toroidal.size,
            :,
            :,
            :,
        ] = radial_upper_recv
    if toroidal_lower_rank != MPI.PROC_NULL:
        with_halo[
            radial_offset : radial_offset + tile.radial.size,
            :toroidal_offset,
            :,
            :,
            :,
        ] = toroidal_lower_recv
    if toroidal_upper_rank != MPI.PROC_NULL:
        with_halo[
            radial_offset : radial_offset + tile.radial.size,
            toroidal_offset + tile.toroidal.size :,
            :,
            :,
            :,
        ] = toroidal_upper_recv
    corner_shape = (
        plan.halo,
        plan.halo,
        plan.n_theta,
        plan.n_vpar,
        plan.n_mu,
    )
    corner_exchange_specs = (
        (
            "radial_lower_toroidal_lower",
            owned[: plan.halo, : plan.halo, :, :, :].copy(),
            corner_ranks["radial_lower_toroidal_lower"],
            31,
            "radial_upper_toroidal_upper",
            corner_ranks["radial_upper_toroidal_upper"],
            31,
            (slice(radial_offset + tile.radial.size, None), slice(toroidal_offset + tile.toroidal.size, None)),
        ),
        (
            "radial_upper_toroidal_upper",
            owned[-plan.halo :, -plan.halo :, :, :, :].copy(),
            corner_ranks["radial_upper_toroidal_upper"],
            37,
            "radial_lower_toroidal_lower",
            corner_ranks["radial_lower_toroidal_lower"],
            37,
            (slice(0, radial_offset), slice(0, toroidal_offset)),
        ),
        (
            "radial_lower_toroidal_upper",
            owned[: plan.halo, -plan.halo :, :, :, :].copy(),
            corner_ranks["radial_lower_toroidal_upper"],
            41,
            "radial_upper_toroidal_lower",
            corner_ranks["radial_upper_toroidal_lower"],
            41,
            (slice(radial_offset + tile.radial.size, None), slice(0, toroidal_offset)),
        ),
        (
            "radial_upper_toroidal_lower",
            owned[-plan.halo :, : plan.halo, :, :, :].copy(),
            corner_ranks["radial_upper_toroidal_lower"],
            43,
            "radial_lower_toroidal_upper",
            corner_ranks["radial_lower_toroidal_upper"],
            43,
            (slice(0, radial_offset), slice(toroidal_offset + tile.toroidal.size, None)),
        ),
    )
    for _, sendbuf, dest, sendtag, _, source, recvtag, target_slices in corner_exchange_specs:
        corner_recv = np.empty(corner_shape, dtype=np.float64)
        comm.Sendrecv(
            sendbuf=sendbuf,
            dest=dest,
            sendtag=sendtag,
            recvbuf=corner_recv,
            source=source,
            recvtag=recvtag,
        )
        if source != MPI.PROC_NULL:
            radial_slice, toroidal_slice = target_slices
            with_halo[radial_slice, toroidal_slice, :, :, :] = corner_recv

    expected_halo = state[
        tile.radial_with_halo.start : tile.radial_with_halo.stop,
        tile.toroidal_with_halo.start : tile.toroidal_with_halo.stop,
        :,
        :,
        :,
    ]
    covered = np.zeros(with_halo.shape[:2], dtype=bool)
    covered[
        radial_offset : radial_offset + tile.radial.size,
        toroidal_offset : toroidal_offset + tile.toroidal.size,
    ] = True
    if radial_lower_rank != MPI.PROC_NULL:
        covered[:radial_offset, toroidal_offset : toroidal_offset + tile.toroidal.size] = True
    if radial_upper_rank != MPI.PROC_NULL:
        covered[
            radial_offset + tile.radial.size :,
            toroidal_offset : toroidal_offset + tile.toroidal.size,
        ] = True
    if toroidal_lower_rank != MPI.PROC_NULL:
        covered[radial_offset : radial_offset + tile.radial.size, :toroidal_offset] = True
    if toroidal_upper_rank != MPI.PROC_NULL:
        covered[
            radial_offset : radial_offset + tile.radial.size,
            toroidal_offset + tile.toroidal.size :,
        ] = True
    if corner_ranks["radial_lower_toroidal_lower"] != MPI.PROC_NULL:
        covered[:radial_offset, :toroidal_offset] = True
    if corner_ranks["radial_lower_toroidal_upper"] != MPI.PROC_NULL:
        covered[:radial_offset, toroidal_offset + tile.toroidal.size :] = True
    if corner_ranks["radial_upper_toroidal_lower"] != MPI.PROC_NULL:
        covered[radial_offset + tile.radial.size :, :toroidal_offset] = True
    if corner_ranks["radial_upper_toroidal_upper"] != MPI.PROC_NULL:
        covered[
            radial_offset + tile.radial.size :,
            toroidal_offset + tile.toroidal.size :,
        ] = True
    halo_linf = float(np.max(np.abs(with_halo[covered] - expected_halo[covered])))
    row: dict[str, Any] = {
        "free_energy": float(np.sum(owned * owned)),
        "halo_linf_error": halo_linf,
        "halo_verified_fraction": float(np.mean(covered)),
        "inventory": float(np.sum(owned)),
        "neighbour_ranks": {
            "radial_lower": None if radial_lower_rank == MPI.PROC_NULL else int(radial_lower_rank),
            "radial_upper": None if radial_upper_rank == MPI.PROC_NULL else int(radial_upper_rank),
            "toroidal_lower": None
            if toroidal_lower_rank == MPI.PROC_NULL
            else int(toroidal_lower_rank),
            "toroidal_upper": None
            if toroidal_upper_rank == MPI.PROC_NULL
            else int(toroidal_upper_rank),
        },
        "corner_neighbour_ranks": {
            key: None if value == MPI.PROC_NULL else int(value)
            for key, value in corner_ranks.items()
        },
        "owned": owned,
        "owned_shape": [int(axis) for axis in owned.shape],
        "parallel_moment": _parallel_moment(owned, plan.n_vpar),
        "rank": rank,
    }
    gathered = comm.gather(row, root=0)
    if rank != 0:
        return 0

    assert gathered is not None
    reconstructed = np.zeros(state_shape, dtype=np.float64)
    local_inventory = 0.0
    local_free_energy = 0.0
    local_parallel_moment = 0.0
    rank_rows: list[dict[str, Any]] = []
    for rank_row in gathered:
        rank_id = int(rank_row["rank"])
        rank_tile = plan.tiles[rank_id]
        reconstructed[
            rank_tile.radial.start : rank_tile.radial.stop,
            rank_tile.toroidal.start : rank_tile.toroidal.stop,
            :,
            :,
            :,
        ] = rank_row["owned"]
        local_inventory += float(rank_row["inventory"])
        local_free_energy += float(rank_row["free_energy"])
        local_parallel_moment += float(rank_row["parallel_moment"])
        rank_rows.append(
            {
                "halo_linf_error": float(rank_row["halo_linf_error"]),
                "halo_verified_fraction": float(rank_row["halo_verified_fraction"]),
                "corner_neighbour_ranks": rank_row["corner_neighbour_ranks"],
                "neighbour_ranks": rank_row["neighbour_ranks"],
                "owned_shape": rank_row["owned_shape"],
                "rank": rank_id,
            }
        )

    global_inventory = float(np.sum(state))
    global_free_energy = float(np.sum(state * state))
    global_parallel_moment = _parallel_moment(state, plan.n_vpar)
    payload = {
        "decomposition_invariant_pass": bool(
            float(np.max(np.abs(reconstructed - state))) == 0.0
            and _relative_error(local_inventory, global_inventory) <= 1.0e-12
            and _relative_error(local_free_energy, global_free_energy) <= 1.0e-12
            and _relative_error(local_parallel_moment, global_parallel_moment) <= 1.0e-12
            and max(float(row["halo_linf_error"]) for row in rank_rows) == 0.0
        ),
        "free_energy_relative_error": _relative_error(local_free_energy, global_free_energy),
        "global_shape": [int(axis) for axis in state_shape],
        "halo_exchange_pass": bool(max(float(row["halo_linf_error"]) for row in rank_rows) == 0.0),
        "inventory_relative_error": _relative_error(local_inventory, global_inventory),
        "max_halo_linf_error": max(float(row["halo_linf_error"]) for row in rank_rows),
        "min_halo_verified_fraction": min(
            float(row["halo_verified_fraction"]) for row in rank_rows
        ),
        "parallel_moment_relative_error": _relative_error(
            local_parallel_moment, global_parallel_moment
        ),
        "radial_parts": radial_parts,
        "rank_count": size,
        "rank_rows": sorted(rank_rows, key=lambda item: int(item["rank"])),
        "reconstruction_linf_error": float(np.max(np.abs(reconstructed - state))),
        "schema": "production-decomposition-mpi-runtime-runner.v1",
        "topology": "radial_toroidal_2d",
        "toroidal_parts": toroidal_parts,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    return run(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
