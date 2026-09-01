# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — DREAM Full-Kinetic Execution Custody
"""Durable, fail-closed custody for long-running DREAM output artefacts."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from validation.dream_full_kinetic_reference import (
    DREAM_COMMIT,
    EXPECTED_FLAGS,
    ION_RATE_AUXILIARY_QUANTITIES,
    REQUIRED_AUXILIARY_QUANTITIES,
)
from validation.reference_data.dream.full_kinetic_radial_parity_deck import (
    CASE,
    REQUESTED_OTHER_QUANTITIES,
    RESOLUTIONS,
)


CUSTODY_SCHEMA: Final[str] = "scpn-fusion.dream-execution-custody.v1"
OUTPUT_INSPECTION_SCHEMA: Final[str] = "scpn-fusion.dream-output-inspection.v1"
FILESYSTEM_ROOT: Final[Path] = Path(os.sep).resolve()
VOLATILE_ROOTS: Final[tuple[Path, ...]] = (
    FILESYSTEM_ROOT / "tmp",
    FILESYSTEM_ROOT / "var" / "tmp",
    FILESYSTEM_ROOT / "run",
    FILESYSTEM_ROOT / "dev" / "shm",
)


@dataclass(frozen=True)
class DreamOutputContract:
    """Exact grid and time contract for one DREAM convergence member."""

    resolution: str
    nr: int
    np: int
    nxi: int
    nt: int
    final_time_s: float


def frozen_output_contract(resolution: str) -> DreamOutputContract:
    """Return the immutable veryfine or superfine output contract."""

    if resolution not in {"veryfine", "superfine"}:
        raise ValueError("custody runner accepts only veryfine then superfine")
    grid = RESOLUTIONS[resolution]
    return DreamOutputContract(
        resolution=resolution,
        nr=grid.nr,
        np=grid.np,
        nxi=grid.nxi,
        nt=grid.nt,
        final_time_s=CASE.simulation_time_s,
    )


def sha256_file(path: Path) -> str:
    """Hash a regular non-symlink file without loading it into memory."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"custody object is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_durable_root(path: Path) -> Path:
    """Resolve a run root and reject known volatile or symlinked custody paths."""

    expanded = path.expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"custody root component is a symlink: {cursor}")
    resolved = absolute.resolve(strict=False)
    if any(resolved == root or root in resolved.parents for root in VOLATILE_ROOTS):
        raise ValueError(f"DREAM custody root is volatile: {resolved}")

    existing = resolved
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    return resolved


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish canonical JSON atomically and durably on the target filesystem."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON object and reject non-object roots."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _decode_hdf5_text(dataset: Any) -> str:
    raw = np.asarray(dataset[()])
    return b"".join(raw.reshape(-1).tolist()).decode("utf-8")


def _dataset_finite_and_activity(dataset: Any) -> tuple[bool, bool]:
    finite = True
    nonzero = False
    for index in range(dataset.shape[0]):
        block = np.asarray(dataset[index])
        if not np.issubdtype(block.dtype, np.number):
            raise ValueError(f"non-numeric DREAM dataset: {dataset.name}")
        finite = finite and bool(np.all(np.isfinite(block)))
        nonzero = nonzero or bool(np.any(block != 0))
        if not finite:
            break
    return finite, nonzero


def _required_shapes(contract: DreamOutputContract) -> dict[str, tuple[int, ...]]:
    nt, nr, nxi, np_ = contract.nt, contract.nr, contract.nxi, contract.np
    shapes: dict[str, tuple[int, ...]] = {
        "grid/t": (nt + 1,),
        "eqsys/f_re": (nt + 1, nr, nxi, np_),
        "eqsys/n_re": (nt + 1, nr),
        "eqsys/j_re": (nt + 1, nr),
        "eqsys/n_tot": (nt + 1, nr),
        "eqsys/E_field": (nt + 1, nr),
        "other/runaway/Ar": (nt, nr + 1, nxi, np_),
        "other/runaway/Drr": (nt, nr + 1, nxi, np_),
        "other/runaway/S_ava": (nt, nr, nxi, np_),
    }
    momentum = {
        "Ap1",
        "Dpp",
        "Dpx",
        "lnLambda_ee_f1",
        "lnLambda_ei_f1",
        "nu_D_f1",
        "nu_s_f1",
        "nu_par_f1",
        "synchrotron_f1",
        "bremsstrahlung_f1",
    }
    pitch = {
        "Ap2",
        "Dxp",
        "Dxx",
        "lnLambda_ee_f2",
        "lnLambda_ei_f2",
        "nu_D_f2",
        "nu_s_f2",
        "nu_par_f2",
        "synchrotron_f2",
    }
    for name in momentum:
        shapes[f"other/runaway/{name}"] = (nt, nr, nxi, np_ + 1)
    for name in pitch:
        shapes[f"other/runaway/{name}"] = (nt, nr, nxi + 1, np_)
    for name in REQUIRED_AUXILIARY_QUANTITIES:
        shape: tuple[int, ...]
        if name in ION_RATE_AUXILIARY_QUANTITIES:
            shape = (nt, 21, nr)
        elif name.startswith("fluid/"):
            shape = (nt, nr)
        else:
            shape = (nt, 1)
        shapes[f"other/{name}"] = shape
    return shapes


def inspect_dream_output(path: Path, contract: DreamOutputContract) -> dict[str, Any]:
    """Validate complete full-kinetic HDF5 structure, finiteness and activity."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"DREAM output is not a regular file: {path}")
    h5py = importlib.import_module("h5py")

    shapes = _required_shapes(contract)
    active = {
        "other/runaway/Drr",
        "other/runaway/Dpp",
        "other/runaway/Dxx",
        "other/runaway/S_ava",
        "other/runaway/synchrotron_f1",
        "other/runaway/synchrotron_f2",
        "other/runaway/bremsstrahlung_f1",
        "other/fluid/GammaAva",
        "other/fluid/runawayRate",
        "other/fluid/W_re",
        "other/scalar/energyloss_f_re",
        "other/scalar/radialloss_f_re",
    }
    activity: dict[str, bool] = {}
    with h5py.File(path, "r") as handle:
        commit = _decode_hdf5_text(handle["code/commit"])
        if commit != DREAM_COMMIT:
            raise ValueError(f"unexpected DREAM commit {commit!r}")
        for key, expected in EXPECTED_FLAGS.items():
            actual = int(np.asarray(handle[key][()]).reshape(-1)[0])
            if actual != expected:
                raise ValueError(f"{key}={actual}, expected {expected}")
        requested = tuple(_decode_hdf5_text(handle["settings/other/include"]).split(";"))
        if requested != REQUESTED_OTHER_QUANTITIES:
            raise ValueError("requested DREAM diagnostic groups are incomplete or reordered")
        for key, expected_shape in shapes.items():
            if key not in handle:
                raise ValueError(f"missing DREAM dataset: {key}")
            dataset = handle[key]
            if tuple(dataset.shape) != expected_shape:
                raise ValueError(
                    f"{key} has shape {tuple(dataset.shape)}, expected {expected_shape}"
                )
            finite, nonzero = _dataset_finite_and_activity(dataset)
            if not finite:
                raise ValueError(f"DREAM dataset contains non-finite values: {key}")
            if key in active:
                activity[key] = nonzero
                if not nonzero:
                    raise ValueError(f"required DREAM physics dataset is inactive: {key}")
        times = np.asarray(handle["grid/t"][()], dtype=np.float64)
        if times[0] != 0.0 or np.any(np.diff(times) <= 0.0):
            raise ValueError("DREAM time grid is not a complete increasing trajectory")
        if not np.isclose(times[-1], contract.final_time_s, rtol=0.0, atol=1.0e-15):
            raise ValueError("DREAM output did not reach the frozen final time")

    return {
        "schema": OUTPUT_INSPECTION_SCHEMA,
        "custody_schema": CUSTODY_SCHEMA,
        "validated": True,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "dream_commit": commit,
        "contract": asdict(contract),
        "required_dataset_count": len(shapes),
        "active_physics": activity,
    }


__all__ = [
    "CUSTODY_SCHEMA",
    "DreamOutputContract",
    "atomic_write_json",
    "frozen_output_contract",
    "inspect_dream_output",
    "read_json_object",
    "sha256_file",
    "validate_durable_root",
]
