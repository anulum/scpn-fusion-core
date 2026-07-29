# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Safe File Loaders
"""Bounded file-loader helpers for untrusted JSON and NumPy archives."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import IO, Any, Callable, cast

import numpy as np
from numpy.lib import format as npy_format

MAX_JSON_BYTES = 10 * 1024 * 1024
MAX_NPZ_BYTES = 10 * 1024 * 1024
MAX_NPZ_MEMBER_BYTES = 10 * 1024 * 1024
MAX_NPZ_TOTAL_BYTES = 10 * 1024 * 1024
MAX_NPY_HEADER_BYTES = 10_000


_read_magic = cast(
    Callable[[IO[bytes]], tuple[int, int]],
    npy_format.read_magic,
)
_read_array_header = cast(
    Callable[
        [IO[bytes], tuple[int, int], int],
        tuple[tuple[int, ...], bool, np.dtype[Any]],
    ],
    vars(npy_format)["_read_array_header"],
)


def _require_nonnegative_limit(value: int, *, name: str) -> None:
    """Reject nonsensical byte limits before inspecting an untrusted file."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _npy_logical_bytes(
    stream: IO[bytes],
    *,
    label: str,
    max_bytes: int,
    max_header_bytes: int,
) -> int:
    """Return bounded logical array bytes after parsing one NPY header."""
    version = _read_magic(stream)
    shape, _, dtype = _read_array_header(
        stream,
        version,
        max_header_bytes,
    )
    if dtype.hasobject:
        raise ValueError(f"NumPy object dtype is not allowed: {label!r}")

    for dimension in shape:
        if dimension < 0:
            raise ValueError(f"NumPy array shape is invalid: {label!r} has {shape!r}")
    if 0 in shape or dtype.itemsize == 0:
        return 0

    max_elements = max_bytes // dtype.itemsize
    element_count = 1
    for dimension in shape:
        if element_count > max_elements // dimension:
            raise ValueError(
                "NumPy array logical payload too large: "
                f"{label!r} declares more than {max_bytes} bytes"
            )
        element_count *= dimension

    logical_bytes = element_count * dtype.itemsize
    if logical_bytes > max_bytes:
        raise ValueError(
            "NumPy array logical payload too large: "
            f"{label!r} declares {logical_bytes} bytes, exceeding {max_bytes}"
        )
    return logical_bytes


def require_file_size_at_most(
    path: str | Path,
    *,
    max_bytes: int,
    label: str,
) -> Path:
    """Return ``path`` after rejecting files larger than ``max_bytes``."""
    path_obj = Path(path)
    size = path_obj.stat().st_size
    if size > max_bytes:
        raise ValueError(f"{label} file too large: {size} bytes exceeds {max_bytes}")
    return path_obj


def checked_json_load(path: str | Path, *, max_bytes: int = MAX_JSON_BYTES) -> Any:
    """Load a JSON document after enforcing a byte-size bound."""
    path_obj = require_file_size_at_most(path, max_bytes=max_bytes, label="JSON")
    with path_obj.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def checked_np_load(
    path: str | Path,
    *,
    max_bytes: int = MAX_NPZ_BYTES,
    max_member_bytes: int = MAX_NPZ_MEMBER_BYTES,
    max_total_bytes: int = MAX_NPZ_TOTAL_BYTES,
    max_header_bytes: int = MAX_NPY_HEADER_BYTES,
    **kwargs: Any,
) -> Any:
    """Open a bounded NumPy file with pickle loading disabled.

    ZIP metadata and every nested NPY header are checked before NumPy can
    materialise any NPZ member. Plain NPY headers receive the same logical-size
    and object-dtype checks before loading.

    Parameters
    ----------
    path:
        NumPy file to open.
    max_bytes:
        Maximum size of the stored file.
    max_member_bytes:
        Maximum uncompressed and logical size of one array.
    max_total_bytes:
        Maximum aggregate uncompressed and logical size of all NPZ members.
    max_header_bytes:
        Maximum size of one parsed NPY header.
    **kwargs:
        Additional keyword arguments forwarded to :func:`numpy.load`, except
        that ``allow_pickle`` is always forced to ``False``.

    Returns
    -------
    Any
        The array or archive returned by :func:`numpy.load`.

    Raises
    ------
    ValueError
        If a stored, expanded, header, or logical array size exceeds its
        configured bound, or if an array uses an object dtype.
    """
    _require_nonnegative_limit(max_bytes, name="max_bytes")
    _require_nonnegative_limit(max_member_bytes, name="max_member_bytes")
    _require_nonnegative_limit(max_total_bytes, name="max_total_bytes")
    _require_nonnegative_limit(max_header_bytes, name="max_header_bytes")
    path_obj = require_file_size_at_most(path, max_bytes=max_bytes, label="NumPy archive")
    if zipfile.is_zipfile(path_obj):
        expanded_total_bytes = 0
        logical_total_bytes = 0
        with zipfile.ZipFile(path_obj) as archive:
            for member in archive.infolist():
                if member.file_size > max_member_bytes:
                    raise ValueError(
                        "NumPy archive member too large: "
                        f"{member.filename!r} expands to {member.file_size} bytes, "
                        f"exceeding {max_member_bytes}"
                    )
                expanded_total_bytes += member.file_size
                if expanded_total_bytes > max_total_bytes:
                    raise ValueError(
                        "NumPy archive expands too large: "
                        f"members total more than {max_total_bytes} bytes"
                    )
                if member.filename.endswith(".npy"):
                    with archive.open(member) as stream:
                        logical_bytes = _npy_logical_bytes(
                            stream,
                            label=member.filename,
                            max_bytes=max_member_bytes,
                            max_header_bytes=max_header_bytes,
                        )
                    if logical_bytes > max_total_bytes - logical_total_bytes:
                        raise ValueError(
                            "NumPy archive logical payload too large: "
                            f"arrays total more than {max_total_bytes} bytes"
                        )
                    logical_total_bytes += logical_bytes
    else:
        with path_obj.open("rb") as stream:
            logical_bytes = _npy_logical_bytes(
                stream,
                label=path_obj.name,
                max_bytes=max_member_bytes,
                max_header_bytes=max_header_bytes,
            )
        if logical_bytes > max_total_bytes:
            raise ValueError(
                "NumPy array logical payload too large: "
                f"{path_obj.name!r} declares {logical_bytes} bytes, "
                f"exceeding aggregate limit {max_total_bytes}"
            )
    kwargs.pop("allow_pickle", None)
    kwargs.pop("max_header_size", None)
    kwargs["max_header_size"] = max_header_bytes
    return np.load(str(path_obj), allow_pickle=False, **kwargs)


__all__ = [
    "MAX_JSON_BYTES",
    "MAX_NPZ_BYTES",
    "MAX_NPZ_MEMBER_BYTES",
    "MAX_NPZ_TOTAL_BYTES",
    "MAX_NPY_HEADER_BYTES",
    "checked_json_load",
    "checked_np_load",
    "require_file_size_at_most",
]
