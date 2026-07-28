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
from typing import Any

import numpy as np

MAX_JSON_BYTES = 10 * 1024 * 1024
MAX_NPZ_BYTES = 10 * 1024 * 1024
MAX_NPZ_MEMBER_BYTES = 10 * 1024 * 1024
MAX_NPZ_TOTAL_BYTES = 10 * 1024 * 1024


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
    **kwargs: Any,
) -> Any:
    """Open a bounded NumPy file with pickle loading disabled.

    ZIP metadata is checked before NumPy can materialise any NPZ member. Plain
    NPY files retain the compressed-file size check and bypass NPZ expansion
    accounting.

    Parameters
    ----------
    path:
        NumPy file to open.
    max_bytes:
        Maximum size of the stored file.
    max_member_bytes:
        Maximum uncompressed size of one NPZ member.
    max_total_bytes:
        Maximum aggregate uncompressed size of all NPZ members.
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
        If a stored or expanded size exceeds its configured bound.
    """
    path_obj = require_file_size_at_most(path, max_bytes=max_bytes, label="NumPy archive")
    if zipfile.is_zipfile(path_obj):
        total_bytes = 0
        with zipfile.ZipFile(path_obj) as archive:
            for member in archive.infolist():
                if member.file_size > max_member_bytes:
                    raise ValueError(
                        "NumPy archive member too large: "
                        f"{member.filename!r} expands to {member.file_size} bytes, "
                        f"exceeding {max_member_bytes}"
                    )
                total_bytes += member.file_size
                if total_bytes > max_total_bytes:
                    raise ValueError(
                        "NumPy archive expands too large: "
                        f"members total more than {max_total_bytes} bytes"
                    )
    kwargs.pop("allow_pickle", None)
    return np.load(str(path_obj), allow_pickle=False, **kwargs)


__all__ = [
    "MAX_JSON_BYTES",
    "MAX_NPZ_BYTES",
    "MAX_NPZ_MEMBER_BYTES",
    "MAX_NPZ_TOTAL_BYTES",
    "checked_json_load",
    "checked_np_load",
    "require_file_size_at_most",
]
