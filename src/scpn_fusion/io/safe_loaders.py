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
from pathlib import Path
from typing import Any

import numpy as np

MAX_JSON_BYTES = 10 * 1024 * 1024
MAX_NPZ_BYTES = 10 * 1024 * 1024


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
    **kwargs: Any,
) -> Any:
    """Open a NumPy archive after enforcing a byte-size bound and disabling pickle."""
    path_obj = require_file_size_at_most(path, max_bytes=max_bytes, label="NumPy archive")
    kwargs.pop("allow_pickle", None)
    return np.load(str(path_obj), allow_pickle=False, **kwargs)


__all__ = [
    "MAX_JSON_BYTES",
    "MAX_NPZ_BYTES",
    "checked_json_load",
    "checked_np_load",
    "require_file_size_at_most",
]
