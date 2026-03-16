# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — IMAS Connector Storage
"""JSON file I/O helpers for IMAS IDS payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_VALID_IDS_TYPES = ("equilibrium", "core_profiles", "core_transport", "summary")


def _numpy_serializer(obj: Any) -> Any:
    """JSON serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_ids(ids_dict: Mapping[str, Any], path: str | Path) -> None:
    """Write an IDS dict to a JSON file with schema validation."""
    if not isinstance(ids_dict, Mapping):
        raise ValueError("ids_dict must be a mapping.")
    if "ids_properties" not in ids_dict:
        raise ValueError("ids_dict must contain 'ids_properties' key.")
    props = ids_dict["ids_properties"]
    if not isinstance(props, Mapping) or "homogeneous_time" not in props:
        raise ValueError("ids_properties must contain 'homogeneous_time'.")

    path_obj = Path(path)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(dict(ids_dict), f, indent=2, default=_numpy_serializer)


def read_ids(path: str | Path) -> dict[str, Any]:
    """Read an IDS JSON file and validate minimal schema."""
    path_obj = Path(path)
    try:
        with path_obj.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupt IDS JSON file: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("IDS file must contain a JSON object at top level.")
    if "ids_properties" not in data:
        raise ValueError("IDS file missing 'ids_properties' key.")
    props = data["ids_properties"]
    if not isinstance(props, dict) or "homogeneous_time" not in props:
        raise ValueError("ids_properties must contain 'homogeneous_time'.")
    return data


__all__ = [
    "_VALID_IDS_TYPES",
    "_numpy_serializer",
    "write_ids",
    "read_ids",
]
