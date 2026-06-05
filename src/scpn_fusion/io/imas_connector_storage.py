# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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
from scpn_fusion.io.safe_loaders import checked_json_load

_VALID_IDS_TYPES = ("equilibrium", "core_profiles", "core_transport", "summary")
MAX_IDS_JSON_DEPTH = 64
MAX_IDS_JSON_LIST_LENGTH = 250_000


def _numpy_serializer(obj: Any) -> Any:
    """JSON serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _validate_json_shape(value: Any, *, path: str = "$", depth: int = 0) -> None:
    """Reject JSON payloads that exceed bounded IDS traversal limits."""
    if depth > MAX_IDS_JSON_DEPTH:
        raise ValueError(f"IDS JSON nesting exceeds safety limit at {path}.")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"IDS JSON object key at {path} must be a string.")
            _validate_json_shape(item, path=f"{path}.{key}", depth=depth + 1)
        return
    if isinstance(value, list):
        if len(value) > MAX_IDS_JSON_LIST_LENGTH:
            raise ValueError(
                f"IDS JSON list at {path} exceeds safety limit {MAX_IDS_JSON_LIST_LENGTH}."
            )
        for idx, item in enumerate(value):
            _validate_json_shape(item, path=f"{path}[{idx}]", depth=depth + 1)


def _validate_minimal_ids_schema(data: Mapping[str, Any]) -> None:
    """Validate the minimum fail-closed IDS storage contract."""
    if "ids_properties" not in data:
        raise ValueError("IDS file missing 'ids_properties' key.")
    props = data["ids_properties"]
    if not isinstance(props, Mapping) or "homogeneous_time" not in props:
        raise ValueError("ids_properties must contain 'homogeneous_time'.")
    homogeneous_time = props["homogeneous_time"]
    if isinstance(homogeneous_time, bool) or not isinstance(homogeneous_time, int):
        raise ValueError("ids_properties.homogeneous_time must be an integer.")
    if "time_slice" in data:
        from scpn_fusion.io.imas_connector_equilibrium import validate_imas_equilibrium_payload

        validate_imas_equilibrium_payload(data)


def write_ids(ids_dict: Mapping[str, Any], path: str | Path) -> None:
    """Write an IDS dict to a JSON file with schema validation."""
    if not isinstance(ids_dict, Mapping):
        raise ValueError("ids_dict must be a mapping.")
    _validate_json_shape(ids_dict)
    _validate_minimal_ids_schema(ids_dict)

    path_obj = Path(path)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(dict(ids_dict), f, indent=2, default=_numpy_serializer)


def read_ids(path: str | Path) -> dict[str, Any]:
    """Read an IDS JSON file and validate minimal schema."""
    path_obj = Path(path)
    try:
        data = checked_json_load(path_obj)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupt IDS JSON file: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("IDS file must contain a JSON object at top level.")
    _validate_json_shape(data)
    _validate_minimal_ids_schema(data)
    return data


__all__ = [
    "_VALID_IDS_TYPES",
    "MAX_IDS_JSON_DEPTH",
    "MAX_IDS_JSON_LIST_LENGTH",
    "_numpy_serializer",
    "write_ids",
    "read_ids",
]
