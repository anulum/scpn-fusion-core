# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — IMAS Connector Storage Tests
"""Tests for IMAS connector JSON storage contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, cast

import numpy as np
import pytest

from scpn_fusion.io import imas_connector
from scpn_fusion.io.imas_connector_storage import (
    MAX_IDS_JSON_LIST_LENGTH,
    read_ids,
    write_ids,
)


def _minimal_ids_payload() -> dict[str, object]:
    """Return the smallest IDS payload accepted by the storage contract."""
    return {"ids_properties": {"homogeneous_time": 1}}


def test_facade_write_and_read_serializes_numpy_values(tmp_path: Path) -> None:
    """The public IMAS facade writes NumPy values and reads plain JSON back."""
    path = tmp_path / "ids.json"
    payload = _minimal_ids_payload()
    payload.update(
        {
            "shot": np.int64(170001),
            "beta_N": np.float32(2.75),
            "rho_norm": np.array([0.0, 0.5, 1.0], dtype=np.float64),
        }
    )

    imas_connector.write_ids(payload, path)
    loaded = imas_connector.read_ids(path)

    assert loaded["ids_properties"] == {"homogeneous_time": 1}
    assert loaded["shot"] == 170001
    assert loaded["beta_N"] == 2.75
    assert loaded["rho_norm"] == [0.0, 0.5, 1.0]


def test_write_ids_rejects_non_mapping_payload(tmp_path: Path) -> None:
    """Storage writes reject non-mapping IDS payloads before file creation."""
    non_mapping = cast(Mapping[str, object], ["not", "a", "mapping"])

    with pytest.raises(ValueError, match="ids_dict must be a mapping"):
        write_ids(non_mapping, tmp_path / "ids.json")


def test_write_ids_rejects_non_string_json_object_key(tmp_path: Path) -> None:
    """Storage writes reject Python mappings that cannot be IDS JSON objects."""
    invalid_key_payload: Mapping[object, object] = {
        "ids_properties": {"homogeneous_time": 1},
        42: "bad-key",
    }

    with pytest.raises(ValueError, match="object key at \\$ must be a string"):
        write_ids(cast(Mapping[str, object], invalid_key_payload), tmp_path / "ids.json")


def test_write_ids_rejects_excessive_list_length(tmp_path: Path) -> None:
    """Storage writes bound list traversal before serializing IDS JSON."""
    payload = _minimal_ids_payload()
    payload["samples"] = [0] * (MAX_IDS_JSON_LIST_LENGTH + 1)

    with pytest.raises(ValueError, match="list at \\$.samples exceeds safety limit"):
        write_ids(payload, tmp_path / "ids.json")


def test_write_ids_rejects_missing_ids_properties(tmp_path: Path) -> None:
    """Storage writes require the IDS properties block."""
    with pytest.raises(ValueError, match="missing 'ids_properties'"):
        write_ids({"time": [0.0]}, tmp_path / "ids.json")


def test_write_ids_rejects_ids_properties_without_homogeneous_time(
    tmp_path: Path,
) -> None:
    """Storage writes require homogeneous-time metadata inside ids_properties."""
    with pytest.raises(ValueError, match="must contain 'homogeneous_time'"):
        write_ids({"ids_properties": {}}, tmp_path / "ids.json")


def test_write_ids_rejects_boolean_homogeneous_time(tmp_path: Path) -> None:
    """Storage writes reject bool values for homogeneous_time despite int subclassing."""
    with pytest.raises(ValueError, match="homogeneous_time must be an integer"):
        write_ids({"ids_properties": {"homogeneous_time": True}}, tmp_path / "ids.json")


def test_write_ids_reports_unsupported_json_value(tmp_path: Path) -> None:
    """Storage writes surface unsupported JSON values from the serializer."""
    payload = _minimal_ids_payload()
    payload["opaque"] = object()

    with pytest.raises(TypeError, match="Object of type object is not JSON serializable"):
        write_ids(payload, tmp_path / "ids.json")


def test_read_ids_rejects_corrupt_json(tmp_path: Path) -> None:
    """Storage reads convert JSON parser failures into IDS file errors."""
    path = tmp_path / "corrupt.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="Corrupt IDS JSON file"):
        read_ids(path)


def test_read_ids_rejects_non_object_top_level(tmp_path: Path) -> None:
    """Storage reads require a JSON object at the IDS file root."""
    path = tmp_path / "list-root.json"
    path.write_text(json.dumps([_minimal_ids_payload()]), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object at top level"):
        read_ids(path)
