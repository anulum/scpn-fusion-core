# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TORAX Runtime Serialization Tests
"""Filesystem and digest tests for TORAX runtime serialization."""

from __future__ import annotations

from pathlib import Path

import pytest

from scpn_fusion.integrations.torax.serialization import (
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    load_json_object,
    write_json_atomic,
)


def test_canonical_json_is_order_independent_and_rejects_nonfinite_values() -> None:
    """Equivalent JSON objects hash identically while NaN and infinity fail closed."""
    left = {"b": [2, 3], "a": {"value": 1.0}}
    right = {"a": {"value": 1.0}, "b": [2, 3]}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_sha256(left) == canonical_sha256(right)
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="non-finite"):
            canonical_json_bytes({"value": value})


def test_atomic_json_round_trip_and_regular_file_custody(tmp_path: Path) -> None:
    """The public writer publishes one bounded regular file with a stable digest."""
    path = tmp_path / "nested/outcome.json"
    write_json_atomic(path, {"schema": "test.v1", "values": [1, 2, 3]})
    assert load_json_object(path) == {"schema": "test.v1", "values": [1, 2, 3]}
    assert len(file_sha256(path)) == 64
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_loader_rejects_symlinks_empty_files_oversize_and_nonobject_roots(tmp_path: Path) -> None:
    """Untrusted runtime JSON must be bounded, object-rooted, and non-symlinked."""
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="regular non-symlink"):
        load_json_object(link)
    empty = tmp_path / "empty.json"
    empty.touch()
    with pytest.raises(ValueError, match="file size"):
        load_json_object(empty)
    oversized = tmp_path / "oversized.json"
    oversized.write_text('{"payload":"12345"}', encoding="utf-8")
    with pytest.raises(ValueError, match="file size"):
        load_json_object(oversized, maximum_bytes=5)
    array_root = tmp_path / "array.json"
    array_root.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be an object"):
        load_json_object(array_root)


def test_loader_rejects_duplicate_object_keys_at_any_depth(tmp_path: Path) -> None:
    """Ambiguous cross-repository JSON cannot silently use the last duplicate value."""
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"v1","nested":{"value":1,"value":2}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key: value"):
        load_json_object(duplicate)
