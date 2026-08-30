# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TORAX Runtime Serialization
"""Canonical JSON, bounded loading, atomic writes, and SHA-256 custody."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, cast

MAX_RUNTIME_JSON_BYTES = 64 * 1024 * 1024


def canonical_json_bytes(value: object) -> bytes:
    """Return the unique UTF-8 representation used by runtime digests."""
    plain_value = _plain_json(value)
    _validate_json_value(plain_value, path="$")
    return json.dumps(
        plain_value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible value with the runtime canonical encoding."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a regular non-symlink file without loading it into memory."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path, *, maximum_bytes: int = MAX_RUNTIME_JSON_BYTES) -> dict[str, Any]:
    """Load one bounded JSON object from a regular non-symlink file."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular non-symlink JSON file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise ValueError(f"JSON file size must be in [1, {maximum_bytes}] bytes: {path}")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_object_without_duplicate_keys,
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    _validate_json_value(value, path="$")
    return cast(dict[str, Any], value)


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically publish canonical JSON in the destination directory."""
    write_bytes_atomic(path, canonical_json_bytes(dict(value)) + b"\n")


def write_bytes_atomic(
    path: Path,
    payload: bytes,
    *,
    maximum_bytes: int = MAX_RUNTIME_JSON_BYTES,
) -> None:
    """Atomically publish exact bounded bytes in the destination directory."""
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("atomic payload must be non-empty bytes")
    if len(payload) > maximum_bytes:
        raise ValueError(f"atomic payload exceeds {maximum_bytes} bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_json_value(value: object, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"non-finite JSON number at {path}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"non-string JSON object key at {path}")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise ValueError(f"non-JSON value at {path}: {type(value).__name__}")


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "MAX_RUNTIME_JSON_BYTES",
    "canonical_json_bytes",
    "canonical_sha256",
    "file_sha256",
    "load_json_object",
    "write_bytes_atomic",
    "write_json_atomic",
]
