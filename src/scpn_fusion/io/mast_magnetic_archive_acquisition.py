# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — complete FAIR-MAST magnetic archive acquisition
"""Materialise a complete FAIR-MAST magnetic group from a tracked manifest."""

from __future__ import annotations

import json
import os
import tempfile
import time
import weakref
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast

from .mast_magnetic_archive import build_mast_complete_magnetic_archive_envelope
from .mast_magnetic_archive_codec import (
    JsonObject,
    JsonValue,
    MastCompleteMagneticArchiveEnvelope,
    MastMagneticArchiveValidationError,
    mast_complete_magnetic_archive_sha256,
)

_FAIR_MAST_ROOT = "https://s3.echo.stfc.ac.uk/mast/level2/shots/"
_OBJECT_KEYS = {"path", "sha256", "size_bytes", "source_url"}


class _S3Reader(Protocol):
    def cat(self, path: str) -> bytes:
        """Return one complete S3 object."""

    def info(self, path: str) -> Mapping[str, object]:
        """Return source-object metadata including exact size."""

    def close(self) -> None:
        """Close retained network sessions."""


class _FairMastS3Reader:
    """Typed lifecycle boundary around the untyped s3fs client."""

    def __init__(self, filesystem: Any) -> None:
        self._filesystem = filesystem

    def cat(self, path: str) -> bytes:
        return cast(bytes, self._filesystem.cat(path))

    def info(self, path: str) -> Mapping[str, object]:
        return cast(Mapping[str, object], self._filesystem.info(path))

    def close(self) -> None:
        loop = self._filesystem.loop
        creator = self._filesystem._s3creator
        registry = getattr(weakref.finalize, "_registry", {})
        for finalizer, info in list(registry.items()):
            arguments = cast(tuple[Any, ...], getattr(info, "args", ()))
            function = getattr(info, "func", None)
            if (
                len(arguments) >= 2
                and arguments[0] is loop
                and arguments[1] is creator
                and getattr(function, "__module__", "") == "s3fs.core"
                and getattr(function, "__name__", "") == "close_session"
            ):
                finalizer.detach()
        self._filesystem.close_session(loop, creator)


class MastMagneticArchiveAcquisitionError(RuntimeError):
    """Raised when a complete authenticated source group cannot be acquired."""


def acquire_mast_complete_magnetic_archive(
    provenance_path: Path,
    archive_parent: Path,
    *,
    attempts: int = 3,
    timeout_seconds: float = 60.0,
) -> MastCompleteMagneticArchiveEnvelope:
    """Download every declared magnetic object and return its verified envelope.

    Existing objects are reused only after exact byte-count and SHA-256 checks. A
    corrupt object is atomically replaced; undeclared local objects are never
    removed and cause the final complete-group verifier to fail closed.
    """
    if attempts < 1:
        raise ValueError("attempts must be positive")
    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive")
    shot_id, records = _read_magnetic_object_inventory(provenance_path)
    destination = archive_parent.resolve() / f"{shot_id}.zarr"
    if destination.is_symlink():
        raise MastMagneticArchiveAcquisitionError("archive destination is a symlink")
    destination.mkdir(parents=True, exist_ok=True)
    filesystem = _open_fair_mast_filesystem(timeout_seconds)
    try:
        for record in records:
            relative = cast(str, record["path"]).removeprefix(f"raw/{shot_id}.zarr/")
            relative_path = PurePosixPath(relative)
            if relative != relative_path.as_posix() or ".." in relative_path.parts:
                raise MastMagneticArchiveValidationError("source object path is unsafe")
            target = destination.joinpath(*relative_path.parts)
            _materialise_object(
                filesystem,
                record,
                target,
                attempts=attempts,
            )
    finally:
        filesystem.close()
    return build_mast_complete_magnetic_archive_envelope(provenance_path, destination)


def _read_magnetic_object_inventory(provenance_path: Path) -> tuple[int, list[JsonObject]]:
    if provenance_path.is_symlink() or not provenance_path.is_file():
        raise MastMagneticArchiveValidationError("provenance manifest is not a regular file")
    try:
        document = json.loads(provenance_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MastMagneticArchiveValidationError("provenance manifest is not valid JSON") from exc
    provenance = _as_object(cast(JsonValue, document), "provenance")
    if provenance.get("schema") != "scpn-fusion-open-disruption-data-provenance.v1":
        raise MastMagneticArchiveValidationError("provenance schema is unsupported")
    license_record = _as_object(provenance.get("license"), "provenance.license")
    if license_record.get("spdx") != "CC-BY-SA-4.0":
        raise MastMagneticArchiveValidationError("provenance licence is unsupported")
    dataset = _as_object(provenance.get("dataset"), "provenance.dataset")
    shot_id = _as_positive_integer(dataset.get("shot_id"), "provenance.dataset.shot_id")
    if dataset.get("device") != "MAST":
        raise MastMagneticArchiveValidationError("provenance device is not MAST")
    downloaded_groups = dataset.get("downloaded_groups")
    if not isinstance(downloaded_groups, list) or "magnetics" not in downloaded_groups:
        raise MastMagneticArchiveValidationError("provenance omits the magnetics group")
    files = dataset.get("files")
    if not isinstance(files, list):
        raise MastMagneticArchiveValidationError("provenance.dataset.files must be an array")
    shot_prefix = f"raw/{shot_id}.zarr/"
    magnetic_prefix = f"{shot_prefix}magnetics/"
    parent_records: list[JsonObject] = []
    for index, item in enumerate(files):
        record = _as_object(item, f"provenance.dataset.files[{index}]")
        if set(record) != _OBJECT_KEYS:
            raise MastMagneticArchiveValidationError("source object keys differ")
        path = record.get("path")
        if not isinstance(path, str) or not path.startswith(shot_prefix):
            raise MastMagneticArchiveValidationError("source object path is cross-shot")
        pure_path = PurePosixPath(path)
        if path != pure_path.as_posix() or ".." in pure_path.parts:
            raise MastMagneticArchiveValidationError("source object path is unsafe")
        source_url = record.get("source_url")
        expected_url = _FAIR_MAST_ROOT + path.removeprefix("raw/")
        if source_url != expected_url:
            raise MastMagneticArchiveValidationError(f"source URL does not bind {path}")
        digest = record.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise MastMagneticArchiveValidationError(f"source digest is invalid for {path}")
        _as_nonnegative_integer(record.get("size_bytes"), f"size for {path}")
        parent_records.append(record)
    parent_records.sort(key=lambda record: cast(str, record["path"]))
    if len({cast(str, record["path"]) for record in parent_records}) != len(parent_records):
        raise MastMagneticArchiveValidationError("parent object inventory is duplicate")
    manifest = "".join(
        f"{record['sha256']}:{record['size_bytes']}:{record['path']}\n" for record in parent_records
    ).encode("utf-8")
    if mast_complete_magnetic_archive_sha256(manifest) != dataset.get("download_manifest_sha256"):
        raise MastMagneticArchiveValidationError("parent download manifest digest differs")
    records = [
        record for record in parent_records if cast(str, record["path"]).startswith(magnetic_prefix)
    ]
    if not records:
        raise MastMagneticArchiveValidationError("magnetic object inventory is empty or duplicate")
    return shot_id, records


def _materialise_object(
    filesystem: _S3Reader,
    record: JsonObject,
    target: Path,
    *,
    attempts: int,
) -> None:
    if target.is_symlink():
        raise MastMagneticArchiveAcquisitionError(f"object destination is a symlink: {target}")
    if target.is_file() and _matches_record(target, record):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    source_key = cast(str, record["source_url"]).removeprefix("https://s3.echo.stfc.ac.uk/")
    for attempt in range(1, attempts + 1):
        temporary_path: Path | None = None
        try:
            source_size = filesystem.info(source_key).get("size")
            if source_size != record["size_bytes"]:
                raise MastMagneticArchiveAcquisitionError(
                    f"remote size differs from manifest: {record['path']}"
                )
            data = filesystem.cat(source_key)
            if len(data) != record["size_bytes"]:
                raise MastMagneticArchiveAcquisitionError(
                    f"downloaded size differs from manifest: {record['path']}"
                )
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(data)
                temporary.flush()
                os.fsync(temporary.fileno())
            if temporary_path is None or not _matches_record(temporary_path, record):
                raise MastMagneticArchiveAcquisitionError(
                    f"downloaded object failed integrity: {record['path']}"
                )
            os.replace(temporary_path, target)
            return
        except (OSError, MastMagneticArchiveAcquisitionError) as exc:
            last_error = exc
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise MastMagneticArchiveAcquisitionError(
        f"cannot acquire {record['path']} after {attempts} attempts: {last_error}"
    )


def _open_fair_mast_filesystem(timeout_seconds: float) -> _S3Reader:
    try:
        s3fs = import_module("s3fs")
        filesystem = s3fs.S3FileSystem(
            anon=True,
            client_kwargs={"endpoint_url": "https://s3.echo.stfc.ac.uk"},
            config_kwargs={
                "connect_timeout": timeout_seconds,
                "read_timeout": timeout_seconds,
                "retries": {"max_attempts": 1},
            },
            skip_instance_cache=True,
        )
    except (AttributeError, ImportError, OSError) as exc:
        raise MastMagneticArchiveAcquisitionError(
            "complete FAIR-MAST acquisition requires the hash-locked mast profile"
        ) from exc
    return _FairMastS3Reader(filesystem)


def _matches_record(path: Path, record: JsonObject) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        data = path.read_bytes()
    except OSError:
        return False
    return (
        len(data) == record["size_bytes"]
        and mast_complete_magnetic_archive_sha256(data) == record["sha256"]
    )


def _as_object(value: object, path: str) -> JsonObject:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise MastMagneticArchiveValidationError(f"{path} must be an object")
    return cast(JsonObject, value)


def _as_nonnegative_integer(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise MastMagneticArchiveValidationError(f"{path} must be a non-negative integer")
    return value


def _as_positive_integer(value: object, path: str) -> int:
    value = _as_nonnegative_integer(value, path)
    if value == 0:
        raise MastMagneticArchiveValidationError(f"{path} must be positive")
    return value


__all__ = [
    "MastMagneticArchiveAcquisitionError",
    "acquire_mast_complete_magnetic_archive",
]
