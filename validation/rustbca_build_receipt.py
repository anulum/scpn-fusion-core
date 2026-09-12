# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — RustBCA local build observation custody
"""Check retained native build inputs without claiming independent attestation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SOURCE_COMMIT = "18920a26263c21b9e337db1bab1fd17f23f0fec5"
SOURCE_MANIFEST_SHA256 = "0c99321189b5228fadce6e398929992ec5d97fc64880772d46a58c6529e1ca9d"
CARGO_LOCK_SHA256 = "f34c453015ac022a34aecce5ef54a67ef886fdb42459241ffd0b4e03a692b1cf"


def verify_rustbca_build_receipt(
    receipt: Path, *, expected_receipt_sha256: str, expected_binary_sha256: str
) -> dict[str, Any]:
    """Verify a pinned local build observation and its retained file dependencies.

    Parameters
    ----------
    receipt : Path
        Native build result.json beside source/, build.log and build_source.py.
        The recorded binary path must still resolve to the observed build output.
    expected_receipt_sha256 : str
        Previously captured SHA256 of the exact observation, supplied by the caller.
    expected_binary_sha256 : str
        SHA256 independently selected for the runtime extension to execute.

    Returns
    -------
    dict[str, Any]
        Input-custody status and pinned identities. Independent attestation and
        physical admission remain false, even when every file matches.

    Raises
    ------
    ValueError
        If identities, source inventory, build outcome or retained bytes disagree.
    OSError
        If any required retained file cannot be read.

    Notes
    -----
    This validates the local record's consistency, not who produced it or whether
    its asserted compilation occurred. The source inventory is pinned separately
    from the caller's record. Host toolchain hermeticity and binary reproducibility
    are not established. Recheck after execution to detect changed retained inputs.
    """
    raw = receipt.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_receipt_sha256:
        raise ValueError("RustBCA build receipt digest mismatch")
    record = json.loads(raw)
    if not isinstance(record, dict):
        raise ValueError("RustBCA build receipt must be an object")
    upstream = record.get("upstream")
    expected_upstream = {
        "repository": "https://github.com/lcpp-org/RustBCA",
        "commit": SOURCE_COMMIT,
        "archive_sha256": "3aded40a7a3282ac41e4a1c90198511f26bf8d418ec2994914b6bf930424f4b7",
        "license_sha256": "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986",
    }
    if upstream != expected_upstream:
        raise ValueError("RustBCA build upstream identity mismatch")
    if (
        record.get("schema") != "scpn-fusion.rustbca-native-build-observation.v1"
        or type(record.get("returncode")) is not int
        or record["returncode"] != 0
        or any(
            record.get(key) is not True
            for key in ("source_unchanged", "lock_unchanged", "tools_unchanged")
        )
    ):
        raise ValueError("RustBCA build observation is not a successful verified build")
    files = record.get("source_files")
    if not isinstance(files, dict):
        raise ValueError("RustBCA source inventory must be an object")
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != SOURCE_MANIFEST_SHA256:
        raise ValueError("RustBCA source inventory mismatch")
    source = receipt.parent / "source"
    actual_files = {str(path.relative_to(source)) for path in source.rglob("*") if path.is_file()}
    if actual_files != set(files) | {"Cargo.lock"}:
        raise ValueError("RustBCA source tree contains missing or unrecorded files")
    if record.get("cargo_lock_sha256") != CARGO_LOCK_SHA256:
        raise ValueError("RustBCA lock identity mismatch")
    if record.get("binary_sha256") != expected_binary_sha256:
        raise ValueError("RustBCA build and runtime binary identities differ")
    binary_name = record.get("binary")
    if not isinstance(binary_name, str) or not Path(binary_name).is_absolute():
        raise ValueError("RustBCA build binary path must be absolute")
    required = {receipt.parent / "source" / name: digest for name, digest in files.items()}
    required.update(
        {
            receipt.parent / "source/Cargo.lock": CARGO_LOCK_SHA256,
            receipt.parent / "build.log": record.get("log_sha256"),
            receipt.parent / "build_source.py": record.get("recipe_sha256"),
            Path(binary_name): expected_binary_sha256,
        }
    )
    for path, expected in required.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"RustBCA retained build file changed: {path}")
    if receipt.read_bytes() != raw:
        raise ValueError("RustBCA build receipt changed during verification")
    return {
        "receipt_sha256": actual,
        "source_commit": SOURCE_COMMIT,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "cargo_lock_sha256": CARGO_LOCK_SHA256,
        "binary_sha256": expected_binary_sha256,
        "retained_build_inputs_verified": True,
        "independent_attestation_verified": False,
        "binary_reproducibility_verified": False,
        "status": "local_build_observation",
    }
