# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — CycloneDX SBOM Validator
"""Fail-closed schema validation for generated CycloneDX JSON SBOMs."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class SbomValidationError(ValueError):
    """Raised when an SBOM cannot be admitted as valid CycloneDX JSON."""


def _cyclonedx_runtime() -> tuple[Any, Any, Any]:
    """Load the build-only CycloneDX runtime when validation is requested."""
    try:
        schema_module = importlib.import_module("cyclonedx.schema")
        validation_module = importlib.import_module("cyclonedx.validation")
    except ImportError as exc:
        raise SbomValidationError(
            "CycloneDX validation requires the hash-pinned requirements/build.txt profile"
        ) from exc
    return (
        schema_module.OutputFormat,
        schema_module.SchemaVersion,
        validation_module.make_schemabased_validator,
    )


def _schema_version(payload: Any, *, path: Path, schema_version_type: Any) -> Any:
    """Resolve the declared CycloneDX version without accepting coercions."""
    if not isinstance(payload, dict):
        raise SbomValidationError(f"{path}: root must be a JSON object")
    if payload.get("bomFormat") != "CycloneDX":
        raise SbomValidationError(f"{path}: bomFormat must be 'CycloneDX'")
    declared = payload.get("specVersion")
    if not isinstance(declared, str):
        raise SbomValidationError(f"{path}: specVersion must be a string")
    try:
        return schema_version_type.from_version(declared)
    except (TypeError, ValueError) as exc:
        raise SbomValidationError(
            f"{path}: unsupported CycloneDX specVersion {declared!r}"
        ) from exc


def validate_sbom(path: Path) -> Any:
    """Validate one UTF-8 JSON document against its declared CycloneDX schema."""
    try:
        document = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SbomValidationError(f"{path}: cannot read UTF-8 JSON: {exc}") from exc
    try:
        payload: Any = json.loads(document)
    except json.JSONDecodeError as exc:
        raise SbomValidationError(f"{path}: malformed JSON: {exc}") from exc

    output_format, schema_version_type, validator_factory = _cyclonedx_runtime()
    schema_version = _schema_version(payload, path=path, schema_version_type=schema_version_type)
    validator = validator_factory(output_format.JSON, schema_version)
    errors = validator.validate_str(document, all_errors=True)
    if errors is not None:
        details = "; ".join(str(error) for error in errors)
        raise SbomValidationError(
            f"{path}: CycloneDX {schema_version.to_version()} validation failed: {details}"
        )
    return schema_version


def main(argv: Sequence[str] | None = None) -> int:
    """Validate every requested SBOM and report the admitted schema version."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="CycloneDX JSON SBOM paths")
    args = parser.parse_args(argv)
    for path in args.paths:
        schema_version = validate_sbom(path)
        print(f"validated {path}: CycloneDX {schema_version.to_version()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
