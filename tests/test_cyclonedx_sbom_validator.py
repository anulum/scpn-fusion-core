# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Dedicated tests for the build-profile CycloneDX schema validator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


pytest.importorskip(
    "cyclonedx",
    reason="exercised authoritatively by the SBOM workflow's hash-pinned build profile",
)

from tools.validate_cyclonedx_sbom import SbomValidationError, validate_sbom


def _write_sbom(tmp_path: Path, payload: object) -> Path:
    """Write one deterministic SBOM fixture for schema-validation tests."""
    path = tmp_path / "sbom.cdx.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_cyclonedx_validator_accepts_declared_valid_schema(tmp_path: Path) -> None:
    """A complete minimal CycloneDX document is admitted."""
    path = _write_sbom(
        tmp_path,
        {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "serialNumber": "urn:uuid:12345678-1234-4234-8234-123456789abc",
            "version": 1,
            "components": [],
        },
    )

    assert validate_sbom(path).to_version() == "1.6"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "root must be a JSON object"),
        ({"bomFormat": "SPDX", "specVersion": "1.6"}, "bomFormat"),
        ({"bomFormat": "CycloneDX", "specVersion": 1.6}, "specVersion must be"),
        ({"bomFormat": "CycloneDX", "specVersion": "99.0"}, "unsupported"),
        (
            {"bomFormat": "CycloneDX", "specVersion": "1.6", "version": 0},
            "validation failed",
        ),
    ],
)
def test_cyclonedx_validator_rejects_invalid_documents(
    tmp_path: Path, payload: object, message: str
) -> None:
    """Format, version, and schema violations all fail closed."""
    with pytest.raises(SbomValidationError, match=message):
        validate_sbom(_write_sbom(tmp_path, payload))
