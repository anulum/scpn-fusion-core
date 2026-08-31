# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — complete FAIR-MAST magnetic archive capture tests
"""Public-surface refusal tests for incomplete magnetic archive sources."""

from pathlib import Path

import pytest

from scpn_fusion.io import (
    MastMagneticArchiveValidationError,
    build_mast_complete_magnetic_archive_envelope,
    verify_mast_complete_magnetic_archive_source,
)

_PROVENANCE = Path(
    "validation/reference_data/mast/disruption_shots/MAGNETIC_ARCHIVE_PROVENANCE.json"
)
_ENVELOPE = Path("validation/reference_data/mast/disruption_shots/MAGNETIC_ARCHIVE_ENVELOPE.json")


def test_builder_rejects_an_absent_complete_archive(tmp_path: Path) -> None:
    """A manifest alone can never be treated as decoded physical source data."""
    with pytest.raises(MastMagneticArchiveValidationError, match="root is missing"):
        build_mast_complete_magnetic_archive_envelope(_PROVENANCE, tmp_path / "27707.zarr")


def test_source_verifier_rejects_an_incomplete_archive(tmp_path: Path) -> None:
    """The tracked envelope cannot validate a directory missing source objects."""
    (tmp_path / "magnetics").mkdir()
    with pytest.raises(MastMagneticArchiveValidationError, match="inventory differs"):
        verify_mast_complete_magnetic_archive_source(_ENVELOPE.read_bytes(), tmp_path)
