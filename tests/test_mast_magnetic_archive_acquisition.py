# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — complete FAIR-MAST magnetic archive acquisition tests
"""Public acquisition contract tests that require no source substitution."""

from pathlib import Path

import pytest

from scpn_fusion.io import acquire_mast_complete_magnetic_archive

_PROVENANCE = Path(
    "validation/reference_data/mast/disruption_shots/MAGNETIC_ARCHIVE_PROVENANCE.json"
)


def test_acquisition_rejects_nonpositive_attempts(tmp_path: Path) -> None:
    """A nonpositive attempt count fails before any remote operation."""
    with pytest.raises(ValueError, match="attempts"):
        acquire_mast_complete_magnetic_archive(
            _PROVENANCE,
            tmp_path,
            attempts=0,
            timeout_seconds=60.0,
        )


def test_acquisition_rejects_nonpositive_timeout(tmp_path: Path) -> None:
    """A nonpositive timeout fails before any remote operation."""
    with pytest.raises(ValueError, match="timeout_seconds"):
        acquire_mast_complete_magnetic_archive(
            _PROVENANCE,
            tmp_path,
            attempts=1,
            timeout_seconds=0.0,
        )
