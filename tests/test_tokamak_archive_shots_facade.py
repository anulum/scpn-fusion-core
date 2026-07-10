# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Facade test for the tokamak archive synthetic-database generation wrapper.

The disruption/synthetic listing and loading wrappers are covered by the
archive facade tests; this closes the ``generate_synthetic_shot_database``
wrapper, which forwards an explicit output directory to the synthetic archive.
"""

from __future__ import annotations

from pathlib import Path

from scpn_fusion.io._tokamak_archive_shots import generate_synthetic_shot_database


def test_generate_synthetic_shot_database_writes_to_explicit_dir(tmp_path: Path) -> None:
    """Generating with an explicit output dir returns a catalogue of written shots."""
    catalogue = generate_synthetic_shot_database(output_dir=tmp_path, seed=1)
    assert len(catalogue) > 0
    assert all("shot_id" in entry and "machine" in entry for entry in catalogue)
    # Every catalogued shot is materialised as an NPZ file in the target directory.
    assert len(sorted(tmp_path.glob("*.npz"))) == len(catalogue)
