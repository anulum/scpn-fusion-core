# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Reference-case tests for the public TGLF interface helpers.

Exercises the in-tree ITG/TEM/ETG reference suite, the filename mapping, the
reference-metadata to transport-input reconstruction, and the reference-data
writer that the module-linkage import never runs.
"""

from __future__ import annotations

import json
from pathlib import Path

from scpn_fusion.core._tglf_interface_reference import (
    REFERENCE_CASES,
    _reference_case_filename,
    validate_reduced_transport_reference_case,
    validate_reduced_transport_reference_suite,
    write_reference_data,
)
from scpn_fusion.core._tglf_interface_types import TGLFReferenceCaseResult


def test_reference_case_filename_normalises_title() -> None:
    """Case titles map to lower-case underscore-joined JSON filenames."""
    assert _reference_case_filename("ITG-dominated") == "itg_dominated.json"
    assert _reference_case_filename("TEM dominated") == "tem_dominated.json"


def test_validate_single_reference_case_returns_result() -> None:
    """Validating a single in-tree reference case yields a populated result."""
    result = validate_reduced_transport_reference_case("ITG-dominated")
    assert isinstance(result, TGLFReferenceCaseResult)
    assert result.reference_mode
    assert result.predicted_mode
    assert result.rel_error_chi_i >= 0.0
    assert result.rel_error_chi_e >= 0.0


def test_validate_reference_suite_covers_all_regimes() -> None:
    """The reference suite validates each of the three canonical regimes."""
    results = validate_reduced_transport_reference_suite()
    assert len(results) == 3
    assert all(isinstance(r, TGLFReferenceCaseResult) for r in results)


def test_write_reference_data_emits_all_cases(tmp_path: Path) -> None:
    """The writer materialises one JSON file per reference case."""
    write_reference_data(output_dir=tmp_path)
    written = sorted(p.name for p in tmp_path.glob("*.json"))
    expected = sorted(_reference_case_filename(name) for name in REFERENCE_CASES)
    assert written == expected
    # Each written payload carries its case name and the source provenance tag.
    sample = json.loads((tmp_path / expected[0]).read_text(encoding="utf-8"))
    assert "case_name" in sample
    assert sample["source"] == "TGLF v4 reference"
