# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — retired partial FAIR-MAST magnetic ingestion test
"""Public refusal contract and real legacy summary-report write surface."""

import json
from pathlib import Path

import pytest

from scpn_fusion.io import MastMagneticArchiveValidationError
from scpn_fusion.io.mast_ingestor import MastIngestor
from tools import acquire_mast_level2_panel


def test_partial_magnetic_reader_is_unconditionally_rejected() -> None:
    """No dependency state can reactivate first-ten magnetic selection."""
    with pytest.raises(MastMagneticArchiveValidationError, match="partial"):
        MastIngestor.load_magnetic_probes(27707)


def test_summary_panel_report_is_written_atomically(tmp_path: Path) -> None:
    """The remaining summary-only tool writes its exact report without substitution."""
    output = tmp_path / "summary-panel.json"
    report = {
        "magnetic_contract": "complete_group_required",
        "schema": "scpn-fusion.mast-level2-summary-panel-run.v1",
    }
    acquire_mast_level2_panel.write_report(output, report)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert not output.with_suffix(".json.tmp").exists()
