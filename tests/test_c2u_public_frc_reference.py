# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — C-2U Public FRC Reference Tests
"""Tests for the public C-2U FRC performance reference table.

Cover the loader, summary, acceptance-status, and CSV parsing/validation
surfaces end to end: the tracked reference artifact, its SI-unit conversions
and claim boundary, the strictly-increasing shot-id contract, the
missing-artifact acceptance path, and every row/column/field validation guard.
The module is a public reference-data loader with no compute hot path and no
Rust/Julia/Go counterpart, so no polyglot parity surface is involved.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scpn_fusion.core import c2u_positive_heating_reference_status
from scpn_fusion.core import public_frc_reference as frc_module
from scpn_fusion.core.public_frc_reference import (
    C2U_REFERENCE_CSV,
    C2U_REFERENCE_METADATA,
    _parse_int,
    _parse_positive,
    load_c2u_positive_heating_shots,
    summarise_c2u_positive_heating_shots,
)

_VALID_ROW: dict[str, str] = {
    "shot": "1",
    "Eth(kJ)": "1.0",
    "Fp(mWb)": "2.0",
    "T(keV)": "0.5",
    "t_max(ms)": "1.0",
    "P_max(MW)": "0.1",
    "E_max(kJ)": "1.1",
    "comment": "ok",
}
_HEADER = ",".join(_VALID_ROW)


def test_c2u_reference_table_loads_with_si_conversions() -> None:
    """The tracked reference table loads with the documented SI conversions."""
    shots = load_c2u_positive_heating_shots()

    assert len(shots) == 30
    assert [shot.shot for shot in shots] == sorted(shot.shot for shot in shots)
    assert shots[0].shot == 45849
    assert shots[-1].shot == 46481

    record = next(shot for shot in shots if shot.shot == 46481)
    assert record.thermal_energy_j == pytest.approx(4359.0)
    assert record.poloidal_flux_wb == pytest.approx(6.670e-3)
    assert record.total_temperature_ev == pytest.approx(623.0)
    assert record.time_of_max_heating_s == pytest.approx(1.02e-3)
    assert record.net_heating_power_w == pytest.approx(0.133e6)
    assert record.energy_at_max_heating_j == pytest.approx(3828.0)
    assert record.energy_per_flux_j_per_wb == pytest.approx(4359.0 / 6.670e-3)


def test_c2u_reference_summary_preserves_observed_ranges_and_claim_boundary() -> None:
    """The summary reports observed ranges and the fixed claim boundary."""
    summary = summarise_c2u_positive_heating_shots()

    assert summary.shot_count == 30
    assert summary.shot_min == 45849
    assert summary.shot_max == 46481
    assert summary.max_thermal_energy_j == pytest.approx(4359.0)
    assert summary.max_total_temperature_ev == pytest.approx(747.0)
    assert summary.max_net_heating_power_w == pytest.approx(0.933e6)
    assert summary.max_poloidal_flux_wb == pytest.approx(7.117e-3)
    assert summary.claim_boundary == (
        "public C-2U positive-net-heating shot table; not Slough Fig. 5 "
        "trajectory parity and not a time-resolved compression benchmark"
    )


def test_c2u_reference_metadata_matches_tracked_artifacts() -> None:
    """The tracked metadata sidecar matches the reference CSV provenance."""
    metadata = json.loads(C2U_REFERENCE_METADATA.read_text(encoding="utf-8"))

    assert C2U_REFERENCE_CSV.exists()
    assert metadata["doi"] == "10.1038/s41598-017-06645-7"
    assert metadata["source_file"] == "frc_public/c2u_optometrist_positive_heating_shots.csv"
    assert metadata["source_sha256"] == (
        "d1168f63d66ba4a47eda52d1a7c8c39191c3af589ce35351881b5c3c42eb4262"
    )
    assert metadata["redistribution_license"] == "CC-BY-4.0"
    assert metadata["claim_boundary"].startswith("Public C-2U positive-net-heating")


def test_c2u_reference_status_reports_bounded_public_reference() -> None:
    """The acceptance status reports a bounded, available public reference."""
    status = c2u_positive_heating_reference_status()

    assert status["status"] == "public_reference_table_available"
    assert status["shot_count"] == 30
    assert status["max_thermal_energy_j"] == pytest.approx(4359.0)
    assert "not Slough Fig. 5" in str(status["claim_boundary"])


def test_c2u_reference_status_blocked_when_artifact_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The status fails closed when the public artifact is not present."""
    monkeypatch.setattr(frc_module, "C2U_REFERENCE_CSV", tmp_path / "absent.csv")
    status = c2u_positive_heating_reference_status()

    assert status["status"] == "blocked_missing_public_reference_artifact"
    assert status["required_artifact"] == "C-2U supplemental shot table plus metadata"


def test_c2u_reference_loader_rejects_nonphysical_rows(tmp_path: Path) -> None:
    """The loader rejects a shot-id sequence that is not strictly increasing."""
    bad = tmp_path / "bad_c2u.csv"
    bad.write_text(
        "\n".join(
            [
                "shot,Eth(kJ),Fp(mWb),T(keV),t_max(ms),P_max(MW),E_max(kJ),comment",
                "10,1.0,2.0,0.5,1.0,0.1,1.1,ok",
                "9,1.0,2.0,0.5,1.0,0.1,1.1,non-monotonic",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        load_c2u_positive_heating_shots(bad)


def test_c2u_reference_loader_rejects_empty_table(tmp_path: Path) -> None:
    """The loader rejects a table whose data rows are all comment-filtered."""
    empty = tmp_path / "empty_c2u.csv"
    empty.write_text(f"{_HEADER}\n# only a comment row\n", encoding="utf-8")

    with pytest.raises(ValueError, match="at least one shot"):
        load_c2u_positive_heating_shots(empty)


def test_c2u_reference_loader_rejects_missing_column(tmp_path: Path) -> None:
    """The loader rejects a row that omits a required column."""
    header = _HEADER.replace(",comment", "")
    incomplete = tmp_path / "missing_column_c2u.csv"
    incomplete.write_text(f"{header}\n1,1.0,2.0,0.5,1.0,0.1,1.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required column"):
        load_c2u_positive_heating_shots(incomplete)


def test_c2u_reference_loader_skips_none_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A ``None`` row from the CSV reader is skipped without aborting the load."""
    source = tmp_path / "none_row_c2u.csv"
    source.write_text(f"{_HEADER}\n", encoding="utf-8")

    def _reader_with_none(*_args: object, **_kwargs: object) -> object:
        return iter([None, dict(_VALID_ROW)])

    monkeypatch.setattr(csv, "DictReader", _reader_with_none)
    shots = load_c2u_positive_heating_shots(source)

    assert len(shots) == 1
    assert shots[0].shot == 1


def test_parse_int_rejects_non_integer() -> None:
    """``_parse_int`` rejects a non-integer cell with a located error."""
    with pytest.raises(ValueError, match="column shot must be an integer"):
        _parse_int("not-an-int", "shot", 7)


def test_parse_int_rejects_non_positive() -> None:
    """``_parse_int`` rejects a non-positive integer cell."""
    with pytest.raises(ValueError, match="column shot must be positive"):
        _parse_int("0", "shot", 7)


def test_parse_positive_rejects_non_numeric() -> None:
    """``_parse_positive`` rejects a non-numeric cell with a located error."""
    with pytest.raises(ValueError, match=r"column Eth\(kJ\) must be numeric"):
        _parse_positive("not-a-float", "Eth(kJ)", 7)


def test_parse_positive_rejects_non_positive() -> None:
    """``_parse_positive`` rejects a zero or negative numeric cell."""
    with pytest.raises(ValueError, match=r"column Eth\(kJ\) must be positive"):
        _parse_positive("0.0", "Eth(kJ)", 7)
