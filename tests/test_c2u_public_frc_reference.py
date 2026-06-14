# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — C-2U Public FRC Reference Tests
"""Tests for the public C-2U FRC performance reference table."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_fusion.core import c2u_positive_heating_reference_status
from scpn_fusion.core.public_frc_reference import (
    C2U_REFERENCE_CSV,
    C2U_REFERENCE_METADATA,
    load_c2u_positive_heating_shots,
    summarise_c2u_positive_heating_shots,
)


def test_c2u_reference_table_loads_with_si_conversions() -> None:
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
    status = c2u_positive_heating_reference_status()

    assert status["status"] == "public_reference_table_available"
    assert status["shot_count"] == 30
    assert status["max_thermal_energy_j"] == pytest.approx(4359.0)
    assert "not Slough Fig. 5" in str(status["claim_boundary"])


def test_c2u_reference_loader_rejects_nonphysical_rows(tmp_path: Path) -> None:
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
