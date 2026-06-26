# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Public FRC Reference Data
"""Public FRC performance-reference datasets with explicit claim boundaries."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from scpn_fusion._data_paths import data_root

REFERENCE_ROOT = data_root() / "validation" / "reference_data"
C2U_REFERENCE_CSV = REFERENCE_ROOT / "frc_public" / "c2u_optometrist_positive_heating_shots.csv"
C2U_REFERENCE_METADATA = (
    REFERENCE_ROOT / "frc_public" / "c2u_optometrist_positive_heating_shots.metadata.json"
)
C2U_CLAIM_BOUNDARY = (
    "public C-2U positive-net-heating shot table; not Slough Fig. 5 "
    "trajectory parity and not a time-resolved compression benchmark"
)


@dataclass(frozen=True)
class C2UPositiveHeatingShot:
    """One public C-2U positive-net-heating shot row in SI-compatible units."""

    shot: int
    thermal_energy_j: float
    poloidal_flux_wb: float
    total_temperature_ev: float
    time_of_max_heating_s: float
    net_heating_power_w: float
    energy_at_max_heating_j: float
    comment: str

    @property
    def energy_per_flux_j_per_wb(self) -> float:
        """Return stored thermal energy per reported poloidal flux."""
        return self.thermal_energy_j / self.poloidal_flux_wb


@dataclass(frozen=True)
class C2UPositiveHeatingSummary:
    """Summary statistics for the public C-2U positive-net-heating table."""

    shot_count: int
    shot_min: int
    shot_max: int
    max_thermal_energy_j: float
    max_poloidal_flux_wb: float
    max_total_temperature_ev: float
    max_net_heating_power_w: float
    claim_boundary: str


def load_c2u_positive_heating_shots(
    path: str | Path = C2U_REFERENCE_CSV,
) -> tuple[C2UPositiveHeatingShot, ...]:
    """Load and validate the public C-2U positive-net-heating shot table."""
    rows: list[C2UPositiveHeatingShot] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        filtered = (line for line in handle if not line.startswith("#"))
        for idx, row in enumerate(csv.DictReader(filtered), start=1):
            if row is None:
                continue
            rows.append(_parse_c2u_row(row, row_number=idx))

    if not rows:
        raise ValueError("C-2U reference table must contain at least one shot")
    shot_ids = [row.shot for row in rows]
    if any(next_id <= shot_id for shot_id, next_id in zip(shot_ids, shot_ids[1:])):
        raise ValueError("C-2U shot identifiers must be strictly increasing")
    return tuple(rows)


def summarise_c2u_positive_heating_shots(
    path: str | Path = C2U_REFERENCE_CSV,
) -> C2UPositiveHeatingSummary:
    """Return summary statistics for the public C-2U reference table."""
    shots = load_c2u_positive_heating_shots(path)
    return C2UPositiveHeatingSummary(
        shot_count=len(shots),
        shot_min=shots[0].shot,
        shot_max=shots[-1].shot,
        max_thermal_energy_j=max(shot.thermal_energy_j for shot in shots),
        max_poloidal_flux_wb=max(shot.poloidal_flux_wb for shot in shots),
        max_total_temperature_ev=max(shot.total_temperature_ev for shot in shots),
        max_net_heating_power_w=max(shot.net_heating_power_w for shot in shots),
        claim_boundary=C2U_CLAIM_BOUNDARY,
    )


def c2u_positive_heating_reference_status() -> dict[str, object]:
    """Return the current acceptance status for the public C-2U table."""
    if not C2U_REFERENCE_CSV.exists() or not C2U_REFERENCE_METADATA.exists():
        return {
            "case": "c2u_positive_net_heating_shots",
            "status": "blocked_missing_public_reference_artifact",
            "required_artifact": "C-2U supplemental shot table plus metadata",
        }
    summary = summarise_c2u_positive_heating_shots()
    return {
        "case": "c2u_positive_net_heating_shots",
        "status": "public_reference_table_available",
        "shot_count": summary.shot_count,
        "shot_min": summary.shot_min,
        "shot_max": summary.shot_max,
        "max_thermal_energy_j": summary.max_thermal_energy_j,
        "max_total_temperature_ev": summary.max_total_temperature_ev,
        "claim_boundary": summary.claim_boundary,
    }


def _parse_c2u_row(row: dict[str, str], *, row_number: int) -> C2UPositiveHeatingShot:
    required = (
        "shot",
        "Eth(kJ)",
        "Fp(mWb)",
        "T(keV)",
        "t_max(ms)",
        "P_max(MW)",
        "E_max(kJ)",
        "comment",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"C-2U row {row_number} missing required column(s): {', '.join(missing)}")
    shot = _parse_int(row["shot"], "shot", row_number)
    thermal_energy_j = _parse_positive(row["Eth(kJ)"], "Eth(kJ)", row_number) * 1.0e3
    poloidal_flux_wb = _parse_positive(row["Fp(mWb)"], "Fp(mWb)", row_number) * 1.0e-3
    total_temperature_ev = _parse_positive(row["T(keV)"], "T(keV)", row_number) * 1.0e3
    time_of_max_heating_s = _parse_positive(row["t_max(ms)"], "t_max(ms)", row_number) * 1.0e-3
    net_heating_power_w = _parse_positive(row["P_max(MW)"], "P_max(MW)", row_number) * 1.0e6
    energy_at_max_heating_j = _parse_positive(row["E_max(kJ)"], "E_max(kJ)", row_number) * 1.0e3
    return C2UPositiveHeatingShot(
        shot=shot,
        thermal_energy_j=thermal_energy_j,
        poloidal_flux_wb=poloidal_flux_wb,
        total_temperature_ev=total_temperature_ev,
        time_of_max_heating_s=time_of_max_heating_s,
        net_heating_power_w=net_heating_power_w,
        energy_at_max_heating_j=energy_at_max_heating_j,
        comment=row["comment"].strip(),
    )


def _parse_int(value: str, column: str, row_number: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"C-2U row {row_number} column {column} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"C-2U row {row_number} column {column} must be positive")
    return parsed


def _parse_positive(value: str, column: str, row_number: int) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"C-2U row {row_number} column {column} must be numeric") from exc
    if not parsed > 0.0:
        raise ValueError(f"C-2U row {row_number} column {column} must be positive")
    return parsed
