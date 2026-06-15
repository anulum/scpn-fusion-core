#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — OpenADAS coronal cooling-rate generator
"""Compute coronal-equilibrium radiated-power cooling rates from OpenADAS adf11.

Reads the OpenADAS ``adf11`` line-power (PLT), recombination/bremsstrahlung
continuum (PRB), ionization (SCD), and recombination (ACD) rate-coefficient files
for an element, solves the coronal ionization balance at the lowest tabulated
density, and sums the charge-state-weighted radiated power to give the cooling
function ``Lz(Te)`` in ``W m^3``. The peak magnitude and location are the values
used to parametrise :class:`scpn_fusion.core.impurity_transport.CoolingCurve`.

The raw adf11 files live under the gitignored ``data/external/openadas_adf11/``
path; their provenance (URLs and SHA-256 checksums) is tracked in
``validation/reference_data/openadas_coronal_lz_manifest.json``. Re-download with:

    curl -sSL -o data/external/openadas_adf11/plt96_c.dat \
        https://open.adas.ac.uk/download/adf11/plt96/plt96_c.dat
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

REPO_ROOT = Path(__file__).resolve().parents[1]
ADF11_DIR = REPO_ROOT / "data" / "external" / "openadas_adf11"

# Supported elements: symbol -> (nuclear charge, adf11 dataset year code). Carbon
# and neon use the 1996 low-Z dataset; argon and tungsten are only available in
# the older 1989 dataset.
ELEMENTS = {
    "c": (6, "96"),
    "ne": (10, "96"),
    "ar": (18, "89"),
    "w": (74, "89"),
}

_FLOAT = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_Z1 = re.compile(r"Z1=\s*(\d+)")


def parse_adf11(path: Path) -> tuple[FloatArray, FloatArray, dict[int, FloatArray]]:
    """Return ``(log10_ne, log10_Te, {z1: log10_coeff[Te, ne]})`` from an adf11 file.

    ``ne`` is in ``cm^-3``, ``Te`` in ``eV``, and the coefficients are stored
    log10 on the ``(Te, ne)`` grid with density varying fastest within each
    temperature, per the adf11 convention.
    """

    lines = path.read_text().splitlines()
    header = lines[0].split()
    n_density, n_temperature = int(header[1]), int(header[2])

    grid: list[float] = []
    index = 2
    while len(grid) < n_density + n_temperature:
        grid.extend(float(value) for value in _FLOAT.findall(lines[index]))
        index += 1
    log_ne = np.asarray(grid[:n_density], dtype=np.float64)
    log_te = np.asarray(grid[n_density : n_density + n_temperature], dtype=np.float64)

    blocks: dict[int, np.ndarray] = {}
    expected = n_density * n_temperature
    for start in range(index, len(lines)):
        match = _Z1.search(lines[start])
        if match is None:
            continue
        z1 = int(match.group(1))
        values: list[float] = []
        cursor = start + 1
        while (
            len(values) < expected
            and cursor < len(lines)
            and "Z1=" not in lines[cursor]
            and not lines[cursor].lstrip().startswith("C")
        ):
            values.extend(float(value) for value in _FLOAT.findall(lines[cursor]))
            cursor += 1
        blocks[z1] = np.asarray(values[:expected], dtype=np.float64).reshape(
            n_temperature, n_density
        )
    return log_ne, log_te, blocks


def coronal_cooling_rate(element: str) -> tuple[FloatArray, FloatArray]:
    """Return ``(Te_eV, Lz_W_m3)`` coronal cooling rate for a supported element."""

    charge, year = ELEMENTS[element]
    _, log_te, plt = parse_adf11(ADF11_DIR / f"plt{year}_{element}.dat")
    _, _, prb = parse_adf11(ADF11_DIR / f"prb{year}_{element}.dat")
    _, _, scd = parse_adf11(ADF11_DIR / f"scd{year}_{element}.dat")
    _, _, acd = parse_adf11(ADF11_DIR / f"acd{year}_{element}.dat")

    te_eV = 10.0**log_te
    coronal_ne_index = 0  # lowest tabulated density is the coronal limit
    cooling = np.zeros_like(te_eV)
    for it in range(te_eV.size):
        # Coronal ionization balance accumulated in log space (robust for the 74
        # tungsten charge states): n_{q+1}/n_q = SCD(q->q+1) / ACD(q+1->q).
        log_abundance = np.zeros(charge + 1, dtype=np.float64)
        for q in range(charge):
            log_abundance[q + 1] = (
                log_abundance[q]
                + scd[q + 1][it, coronal_ne_index]
                - acd[q + 1][it, coronal_ne_index]
            )
        abundance = 10.0 ** (log_abundance - log_abundance.max())
        abundance /= abundance.sum()

        line = sum(
            abundance[q] * 10.0 ** plt[q + 1][it, coronal_ne_index] for q in range(charge)
        )
        continuum = sum(
            abundance[q] * 10.0 ** prb[q][it, coronal_ne_index] for q in range(1, charge + 1)
        )
        cooling[it] = (line + continuum) * 1.0e-6  # W cm^3 -> W m^3
    return te_eV, cooling


def main() -> None:
    for element in ELEMENTS:
        te_eV, cooling = coronal_cooling_rate(element)
        peak = int(np.argmax(cooling))
        print(
            f"{element.upper()}: coronal Lz peak {cooling[peak]:.3e} W m^3 "
            f"at Te = {te_eV[peak]:.1f} eV"
        )


if __name__ == "__main__":
    main()
