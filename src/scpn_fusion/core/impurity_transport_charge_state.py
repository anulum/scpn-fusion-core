# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Charge-State Collisional-Radiative Math
"""Native ADAS-style charge-state collisional-radiative coefficient and transfer math.

This cluster holds the analytic ADAS-style coefficient contract, the ionisation
and recombination source/sink matrices, the conservative single-step charge-state
advance, and the conservative charge-state transfer matrix. It depends only on the
data contracts (:mod:`impurity_transport_contracts`).
"""

from __future__ import annotations

import numpy as np

from scpn_fusion.core.impurity_transport_contracts import AdasChargeStateCoefficients, FloatArray


def adas_style_charge_state_coefficients(
    element: str,
    charge_states: FloatArray | list[int] | tuple[int, ...],
    Te_eV: FloatArray,
) -> AdasChargeStateCoefficients:
    """Return finite ADAS-style CR coefficients on a charge-state axis.

    The tables are analytic, deterministic coefficient contracts for native
    testing. They are shaped and unit-labelled for ADAS/Aurora/STRAHL artifact
    ingestion, but they are not a substitute for licensed Open-ADAS datasets.
    """
    charge = np.asarray(charge_states, dtype=np.float64)
    te = np.asarray(Te_eV, dtype=np.float64)
    if te.shape != charge.shape:
        raise ValueError("Te_eV must match charge_states shape")
    if not np.all(np.isfinite(te)) or np.any(te <= 0.0):
        raise ValueError("Te_eV must be finite and positive")
    if charge.ndim != 1 or charge.size < 2:
        raise ValueError("charge_states must be a 1D axis with at least two states")
    if not np.all(np.isfinite(charge)) or not np.all(np.diff(charge) > 0.0):
        raise ValueError("charge_states must be finite and strictly increasing")

    element_scale = {"C": 0.65, "Ne": 0.9, "Ar": 1.15, "W": 2.8}.get(element, 1.0)
    z_norm = charge / max(float(np.max(charge)), 1.0)
    ion_peak = 30.0 * (1.0 + 12.0 * z_norm) ** 1.35
    rec_peak = 5.0e3 / (1.0 + 8.0 * z_norm)
    ionisation = element_scale * 2.5e-14 * np.sqrt(te / 100.0) * np.exp(-ion_peak / te)
    recombination = element_scale * 1.8e-14 * (rec_peak / (te + rec_peak)) ** 0.75
    recombination[-1] = 0.0
    ionisation[-1] = 0.0
    line_radiation = (
        element_scale
        * 1.0e-32
        * (1.0 + 0.15 * charge)
        * np.exp(-(((np.log(te) - np.log(np.maximum(ion_peak, 2.0))) / 1.8) ** 2))
    )
    return AdasChargeStateCoefficients(
        charge_states=charge,
        ionisation_m3_s=ionisation,
        recombination_m3_s=recombination,
        line_radiation_w_m3=line_radiation,
    )


def collisional_radiative_source_sink_matrices(
    charge_state_density_rz: FloatArray,
    ne: FloatArray,
    coeffs: AdasChargeStateCoefficients,
) -> tuple[FloatArray, FloatArray]:
    """Return finite ionisation and recombination transfer matrices."""
    density = np.asarray(charge_state_density_rz, dtype=np.float64)
    ne_arr = np.asarray(ne, dtype=np.float64)
    ion = np.asarray(coeffs.ionisation_m3_s, dtype=np.float64)
    rec = np.asarray(coeffs.recombination_m3_s, dtype=np.float64)
    if density.ndim != 2:
        raise ValueError("charge_state_density_rz must have shape radius x charge_state")
    if ne_arr.ndim != 1 or ne_arr.shape[0] != density.shape[0]:
        raise ValueError("ne must match the radius dimension")
    if density.shape[1] != ion.shape[0] or density.shape[1] != rec.shape[0]:
        raise ValueError("coefficient charge axis must match density charge axis")
    if not np.all(np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError("charge_state_density_rz must be finite and non-negative")
    if not np.all(np.isfinite(ne_arr)) or np.any(ne_arr <= 0.0):
        raise ValueError("ne must be finite and positive")

    ionisation = ne_arr[:, np.newaxis] * density * ion[np.newaxis, :]
    recombination = ne_arr[:, np.newaxis] * density * rec[np.newaxis, :]
    ionisation[:, -1] = 0.0
    recombination[:, 0] = 0.0
    return ionisation, recombination


def advance_charge_state_collisional_radiative(
    charge_state_density_rz: FloatArray,
    ne: FloatArray,
    coeffs: AdasChargeStateCoefficients,
    dt: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Advance one conservative charge-state CR step.

    Transfers are pairwise between neighbouring charge states and limited by
    donor inventory, preserving total impurity density per radius without
    introducing negative densities.
    """
    dt_s = float(dt)
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt must be finite and positive")
    density = np.asarray(charge_state_density_rz, dtype=np.float64)
    ionisation, recombination = collisional_radiative_source_sink_matrices(density, ne, coeffs)
    updated = density.copy()
    for z_idx in range(density.shape[1] - 1):
        ion_flux = np.minimum(ionisation[:, z_idx], updated[:, z_idx] / dt_s)
        rec_flux = np.minimum(recombination[:, z_idx + 1], updated[:, z_idx + 1] / dt_s)
        updated[:, z_idx] += dt_s * (rec_flux - ion_flux)
        updated[:, z_idx + 1] += dt_s * (ion_flux - rec_flux)
    return np.maximum(updated, 0.0), ionisation, recombination


def _source_sink_transfer_matrix(ionisation: FloatArray, recombination: FloatArray) -> FloatArray:
    """Return conservative charge-state transfer matrix with from-charge rows."""
    ion = np.asarray(ionisation, dtype=np.float64)
    rec = np.asarray(recombination, dtype=np.float64)
    if ion.shape != rec.shape or ion.ndim != 2:
        raise ValueError("ionisation and recombination must have matching radius x charge shapes")
    matrix = np.zeros((ion.shape[0], ion.shape[1], ion.shape[1]), dtype=np.float64)
    for z_idx in range(ion.shape[1] - 1):
        ion_flux = ion[:, z_idx]
        rec_flux = rec[:, z_idx + 1]
        matrix[:, z_idx, z_idx + 1] += ion_flux
        matrix[:, z_idx + 1, z_idx] += rec_flux
    for z_idx in range(ion.shape[1]):
        off_diagonal_sum = np.sum(matrix[:, z_idx, :], axis=1)
        matrix[:, z_idx, z_idx] = -off_diagonal_sum
    for _ in range(3):
        residual = np.sum(matrix, axis=2)
        for z_idx in range(ion.shape[1]):
            target_idx = z_idx + 1 if z_idx < ion.shape[1] - 1 else z_idx - 1
            matrix[:, z_idx, target_idx] -= residual[:, z_idx]
    return matrix
