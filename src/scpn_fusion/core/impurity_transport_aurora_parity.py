# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Aurora/STRAHL Parity Solver
"""Native Aurora/STRAHL-compatible charge-state parity solver and artifact builder.

This cluster holds the native finite-volume Aurora-parity solver, the
volume-integral geometry helper, and the Aurora/STRAHL charge-state artifact
builder. It depends on the data contracts
(:mod:`impurity_transport_contracts`) and the charge-state math
(:mod:`impurity_transport_charge_state`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scpn_fusion.core.impurity_transport_charge_state import (
    _source_sink_transfer_matrix,
    adas_style_charge_state_coefficients,
    advance_charge_state_collisional_radiative,
    collisional_radiative_source_sink_matrices,
)
from scpn_fusion.core.impurity_transport_contracts import (
    AdasChargeStateCoefficients,
    AuroraParityCase,
    AuroraStrahlArtifact,
    FloatArray,
    _strict_axis,
)


def _volume_integral(profile: FloatArray, radius_m: FloatArray, R0: float) -> float:
    trapz: Any = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    vol_element = 4.0 * np.pi**2 * R0 * radius_m
    return float(trapz(profile * vol_element, radius_m))


class AuroraParityImpuritySolver:
    """Native Aurora-compatible parity solver, separate from the general solver.

    The class is not an Aurora wrapper. It implements a conservative native
    finite-volume radial transport step, neighbouring charge-state CR transfer,
    and line-radiation export on an Aurora-style case contract. The purpose is
    strict same-case parity testing without changing the general impurity
    solver's development path.
    """

    def __init__(self, case: AuroraParityCase):
        self.case = case
        self.charge_states = np.asarray(case.charge_states, dtype=np.float64)
        self.radius_m = np.asarray(case.radius_m, dtype=np.float64)
        self.time_s = np.asarray(case.time_s, dtype=np.float64)

    def _rate_tables(self, time_idx: int, density: FloatArray) -> tuple[FloatArray, FloatArray]:
        if self.case.ionisation_m3_s_t_r_z is not None:
            ion_coeff = np.asarray(self.case.ionisation_m3_s_t_r_z[time_idx], dtype=np.float64)
        else:
            ion_coeff = self._parametric_coefficients(time_idx).ionisation_m3_s[np.newaxis, :]
        if self.case.recombination_m3_s_t_r_z is not None:
            rec_coeff = np.asarray(self.case.recombination_m3_s_t_r_z[time_idx], dtype=np.float64)
        else:
            rec_coeff = self._parametric_coefficients(time_idx).recombination_m3_s[np.newaxis, :]
        ne = np.asarray(self.case.ne_t_r[time_idx], dtype=np.float64)
        ionisation = ne[:, np.newaxis] * density * ion_coeff
        recombination = ne[:, np.newaxis] * density * rec_coeff
        ionisation[:, -1] = 0.0
        recombination[:, 0] = 0.0
        return ionisation, recombination

    def _parametric_coefficients(self, time_idx: int) -> AdasChargeStateCoefficients:
        te = np.asarray(self.case.Te_t_r[time_idx], dtype=np.float64)
        te_charge = np.interp(
            self.charge_states,
            [self.charge_states[0], self.charge_states[-1]],
            [float(np.min(te)), float(np.max(te))],
        )
        return adas_style_charge_state_coefficients(
            self.case.element,
            self.charge_states,
            te_charge,
        )

    def _radial_transport_step(self, density: FloatArray, dt_s: float) -> FloatArray:
        radius = self.radius_m
        edges = self._radial_edges()
        annulus = 0.5 * (edges[1:] ** 2 - edges[:-1] ** 2)
        updated = density.copy()
        for charge_idx in range(density.shape[1]):
            D = np.asarray(self.case.diffusion_m2_s_r_z[:, charge_idx], dtype=np.float64)
            V = np.asarray(self.case.convection_m_s_r_z[:, charge_idx], dtype=np.float64)
            flux = np.zeros(radius.size + 1, dtype=np.float64)
            for iface in range(1, radius.size):
                dr = radius[iface] - radius[iface - 1]
                grad = (density[iface, charge_idx] - density[iface - 1, charge_idx]) / dr
                D_face = 0.5 * (D[iface] + D[iface - 1])
                V_face = 0.5 * (V[iface] + V[iface - 1])
                upwind = (
                    density[iface - 1, charge_idx] if V_face >= 0.0 else density[iface, charge_idx]
                )
                flux[iface] = -D_face * grad + V_face * upwind
            updated[:, charge_idx] -= (
                dt_s * (edges[1:] * flux[1:] - edges[:-1] * flux[:-1]) / annulus
            )
        return np.maximum(updated, 0.0)

    def _radial_edges(self) -> FloatArray:
        radius = self.radius_m
        edges = np.empty(radius.size + 1, dtype=np.float64)
        edges[1:-1] = 0.5 * (radius[:-1] + radius[1:])
        edges[0] = max(0.0, radius[0] - 0.5 * (radius[1] - radius[0]))
        edges[-1] = radius[-1] + 0.5 * (radius[-1] - radius[-2])
        return edges

    def _finite_volume_inventory(self, total_density_r: FloatArray) -> float:
        edges = self._radial_edges()
        annulus = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        volume = 2.0 * np.pi * self.case.major_radius_m * annulus
        return float(np.sum(np.asarray(total_density_r, dtype=np.float64) * volume))

    def _advance_transport_and_cr(
        self, density: FloatArray, step: int, dt_s: float
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Advance native radial transport and neighbouring CR transfer once."""
        advanced = self._radial_transport_step(density, dt_s)
        ionisation, recombination = self._rate_tables(step, advanced)
        for charge_idx in range(advanced.shape[1] - 1):
            ion_flux = np.minimum(
                ionisation[:, charge_idx],
                advanced[:, charge_idx] / dt_s,
            )
            rec_flux = np.minimum(
                recombination[:, charge_idx + 1],
                advanced[:, charge_idx + 1] / dt_s,
            )
            advanced[:, charge_idx] += dt_s * (rec_flux - ion_flux)
            advanced[:, charge_idx + 1] += dt_s * (ion_flux - rec_flux)
        return np.maximum(advanced, 0.0), ionisation, recombination

    def derive_effective_source_closure(self, reference_density_t_r_z: FloatArray) -> FloatArray:
        """Derive the same-case effective source/recycling residual.

        The returned array is a diagnostic closure, not a mechanistic Aurora or
        STRAHL source model.  For each output step it records the density-rate
        residual needed after the native finite-volume transport and CR
        predictor to reproduce the supplied Aurora density trajectory.
        """
        reference = np.asarray(reference_density_t_r_z, dtype=np.float64)
        expected_shape = (
            self.time_s.size,
            self.radius_m.size,
            self.charge_states.size,
        )
        if reference.shape != expected_shape:
            raise ValueError(f"reference_density_t_r_z must have shape {expected_shape}")
        if not np.all(np.isfinite(reference)) or np.any(reference < 0.0):
            raise ValueError("reference_density_t_r_z must be finite and non-negative")

        closure = np.zeros_like(reference)
        density = reference[0].copy()
        for step in range(1, self.time_s.size):
            dt_s = float(self.time_s[step] - self.time_s[step - 1])
            predicted, _, _ = self._advance_transport_and_cr(density, step, dt_s)
            closure[step] = (reference[step] - predicted) / dt_s
            density = reference[step].copy()
        return closure

    def radial_transport_budget_diagnostic(
        self, density_r_z: FloatArray, dt_s: float
    ) -> dict[str, float | bool]:
        """Return finite-volume radial-operator conservation diagnostics.

        The radial operator uses zero-flux boundary conditions.  This diagnostic
        validates that applying the radial transport step to an evolved
        charge-state density preserves total impurity inventory up to numerical
        roundoff.
        """
        density = np.asarray(density_r_z, dtype=np.float64)
        expected_shape = (self.radius_m.size, self.charge_states.size)
        if density.shape != expected_shape:
            raise ValueError(f"density_r_z must have shape {expected_shape}")
        if not np.all(np.isfinite(density)) or np.any(density < 0.0):
            raise ValueError("density_r_z must be finite and non-negative")
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")

        before = self._finite_volume_inventory(np.sum(density, axis=1))
        after_density = self._radial_transport_step(density, dt_s)
        after = self._finite_volume_inventory(np.sum(after_density, axis=1))
        relative_error = abs(after - before) / max(abs(before), 1.0)
        return {
            "inventory_before": before,
            "inventory_after": after,
            "relative_inventory_error": float(relative_error),
            "passed": bool(relative_error <= 1.0e-12),
        }

    def solve(self) -> AuroraStrahlArtifact:
        """Return Aurora-style observables for the native parity contract."""
        density = np.asarray(self.case.initial_charge_state_density_rz, dtype=np.float64).copy()
        density_history = [density.copy()]
        source_sink_history: list[FloatArray] = []
        line_power_by_charge: list[FloatArray] = []
        line_power: list[float] = []
        inventory_history: list[float] = [self._finite_volume_inventory(np.sum(density, axis=1))]
        final_ion, final_rec = self._rate_tables(0, density)
        source_sink_history.append(_source_sink_transfer_matrix(final_ion, final_rec))
        line_density = self._line_radiation_density(0, density)
        line_power_by_charge.append(line_density)
        line_power.append(self._line_power_total(line_density))
        for step in range(1, self.time_s.size):
            dt_s = float(self.time_s[step] - self.time_s[step - 1])
            density, final_ion, final_rec = self._advance_transport_and_cr(
                density,
                step,
                dt_s,
            )
            if self.case.effective_source_m3_s_t_r_z is not None:
                density += dt_s * np.asarray(
                    self.case.effective_source_m3_s_t_r_z[step],
                    dtype=np.float64,
                )
            density = np.maximum(density, 0.0)
            final_ion, final_rec = self._rate_tables(step, density)
            source_sink_history.append(_source_sink_transfer_matrix(final_ion, final_rec))
            density_history.append(density.copy())
            line_density = self._line_radiation_density(step, density)
            line_power_by_charge.append(line_density)
            line_power.append(self._line_power_total(line_density))
            inventory_history.append(self._finite_volume_inventory(np.sum(density, axis=1)))

        density_t_r_z = np.stack(density_history, axis=0)
        total_t_r = np.sum(density_t_r_z, axis=2)
        initial_inventory = float(inventory_history[0])
        final_inventory = float(inventory_history[-1])
        conservation_error = abs(final_inventory - initial_inventory) / max(
            abs(initial_inventory), 1.0
        )
        return AuroraStrahlArtifact(
            coordinates={
                "time_s": [float(v) for v in self.time_s],
                "radius_m": [float(v) for v in self.radius_m],
                "charge_state": [float(v) for v in self.charge_states],
            },
            coordinate_units={
                "time_s": "s",
                "radius_m": "m",
                "charge_state": "integer_charge",
            },
            observable_axes={
                "charge_state_density_r_t": ["time_s", "radius_m", "charge_state"],
                "total_impurity_density_r_t": ["time_s", "radius_m"],
                "line_radiation_power_t": ["time_s"],
                "line_radiation_power_t_r_z": ["time_s", "radius_m", "charge_state"],
                "source_sink_matrix_t_r_z_z": [
                    "time_s",
                    "radius_m",
                    "charge_state",
                    "charge_state",
                ],
                "total_impurity_inventory_t": ["time_s"],
                "ionisation_source_matrix": ["radius_m", "charge_state"],
                "recombination_sink_matrix": ["radius_m", "charge_state"],
            },
            observable_units={
                "charge_state_density_r_t": "m^-3",
                "total_impurity_density_r_t": "m^-3",
                "line_radiation_power_t": "W",
                "line_radiation_power_t_r_z": "W",
                "source_sink_matrix_t_r_z_z": "m^-3_s^-1",
                "total_impurity_inventory_t": "particles",
                "ionisation_source_matrix": "m^-3_s^-1",
                "recombination_sink_matrix": "m^-3_s^-1",
            },
            observables={
                "charge_state_density_r_t": density_t_r_z.tolist(),
                "total_impurity_density_r_t": total_t_r.tolist(),
                "line_radiation_power_t": [float(v) for v in line_power],
                "line_radiation_power_t_r_z": np.stack(line_power_by_charge, axis=0).tolist(),
                "source_sink_matrix_t_r_z_z": np.stack(source_sink_history, axis=0).tolist(),
                "total_impurity_inventory_t": [float(v) for v in inventory_history],
                "ionisation_source_matrix": final_ion.tolist(),
                "recombination_sink_matrix": final_rec.tolist(),
            },
            conservation={
                "initial_inventory": initial_inventory,
                "final_inventory": final_inventory,
                "relative_inventory_error": float(conservation_error),
            },
            provenance={
                "reference_family": "Aurora/STRAHL",
                "implementation": "native_aurora_compatibility_parity_solver",
                "parity_status": "native_aurora_compatibility_mode_threshold_gated",
            },
        )

    def _line_radiation_density(self, time_idx: int, density: FloatArray) -> FloatArray:
        if self.case.line_radiation_w_m3_t_r_z is not None:
            coeff = np.asarray(self.case.line_radiation_w_m3_t_r_z[time_idx], dtype=np.float64)
        else:
            coeff = self._parametric_coefficients(time_idx).line_radiation_w_m3[np.newaxis, :]
        ne = np.asarray(self.case.ne_t_r[time_idx], dtype=np.float64)
        return ne[:, np.newaxis] * density * coeff

    def _line_power_total(self, line_density_r_z: FloatArray) -> float:
        if self.case.line_radiation_w_m3_t_r_z is not None:
            return float(np.sum(line_density_r_z))
        return _volume_integral(
            np.sum(line_density_r_z, axis=1),
            self.radius_m,
            self.case.major_radius_m,
        )


def build_aurora_strahl_charge_state_artifact(
    *,
    element: str,
    charge_states: FloatArray | list[int] | tuple[int, ...],
    radius_m: FloatArray,
    time_s: FloatArray,
    ne_t_r: FloatArray,
    Te_t_r: FloatArray,
    initial_charge_state_density_rz: FloatArray,
    major_radius_m: float,
) -> AuroraStrahlArtifact:
    """Simulate and export an Aurora/STRAHL-style charge-state artifact."""
    charge_axis = _strict_axis("charge_state", np.asarray(charge_states, dtype=np.float64))
    radius_axis = _strict_axis("radius_m", np.asarray(radius_m, dtype=np.float64))
    time_axis = _strict_axis("time_s", np.asarray(time_s, dtype=np.float64))
    if radius_axis[0] < 0.0:
        raise ValueError("radius_m must be non-negative")
    if not np.isfinite(major_radius_m) or major_radius_m <= 0.0:
        raise ValueError("major_radius_m must be finite and positive")
    ne_hist = np.asarray(ne_t_r, dtype=np.float64)
    te_hist = np.asarray(Te_t_r, dtype=np.float64)
    density = np.asarray(initial_charge_state_density_rz, dtype=np.float64)
    expected_2d = (time_axis.size, radius_axis.size)
    expected_density = (radius_axis.size, charge_axis.size)
    if ne_hist.shape != expected_2d or te_hist.shape != expected_2d:
        raise ValueError("ne_t_r and Te_t_r must have shape time x radius")
    if density.shape != expected_density:
        raise ValueError("initial_charge_state_density_rz must have shape radius x charge_state")
    if not np.all(np.isfinite(ne_hist)) or np.any(ne_hist <= 0.0):
        raise ValueError("ne_t_r must be finite and positive")
    if not np.all(np.isfinite(te_hist)) or np.any(te_hist <= 0.0):
        raise ValueError("Te_t_r must be finite and positive")
    if not np.all(np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError("initial_charge_state_density_rz must be finite and non-negative")

    density_history = [density.copy()]
    line_power_by_charge: list[FloatArray] = []
    line_power: list[float] = []
    source_sink_history: list[FloatArray] = []
    inventory_history: list[float] = [
        _volume_integral(np.sum(density, axis=1), radius_axis, major_radius_m)
    ]
    final_ion = np.zeros_like(density)
    final_rec = np.zeros_like(density)
    te_charge = np.interp(
        charge_axis,
        [charge_axis[0], charge_axis[-1]],
        [te_hist[0].min(), te_hist[0].max()],
    )
    coeffs = adas_style_charge_state_coefficients(element, charge_axis, te_charge)
    final_ion, final_rec = collisional_radiative_source_sink_matrices(density, ne_hist[0], coeffs)
    source_sink_history.append(_source_sink_transfer_matrix(final_ion, final_rec))
    rad_density = ne_hist[0, :, np.newaxis] * density * coeffs.line_radiation_w_m3[np.newaxis, :]
    line_power_by_charge.append(rad_density)
    line_power.append(_volume_integral(np.sum(rad_density, axis=1), radius_axis, major_radius_m))
    for step in range(1, time_axis.size):
        dt_s = float(time_axis[step] - time_axis[step - 1])
        te_charge = np.interp(
            charge_axis,
            [charge_axis[0], charge_axis[-1]],
            [te_hist[step].min(), te_hist[step].max()],
        )
        coeffs = adas_style_charge_state_coefficients(element, charge_axis, te_charge)
        density, final_ion, final_rec = advance_charge_state_collisional_radiative(
            density, ne_hist[step], coeffs, dt_s
        )
        source_sink_history.append(_source_sink_transfer_matrix(final_ion, final_rec))
        density_history.append(density.copy())
        rad_density = (
            ne_hist[step, :, np.newaxis] * density * coeffs.line_radiation_w_m3[np.newaxis, :]
        )
        line_power_by_charge.append(rad_density)
        line_power.append(
            _volume_integral(np.sum(rad_density, axis=1), radius_axis, major_radius_m)
        )
        inventory_history.append(
            _volume_integral(np.sum(density, axis=1), radius_axis, major_radius_m)
        )

    density_t_r_z = np.stack(density_history, axis=0)
    total_t_r = np.sum(density_t_r_z, axis=2)
    initial_inventory = _volume_integral(total_t_r[0], radius_axis, major_radius_m)
    final_inventory = _volume_integral(total_t_r[-1], radius_axis, major_radius_m)
    conservation_error = abs(final_inventory - initial_inventory) / max(abs(initial_inventory), 1.0)

    return AuroraStrahlArtifact(
        coordinates={
            "time_s": [float(v) for v in time_axis],
            "radius_m": [float(v) for v in radius_axis],
            "charge_state": [float(v) for v in charge_axis],
        },
        coordinate_units={
            "time_s": "s",
            "radius_m": "m",
            "charge_state": "integer_charge",
        },
        observable_axes={
            "charge_state_density_r_t": ["time_s", "radius_m", "charge_state"],
            "total_impurity_density_r_t": ["time_s", "radius_m"],
            "line_radiation_power_t": ["time_s"],
            "line_radiation_power_t_r_z": ["time_s", "radius_m", "charge_state"],
            "source_sink_matrix_t_r_z_z": [
                "time_s",
                "radius_m",
                "charge_state",
                "charge_state",
            ],
            "total_impurity_inventory_t": ["time_s"],
            "ionisation_source_matrix": ["radius_m", "charge_state"],
            "recombination_sink_matrix": ["radius_m", "charge_state"],
        },
        observable_units={
            "charge_state_density_r_t": "m^-3",
            "total_impurity_density_r_t": "m^-3",
            "line_radiation_power_t": "W",
            "line_radiation_power_t_r_z": "W",
            "source_sink_matrix_t_r_z_z": "m^-3_s^-1",
            "total_impurity_inventory_t": "particles",
            "ionisation_source_matrix": "m^-3_s^-1",
            "recombination_sink_matrix": "m^-3_s^-1",
        },
        observables={
            "charge_state_density_r_t": density_t_r_z.tolist(),
            "total_impurity_density_r_t": total_t_r.tolist(),
            "line_radiation_power_t": [float(v) for v in line_power],
            "line_radiation_power_t_r_z": np.stack(line_power_by_charge, axis=0).tolist(),
            "source_sink_matrix_t_r_z_z": np.stack(source_sink_history, axis=0).tolist(),
            "total_impurity_inventory_t": [float(v) for v in inventory_history],
            "ionisation_source_matrix": final_ion.tolist(),
            "recombination_sink_matrix": final_rec.tolist(),
        },
        conservation={
            "initial_inventory": float(initial_inventory),
            "final_inventory": float(final_inventory),
            "relative_inventory_error": float(conservation_error),
        },
        provenance={
            "reference_family": "Aurora/STRAHL",
            "implementation": "native_charge_state_collisional_radiative_contract_with_adas_style_coefficients",
            "parity_status": "artifact_contract_only_not_public_aurora_strahl_parity",
        },
    )
