# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — DREAM Full Kinetic Metrics
"""Native residual and convergence metrics for full-kinetic DREAM parity."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from scpn_fusion.core.runaway_kinetic_diagnostics import (
    interval_residual,
    weighted_relative_l2,
)
from scpn_fusion.core.runaway_kinetic_grid import FloatArray
from validation.dream_full_kinetic_reference import DreamFullKineticOutput
from validation.dream_full_kinetic_report_contract import (
    INITIAL_CURRENT_DEFECT_INTERPRETATION,
)


def _relative_l2(actual: FloatArray, expected: FloatArray) -> float:
    denominator = max(float(np.linalg.norm(expected)), 1.0)
    return float(np.linalg.norm(actual - expected) / denominator)


def _interpolate(
    source_axes: tuple[FloatArray, FloatArray, FloatArray],
    source: FloatArray,
    target_axes: tuple[FloatArray, FloatArray, FloatArray],
) -> FloatArray:
    """Interpolate within the shared grid domain, tolerating endpoint ULP drift.

    DREAM constructs each resolution independently, so mathematically identical
    domain endpoints can differ by one or two floating-point ULPs.  Snap only
    those round-off-sized excursions to the authenticated source boundary and
    continue to reject every material extrapolation.
    """
    bounded_target_axes: list[FloatArray] = []
    for dimension, (source_axis, target_axis) in enumerate(
        zip(source_axes, target_axes, strict=True)
    ):
        lower = float(source_axis[0])
        upper = float(source_axis[-1])
        tolerance = (
            8.0
            * float(np.finfo(np.float64).eps)
            * max(
                1.0,
                abs(lower),
                abs(upper),
            )
        )
        target = np.asarray(target_axis, dtype=np.float64)
        if np.any(target < lower - tolerance) or np.any(target > upper + tolerance):
            raise ValueError(f"target axis {dimension} extends outside source interpolation domain")
        bounded_target_axes.append(cast(FloatArray, np.clip(target, lower, upper)))

    interpolator = RegularGridInterpolator(
        source_axes,
        source,
        method="linear",
        bounds_error=True,
    )
    mesh = np.meshgrid(*bounded_target_axes, indexing="ij")
    points = np.stack([axis.reshape(-1) for axis in mesh], axis=-1)
    shape = tuple(axis.size for axis in bounded_target_axes)
    return cast(FloatArray, interpolator(points).reshape(shape))


def _state_axes(output: DreamFullKineticOutput) -> tuple[FloatArray, ...]:
    return (output.grid.radius_m, output.grid.pitch, output.grid.momentum_mc)


def _native_metrics(output: DreamFullKineticOutput) -> dict[str, Any]:
    distribution_residuals: list[float] = []
    density_residuals: list[float] = []
    pitch_reconstruction: list[float] = []
    source_reconstruction: list[float] = []
    pitch_conservation: list[float] = []
    radial_loss_reconstruction: list[float] = []
    density_source_budgets: list[dict[str, Any]] = []
    current_reconstruction = [
        _relative_l2(moment, expected)
        for moment, expected in zip(
            output.current_moment()[1:],
            output.current_density_a_m2[1:],
            strict=True,
        )
    ]
    radiation_budgets: list[dict[str, float]] = []

    for step, dt_s in enumerate(np.diff(output.times_s)):
        operator = output.native_operator(step)
        tendencies = operator.evaluate(
            output.distribution[step + 1],
            output.density_m3[step + 1],
        )
        distribution_residuals.append(
            interval_residual(
                output.distribution[step],
                output.distribution[step + 1],
                tendencies.total,
                output.geometry,
                float(dt_s),
            )
        )
        density_difference = (output.density_m3[step + 1] - output.density_m3[step]) / dt_s
        density_residuals.append(
            _relative_l2(
                tendencies.runaway_density_tendency_m3_s,
                density_difference,
            )
        )
        density_source_budgets.append(
            {
                "radial_transport_m3_s": (
                    tendencies.runaway_density_radial_transport_m3_s.tolist()
                ),
                "avalanche_generation_m3_s": (
                    tendencies.runaway_density_avalanche_generation_m3_s.tolist()
                ),
                "external_source_m3_s": (tendencies.runaway_density_external_source_m3_s.tolist()),
                "total_tendency_m3_s": (tendencies.runaway_density_tendency_m3_s.tolist()),
                "finite_difference_m3_s": density_difference.tolist(),
            }
        )
        pitch_reconstruction.append(
            _relative_l2(
                operator.coefficients.pitch_advection,
                output.coefficients["Ap2"][step],
            )
        )
        source_reconstruction.append(
            _relative_l2(
                tendencies.avalanche_generation,
                -output.coefficients["S_ava"][step],
            )
        )
        pitch_rate = np.sum(tendencies.pitch_scattering * output.geometry.cell_measure)
        pitch_scale = max(
            float(np.sum(np.abs(tendencies.pitch_scattering) * output.geometry.cell_measure)),
            1.0,
        )
        pitch_conservation.append(float(abs(pitch_rate) / pitch_scale))
        radial_loss_reconstruction.append(
            _relative_l2(
                np.asarray([-np.sum(tendencies.radial_transport * output.geometry.cell_measure)]),
                np.asarray([output.auxiliary_diagnostics["scalar/radialloss_f_re"][step, 0]]),
            )
        )
        radiation_budgets.append(
            {
                "synchrotron_phase_space_rate": float(
                    np.sum(tendencies.synchrotron_loss * output.geometry.cell_measure)
                ),
                "bremsstrahlung_phase_space_rate": float(
                    np.sum(tendencies.bremsstrahlung_loss * output.geometry.cell_measure)
                ),
            }
        )

    return {
        "distribution_residual_relative_l2": distribution_residuals,
        "distribution_residual_max": max(distribution_residuals),
        "density_residual_relative_l2": density_residuals,
        "density_residual_max": max(density_residuals),
        "density_source_budgets": density_source_budgets,
        "pitch_advection_reconstruction_relative_l2": pitch_reconstruction,
        "pitch_advection_reconstruction_max": max(pitch_reconstruction),
        "avalanche_source_reconstruction_relative_l2": source_reconstruction,
        "avalanche_source_reconstruction_max": max(source_reconstruction),
        "pitch_scattering_particle_conservation": pitch_conservation,
        "pitch_scattering_particle_conservation_max": max(pitch_conservation),
        "radial_loss_reconstruction_relative_l2": radial_loss_reconstruction,
        "radial_loss_reconstruction_max": max(radial_loss_reconstruction),
        "current_moment_reconstruction_relative_l2": current_reconstruction,
        "current_moment_reconstruction_max": max(current_reconstruction),
        "initial_current_initialization_defect": {
            "saved_j_re_norm": float(np.linalg.norm(output.current_density_a_m2[0])),
            "distribution_moment_norm": float(np.linalg.norm(output.current_moment()[0])),
            "interpretation": INITIAL_CURRENT_DEFECT_INTERPRETATION,
        },
        "radiation_budgets": radiation_budgets,
    }


def _state_convergence(
    coarse: DreamFullKineticOutput,
    fine: DreamFullKineticOutput,
) -> dict[str, float]:
    interpolated_distribution = _interpolate(
        cast(tuple[FloatArray, FloatArray, FloatArray], _state_axes(fine)),
        fine.distribution[-1],
        cast(tuple[FloatArray, FloatArray, FloatArray], _state_axes(coarse)),
    )
    distribution_error = weighted_relative_l2(
        coarse.distribution[-1],
        interpolated_distribution,
        coarse.geometry.cell_measure,
    )
    density_target = np.interp(
        coarse.grid.radius_m,
        fine.grid.radius_m,
        fine.density_m3[-1],
    )
    current_target = np.interp(
        coarse.grid.radius_m,
        fine.grid.radius_m,
        fine.current_density_a_m2[-1],
    )
    return {
        "distribution_relative_l2": distribution_error,
        "density_relative_l2": _relative_l2(coarse.density_m3[-1], density_target),
        "current_relative_l2": _relative_l2(coarse.current_density_a_m2[-1], current_target),
        "growth_ratio_absolute_error": abs(
            float(np.sum(coarse.density_m3[-1]) / np.sum(coarse.density_m3[0]))
            - float(np.sum(fine.density_m3[-1]) / np.sum(fine.density_m3[0]))
        ),
    }


def _operator_convergence(
    coarse: DreamFullKineticOutput,
    fine: DreamFullKineticOutput,
) -> dict[str, float]:
    step_coarse = coarse.times_s.size - 2
    step_fine = fine.times_s.size - 2
    comparisons: dict[str, float] = {}
    specs = {
        "radial_transport_Drr": (
            "Drr",
            (coarse.grid.radius_faces_m, coarse.grid.pitch, coarse.grid.momentum_mc),
            (fine.grid.radius_faces_m, fine.grid.pitch, fine.grid.momentum_mc),
        ),
        "pitch_scattering_Dxx": (
            "Dxx",
            (coarse.grid.radius_m, coarse.grid.pitch_faces, coarse.grid.momentum_mc),
            (fine.grid.radius_m, fine.grid.pitch_faces, fine.grid.momentum_mc),
        ),
        "synchrotron_momentum": (
            "synchrotron_f1",
            (
                coarse.grid.radius_m,
                coarse.grid.pitch,
                coarse.grid.momentum_faces_mc,
            ),
            (fine.grid.radius_m, fine.grid.pitch, fine.grid.momentum_faces_mc),
        ),
        "bremsstrahlung_momentum": (
            "bremsstrahlung_f1",
            (
                coarse.grid.radius_m,
                coarse.grid.pitch,
                coarse.grid.momentum_faces_mc,
            ),
            (fine.grid.radius_m, fine.grid.pitch, fine.grid.momentum_faces_mc),
        ),
        "partial_screening_nu_D": (
            "nu_D_f2",
            (coarse.grid.radius_m, coarse.grid.pitch_faces, coarse.grid.momentum_mc),
            (fine.grid.radius_m, fine.grid.pitch_faces, fine.grid.momentum_mc),
        ),
    }
    for label, (quantity, target_axes, source_axes) in specs.items():
        target = coarse.coefficients[quantity][step_coarse]
        interpolated = _interpolate(
            source_axes,
            fine.coefficients[quantity][step_fine],
            target_axes,
        )
        comparisons[label] = _relative_l2(target, interpolated)
    return comparisons


def _ratio(refined: float, coarse: float) -> float:
    return refined / max(coarse, float(np.finfo(np.float64).tiny))


def _same_encoded_scalar(actual: float, expected: float) -> bool:
    """Compare one serialized scalar modulo round-off-sized HDF5 grid drift."""

    tolerance = (
        8.0
        * float(np.finfo(np.float64).eps)
        * max(
            1.0,
            abs(actual),
            abs(expected),
        )
    )
    return bool(np.isclose(actual, expected, rtol=0.0, atol=tolerance))
