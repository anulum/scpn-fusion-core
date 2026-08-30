# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Runaway Kinetic Operator Coefficients
"""Complete coefficient contract for a radius-momentum-pitch operator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scpn_fusion.core.runaway_kinetic_grid import FloatArray, RunawayKineticGrid


def _finite_array(
    name: str,
    values: FloatArray,
    shape: tuple[int, ...],
    *,
    nonnegative: bool = False,
) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RunawayKineticCoefficients:
    """Flux, diffusion and source coefficients for every evolved axis.

    Advection follows the flux convention ``Gamma = A*f - D*grad(f)``.
    The radiation components remain separate so synchrotron and
    bremsstrahlung budgets cannot disappear inside one opaque total.  The
    kinetic avalanche kernel and the momentum-integrated total-density rate
    are distinct because a finite kinetic grid need not contain the complete
    knock-on spectrum represented by the total runaway density equation.
    """

    radial_advection: FloatArray
    momentum_electric_advection: FloatArray
    momentum_collision_advection: FloatArray
    momentum_synchrotron_advection: FloatArray
    momentum_bremsstrahlung_advection: FloatArray
    pitch_electric_advection: FloatArray
    pitch_synchrotron_advection: FloatArray
    radial_diffusion: FloatArray
    momentum_diffusion: FloatArray
    pitch_diffusion: FloatArray
    momentum_pitch_diffusion: FloatArray
    pitch_momentum_diffusion: FloatArray
    avalanche_source_kernel: FloatArray
    total_electron_density_m3: FloatArray
    total_density_avalanche_rate_s_inv: FloatArray
    total_density_external_source_m3_s: FloatArray
    external_source: FloatArray

    @classmethod
    def checked(
        cls,
        grid: RunawayKineticGrid,
        *,
        radial_advection: FloatArray,
        momentum_electric_advection: FloatArray,
        momentum_collision_advection: FloatArray,
        momentum_synchrotron_advection: FloatArray,
        momentum_bremsstrahlung_advection: FloatArray,
        pitch_electric_advection: FloatArray,
        pitch_synchrotron_advection: FloatArray,
        radial_diffusion: FloatArray,
        momentum_diffusion: FloatArray,
        pitch_diffusion: FloatArray,
        momentum_pitch_diffusion: FloatArray,
        pitch_momentum_diffusion: FloatArray,
        avalanche_source_kernel: FloatArray,
        total_electron_density_m3: FloatArray,
        total_density_avalanche_rate_s_inv: FloatArray,
        total_density_external_source_m3_s: FloatArray,
        external_source: FloatArray,
    ) -> RunawayKineticCoefficients:
        """Construct a shape-checked immutable coefficient bundle."""

        cell = grid.shape
        radial_face = (grid.nr + 1, grid.nxi, grid.np)
        momentum_face = (grid.nr, grid.nxi, grid.np + 1)
        pitch_face = (grid.nr, grid.nxi + 1, grid.np)
        return cls(
            radial_advection=_finite_array("radial_advection", radial_advection, radial_face),
            momentum_electric_advection=_finite_array(
                "momentum_electric_advection",
                momentum_electric_advection,
                momentum_face,
            ),
            momentum_collision_advection=_finite_array(
                "momentum_collision_advection",
                momentum_collision_advection,
                momentum_face,
            ),
            momentum_synchrotron_advection=_finite_array(
                "momentum_synchrotron_advection",
                momentum_synchrotron_advection,
                momentum_face,
            ),
            momentum_bremsstrahlung_advection=_finite_array(
                "momentum_bremsstrahlung_advection",
                momentum_bremsstrahlung_advection,
                momentum_face,
            ),
            pitch_electric_advection=_finite_array(
                "pitch_electric_advection", pitch_electric_advection, pitch_face
            ),
            pitch_synchrotron_advection=_finite_array(
                "pitch_synchrotron_advection",
                pitch_synchrotron_advection,
                pitch_face,
            ),
            radial_diffusion=_finite_array(
                "radial_diffusion", radial_diffusion, radial_face, nonnegative=True
            ),
            momentum_diffusion=_finite_array(
                "momentum_diffusion", momentum_diffusion, momentum_face, nonnegative=True
            ),
            pitch_diffusion=_finite_array(
                "pitch_diffusion", pitch_diffusion, pitch_face, nonnegative=True
            ),
            momentum_pitch_diffusion=_finite_array(
                "momentum_pitch_diffusion",
                momentum_pitch_diffusion,
                momentum_face,
            ),
            pitch_momentum_diffusion=_finite_array(
                "pitch_momentum_diffusion",
                pitch_momentum_diffusion,
                pitch_face,
            ),
            avalanche_source_kernel=_finite_array(
                "avalanche_source_kernel",
                avalanche_source_kernel,
                cell,
                nonnegative=True,
            ),
            total_electron_density_m3=_finite_array(
                "total_electron_density_m3",
                total_electron_density_m3,
                (grid.nr,),
                nonnegative=True,
            ),
            total_density_avalanche_rate_s_inv=_finite_array(
                "total_density_avalanche_rate_s_inv",
                total_density_avalanche_rate_s_inv,
                (grid.nr,),
                nonnegative=True,
            ),
            total_density_external_source_m3_s=_finite_array(
                "total_density_external_source_m3_s",
                total_density_external_source_m3_s,
                (grid.nr,),
            ),
            external_source=_finite_array("external_source", external_source, cell),
        )

    @property
    def momentum_advection(self) -> FloatArray:
        """Total momentum advection with every declared loss term included."""

        result = (
            self.momentum_electric_advection
            + self.momentum_collision_advection
            + self.momentum_synchrotron_advection
            + self.momentum_bremsstrahlung_advection
        )
        result.setflags(write=False)
        return result

    @property
    def pitch_advection(self) -> FloatArray:
        """Total pitch advection including electric and synchrotron terms."""

        result = self.pitch_electric_advection + self.pitch_synchrotron_advection
        result.setflags(write=False)
        return result


__all__ = ["RunawayKineticCoefficients"]
