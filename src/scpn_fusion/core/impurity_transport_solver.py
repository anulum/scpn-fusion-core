# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Implicit Impurity Transport Solver
"""Implicit banded radial impurity transport solver for multiple species.

This cluster holds the general multi-species implicit radial transport solver
:class:`ImpurityTransportSolver`. It depends only on the data contracts
(:mod:`impurity_transport_contracts`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scpn_fusion.core.impurity_transport_contracts import FloatArray, ImpuritySpecies


class ImpurityTransportSolver:
    """Implicit radial impurity transport solver for multiple species."""

    def __init__(self, rho: FloatArray, R0: float, a: float, species: list[ImpuritySpecies]):
        """Initialize geometry, species inventory, and radial impurity state."""
        self.rho = np.asarray(rho, dtype=float)
        self.R0 = R0
        self.a = a
        self.species = species

        self.nr = len(self.rho)
        if self.nr < 3:
            raise ValueError("rho must contain at least three radial points")
        if not np.all(np.isfinite(self.rho)):
            raise ValueError("rho must contain only finite values")
        if not np.all(np.diff(self.rho) > 0.0):
            raise ValueError("rho must be strictly increasing")
        if not np.isclose(self.rho[0], 0.0) or not np.isclose(self.rho[-1], 1.0):
            raise ValueError("rho must span the normalised interval [0, 1]")
        if not np.isfinite(R0) or R0 <= 0.0:
            raise ValueError("R0 must be finite and positive")
        if not np.isfinite(a) or a <= 0.0:
            raise ValueError("a must be finite and positive")

        drho = np.diff(self.rho)
        if not np.allclose(drho, drho[0], rtol=1e-6, atol=1e-12):
            raise ValueError("rho grid must be uniformly spaced for the banded transport solve")
        self.drho = float(drho[0])

        self.n_z: dict[str, FloatArray] = {
            s.element: np.zeros(self.nr, dtype=np.float64) for s in species
        }

    def step(
        self,
        dt: float,
        ne: FloatArray,
        Te_eV: FloatArray,
        Ti_eV: FloatArray,
        D_anom: float,
        V_pinch: dict[str, FloatArray],
    ) -> dict[str, FloatArray]:
        """Advance the 1D impurity transport one step for each species.

        Uses simple upwind/centered differences.
        """
        import scipy.linalg

        dr = self.drho * self.a

        for s in self.species:
            n = self.n_z[s.element]
            V = V_pinch.get(s.element, np.zeros(self.nr))

            # D_total = D_anom + D_neo
            D = D_anom * np.ones(self.nr)

            # Implicit advance
            diag = np.zeros(self.nr)
            upper = np.zeros(self.nr)
            lower = np.zeros(self.nr)
            rhs = np.zeros(self.nr)
            source = self._edge_source_density(s)

            # Boundary conditions
            diag[0] = 1.0
            upper[0] = -1.0
            rhs[0] = 0.0  # dn/dr = 0 at axis

            diag[-1] = 1.0
            rhs[-1] = n[-1] + dt * source[-1]

            # Interior
            for i in range(1, self.nr - 1):
                r_val = self.rho[i] * self.a

                # Diffusion term
                coeff_D_plus = D[i] / dr**2 + D[i] / (2.0 * r_val * dr)
                coeff_D_minus = D[i] / dr**2 - D[i] / (2.0 * r_val * dr)
                coeff_D_0 = -2.0 * D[i] / dr**2

                # Convection term (upwind)
                if V[i] > 0:
                    coeff_V_0 = -V[i] / dr - V[i] / r_val
                    coeff_V_minus = V[i] / dr
                    coeff_V_plus = 0.0
                else:
                    coeff_V_0 = V[i] / dr - V[i] / r_val
                    coeff_V_plus = -V[i] / dr
                    coeff_V_minus = 0.0

                lower[i] = -dt * (coeff_D_minus + coeff_V_minus)
                diag[i] = 1.0 - dt * (coeff_D_0 + coeff_V_0)
                upper[i] = -dt * (coeff_D_plus + coeff_V_plus)

                rhs[i] = n[i] + dt * source[i]

            # Solve
            ab = np.zeros((3, self.nr))
            ab[0, 1:] = upper[:-1]
            ab[1, :] = diag
            ab[2, :-1] = lower[1:]

            n_new = scipy.linalg.solve_banded((1, 1), ab, rhs)

            # Replace
            self.n_z[s.element] = np.maximum(n_new, 0.0)

        return self.n_z

    def _edge_source_density(self, species: ImpuritySpecies) -> FloatArray:
        """Return a volume-normalised edge source density [m^-3 s^-1]."""
        if species.source_rate == 0.0:
            return np.zeros(self.nr)

        width = max(species.source_decay_width_rho, self.drho)
        profile = np.exp(-(1.0 - self.rho) / width)
        profile[self.rho < max(0.0, 1.0 - 8.0 * width)] = 0.0

        vol_element = 4.0 * np.pi**2 * self.R0 * self.a**2 * self.rho
        _trapz: Any = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
        profile_volume = float(_trapz(profile * vol_element, self.rho))
        if (  # pragma: no cover - unreachable: the constructor pins rho[-1]==1, so
            # profile[edge]==exp(0)==1 and vol_element[edge]>0 keep the trapezoidal
            # normalisation strictly positive for every admissible grid.
            profile_volume <= 0.0 or not np.isfinite(profile_volume)
        ):
            raise ValueError("edge source profile has zero normalisation")

        edge_area = 4.0 * np.pi**2 * self.R0 * self.a
        total_particles_per_second = species.source_rate * edge_area
        return np.asarray(profile * total_particles_per_second / profile_volume)
