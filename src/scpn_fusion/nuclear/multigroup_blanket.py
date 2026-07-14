# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Multi-Group Blanket Neutronics
"""Three-group cylindrical neutron transport for tritium breeding.

Energy-resolved (fast / epithermal / thermal) diffusion model with down-scatter,
beryllium (n,2n) multiplication, and Li-6(n,t) capture — an upgrade over the
single-group :class:`~scpn_fusion.nuclear.blanket_neutronics.BreedingBlanket`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._blanket_validators import _require_finite_float, _require_int


class MultiGroupBlanket:
    """3-group neutron transport for tritium breeding ratio calculation.

    Energy groups:
        Group 1 (fast):       E > 1 MeV  (source: 14.1 MeV D-T neutrons)
        Group 2 (epithermal): 1 eV < E < 1 MeV  (down-scattered)
        Group 3 (thermal):    E < 1 eV  (thermalised, main Li-6 capture)

    Includes:
        - Energy-dependent cross sections per group
        - Down-scatter from fast → epithermal → thermal
        - Beryllium (n,2n) multiplication in fast group
        - Li-6(n,t) capture in all groups (dominant in thermal)

    This is a significant upgrade over the single-group BreedingBlanket above.
    """

    def __init__(
        self,
        thickness_cm: float = 80.0,
        li6_enrichment: float = 0.9,
        n_cells: int = 100,
        r_inner_cm: float = 200.0,
    ) -> None:
        self.thickness = _require_finite_float("thickness_cm", thickness_cm, min_value=0.1)
        self.r_inner = _require_finite_float("r_inner_cm", r_inner_cm, min_value=10.0)
        self.li6_enrich = _require_finite_float(
            "li6_enrichment",
            li6_enrichment,
            min_value=0.0,
            max_value=1.0,
        )
        self.n_cells = max(_require_int("n_cells", n_cells, 3), int(self.thickness * 2.5))
        self.r = np.linspace(self.r_inner, self.r_inner + self.thickness, self.n_cells)
        self.dx = self.r[1] - self.r[0]
        # For compatibility
        self.x = self.r - self.r_inner

        # ── Cross sections (cm^-1) per group ─────────────────────────
        # Group 1: fast (14 MeV)
        # Li-6(n,t) at 14 MeV is small (~25 mb); Be(n,2n) threshold ~1.8 MeV.
        self.sigma_capture_g1 = 0.005 * self.li6_enrich  # Li-6 capture at 14 MeV (small)
        self.sigma_scatter_g1 = 0.20  # elastic scatter
        self.sigma_multiply_g1 = 0.10  # Be (n,2n) at 14 MeV
        self.sigma_downscatter_12 = 0.20  # fast → epithermal (inelastic)
        self.sigma_parasitic_g1 = 0.005  # structural parasitic

        # Group 2: epithermal (keV–MeV)
        # Li-6 has resonance capture in the keV range.
        self.sigma_capture_g2 = 0.05 * self.li6_enrich  # Li-6 resonance capture
        self.sigma_scatter_g2 = 0.15
        self.sigma_downscatter_23 = 0.18  # epithermal → thermal (moderation)
        self.sigma_parasitic_g2 = 0.01

        # Group 3: thermal (< 1 eV)
        # Li-6(n,t) at thermal: ~940 barns, dominant capture pathway.
        # LiPb atom density × micro-sigma → macro ~0.8 cm^-1 at 90% enrichment.
        self.sigma_capture_g3 = 0.80 * self.li6_enrich  # Li-6 dominant at thermal
        self.sigma_scatter_g3 = 0.05
        self.sigma_parasitic_g3 = 0.01

        self.multiplier_gain = 1.8  # Be(n,2n) neutron gain

    def _solve_cylindrical_group(
        self,
        D: float,
        sigma_rem: float,
        source: NDArray[np.float64],
        bc_left: tuple[str, float],
        bc_right: tuple[str, float],
    ) -> NDArray[np.float64]:
        """Solve 1D cylindrical diffusion for a single group."""
        N = self.n_cells
        dr = self.dx
        A = np.zeros((N, N))
        b = source.copy()

        for i in range(1, N - 1):
            r_i = self.r[i]
            r_p = r_i + 0.5 * dr
            r_m = r_i - 0.5 * dr
            c_p = (D * r_p) / (r_i * dr**2)
            c_m = (D * r_m) / (r_i * dr**2)
            A[i, i - 1] = -c_m
            A[i, i] = c_p + c_m + sigma_rem
            A[i, i + 1] = -c_p

        # Left BC (r = r_inner)
        if bc_left[0] == "dirichlet":
            A[0, 0] = 1.0
            b[0] = bc_left[1]
        elif bc_left[0] == "neumann":
            A[0, 0] = 1.0
            A[0, 1] = -1.0
            b[0] = bc_left[1] * dr

        # Right BC (r = r_outer)
        if bc_right[0] == "dirichlet":
            A[-1, -1] = 1.0
            b[-1] = bc_right[1]
        elif bc_right[0] == "neumann":
            A[-1, -1] = 1.0
            A[-1, -2] = -1.0
            b[-1] = bc_right[1] * dr

        return np.asarray(np.linalg.solve(A, b), dtype=np.float64)

    def solve_transport(
        self,
        incident_flux: float = 1e14,
        port_coverage_factor: float = 0.80,
        streaming_factor: float = 0.85,
    ) -> dict[str, object]:
        """Solve 3-group steady-state cylindrical neutron diffusion."""
        incident_flux = _require_finite_float("incident_flux", incident_flux, min_value=1.0)
        port_coverage_factor = float(port_coverage_factor)
        if not (0.0 < port_coverage_factor <= 1.0):
            raise ValueError("port_coverage_factor must be in (0, 1].")
        streaming_factor = float(streaming_factor)
        if not (0.0 < streaming_factor <= 1.0):
            raise ValueError("streaming_factor must be in (0, 1].")

        # Group 1 (fast)
        sigma_tot_1 = (
            self.sigma_capture_g1
            + self.sigma_scatter_g1
            + self.sigma_multiply_g1
            + self.sigma_downscatter_12
            + self.sigma_parasitic_g1
        )
        D1 = 1.0 / (3.0 * sigma_tot_1)
        sigma_rem_1 = (
            self.sigma_capture_g1
            + self.sigma_downscatter_12
            + self.sigma_parasitic_g1
            - self.sigma_multiply_g1 * (self.multiplier_gain - 1.0)
        )

        phi_g1 = self._solve_cylindrical_group(
            D1,
            sigma_rem_1,
            np.zeros(self.n_cells),
            ("dirichlet", incident_flux),
            ("dirichlet", 0.0),
        )
        n_clamped_g1 = int(np.sum(phi_g1 < 0))
        phi_g1 = np.maximum(phi_g1, 0.0)

        # Group 2 (epithermal)
        sigma_tot_2 = (
            self.sigma_capture_g2
            + self.sigma_scatter_g2
            + self.sigma_downscatter_23
            + self.sigma_parasitic_g2
        )
        D2 = 1.0 / (3.0 * sigma_tot_2)
        sigma_rem_2 = self.sigma_capture_g2 + self.sigma_downscatter_23 + self.sigma_parasitic_g2

        source2 = self.sigma_downscatter_12 * phi_g1
        phi_g2 = self._solve_cylindrical_group(
            D2, sigma_rem_2, source2, ("neumann", 0.0), ("dirichlet", 0.0)
        )
        n_clamped_g2 = int(np.sum(phi_g2 < 0))
        phi_g2 = np.maximum(phi_g2, 0.0)

        # Group 3 (thermal)
        sigma_tot_3 = self.sigma_capture_g3 + self.sigma_scatter_g3 + self.sigma_parasitic_g3
        D3 = 1.0 / (3.0 * sigma_tot_3)
        sigma_rem_3 = self.sigma_capture_g3 + self.sigma_parasitic_g3

        source3 = self.sigma_downscatter_23 * phi_g2
        phi_g3 = self._solve_cylindrical_group(
            D3, sigma_rem_3, source3, ("neumann", 0.0), ("dirichlet", 0.0)
        )
        n_clamped_g3 = int(np.sum(phi_g3 < 0))
        phi_g3 = np.maximum(phi_g3, 0.0)

        prod_g1 = self.sigma_capture_g1 * phi_g1
        prod_g2 = self.sigma_capture_g2 * phi_g2
        prod_g3 = self.sigma_capture_g3 * phi_g3
        total_prod = prod_g1 + prod_g2 + prod_g3

        if hasattr(np, "trapezoid"):
            total_tritium = float(
                np.trapezoid(total_prod * 2.0 * np.pi * self.r, self.r)
            )  # pragma: no cover - numpy>=2.0 trapezoid path
        else:
            edge = np.diff(self.r)
            total_integrand = total_prod * 2.0 * np.pi * self.r
            total_tritium = float(np.sum(0.5 * (total_integrand[1:] + total_integrand[:-1]) * edge))

        # Incident current (per unit length): J+ * Area_inner
        incident_current_total = (phi_g1[0] / 4.0) * (2.0 * np.pi * self.r_inner)
        tbr_ideal = total_tritium / max(incident_current_total, 1e-12)
        tbr = tbr_ideal * port_coverage_factor * streaming_factor

        # Per-group TBR breakdown (with same correction factors as total)
        corr = port_coverage_factor * streaming_factor
        if hasattr(np, "trapezoid"):
            tbr_g1_raw = float(
                np.trapezoid(prod_g1 * 2.0 * np.pi * self.r, self.r)
            )  # pragma: no cover - numpy>=2.0 trapezoid path
            tbr_g2_raw = float(
                np.trapezoid(prod_g2 * 2.0 * np.pi * self.r, self.r)
            )  # pragma: no cover - numpy>=2.0 trapezoid path
            tbr_g3_raw = float(
                np.trapezoid(prod_g3 * 2.0 * np.pi * self.r, self.r)
            )  # pragma: no cover - numpy>=2.0 trapezoid path
        else:
            edge = np.diff(self.r)
            g1 = prod_g1 * 2.0 * np.pi * self.r
            g2 = prod_g2 * 2.0 * np.pi * self.r
            g3 = prod_g3 * 2.0 * np.pi * self.r
            tbr_g1_raw = float(np.sum(0.5 * (g1[1:] + g1[:-1]) * edge))
            tbr_g2_raw = float(np.sum(0.5 * (g2[1:] + g2[:-1]) * edge))
            tbr_g3_raw = float(np.sum(0.5 * (g3[1:] + g3[:-1]) * edge))

        tbr_g1 = tbr_g1_raw / max(incident_current_total, 1e-12) * corr
        tbr_g2 = tbr_g2_raw / max(incident_current_total, 1e-12) * corr
        tbr_g3 = tbr_g3_raw / max(incident_current_total, 1e-12) * corr

        # Flux clamping telemetry (tracked before np.maximum above)
        flux_clamp_total = n_clamped_g1 + n_clamped_g2 + n_clamped_g3
        clamp_events = {
            "fast": n_clamped_g1,
            "epithermal": n_clamped_g2,
            "thermal": n_clamped_g3,
        }

        # Incident current density (cm^-2 s^-1)
        area_cm2 = 2.0 * np.pi * self.r_inner * 1e4
        incident_current_cm2_s = float(incident_current_total / max(area_cm2, 1e-12))

        return {
            "phi_g1": phi_g1,
            "phi_g2": phi_g2,
            "phi_g3": phi_g3,
            "total_production": total_prod,
            "tbr": float(tbr),
            "tbr_ideal": float(tbr_ideal),
            "tbr_by_group": {
                "fast": float(tbr_g1),
                "epithermal": float(tbr_g2),
                "thermal": float(tbr_g3),
            },
            "incident_current_total": float(incident_current_total),
            "incident_current_cm2_s": incident_current_cm2_s,
            "flux_clamp_total": flux_clamp_total,
            "flux_clamp_events": clamp_events,
        }
