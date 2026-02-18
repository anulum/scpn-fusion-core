# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Fusion Kernel
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Non-linear free-boundary Grad-Shafranov equilibrium solver.

Solves the Grad-Shafranov equation for toroidal plasma equilibrium using
Picard iteration with under-relaxation.  Supports both L-mode (linear)
and H-mode (mtanh pedestal) pressure/current profiles.

The solver can optionally offload the inner elliptic solve to a compiled
C++ library via :class:`~scpn_fusion.hpc.hpc_bridge.HPCBridge`, or to
the Rust multigrid backend when available.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.special import ellipe, ellipk

from scpn_fusion.hpc.hpc_bridge import HPCBridge

logger = logging.getLogger(__name__)

# ── Type aliases ──────────────────────────────────────────────────────
FloatArray = NDArray[np.float64]

from dataclasses import dataclass, field

@dataclass
class CoilSet:
    """External coil set for free-boundary solve.

    Attributes
    ----------
    positions : list of (R, Z) tuples
        Coil centre coordinates [m].
    currents : NDArray
        Current per coil [A].
    turns : list of int
        Number of turns per coil.
    current_limits : NDArray or None
        Per-coil maximum absolute current [A].  Shape ``(n_coils,)``.
        When set, ``optimize_coil_currents`` enforces these bounds.
    target_flux_points : NDArray or None
        Points ``(R, Z)`` on the desired separatrix for shape optimisation.
        Shape ``(n_pts, 2)``.
    """
    positions: list[tuple[float, float]] = field(default_factory=list)
    currents: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    turns: list[int] = field(default_factory=list)
    current_limits: NDArray[np.float64] | None = None
    target_flux_points: NDArray[np.float64] | None = None


class FusionKernel:
    """Non-linear free-boundary Grad-Shafranov equilibrium solver.

    Parameters
    ----------
    config_path : str | Path
        Path to a JSON configuration file describing the reactor geometry,
        coil set, physics parameters and solver settings.

    Attributes
    ----------
    Psi : FloatArray
        Poloidal flux on the (Z, R) grid.
    J_phi : FloatArray
        Toroidal current density on the (Z, R) grid.
    B_R, B_Z : FloatArray
        Radial and vertical magnetic field components (set after solve).
    """

    # ── construction ──────────────────────────────────────────────────

    def __init__(self, config_path: str | Path) -> None:
        self._config_path = str(config_path)
        self.load_config(config_path)
        self.initialize_grid()
        self.setup_accelerator()

    def load_config(self, path: str | Path) -> None:
        """Load reactor configuration from a JSON file.

        Parameters
        ----------
        path : str | Path
            Filesystem path to the configuration JSON.
        """
        with open(path, "r") as f:
            self.cfg: dict[str, Any] = json.load(f)
        logger.info("Loaded configuration for: %s", self.cfg["reactor_name"])

    def initialize_grid(self) -> None:
        """Build the computational (R, Z) mesh from the loaded config."""
        dims = self.cfg["dimensions"]
        res = self.cfg["grid_resolution"]

        self.NR: int = res[0]
        self.NZ: int = res[1]
        self.R: FloatArray = np.linspace(dims["R_min"], dims["R_max"], self.NR)
        self.Z: FloatArray = np.linspace(dims["Z_min"], dims["Z_max"], self.NZ)
        self.dR: float = float(self.R[1] - self.R[0])
        self.dZ: float = float(self.Z[1] - self.Z[0])
        self.RR: FloatArray
        self.ZZ: FloatArray
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)

        self.Psi = np.zeros((self.NZ, self.NR))
        self.J_phi = np.zeros((self.NZ, self.NR))

        self.p_prime_0: float = -1.0
        self.ff_prime_0: float = -1.0

        # Profile mode
        self.profile_mode: str = "l-mode"
        self.ped_params_p: dict[str, float] = {
            "ped_top": 0.92,
            "ped_width": 0.05,
            "ped_height": 1.0,
            "core_alpha": 0.3,
        }
        self.ped_params_ff: dict[str, float] = {
            "ped_top": 0.92,
            "ped_width": 0.05,
            "ped_height": 1.0,
            "core_alpha": 0.3,
        }

        profiles_cfg = self.cfg.get("physics", {}).get("profiles")
        if profiles_cfg:
            self.profile_mode = profiles_cfg.get("mode", "l-mode")
            if "p_prime" in profiles_cfg:
                self.ped_params_p.update(profiles_cfg["p_prime"])
            if "ff_prime" in profiles_cfg:
                self.ped_params_ff.update(profiles_cfg["ff_prime"])

    def setup_accelerator(self) -> None:
        """Initialise the optional C++ HPC acceleration bridge."""
        self.hpc = HPCBridge()
        if self.hpc.is_available():
            logger.info("HPC Acceleration ENABLED.")
            self.hpc.initialize(
                self.NR,
                self.NZ,
                (self.R[0], self.R[-1]),
                (self.Z[0], self.Z[-1]),
            )
        else:
            logger.info(
                "HPC Acceleration UNAVAILABLE (using Python fallback)."
            )

    # ── vacuum field ──────────────────────────────────────────────────

    def calculate_vacuum_field(self) -> FloatArray:
        """Compute the vacuum poloidal flux from the external coil set.

        Uses elliptic integrals (toroidal Green's function) for each coil.

        Returns
        -------
        FloatArray
            Vacuum flux Psi_vac on the (NZ, NR) grid.
        """
        logger.debug("Computing vacuum field (toroidal exact)…")
        Psi_vac = np.zeros((self.NZ, self.NR))
        mu0: float = self.cfg["physics"].get("vacuum_permeability", 1.0)

        for coil in self.cfg["coils"]:
            Rc, Zc = coil["r"], coil["z"]
            I = coil["current"]

            dZ = self.ZZ - Zc
            R_plus_Rc_sq = (self.RR + Rc) ** 2

            k2 = (4.0 * self.RR * Rc) / (R_plus_Rc_sq + dZ**2)
            k2 = np.clip(k2, 1e-9, 0.999999)

            K = ellipk(k2)
            E = ellipe(k2)

            prefactor = (mu0 * I) / (2 * np.pi)
            sqrt_term = np.sqrt(R_plus_Rc_sq + dZ**2)
            term = ((2.0 - k2) * K - 2.0 * E) / k2
            Psi_vac += prefactor * sqrt_term * term

        return Psi_vac

    # ── topology analysis ─────────────────────────────────────────────

    def find_x_point(
        self, Psi: FloatArray
    ) -> tuple[tuple[float, float], float]:
        """Locate the X-point (magnetic null) in the lower divertor region.

        Parameters
        ----------
        Psi : FloatArray
            Poloidal flux array on the (NZ, NR) grid.

        Returns
        -------
        tuple[tuple[float, float], float]
            ``((R_x, Z_x), Psi_x)`` — position and flux value at the
            X-point.
        """
        dPsi_dR, dPsi_dZ = np.gradient(Psi, self.dR, self.dZ)
        B_mag = np.sqrt(dPsi_dR**2 + dPsi_dZ**2)

        mask_divertor = self.ZZ < (self.cfg["dimensions"]["Z_min"] * 0.5)
        if np.any(mask_divertor):
            masked_B = np.where(mask_divertor, B_mag, 1e9)
            idx_min = np.argmin(masked_B)
            iz, ir = np.unravel_index(idx_min, Psi.shape)
            return (float(self.R[ir]), float(self.Z[iz])), float(Psi[iz, ir])

        return (0.0, 0.0), float(np.min(Psi))

    def _find_magnetic_axis(self) -> tuple[int, int, float]:
        """Find the O-point (magnetic axis) as the global Psi maximum.

        Returns
        -------
        tuple[int, int, float]
            ``(iz, ir, Psi_axis)`` — grid indices and flux value.
        """
        idx_max = int(np.argmax(self.Psi))
        iz, ir = np.unravel_index(idx_max, self.Psi.shape)
        psi_axis = float(self.Psi[iz, ir])
        if abs(psi_axis) < 1e-6:
            psi_axis = 1e-6
        return int(iz), int(ir), psi_axis

    # ── profile functions ─────────────────────────────────────────────

    @staticmethod
    def mtanh_profile(
        psi_norm: FloatArray, params: dict[str, float]
    ) -> FloatArray:
        """Evaluate a modified-tanh pedestal profile (vectorised).

        Parameters
        ----------
        psi_norm : FloatArray
            Normalised poloidal flux (0 at axis, 1 at separatrix).
        params : dict[str, float]
            Profile shape parameters with keys ``ped_top``, ``ped_width``,
            ``ped_height``, ``core_alpha``.

        Returns
        -------
        FloatArray
            Profile value; zero outside the plasma region.
        """
        result = np.zeros_like(psi_norm)
        mask = (psi_norm >= 0) & (psi_norm < 1.0)
        x = psi_norm[mask]

        y = np.clip((params["ped_top"] - x) / params["ped_width"], -20, 20)
        pedestal = 0.5 * params["ped_height"] * (1.0 + np.tanh(y))

        core = np.where(
            x < params["ped_top"],
            np.maximum(0.0, 1.0 - (x / params["ped_top"]) ** 2),
            0.0,
        )

        result[mask] = pedestal + params["core_alpha"] * core
        return result

    # ── source term ───────────────────────────────────────────────────

    def update_plasma_source_nonlinear(
        self, Psi_axis: float, Psi_boundary: float
    ) -> FloatArray:
        """Compute the toroidal current density J_phi from the GS source.

        Uses ``J_phi = R p'(psi) + FF'(psi) / (mu0 R)`` with either
        L-mode (linear) or H-mode (mtanh) profiles, then renormalises to
        match the target plasma current.

        Parameters
        ----------
        Psi_axis : float
            Poloidal flux at the magnetic axis (O-point).
        Psi_boundary : float
            Poloidal flux at the separatrix (X-point or limiter).

        Returns
        -------
        FloatArray
            Updated ``J_phi`` on the (NZ, NR) grid.
        """
        mu0: float = self.cfg["physics"]["vacuum_permeability"]

        denom = Psi_boundary - Psi_axis
        if abs(denom) < 1e-9:
            denom = 1e-9

        Psi_norm = (self.Psi - Psi_axis) / denom
        mask_plasma = (Psi_norm >= 0) & (Psi_norm < 1.0)

        if self.profile_mode in ("h-mode", "H-mode", "hmode"):
            p_profile = self.mtanh_profile(Psi_norm, self.ped_params_p)
            ff_profile = self.mtanh_profile(Psi_norm, self.ped_params_ff)
        else:
            p_profile = np.zeros_like(self.Psi)
            p_profile[mask_plasma] = 1.0 - Psi_norm[mask_plasma]
            ff_profile = p_profile.copy()

        J_p = self.RR * p_profile
        J_f = (1.0 / (mu0 * self.RR)) * ff_profile

        beta_mix = 0.5
        J_raw = beta_mix * J_p + (1 - beta_mix) * J_f

        I_current = float(np.sum(J_raw)) * self.dR * self.dZ
        I_target: float = self.cfg["physics"]["plasma_current_target"]

        if abs(I_current) > 1e-9:
            self.J_phi = J_raw * (I_target / I_current)
        else:
            self.J_phi = np.zeros_like(self.Psi)

        return self.J_phi

    # ── elliptic sub-solvers ──────────────────────────────────────────

    def _jacobi_step(
        self, Psi: FloatArray, Source: FloatArray
    ) -> FloatArray:
        """Perform one Jacobi iteration on the interior grid points.

        Parameters
        ----------
        Psi : FloatArray
            Current flux estimate.
        Source : FloatArray
            Right-hand-side source term ``-mu0 R J_phi``.

        Returns
        -------
        FloatArray
            Updated flux array (boundaries unchanged).
        """
        Psi_new = Psi.copy()
        Psi_new[1:-1, 1:-1] = 0.25 * (
            Psi[0:-2, 1:-1]
            + Psi[2:, 1:-1]
            + Psi[1:-1, 0:-2]
            + Psi[1:-1, 2:]
            - (self.dR**2) * Source[1:-1, 1:-1]
        )
        return Psi_new

    def _sor_step(
        self,
        Psi: FloatArray,
        Source: FloatArray,
        omega: float = 1.6,
    ) -> FloatArray:
        """Vectorised Red-Black SOR iteration with toroidal 1/R stencil.

        The GS* operator in cylindrical (R, Z) coordinates is:

            ∂²ψ/∂R² - (1/R) ∂ψ/∂R + ∂²ψ/∂Z² = Source

        Discretised with central differences this gives R-dependent
        coefficients a_E, a_W (east/west neighbours in R) and constant
        a_N, a_S (north/south neighbours in Z).

        Parameters
        ----------
        Psi : FloatArray
            Current flux estimate.
        Source : FloatArray
            Right-hand-side source term ``-mu0 R J_phi``.
        omega : float
            Over-relaxation factor.  Must satisfy 1.0 <= omega < 2.0.

        Returns
        -------
        FloatArray
            Updated flux array after one full red-black sweep.
        """
        Psi_new = Psi.copy()
        NZ, NR = Psi.shape
        dR2 = self.dR ** 2
        dZ2 = self.dZ ** 2

        # Toroidal stencil coefficients (arrays over interior grid)
        R_int = self.RR[1:-1, 1:-1]
        R_safe = np.maximum(R_int, 1e-10)
        a_E = 1.0 / dR2 + 1.0 / (2.0 * R_safe * self.dR)  # (NZ-2, NR-2)
        a_W = 1.0 / dR2 - 1.0 / (2.0 * R_safe * self.dR)  # (NZ-2, NR-2)
        a_NS = 1.0 / dZ2  # scalar — same for north and south
        a_C = 2.0 / dR2 + 2.0 / dZ2  # scalar

        # Checkerboard mask for interior points
        ii, jj = np.mgrid[1:NZ - 1, 1:NR - 1]

        for parity in (0, 1):  # 0 = red, 1 = black
            mask = ((ii + jj) % 2) == parity
            gs_update = (
                a_E[mask] * Psi_new[1:-1, 2:][mask]
                + a_W[mask] * Psi_new[1:-1, 0:-2][mask]
                + a_NS * Psi_new[0:-2, 1:-1][mask]
                + a_NS * Psi_new[2:, 1:-1][mask]
                - Source[1:-1, 1:-1][mask]
            ) / a_C

            old_vals = Psi_new[1:-1, 1:-1][mask]
            interior = Psi_new[1:-1, 1:-1]
            interior[mask] = (1.0 - omega) * old_vals + omega * gs_update
            Psi_new[1:-1, 1:-1] = interior

        return Psi_new

    # ── multigrid sub-solvers ─────────────────────────────────────────

    @staticmethod
    def _restrict_full_weight(fine: FloatArray) -> FloatArray:
        """Full-weighting restriction operator (fine → coarse).

        Standard 9-point stencil:
            coarse[i,j] = 1/16 * (4*fine[2i,2j]
                          + 2*(fine[2i-1,2j] + fine[2i+1,2j]
                               + fine[2i,2j-1] + fine[2i,2j+1])
                          + (fine[2i-1,2j-1] + fine[2i-1,2j+1]
                             + fine[2i+1,2j-1] + fine[2i+1,2j+1]))
        """
        nz_f, nr_f = fine.shape
        nz_c = (nz_f + 1) // 2
        nr_c = (nr_f + 1) // 2
        coarse = np.zeros((nz_c, nr_c))

        # Interior points via full-weighting
        for ic in range(1, nz_c - 1):
            for jc in range(1, nr_c - 1):
                i = 2 * ic
                j = 2 * jc
                coarse[ic, jc] = (
                    4.0 * fine[i, j]
                    + 2.0 * (fine[i - 1, j] + fine[i + 1, j]
                             + fine[i, j - 1] + fine[i, j + 1])
                    + (fine[i - 1, j - 1] + fine[i - 1, j + 1]
                       + fine[i + 1, j - 1] + fine[i + 1, j + 1])
                ) / 16.0

        # Boundary: inject directly
        coarse[0, :] = fine[0, ::2][:nr_c]
        coarse[-1, :] = fine[-1, ::2][:nr_c]
        coarse[:, 0] = fine[::2, 0][:nz_c]
        coarse[:, -1] = fine[::2, -1][:nr_c]

        return coarse

    @staticmethod
    def _prolongate_bilinear(coarse: FloatArray, nz_f: int, nr_f: int) -> FloatArray:
        """Bilinear prolongation operator (coarse → fine).

        Direct injection at coincident points, linear interpolation elsewhere.
        """
        nz_c, nr_c = coarse.shape
        fine = np.zeros((nz_f, nr_f))

        for ic in range(nz_c):
            for jc in range(nr_c):
                i = 2 * ic
                j = 2 * jc
                if i < nz_f and j < nr_f:
                    fine[i, j] = coarse[ic, jc]

        # Interpolate horizontal midpoints
        for ic in range(nz_c):
            i = 2 * ic
            if i >= nz_f:
                continue
            for jc in range(nr_c - 1):
                j = 2 * jc + 1
                if j < nr_f:
                    fine[i, j] = 0.5 * (coarse[ic, jc] + coarse[ic, jc + 1])

        # Interpolate vertical midpoints
        for ic in range(nz_c - 1):
            i = 2 * ic + 1
            if i >= nz_f:
                continue
            for jc in range(nr_c):
                j = 2 * jc
                if j < nr_f:
                    fine[i, j] = 0.5 * (coarse[ic, jc] + coarse[ic + 1, jc])

        # Interpolate center points
        for ic in range(nz_c - 1):
            i = 2 * ic + 1
            if i >= nz_f:
                continue
            for jc in range(nr_c - 1):
                j = 2 * jc + 1
                if j < nr_f:
                    fine[i, j] = 0.25 * (
                        coarse[ic, jc] + coarse[ic + 1, jc]
                        + coarse[ic, jc + 1] + coarse[ic + 1, jc + 1]
                    )

        return fine

    def _mg_smooth(
        self,
        Psi: FloatArray,
        Source: FloatArray,
        R_grid: FloatArray,
        dR: float,
        dZ: float,
        omega: float,
        n_sweeps: int,
    ) -> FloatArray:
        """Red-Black SOR smoother with toroidal 1/R stencil for multigrid.

        Works on arbitrary grid sizes (not just the root grid).
        """
        NZ, NR = Psi.shape
        dR2 = dR ** 2
        dZ2 = dZ ** 2

        R_int = R_grid[1:-1, 1:-1]
        R_safe = np.maximum(R_int, 1e-10)
        a_E = 1.0 / dR2 + 1.0 / (2.0 * R_safe * dR)
        a_W = 1.0 / dR2 - 1.0 / (2.0 * R_safe * dR)
        a_NS = 1.0 / dZ2
        a_C = 2.0 / dR2 + 2.0 / dZ2

        ii, jj = np.mgrid[1:NZ - 1, 1:NR - 1]

        for _ in range(n_sweeps):
            for parity in (0, 1):
                mask = ((ii + jj) % 2) == parity
                gs_update = (
                    a_E[mask] * Psi[1:-1, 2:][mask]
                    + a_W[mask] * Psi[1:-1, 0:-2][mask]
                    + a_NS * Psi[0:-2, 1:-1][mask]
                    + a_NS * Psi[2:, 1:-1][mask]
                    - Source[1:-1, 1:-1][mask]
                ) / a_C
                old_vals = Psi[1:-1, 1:-1][mask]
                interior = Psi[1:-1, 1:-1]
                interior[mask] = (1.0 - omega) * old_vals + omega * gs_update
                Psi[1:-1, 1:-1] = interior

        return Psi

    def _mg_residual(
        self,
        Psi: FloatArray,
        Source: FloatArray,
        R_grid: FloatArray,
        dR: float,
        dZ: float,
    ) -> FloatArray:
        """Compute GS* residual r = L*[Psi] - Source on given grid."""
        NZ, NR = Psi.shape
        dR2 = dR ** 2
        dZ2 = dZ ** 2

        residual = np.zeros_like(Psi)
        R_int = R_grid[1:-1, 1:-1]
        R_safe = np.maximum(R_int, 1e-10)

        d2R = (Psi[1:-1, 2:] - 2.0 * Psi[1:-1, 1:-1] + Psi[1:-1, 0:-2]) / dR2
        d1R = (Psi[1:-1, 2:] - Psi[1:-1, 0:-2]) / (2.0 * dR)
        d2Z = (Psi[2:, 1:-1] - 2.0 * Psi[1:-1, 1:-1] + Psi[0:-2, 1:-1]) / dZ2

        Lpsi = d2R - d1R / R_safe + d2Z
        residual[1:-1, 1:-1] = Lpsi - Source[1:-1, 1:-1]
        return residual

    def _multigrid_vcycle(
        self,
        Psi: FloatArray,
        Source: FloatArray,
        R_grid: FloatArray,
        dR: float,
        dZ: float,
        *,
        omega: float = 1.6,
        pre_smooth: int = 3,
        post_smooth: int = 3,
        min_grid: int = 5,
    ) -> FloatArray:
        """One V-cycle of geometric multigrid for the GS* operator.

        Parameters
        ----------
        Psi : FloatArray
            Current solution estimate.
        Source : FloatArray
            Right-hand-side source term.
        R_grid : FloatArray
            R-coordinate meshgrid matching Psi shape.
        dR, dZ : float
            Grid spacings.
        omega : float
            SOR over-relaxation factor.
        pre_smooth, post_smooth : int
            Number of smoothing sweeps before/after coarse correction.
        min_grid : int
            Minimum grid dimension before switching to direct solve.

        Returns
        -------
        FloatArray
            Improved solution estimate.
        """
        NZ, NR = Psi.shape

        # Base case: grid too coarse — solve directly with many SOR sweeps
        if NZ <= min_grid or NR <= min_grid:
            return self._mg_smooth(
                Psi.copy(), Source, R_grid, dR, dZ, omega, n_sweeps=50
            )

        # 1. Pre-smooth
        Psi = self._mg_smooth(Psi.copy(), Source, R_grid, dR, dZ, omega, pre_smooth)

        # 2. Compute residual
        residual = self._mg_residual(Psi, Source, R_grid, dR, dZ)

        # 3. Restrict residual and R-grid to coarse level
        r_coarse = self._restrict_full_weight(residual)
        R_coarse = self._restrict_full_weight(R_grid)
        nz_c, nr_c = r_coarse.shape

        # Coarse grid spacings (doubled)
        dR_c = dR * 2.0
        dZ_c = dZ * 2.0

        # 4. Solve coarse-grid correction: L*[e] = r
        e_coarse = np.zeros((nz_c, nr_c))
        e_coarse = self._multigrid_vcycle(
            e_coarse, r_coarse, R_coarse, dR_c, dZ_c,
            omega=omega, pre_smooth=pre_smooth, post_smooth=post_smooth,
            min_grid=min_grid,
        )

        # 5. Prolongate correction and apply
        correction = self._prolongate_bilinear(e_coarse, NZ, NR)
        Psi = Psi + correction

        # 6. Post-smooth
        Psi = self._mg_smooth(Psi, Source, R_grid, dR, dZ, omega, post_smooth)

        return Psi

    def _anderson_step(
        self,
        psi_history: list[FloatArray],
        res_history: list[FloatArray],
        m: int = 5,
    ) -> FloatArray:
        """Anderson acceleration (mixing) for the Picard iterate sequence.

        Computes optimal coefficients from the last *m* residuals via a
        least-squares solve, then returns the mixed iterate.

        Parameters
        ----------
        psi_history : list[FloatArray]
            List of recent Psi iterates.
        res_history : list[FloatArray]
            Corresponding residuals ``Psi_{k+1} - Psi_k`` for each iterate.
        m : int
            Mixing depth (number of previous iterates to use).

        Returns
        -------
        FloatArray
            Anderson-mixed Psi iterate.
        """
        k = len(res_history)
        mk = min(m, k)

        if mk < 2:
            # Not enough history — fall back to latest iterate
            return psi_history[-1].copy()

        # Stack the last mk residuals as column vectors
        res_cols = [r.ravel() for r in res_history[-mk:]]
        F = np.column_stack(res_cols)  # (N, mk)

        # Solve min ||F @ alpha||^2 s.t. sum(alpha) = 1
        # via: delta_F[:,j] = F[:,j+1] - F[:,j], then solve normal equations
        dF = np.diff(F, axis=1)  # (N, mk-1)
        rhs = F[:, -1]           # latest residual

        # Tikhonov regularisation for numerical stability
        gram = dF.T @ dF
        gram += 1e-10 * np.eye(gram.shape[0])
        try:
            gamma = np.linalg.solve(gram, dF.T @ rhs)
        except np.linalg.LinAlgError:
            return psi_history[-1].copy()

        # Reconstruct alpha from gamma
        alpha = np.zeros(mk)
        alpha[-1] = 1.0 - np.sum(gamma)
        alpha[:-1] -= gamma  # alpha_j -= gamma_j
        # But alpha must sum to 1; fix via normalisation
        alpha_sum = np.sum(alpha)
        if abs(alpha_sum) < 1e-12:
            return psi_history[-1].copy()
        alpha /= alpha_sum

        # Mix iterates
        shape = psi_history[-1].shape
        mixed = np.zeros_like(psi_history[-1])
        psi_cols = psi_history[-mk:]
        for j in range(mk):
            mixed += alpha[j] * psi_cols[j]

        return mixed

    def _apply_boundary_conditions(
        self, Psi: FloatArray, Psi_bc: FloatArray
    ) -> None:
        """Copy vacuum-field boundary values onto the edges of *Psi*.

        Parameters
        ----------
        Psi : FloatArray
            Array to update (modified in place).
        Psi_bc : FloatArray
            Boundary-condition source (typically the vacuum field).
        """
        Psi[0, :] = Psi_bc[0, :]
        Psi[-1, :] = Psi_bc[-1, :]
        Psi[:, 0] = Psi_bc[:, 0]
        Psi[:, -1] = Psi_bc[:, -1]

    def _elliptic_solve(
        self, Source: FloatArray, Psi_bc: FloatArray
    ) -> FloatArray:
        """Run the inner elliptic solve (HPC or Python fallback).

        The solver method is chosen from the config key
        ``solver.solver_method``:

        - ``"jacobi"`` — legacy single Jacobi sweep (fastest per step,
          slowest convergence).
        - ``"sor"`` — Red-Black SOR with toroidal 1/R stencil (default).
        - ``"anderson"`` — SOR + Anderson acceleration (best convergence).

        Parameters
        ----------
        Source : FloatArray
            Right-hand-side ``-mu0 R J_phi``.
        Psi_bc : FloatArray
            Vacuum-field array used for Dirichlet boundary conditions.

        Returns
        -------
        FloatArray
            Updated Psi after elliptic solve + boundary enforcement.
        """
        if self.hpc.is_available():
            Psi_acc = self.hpc.solve(self.J_phi, iterations=50)
            if Psi_acc is not None:
                self._apply_boundary_conditions(Psi_acc, Psi_bc)
                return Psi_acc

        method = self.cfg["solver"].get("solver_method", "multigrid")
        omega = self.cfg["solver"].get("sor_omega", 1.6)

        if method == "jacobi":
            Psi_new = self._jacobi_step(self.Psi, Source)
        elif method == "multigrid":
            Psi_new = self._multigrid_vcycle(
                self.Psi.copy(), Source, self.RR, self.dR, self.dZ,
                omega=omega,
            )
        else:
            # Both "sor" and "anderson" use SOR as the inner sweep
            Psi_new = self._sor_step(self.Psi, Source, omega=omega)

        self._apply_boundary_conditions(Psi_new, Psi_bc)
        return Psi_new

    # ── seed plasma ───────────────────────────────────────────────────

    def _seed_plasma(self, mu0: float) -> None:
        """Create an initial Gaussian current seed and do preliminary solves.

        Parameters
        ----------
        mu0 : float
            Vacuum permeability.
        """
        R_center = (
            self.cfg["dimensions"]["R_min"]
            + self.cfg["dimensions"]["R_max"]
        ) / 2.0
        dist_sq = (self.RR - R_center) ** 2 + self.ZZ**2
        self.J_phi = np.exp(-dist_sq / 2.0)

        I_seed = float(np.sum(self.J_phi)) * self.dR * self.dZ
        I_target: float = self.cfg["physics"]["plasma_current_target"]
        if I_seed > 0:
            self.J_phi *= I_target / I_seed

        Source = -mu0 * self.RR * self.J_phi
        for _ in range(50):
            self.Psi = self._jacobi_step(self.Psi, Source)

    def _prepare_initial_flux(
        self,
        preserve_initial_state: bool,
        boundary_flux: FloatArray | None,
    ) -> FloatArray:
        """Prepare initial Psi and boundary map for iterative GS solves.

        Parameters
        ----------
        preserve_initial_state : bool
            When True, keep the existing interior ``self.Psi`` values and only
            enforce the provided boundary map.
        boundary_flux : FloatArray | None
            Explicit boundary map to enforce. Must match ``self.Psi.shape``
            when provided.

        Returns
        -------
        FloatArray
            Boundary flux map used by boundary-condition enforcement.
        """
        if boundary_flux is not None:
            psi_boundary = np.asarray(boundary_flux, dtype=np.float64)
            if psi_boundary.shape != self.Psi.shape:
                raise ValueError(
                    f"boundary_flux shape {psi_boundary.shape} "
                    f"must match Psi shape {self.Psi.shape}"
                )
            psi_boundary = psi_boundary.copy()
        elif preserve_initial_state:
            psi_boundary = self.Psi.copy()
        else:
            psi_boundary = self.calculate_vacuum_field()

        if preserve_initial_state:
            self._apply_boundary_conditions(self.Psi, psi_boundary)
        else:
            self.Psi = psi_boundary.copy()

        return psi_boundary

    # ── Newton-Kantorovich equilibrium solver ────────────────────────

    def _compute_gs_residual(self, Source: FloatArray) -> FloatArray:
        """Compute the GS residual r = L*[psi] - Source on interior points.

        The toroidal GS* operator is:
            L* = d2/dR2 - (1/R) d/dR + d2/dZ2
        """
        Psi = self.Psi
        NZ, NR = Psi.shape
        dR2 = self.dR ** 2
        dZ2 = self.dZ ** 2

        residual = np.zeros_like(Psi)
        R_int = self.RR[1:-1, 1:-1]
        R_safe = np.maximum(R_int, 1e-10)

        # 5-point toroidal stencil
        d2R = (Psi[1:-1, 2:] - 2.0 * Psi[1:-1, 1:-1] + Psi[1:-1, 0:-2]) / dR2
        d1R = (Psi[1:-1, 2:] - Psi[1:-1, 0:-2]) / (2.0 * self.dR)
        d2Z = (Psi[2:, 1:-1] - 2.0 * Psi[1:-1, 1:-1] + Psi[0:-2, 1:-1]) / dZ2

        Lpsi = d2R - d1R / R_safe + d2Z
        residual[1:-1, 1:-1] = Lpsi - Source[1:-1, 1:-1]
        return residual

    def _compute_gs_residual_rms(self, Source: FloatArray) -> float:
        """Return RMS GS residual over interior points."""
        residual = self._compute_gs_residual(Source)
        interior = residual[1:-1, 1:-1]
        if interior.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(interior * interior)))

    def _apply_gs_operator(self, v: FloatArray) -> FloatArray:
        """Apply the discrete GS* operator to array *v*.

        Used as the matvec in the GMRES LinearOperator for Newton.
        """
        NZ, NR = v.shape
        dR2 = self.dR ** 2
        dZ2 = self.dZ ** 2

        result = np.zeros_like(v)
        R_int = self.RR[1:-1, 1:-1]
        R_safe = np.maximum(R_int, 1e-10)

        d2R = (v[1:-1, 2:] - 2.0 * v[1:-1, 1:-1] + v[1:-1, 0:-2]) / dR2
        d1R = (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2.0 * self.dR)
        d2Z = (v[2:, 1:-1] - 2.0 * v[1:-1, 1:-1] + v[0:-2, 1:-1]) / dZ2

        result[1:-1, 1:-1] = d2R - d1R / R_safe + d2Z
        return result

    def _compute_profile_jacobian(
        self, Psi_axis: float, Psi_boundary: float, mu0: float
    ) -> FloatArray:
        """Compute dJ_phi/dpsi as a 2D diagonal scaling field.

        For L-mode linear profiles:
            p'(psi_norm) = const => J_phi ∝ (1 - psi_norm) * R
            dJ_phi/dpsi = -c / (Psi_boundary - Psi_axis) for points inside plasma

        Returns a 2D array of the same shape as self.Psi.
        """
        denom = Psi_boundary - Psi_axis
        if abs(denom) < 1e-9:
            denom = 1e-9

        Psi_norm = (self.Psi - Psi_axis) / denom
        mask_plasma = (Psi_norm >= 0) & (Psi_norm < 1.0)

        # For the linear L-mode profile: Source = -mu0 * R * J_phi
        # J_phi = c * (1 - psi_norm) * R  =>  dJ_phi/dpsi_norm = -c * R
        # dJ_phi/dpsi = dJ_phi/dpsi_norm * dpsi_norm/dpsi = -c * R / denom
        # We compute c from the current normalisation
        I_target = self.cfg["physics"]["plasma_current_target"]
        # Approximate: I = integral(J_phi) dA ≈ c * sum_plasma((1-psi_norm)*R) * dR*dZ
        s = float(np.sum(np.where(mask_plasma, (1 - Psi_norm) * self.RR, 0.0))) * self.dR * self.dZ
        c = I_target / max(abs(s), 1e-9)

        dJ_dpsi = np.zeros_like(self.Psi)
        dJ_dpsi[mask_plasma] = -c * self.RR[mask_plasma] / denom

        return dJ_dpsi

    def _newton_solve_dispatch(
        self,
        preserve_initial_state: bool = False,
        boundary_flux: FloatArray | None = None,
    ) -> dict[str, Any]:
        """Newton-Kantorovich equilibrium solver with Picard warmup.

        1. Run 15 Picard warmup steps to get a reasonable initial guess.
        2. Switch to Newton: at each step solve J_k * delta = -r_k via
           scipy GMRES, then update psi += alpha * delta.

        Returns the standard result dict.
        """
        from scipy.sparse.linalg import LinearOperator, gmres

        t0 = time.time()
        Psi_vac_boundary = self._prepare_initial_flux(
            preserve_initial_state=preserve_initial_state,
            boundary_flux=boundary_flux,
        )

        max_iter: int = self.cfg["solver"]["max_iterations"]
        tol: float = self.cfg["solver"]["convergence_threshold"]
        picard_alpha: float = self.cfg["solver"].get("relaxation_factor", 0.1)
        fail_on_diverge: bool = bool(self.cfg["solver"].get("fail_on_diverge", False))
        require_gs_residual: bool = bool(
            self.cfg["solver"].get("require_gs_residual", False)
        )
        gs_tol: float = float(self.cfg["solver"].get("gs_residual_threshold", tol))
        if require_gs_residual and gs_tol <= 0.0:
            raise ValueError("solver.gs_residual_threshold must be > 0")
        mu0: float = self.cfg["physics"]["vacuum_permeability"]
        warmup_steps: int = min(15, max_iter // 2)
        newton_alpha: float = 0.5  # damped Newton

        residual_history: list[float] = []
        gs_residual_history: list[float] = []
        converged = False
        final_iter = 0
        gs_best: float = float("inf")
        final_source: FloatArray | None = None

        self._seed_plasma(mu0)

        # ── Phase A: Picard warmup ──
        for k in range(warmup_steps):
            final_iter = k
            _, _, Psi_axis = self._find_magnetic_axis()
            _, Psi_boundary = self.find_x_point(self.Psi)
            if abs(Psi_axis - Psi_boundary) < 0.1:
                Psi_boundary = Psi_axis * 0.1

            if not getattr(self, "external_profile_mode", False):
                self.J_phi = self.update_plasma_source_nonlinear(Psi_axis, Psi_boundary)

            Source = -mu0 * self.RR * self.J_phi
            final_source = Source
            Psi_new = self._elliptic_solve(Source, Psi_vac_boundary)

            if np.isnan(Psi_new).any() or np.isinf(Psi_new).any():
                if fail_on_diverge:
                    raise RuntimeError(f"Newton warmup diverged at iter={k}")
                break

            diff = float(np.mean(np.abs(Psi_new - self.Psi)))
            residual_history.append(diff)
            self.Psi = (1.0 - picard_alpha) * self.Psi + picard_alpha * Psi_new
            self._apply_boundary_conditions(self.Psi, Psi_vac_boundary)
            gs_residual = self._compute_gs_residual_rms(Source)
            gs_residual_history.append(gs_residual)
            if gs_residual < gs_best:
                gs_best = gs_residual

            update_converged = diff < tol
            gs_converged = (not require_gs_residual) or (gs_residual < gs_tol)
            if update_converged and gs_converged:
                converged = True
                break

        # ── Phase B: Newton iterations ──
        if not converged:
            NZ, NR_grid = self.Psi.shape
            n_interior = (NZ - 2) * (NR_grid - 2)

            for k in range(warmup_steps, max_iter):
                final_iter = k

                _, _, Psi_axis = self._find_magnetic_axis()
                _, Psi_boundary = self.find_x_point(self.Psi)
                if abs(Psi_axis - Psi_boundary) < 0.1:
                    Psi_boundary = Psi_axis * 0.1

                if not getattr(self, "external_profile_mode", False):
                    self.J_phi = self.update_plasma_source_nonlinear(Psi_axis, Psi_boundary)

                # Residual: r_k = L[psi] + mu0*R*J_phi  (Source = -mu0*R*J)
                Source = -mu0 * self.RR * self.J_phi
                final_source = Source
                r_k = self._compute_gs_residual(Source)
                res_norm = float(np.sqrt(np.sum(r_k[1:-1, 1:-1] ** 2)))
                gs_residual = float(np.sqrt(np.mean(r_k[1:-1, 1:-1] ** 2)))
                residual_history.append(res_norm)
                gs_residual_history.append(gs_residual)
                if gs_residual < gs_best:
                    gs_best = gs_residual

                update_converged = res_norm < tol
                gs_converged = (not require_gs_residual) or (gs_residual < gs_tol)
                if update_converged and gs_converged:
                    converged = True
                    break

                # Build Jacobian operator: J_k = L + mu0*R*dJ/dpsi
                dJ_dpsi = self._compute_profile_jacobian(Psi_axis, Psi_boundary, mu0)
                diag_term = -mu0 * self.RR * dJ_dpsi  # the source derivative

                def matvec(v_flat: np.ndarray) -> np.ndarray:
                    v2d = np.zeros((NZ, NR_grid))
                    v2d[1:-1, 1:-1] = v_flat.reshape(NZ - 2, NR_grid - 2)
                    Lv = self._apply_gs_operator(v2d)
                    Lv[1:-1, 1:-1] -= diag_term[1:-1, 1:-1] * v2d[1:-1, 1:-1]
                    return Lv[1:-1, 1:-1].ravel()

                J_op = LinearOperator(
                    shape=(n_interior, n_interior),
                    matvec=matvec,
                    dtype=np.float64,
                )

                # Solve J_k * delta = -r_k
                rhs = -r_k[1:-1, 1:-1].ravel()
                delta_flat, info = gmres(J_op, rhs, maxiter=100, restart=50,
                                         atol=1e-8, rtol=1e-6)

                if info != 0:
                    logger.warning("GMRES did not converge at Newton iter %d (info=%d)", k, info)

                delta = np.zeros_like(self.Psi)
                delta[1:-1, 1:-1] = delta_flat.reshape(NZ - 2, NR_grid - 2)

                # Damped Newton update
                self.Psi += newton_alpha * delta
                self._apply_boundary_conditions(self.Psi, Psi_vac_boundary)

                # NaN check
                if np.isnan(self.Psi).any() or np.isinf(self.Psi).any():
                    logger.warning("Newton diverged at iter %d", k)
                    if fail_on_diverge:
                        raise RuntimeError(f"Newton solver diverged at iter={k}")
                    break

        if final_source is None:
            gs_final = float("inf")
            gs_best_out = float("inf")
        elif gs_residual_history:
            gs_final = gs_residual_history[-1]
            gs_best_out = gs_best
        else:
            gs_final = self._compute_gs_residual_rms(final_source)
            gs_best_out = gs_final

        self.compute_b_field()
        elapsed = time.time() - t0
        logger.info("Newton solved in %.2fs, %d iters", elapsed, final_iter + 1)

        return {
            "psi": self.Psi,
            "converged": converged,
            "iterations": final_iter + 1,
            "residual": residual_history[-1] if residual_history else float("inf"),
            "residual_history": residual_history,
            "gs_residual": gs_final,
            "gs_residual_best": gs_best_out,
            "gs_residual_history": gs_residual_history,
            "wall_time_s": elapsed,
            "solver_method": "newton",
        }

    # ── Rust multigrid delegation ────────────────────────────────────

    def _solve_via_rust_multigrid(
        self,
        preserve_initial_state: bool = False,
        boundary_flux: FloatArray | None = None,
    ) -> dict[str, Any]:
        """Delegate the full equilibrium solve to the Rust multigrid backend.

        Falls back to Python SOR if the Rust extension is not installed.
        """
        from scpn_fusion.core._rust_compat import _rust_available, RustAcceleratedKernel

        if preserve_initial_state or boundary_flux is not None:
            logger.warning(
                "Boundary-constrained solve requested with rust_multigrid; "
                "falling back to Python SOR."
            )
            prior_method = self.cfg["solver"].get("solver_method", "rust_multigrid")
            self.cfg["solver"]["solver_method"] = "sor"
            try:
                return self.solve_equilibrium(
                    preserve_initial_state=preserve_initial_state,
                    boundary_flux=boundary_flux,
                )
            finally:
                self.cfg["solver"]["solver_method"] = prior_method

        if not _rust_available():
            logger.warning("Rust unavailable; falling back to Python SOR.")
            prior_method = self.cfg["solver"].get("solver_method", "rust_multigrid")
            self.cfg["solver"]["solver_method"] = "sor"
            try:
                return self.solve_equilibrium()
            finally:
                self.cfg["solver"]["solver_method"] = prior_method

        t0 = time.time()
        rk = RustAcceleratedKernel(self._config_path)
        rk.set_solver_method("multigrid")
        rust_result = rk.solve_equilibrium()

        # Sync state back
        self.Psi = rk.Psi
        self.J_phi = rk.J_phi
        self.B_R = rk.B_R
        self.B_Z = rk.B_Z

        mu0: float = self.cfg["physics"]["vacuum_permeability"]
        source = -mu0 * self.RR * self.J_phi
        gs_residual = self._compute_gs_residual_rms(source)
        elapsed = time.time() - t0
        solver_tol = float(self.cfg.get("solver", {}).get("convergence_threshold", 1e-4))
        practical_tol = max(solver_tol, 2e-3)
        converged = bool(rust_result.converged or rust_result.residual <= practical_tol)
        return {
            "psi": self.Psi,
            "converged": converged,
            "iterations": rust_result.iterations,
            "residual": rust_result.residual,
            "residual_history": [],
            "gs_residual": gs_residual,
            "gs_residual_best": gs_residual,
            "gs_residual_history": [],
            "wall_time_s": elapsed,
            "solver_method": "rust_multigrid",
        }

    # ── main solver ───────────────────────────────────────────────────

    def solve_equilibrium(
        self,
        preserve_initial_state: bool = False,
        boundary_flux: FloatArray | None = None,
    ) -> dict[str, Any]:
        """Run the full Picard-iteration equilibrium solver.

        Iterates: topology analysis -> source update -> elliptic solve ->
        under-relaxation until the residual drops below the configured
        convergence threshold or the maximum iteration count is reached.

        When ``solver.solver_method`` is ``"anderson"``, Anderson
        acceleration is applied every few Picard steps to speed up
        convergence.

        Returns
        -------
        dict[str, Any]
            ``{"psi": FloatArray, "converged": bool, "iterations": int,
            "residual": float, "residual_history": list[float],
            "gs_residual": float, "gs_residual_best": float,
            "gs_residual_history": list[float],
            "wall_time_s": float, "solver_method": str}``

        Parameters
        ----------
        preserve_initial_state : bool, optional
            Keep current interior ``self.Psi`` values and only enforce boundary
            conditions before the iterative solve. Default is ``False``.
        boundary_flux : FloatArray | None, optional
            Explicit boundary map to enforce during solve. Must have shape
            ``(NZ, NR)`` when provided.

        Raises
        ------
        RuntimeError
            If the solver produces NaN or Inf and ``solver.fail_on_diverge``
            is enabled in the active configuration.
        """
        t0 = time.time()

        method: str = self.cfg["solver"].get("solver_method", "multigrid")

        # ── Fast-path dispatches ──
        if method == "rust_multigrid":
            return self._solve_via_rust_multigrid(
                preserve_initial_state=preserve_initial_state,
                boundary_flux=boundary_flux,
            )
        if method == "newton":
            return self._newton_solve_dispatch(
                preserve_initial_state=preserve_initial_state,
                boundary_flux=boundary_flux,
            )

        Psi_vac_boundary = self._prepare_initial_flux(
            preserve_initial_state=preserve_initial_state,
            boundary_flux=boundary_flux,
        )

        max_iter: int = self.cfg["solver"]["max_iterations"]
        tol: float = self.cfg["solver"]["convergence_threshold"]
        alpha: float = self.cfg["solver"].get("relaxation_factor", 0.1)
        fail_on_diverge: bool = bool(
            self.cfg["solver"].get("fail_on_diverge", False)
        )
        require_gs_residual: bool = bool(
            self.cfg["solver"].get("require_gs_residual", False)
        )
        gs_tol: float = float(self.cfg["solver"].get("gs_residual_threshold", tol))
        if require_gs_residual and gs_tol <= 0.0:
            raise ValueError("solver.gs_residual_threshold must be > 0")
        mu0: float = self.cfg["physics"]["vacuum_permeability"]
        anderson_m: int = self.cfg["solver"].get("anderson_depth", 5)

        x_point_pos: tuple[float, float] = (0.0, 0.0)
        Psi_best = self.Psi.copy()
        diff_best: float = 1e9
        residual_history: list[float] = []
        gs_residual_history: list[float] = []
        converged = False
        gs_best: float = float("inf")
        final_source: FloatArray | None = None

        # Anderson acceleration history buffers
        psi_history: list[FloatArray] = []
        res_history: list[FloatArray] = []

        self._seed_plasma(mu0)

        final_iter = 0
        for k in range(max_iter):
            final_iter = k
            # 1. Topology
            _, _, Psi_axis = self._find_magnetic_axis()
            x_point_pos, Psi_boundary = self.find_x_point(self.Psi)

            if abs(Psi_axis - Psi_boundary) < 0.1:
                Psi_boundary = Psi_axis * 0.1

            # 2. Source update
            if not getattr(self, "external_profile_mode", False):
                self.J_phi = self.update_plasma_source_nonlinear(
                    Psi_axis, Psi_boundary
                )

            # 3. Elliptic solve
            Source = -mu0 * self.RR * self.J_phi
            final_source = Source
            Psi_new = self._elliptic_solve(Source, Psi_vac_boundary)

            # Divergence check
            if np.isnan(Psi_new).any() or np.isinf(Psi_new).any():
                logger.warning(
                    "Solver diverged at iter %d — reverting to best state.",
                    k,
                )
                self.Psi = Psi_best
                if fail_on_diverge:
                    raise RuntimeError(
                        f"Equilibrium solver diverged at iter={k}"
                    )
                break

            # 4. Under-relaxation
            diff = float(np.mean(np.abs(Psi_new - self.Psi)))
            residual_history.append(diff)
            self.Psi = (1.0 - alpha) * self.Psi + alpha * Psi_new

            # 5. Anderson acceleration (optional)
            if method == "anderson":
                psi_history.append(self.Psi.copy())
                res_history.append(Psi_new - self.Psi)
                if len(psi_history) >= 3 and k % 3 == 0:
                    mixed = self._anderson_step(
                        psi_history, res_history, m=anderson_m
                    )
                    self._apply_boundary_conditions(mixed, Psi_vac_boundary)
                    self.Psi = mixed
                # Trim history to avoid unbounded memory growth
                if len(psi_history) > anderson_m + 2:
                    psi_history.pop(0)
                    res_history.pop(0)

            gs_residual = self._compute_gs_residual_rms(Source)
            gs_residual_history.append(gs_residual)
            if gs_residual < gs_best:
                gs_best = gs_residual

            if diff < diff_best:
                diff_best = diff
                Psi_best = self.Psi.copy()

            update_converged = diff < tol
            gs_converged = (not require_gs_residual) or (gs_residual < gs_tol)
            if update_converged and gs_converged:
                logger.info(
                    "Converged at iter %d.  Update residual: %.6e | GS RMS: %.6e",
                    k,
                    diff,
                    gs_residual,
                )
                converged = True
                break

            if k % 100 == 0:
                logger.debug(
                    "Iter %d: res=%.2e | axis=%.2f | X-pt=%.2f "
                    "at R=%.2f, Z=%.2f",
                    k,
                    diff,
                    Psi_axis,
                    Psi_boundary,
                    x_point_pos[0],
                    x_point_pos[1],
                )

        if final_source is None:
            gs_final = float("inf")
            gs_best_out = float("inf")
        elif gs_residual_history:
            gs_final = gs_residual_history[-1]
            gs_best_out = gs_best
        else:
            gs_final = self._compute_gs_residual_rms(final_source)
            gs_best_out = gs_final

        self.compute_b_field()
        elapsed = time.time() - t0
        logger.info(
            "Solved in %.2fs (%s).  X-point: R=%.2f, Z=%.2f",
            elapsed,
            method,
            x_point_pos[0],
            x_point_pos[1],
        )

        return {
            "psi": self.Psi,
            "converged": converged,
            "iterations": final_iter + 1,
            "residual": diff_best,
            "residual_history": residual_history,
            "gs_residual": gs_final,
            "gs_residual_best": gs_best_out,
            "gs_residual_history": gs_residual_history,
            "wall_time_s": elapsed,
            "solver_method": method,
        }

    # ── post-processing ───────────────────────────────────────────────

    def compute_b_field(self) -> None:
        """Derive the magnetic field components from the solved Psi."""
        dPsi_dR, dPsi_dZ = np.gradient(self.Psi, self.dR, self.dZ)
        R_safe = np.maximum(self.RR, 1e-6)
        self.B_R = -(1.0 / R_safe) * dPsi_dZ
        self.B_Z = (1.0 / R_safe) * dPsi_dR

    @staticmethod
    def _green_function(R_src, Z_src, R_obs, Z_obs):
        """Toroidal Green's function using elliptic integrals."""
        mu0 = 4e-7 * np.pi
        denom = (R_obs + R_src)**2 + (Z_obs - Z_src)**2
        if denom < 1e-30:
            return 0.0
        k2 = 4.0 * R_obs * R_src / denom
        k2 = np.clip(k2, 1e-9, 0.999999)
        k = np.sqrt(k2)
        from scipy.special import ellipk, ellipe
        K_val = ellipk(k2)
        E_val = ellipe(k2)
        prefactor = mu0 / (2.0 * np.pi) * np.sqrt(R_obs * R_src)
        psi = prefactor * ((2.0 - k2) * K_val - 2.0 * E_val) / k
        return float(psi)

    def _compute_external_flux(self, coils):
        """Sum Green's function contributions on boundary from CoilSet."""
        NR, NZ = len(self.R), len(self.Z)
        psi_ext = np.zeros((NZ, NR))
        for idx, (pos, current) in enumerate(zip(coils.positions, coils.currents)):
            R_c, Z_c = pos
            turns = coils.turns[idx] if idx < len(coils.turns) else 1
            I_eff = current * turns
            for iz in range(NZ):
                for ir in range(NR):
                    psi_ext[iz, ir] += I_eff * self._green_function(
                        R_c, Z_c, self.R[ir], self.Z[iz]
                    )
        return psi_ext

    def _build_mutual_inductance_matrix(
        self,
        coils: CoilSet,
        obs_points: FloatArray,
    ) -> FloatArray:
        """Build mutual-inductance matrix M[k, p] for coil optimisation.

        ``M[k, p]`` is the flux at observation point *p* due to unit current
        in coil *k*.  Uses the toroidal Green's function.

        Parameters
        ----------
        coils : CoilSet
            Coil geometry.
        obs_points : FloatArray, shape (n_pts, 2)
            Observation points ``(R, Z)`` — typically the target separatrix.

        Returns
        -------
        FloatArray, shape (n_coils, n_pts)
        """
        n_coils = len(coils.positions)
        n_pts = obs_points.shape[0]
        M = np.zeros((n_coils, n_pts))

        for k, (Rc, Zc) in enumerate(coils.positions):
            turns = coils.turns[k] if k < len(coils.turns) else 1
            for p in range(n_pts):
                R_obs, Z_obs = obs_points[p]
                M[k, p] = turns * self._green_function(Rc, Zc, R_obs, Z_obs)

        return M

    def optimize_coil_currents(
        self,
        coils: CoilSet,
        target_flux: FloatArray,
        tikhonov_alpha: float = 1e-4,
    ) -> FloatArray:
        """Find coil currents that best reproduce *target_flux* at the control points.

        Solves the bounded linear least-squares problem:

            min_I || M^T I - target_flux ||^2 + alpha * ||I||^2
            s.t.  -I_max <= I <= I_max  (per coil)

        where ``M`` is the mutual-inductance matrix.

        Parameters
        ----------
        coils : CoilSet
            Must have ``target_flux_points`` set (shape ``(n_pts, 2)``).
        target_flux : FloatArray, shape (n_pts,)
            Desired poloidal flux at each control point.
        tikhonov_alpha : float
            Regularisation strength to penalise large currents.

        Returns
        -------
        FloatArray, shape (n_coils,)
            Optimised coil currents [A].
        """
        from scipy.optimize import lsq_linear

        if coils.target_flux_points is None:
            raise ValueError("CoilSet.target_flux_points must be set for optimisation.")

        obs = coils.target_flux_points
        M = self._build_mutual_inductance_matrix(coils, obs)  # (n_coils, n_pts)

        # Build augmented system: [M^T; sqrt(alpha)*I] I = [target; 0]
        n_coils = M.shape[0]
        A = np.vstack([M.T, np.sqrt(tikhonov_alpha) * np.eye(n_coils)])
        b = np.concatenate([target_flux, np.zeros(n_coils)])

        # Bounds
        if coils.current_limits is not None:
            lb = -np.abs(coils.current_limits)
            ub = np.abs(coils.current_limits)
        else:
            lb = -np.inf * np.ones(n_coils)
            ub = np.inf * np.ones(n_coils)

        result = lsq_linear(A, b, bounds=(lb, ub), method='trf')
        logger.info(
            "Coil optimisation: cost=%.4e, status=%d (%s)",
            result.cost, result.status, result.message,
        )
        return result.x.astype(np.float64)

    def solve_free_boundary(
        self,
        coils: CoilSet,
        max_outer_iter: int = 20,
        tol: float = 1e-4,
        optimize_shape: bool = False,
        tikhonov_alpha: float = 1e-4,
    ) -> dict[str, Any]:
        """Free-boundary GS solve with external coil currents.

        Iterates between updating boundary flux from coils and solving the
        internal GS equation.  When ``optimize_shape=True`` and the coil set
        has ``target_flux_points``, an additional outer loop optimises the
        coil currents to match the desired plasma boundary shape.

        Parameters
        ----------
        coils : CoilSet
            External coil set.
        max_outer_iter : int
            Maximum free-boundary iterations.
        tol : float
            Convergence tolerance on max |delta psi|.
        optimize_shape : bool
            When True, run coil-current optimisation at each outer step.
        tikhonov_alpha : float
            Tikhonov regularisation for coil optimisation.

        Returns
        -------
        dict
            ``{"outer_iterations": int, "final_diff": float,
            "coil_currents": NDArray}``
        """
        psi_ext = self._compute_external_flux(coils)

        for outer in range(max_outer_iter):
            # Apply external flux as boundary condition
            self.Psi[0, :] = psi_ext[0, :]
            self.Psi[-1, :] = psi_ext[-1, :]
            self.Psi[:, 0] = psi_ext[:, 0]
            self.Psi[:, -1] = psi_ext[:, -1]

            # Inner GS solve (use existing Picard iteration)
            psi_old = self.Psi.copy()
            self.solve_equilibrium(
                preserve_initial_state=True,
                boundary_flux=psi_ext,
            )

            # Optional: optimise coil currents to match target shape
            if optimize_shape and coils.target_flux_points is not None:
                obs = coils.target_flux_points
                # Extract current flux at target points via interpolation
                target_psi = np.array([
                    float(self._interp_psi(R_t, Z_t))
                    for R_t, Z_t in obs
                ])
                new_currents = self.optimize_coil_currents(
                    coils, target_psi, tikhonov_alpha=tikhonov_alpha,
                )
                coils.currents = new_currents
                psi_ext = self._compute_external_flux(coils)

            # Check convergence
            diff = float(np.max(np.abs(self.Psi - psi_old)))
            if diff < tol:
                logger.info("Free-boundary converged at outer iter %d (diff=%.2e)", outer, diff)
                break

        return {
            'outer_iterations': outer + 1,
            'final_diff': diff,
            'coil_currents': coils.currents.copy(),
        }

    def _interp_psi(self, R_pt: float, Z_pt: float) -> float:
        """Bilinear interpolation of Psi at an arbitrary (R, Z) point."""
        # Find enclosing cell
        ir = np.searchsorted(self.R, R_pt) - 1
        iz = np.searchsorted(self.Z, Z_pt) - 1
        ir = max(0, min(ir, self.NR - 2))
        iz = max(0, min(iz, self.NZ - 2))

        # Local coordinates
        t_r = (R_pt - self.R[ir]) / self.dR
        t_z = (Z_pt - self.Z[iz]) / self.dZ
        t_r = max(0.0, min(1.0, t_r))
        t_z = max(0.0, min(1.0, t_z))

        # Bilinear
        psi = (
            (1 - t_r) * (1 - t_z) * self.Psi[iz, ir]
            + t_r * (1 - t_z) * self.Psi[iz, ir + 1]
            + (1 - t_r) * t_z * self.Psi[iz + 1, ir]
            + t_r * t_z * self.Psi[iz + 1, ir + 1]
        )
        return float(psi)

    def save_results(self, filename: str = "equilibrium_nonlinear.npz") -> None:
        """Save the equilibrium state to a compressed NumPy archive.

        Parameters
        ----------
        filename : str
            Output file path.
        """
        np.savez(filename, R=self.R, Z=self.Z, Psi=self.Psi, J_phi=self.J_phi)
        logger.info("Saved: %s", filename)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    config_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "03_CODE/SCPN-Fusion-Core/iter_config.json"
    )
    fk = FusionKernel(config_file)
    fk.solve_equilibrium()
    fk.save_results("03_CODE/SCPN-Fusion-Core/final_state_nonlinear.npz")
