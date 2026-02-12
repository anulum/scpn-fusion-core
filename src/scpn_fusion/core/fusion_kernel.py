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

        # Python fallback: single Jacobi sweep
        Psi_new = self._jacobi_step(self.Psi, Source)
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

    # ── main solver ───────────────────────────────────────────────────

    def solve_equilibrium(self) -> None:
        """Run the full Picard-iteration equilibrium solver.

        Iterates: topology analysis -> source update -> elliptic solve ->
        under-relaxation until the residual drops below the configured
        convergence threshold or the maximum iteration count is reached.

        Raises
        ------
        RuntimeError
            If the solver produces NaN or Inf (logged as a warning and
            reverts to the best-known state rather than raising).
        """
        t0 = time.time()
        self.Psi = self.calculate_vacuum_field()
        Psi_vac_boundary = self.Psi.copy()

        max_iter: int = self.cfg["solver"]["max_iterations"]
        tol: float = self.cfg["solver"]["convergence_threshold"]
        alpha: float = self.cfg["solver"].get("relaxation_factor", 0.1)
        mu0: float = self.cfg["physics"]["vacuum_permeability"]

        x_point_pos: tuple[float, float] = (0.0, 0.0)
        Psi_best = self.Psi.copy()
        diff_best: float = 1e9

        self._seed_plasma(mu0)

        for k in range(max_iter):
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
            Psi_new = self._elliptic_solve(Source, Psi_vac_boundary)

            # Divergence check
            if np.isnan(Psi_new).any() or np.isinf(Psi_new).any():
                logger.warning(
                    "Solver diverged at iter %d — reverting to best state.",
                    k,
                )
                self.Psi = Psi_best
                break

            # 4. Under-relaxation
            diff = float(np.mean(np.abs(Psi_new - self.Psi)))
            self.Psi = (1.0 - alpha) * self.Psi + alpha * Psi_new

            if diff < diff_best:
                diff_best = diff
                Psi_best = self.Psi.copy()

            if diff < tol:
                logger.info("Converged at iter %d.  Residual: %.6e", k, diff)
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

        self.compute_b_field()
        elapsed = time.time() - t0
        logger.info(
            "Solved in %.2fs.  X-point: R=%.2f, Z=%.2f",
            elapsed,
            x_point_pos[0],
            x_point_pos[1],
        )

    # ── post-processing ───────────────────────────────────────────────

    def compute_b_field(self) -> None:
        """Derive the magnetic field components from the solved Psi."""
        dPsi_dR, dPsi_dZ = np.gradient(self.Psi, self.dR, self.dZ)
        R_safe = np.maximum(self.RR, 1e-6)
        self.B_R: FloatArray = -(1.0 / R_safe) * dPsi_dZ
        self.B_Z: FloatArray = (1.0 / R_safe) * dPsi_dR

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
