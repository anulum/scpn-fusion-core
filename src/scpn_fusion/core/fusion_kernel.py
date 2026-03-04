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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import ellipe, ellipk

from scpn_fusion.core.config_schema import validate_config
from scpn_fusion.core.fusion_kernel_free_boundary import (
    build_mutual_inductance_matrix as _build_mutual_inductance_matrix_runtime,
    compute_external_flux as _compute_external_flux_runtime,
    green_function as _green_function_runtime,
    interp_psi as _interp_psi_runtime,
    optimize_coil_currents as _optimize_coil_currents_runtime,
    resolve_shape_target_flux as _resolve_shape_target_flux_runtime,
    solve_free_boundary as _solve_free_boundary_runtime,
)
from scpn_fusion.core.fusion_kernel_iterative_solver import FusionKernelIterativeSolverMixin
from scpn_fusion.core.fusion_kernel_newton_solver import FusionKernelNewtonSolverMixin
from scpn_fusion.core.fusion_kernel_numerics import (
    NUMERIC_SANITIZE_CAP as _NUMERIC_SANITIZE_CAP,
    FloatArray,
    sanitize_numeric_array as _sanitize_numeric_array_impl,
    stable_rms as _stable_rms_impl,
)
from scpn_fusion.hpc.hpc_bridge import HPCBridge

logger = logging.getLogger(__name__)


def _sanitize_numeric_array(arr: FloatArray, *, cap: float = _NUMERIC_SANITIZE_CAP) -> FloatArray:
    """Return finite array with values clipped to ``[-cap, cap]``."""
    return _sanitize_numeric_array_impl(arr, cap=cap)


def _stable_rms(arr: FloatArray) -> float:
    """Compute RMS without overflow from squaring very large magnitudes."""
    return _stable_rms_impl(arr)


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
    target_flux_values : NDArray or None
        Optional target flux values [Wb] at ``target_flux_points``.
        Shape ``(n_pts,)``. When omitted, ``solve_free_boundary`` uses an
        isoflux target inferred from current interpolation.
    """
    positions: list[tuple[float, float]] = field(default_factory=list)
    currents: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    turns: list[int] = field(default_factory=list)
    current_limits: NDArray[np.float64] | None = None
    target_flux_points: NDArray[np.float64] | None = None
    target_flux_values: NDArray[np.float64] | None = None


class FusionKernel(FusionKernelNewtonSolverMixin, FusionKernelIterativeSolverMixin):
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
            raw_cfg = json.load(f)
        
        # Hardening: Strict schema validation at the entry point
        validated_cfg = validate_config(raw_cfg)
        self.cfg = validated_cfg.model_dump()
        
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
            "HPC Acceleration UNAVAILABLE (using Python compatibility backend)."
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
            current = coil["current"]

            dZ = self.ZZ - Zc
            R_plus_Rc_sq = (self.RR + Rc) ** 2

            k2 = (4.0 * self.RR * Rc) / (R_plus_Rc_sq + dZ**2)
            k2 = np.clip(k2, 1e-12, 1.0 - 1e-12)

            K = ellipk(k2)
            E = ellipe(k2)

            prefactor = (mu0 * current) / (2 * np.pi)
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
        psi_raw = np.asarray(Psi, dtype=np.float64)
        finite_mask = np.isfinite(psi_raw)
        if not np.any(finite_mask):
            return (0.0, 0.0), 0.0
        psi_safe = np.nan_to_num(psi_raw, nan=0.0, posinf=1e300, neginf=-1e300)

        dPsi_dR, dPsi_dZ = np.gradient(psi_safe, self.dR, self.dZ)
        # ``hypot`` avoids overflow/underflow in extreme gradient excursions.
        B_mag = np.hypot(dPsi_dR, dPsi_dZ)

        mask_divertor = self.ZZ < (self.cfg["dimensions"]["Z_min"] * 0.5)
        if np.any(mask_divertor):
            masked_B = np.where(mask_divertor, B_mag, np.inf)
            finite_count = int(np.isfinite(masked_B).sum())
            if finite_count > 0:
                use_saddle_detection = bool(
                    self.cfg.get("solver", {}).get("xpoint_use_saddle_detection", False)
                )
                if not use_saddle_detection:
                    idx_min = int(np.argmin(masked_B))
                    iz, ir = np.unravel_index(idx_min, Psi.shape)
                    psi_x = float(psi_raw[iz, ir])
                    if not np.isfinite(psi_x):
                        psi_x = float(psi_safe[iz, ir])
                    return (float(self.R[ir]), float(self.Z[iz])), psi_x

                # Prefer true magnetic saddles over pure |grad(Psi)| minima.
                n_candidates = min(16, finite_count)
                flat = masked_B.ravel()
                candidate_idx = np.argpartition(flat, n_candidates - 1)[:n_candidates]
                saddle_hits: list[tuple[float, int, int]] = []
                nz, nr = Psi.shape
                for idx in candidate_idx:
                    iz, ir = np.unravel_index(int(idx), Psi.shape)
                    if iz <= 0 or iz >= nz - 1 or ir <= 0 or ir >= nr - 1:
                        continue
                    d2R = (psi_safe[iz, ir + 1] - 2.0 * psi_safe[iz, ir] + psi_safe[iz, ir - 1]) / (
                        self.dR**2
                    )
                    d2Z = (psi_safe[iz + 1, ir] - 2.0 * psi_safe[iz, ir] + psi_safe[iz - 1, ir]) / (
                        self.dZ**2
                    )
                    dRZ = (
                        psi_safe[iz + 1, ir + 1]
                        - psi_safe[iz + 1, ir - 1]
                        - psi_safe[iz - 1, ir + 1]
                        + psi_safe[iz - 1, ir - 1]
                    ) / (4.0 * self.dR * self.dZ)
                    det_hessian = float(d2R * d2Z - dRZ * dRZ)
                    if np.isfinite(det_hessian) and det_hessian < 0.0:
                        saddle_hits.append((float(masked_B[iz, ir]), int(iz), int(ir)))

                if saddle_hits:
                    _, iz_best, ir_best = min(saddle_hits, key=lambda x: x[0])
                    psi_x = float(psi_raw[iz_best, ir_best])
                    if not np.isfinite(psi_x):
                        psi_x = float(psi_safe[iz_best, ir_best])
                    return (
                        (float(self.R[ir_best]), float(self.Z[iz_best])),
                        psi_x,
                    )

                # Fallback to minimum-gradient candidate when saddle test fails.
                idx_min = int(np.argmin(masked_B))
                iz, ir = np.unravel_index(idx_min, Psi.shape)
                psi_x = float(psi_raw[iz, ir])
                if not np.isfinite(psi_x):
                    psi_x = float(psi_safe[iz, ir])
                return (float(self.R[ir]), float(self.Z[iz])), psi_x

        psi_min = float(np.min(psi_raw[finite_mask]))
        return (0.0, 0.0), psi_min

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

    # Iterative and Newton solver internals moved to dedicated mixins.

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
        return _green_function_runtime(R_src, Z_src, R_obs, Z_obs)

    def _compute_external_flux(self, coils):
        """Sum Green's function contributions on boundary from CoilSet."""
        return _compute_external_flux_runtime(self, coils)

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
        return _build_mutual_inductance_matrix_runtime(self, coils, obs_points)

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
        return _optimize_coil_currents_runtime(
            self,
            coils,
            target_flux,
            tikhonov_alpha=tikhonov_alpha,
        )

    def _resolve_shape_target_flux(self, coils: CoilSet) -> FloatArray:
        """Resolve target flux vector for shape optimisation control points.

        Priority:
        1) ``coils.target_flux_values`` if provided and valid.
        2) Inferred isoflux target from current interpolation over
           ``coils.target_flux_points``.
        """
        return _resolve_shape_target_flux_runtime(self, coils)

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
        return _solve_free_boundary_runtime(
            self,
            coils,
            max_outer_iter=max_outer_iter,
            tol=tol,
            optimize_shape=optimize_shape,
            tikhonov_alpha=tikhonov_alpha,
        )

    def _interp_psi(self, R_pt: float, Z_pt: float) -> float:
        """Bilinear interpolation of Psi at an arbitrary (R, Z) point."""
        return _interp_psi_runtime(self, R_pt, Z_pt)

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
