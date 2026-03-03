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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.special import ellipe, ellipk

from scpn_fusion.core.config_schema import validate_config
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
            I = coil["current"]

            dZ = self.ZZ - Zc
            R_plus_Rc_sq = (self.RR + Rc) ** 2

            k2 = (4.0 * self.RR * Rc) / (R_plus_Rc_sq + dZ**2)
            k2 = np.clip(k2, 1e-12, 1.0 - 1e-12)

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
                    return (float(self.R[ir]), float(self.Z[iz])), float(Psi[iz, ir])

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
                    d2R = (Psi[iz, ir + 1] - 2.0 * Psi[iz, ir] + Psi[iz, ir - 1]) / (
                        self.dR**2
                    )
                    d2Z = (Psi[iz + 1, ir] - 2.0 * Psi[iz, ir] + Psi[iz - 1, ir]) / (
                        self.dZ**2
                    )
                    dRZ = (
                        Psi[iz + 1, ir + 1]
                        - Psi[iz + 1, ir - 1]
                        - Psi[iz - 1, ir + 1]
                        + Psi[iz - 1, ir - 1]
                    ) / (4.0 * self.dR * self.dZ)
                    det_hessian = float(d2R * d2Z - dRZ * dRZ)
                    if np.isfinite(det_hessian) and det_hessian < 0.0:
                        saddle_hits.append((float(masked_B[iz, ir]), int(iz), int(ir)))

                if saddle_hits:
                    _, iz_best, ir_best = min(saddle_hits, key=lambda x: x[0])
                    return (
                        (float(self.R[ir_best]), float(self.Z[iz_best])),
                        float(Psi[iz_best, ir_best]),
                    )

                # Fallback to minimum-gradient candidate when saddle test fails.
                idx_min = int(np.argmin(masked_B))
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
        mu0 = 4e-7 * np.pi
        denom = (R_obs + R_src)**2 + (Z_obs - Z_src)**2
        if denom < 1e-30:
            return 0.0
        k2 = 4.0 * R_obs * R_src / denom
        k2 = np.clip(k2, 1e-12, 1.0 - 1e-12)
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
        if len(coils.positions) == 0:
            raise ValueError("CoilSet.positions must contain at least one coil.")
        if len(coils.currents) != len(coils.positions):
            raise ValueError(
                "CoilSet.currents length must match number of coil positions."
            )
        if not np.isfinite(tikhonov_alpha) or tikhonov_alpha < 0.0:
            raise ValueError("tikhonov_alpha must be finite and non-negative.")

        obs = np.asarray(coils.target_flux_points, dtype=np.float64)
        if obs.ndim != 2 or obs.shape[1] != 2 or obs.shape[0] == 0:
            raise ValueError("target_flux_points must have shape (n_points, 2) with n_points > 0.")
        if not np.all(np.isfinite(obs)):
            raise ValueError("target_flux_points must contain finite values only.")

        target = np.asarray(target_flux, dtype=np.float64).reshape(-1)
        if target.shape[0] != obs.shape[0]:
            raise ValueError(
                "target_flux must have the same length as target_flux_points."
            )
        if not np.all(np.isfinite(target)):
            raise ValueError("target_flux must contain finite values only.")

        M = self._build_mutual_inductance_matrix(coils, obs)  # (n_coils, n_pts)
        if not np.all(np.isfinite(M)):
            raise ValueError("Mutual inductance matrix contains non-finite entries.")

        # Build augmented system: [M^T; sqrt(alpha)*I] I = [target; 0]
        n_coils = M.shape[0]
        A = np.vstack([M.T, np.sqrt(tikhonov_alpha) * np.eye(n_coils)])
        b = np.concatenate([target, np.zeros(n_coils)])

        # Bounds
        if coils.current_limits is not None:
            limits = np.asarray(coils.current_limits, dtype=np.float64).reshape(-1)
            if limits.shape[0] != n_coils:
                raise ValueError(
                    "current_limits must have one entry per coil."
                )
            if not np.all(np.isfinite(limits)):
                raise ValueError("current_limits must contain finite values only.")
            lb = -np.abs(limits)
            ub = np.abs(limits)
        else:
            lb = -np.inf * np.ones(n_coils)
            ub = np.inf * np.ones(n_coils)

        result = lsq_linear(A, b, bounds=(lb, ub), method='trf')
        if not bool(getattr(result, "success", False)) or not np.all(np.isfinite(result.x)):
            logger.warning(
                "Coil optimisation failed (status=%s): %s. Falling back to prior currents.",
                getattr(result, "status", "unknown"),
                getattr(result, "message", "no message"),
            )
            prior_currents = np.asarray(coils.currents, dtype=np.float64).copy()
            return np.clip(prior_currents, lb, ub).astype(np.float64)

        logger.info(
            "Coil optimisation: cost=%.4e, status=%d (%s)",
            result.cost, result.status, result.message,
        )
        return np.asarray(result.x, dtype=np.float64)

    def _resolve_shape_target_flux(self, coils: CoilSet) -> FloatArray:
        """Resolve target flux vector for shape optimisation control points.

        Priority:
        1) ``coils.target_flux_values`` if provided and valid.
        2) Inferred isoflux target from current interpolation over
           ``coils.target_flux_points``.
        """
        if coils.target_flux_points is None:
            raise ValueError("CoilSet.target_flux_points must be set for shape optimisation.")

        obs = np.asarray(coils.target_flux_points, dtype=np.float64)
        n_pts = int(obs.shape[0])
        if obs.ndim != 2 or obs.shape[1] != 2 or n_pts == 0:
            raise ValueError("target_flux_points must have shape (n_points, 2) with n_points > 0.")

        if coils.target_flux_values is not None:
            target = np.asarray(coils.target_flux_values, dtype=np.float64).reshape(-1)
            if target.shape[0] != n_pts:
                raise ValueError(
                    "target_flux_values must have the same length as target_flux_points."
                )
            if not np.all(np.isfinite(target)):
                raise ValueError("target_flux_values must contain finite values only.")
            return target

        psi_samples = np.array(
            [float(self._interp_psi(R_t, Z_t)) for R_t, Z_t in obs], dtype=np.float64
        )
        iso_level = float(np.mean(psi_samples))
        return np.full(n_pts, iso_level, dtype=np.float64)

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
        if max_outer_iter < 1:
            raise ValueError("max_outer_iter must be >= 1.")
        if not np.isfinite(tol) or tol < 0.0:
            raise ValueError("tol must be finite and >= 0.")

        psi_ext = self._compute_external_flux(coils)
        diff = float("inf")

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
                target_psi = self._resolve_shape_target_flux(coils)
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
