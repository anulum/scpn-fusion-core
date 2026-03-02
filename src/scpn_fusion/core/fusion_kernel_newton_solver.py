"""Newton and top-level equilibrium dispatch routines for FusionKernel."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from scpn_fusion.core.fusion_kernel_numerics import (
    FloatArray,
    stable_rms as _stable_rms,
)

logger = logging.getLogger(__name__)


class FusionKernelNewtonSolverMixin:
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
        return _stable_rms(interior)

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

        # Zero-current short-circuit: the vacuum field is the exact
        # equilibrium; iterating the discrete stencil would only drift
        # toward the discretised vacuum (different from analytical).
        I_target: float = self.cfg["physics"]["plasma_current_target"]
        if abs(I_target) < 1e-12 and not preserve_initial_state:
            self.Psi = self.calculate_vacuum_field()
            self.J_phi = np.zeros_like(self.Psi)
            self.compute_b_field()
            elapsed = time.time() - t0
            return {
                "psi": self.Psi,
                "converged": True,
                "iterations": 0,
                "residual": 0.0,
                "residual_history": [],
                "gs_residual": 0.0,
                "gs_residual_best": 0.0,
                "gs_residual_history": [],
                "wall_time_s": elapsed,
                "solver_method": method,
            }

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
                    "Solver diverged — reverting to best state.",
                    extra={"physics_context": {
                        "iteration": k,
                        "psi_axis": float(Psi_axis),
                        "psi_boundary": float(Psi_boundary)
                    }}
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

