"""Iterative linear/nonlinear GS sub-solvers for FusionKernel."""

from __future__ import annotations

import numpy as np

from scpn_fusion.core.fusion_kernel_numerics import (
    NUMERIC_SANITIZE_CAP as _NUMERIC_SANITIZE_CAP,
    FloatArray,
    sanitize_numeric_array as _sanitize_numeric_array,
)


class FusionKernelIterativeSolverMixin:
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
        Psi_new = _sanitize_numeric_array(Psi)
        Source = _sanitize_numeric_array(Source)
        NZ, NR = Psi.shape
        dR2 = self.dR ** 2
        dZ2 = self.dZ ** 2

        # Toroidal stencil coefficients (arrays over interior grid)
        R_int = self.RR[1:-1, 1:-1]
        R_safe = np.maximum(R_int, 1e-10)
        a_E = 1.0 / dR2 - 1.0 / (2.0 * R_safe * self.dR)  # (NZ-2, NR-2)
        a_W = 1.0 / dR2 + 1.0 / (2.0 * R_safe * self.dR)  # (NZ-2, NR-2)
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
            interior[mask] = np.clip(interior[mask], -_NUMERIC_SANITIZE_CAP, _NUMERIC_SANITIZE_CAP)
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

        # Interior: vectorised 9-point stencil via even-index slicing
        coarse[1:-1, 1:-1] = (
            4.0 * fine[2:-2:2, 2:-2:2]
            + 2.0 * (fine[1:-3:2, 2:-2:2] + fine[3:-1:2, 2:-2:2]
                     + fine[2:-2:2, 1:-3:2] + fine[2:-2:2, 3:-1:2])
            + (fine[1:-3:2, 1:-3:2] + fine[1:-3:2, 3:-1:2]
               + fine[3:-1:2, 1:-3:2] + fine[3:-1:2, 3:-1:2])
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

        # Coincident points (even rows, even cols)
        nz_used = min(nz_c, (nz_f + 1) // 2)
        nr_used = min(nr_c, (nr_f + 1) // 2)
        fine[:2 * nz_used - 1:2, :2 * nr_used - 1:2] = coarse[:nz_used, :nr_used]

        # Horizontal midpoints (even rows, odd cols)
        h_end = min(2 * (nr_c - 1), nr_f - 1)
        fine[:2 * nz_used - 1:2, 1:h_end:2] = 0.5 * (
            coarse[:nz_used, :-1] + coarse[:nz_used, 1:]
        )[:, :(h_end - 1) // 2 + 1]

        # Vertical midpoints (odd rows, even cols)
        v_end = min(2 * (nz_c - 1), nz_f - 1)
        fine[1:v_end:2, :2 * nr_used - 1:2] = 0.5 * (
            coarse[:-1, :nr_used] + coarse[1:, :nr_used]
        )[:((v_end - 1) // 2 + 1), :]

        # Centre points (odd rows, odd cols)
        fine[1:v_end:2, 1:h_end:2] = 0.25 * (
            coarse[:-1, :-1] + coarse[1:, :-1]
            + coarse[:-1, 1:] + coarse[1:, 1:]
        )[:((v_end - 1) // 2 + 1), :(h_end - 1) // 2 + 1]

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
        a_E = 1.0 / dR2 - 1.0 / (2.0 * R_safe * dR)
        a_W = 1.0 / dR2 + 1.0 / (2.0 * R_safe * dR)
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
        """Run the inner elliptic solve (HPC or Python compatibility backend).

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
        I_target: float = self.cfg["physics"]["plasma_current_target"]
        if abs(I_target) < 1e-12:
            # No plasma current — keep Psi at the vacuum field set by
            # _prepare_initial_flux; Jacobi steps with the flat-Laplacian
            # stencil would corrupt the toroidal vacuum solution.
            self.J_phi = np.zeros_like(self.Psi)
            return

        R_center = (
            self.cfg["dimensions"]["R_min"]
            + self.cfg["dimensions"]["R_max"]
        ) / 2.0
        dist_sq = (self.RR - R_center) ** 2 + self.ZZ**2
        self.J_phi = np.exp(-dist_sq / 2.0)

        I_seed = float(np.sum(self.J_phi)) * self.dR * self.dZ
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

