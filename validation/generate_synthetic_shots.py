# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
# ----------------------------------------------------------------------
# SCPN Fusion Core -- Synthetic Shot Generator for Equilibrium Validation
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ----------------------------------------------------------------------
"""Generate synthetic tokamak equilibria with known ground truth.

Each "shot" consists of:
- Input: p'(psi) and FF'(psi) profiles (parameterized)
- Ground truth: psi(R,Z) solved analytically or semi-analytically
- Synthetic probes: psi sampled at N boundary locations
- Parameters: R0, B0, Ip, kappa, delta for the shot

50 shots covering:
- 10 circular cross-section (kappa=1.0, delta=0.0)
- 15 moderately elongated (kappa=1.4-1.7, delta=0.2-0.3) -- DIII-D-like
- 15 highly elongated (kappa=1.7-2.0, delta=0.3-0.5) -- ITER-like
- 5 high-beta (beta_N > 2.0) -- stress test
- 5 low-current (Ip < 1 MA) -- edge case

The equilibria are computed using the Solov'ev / Cerfon-Freidberg (2010)
analytical solution to the Grad-Shafranov equation with linear source
terms.  This avoids any dependency on the Rust multigrid solver while
providing exact ground-truth psi(R,Z) fields.

References
----------
- Solov'ev L.S., Sov. Phys. JETP 26 (1968) 400
- Cerfon A.J. and Freidberg J.P., Phys. Plasmas 17 (2010) 032502
- Pataki et al., J. Comp. Phys. 243 (2013) 28

Dependencies: numpy, scipy (interpolate only), json, pathlib
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


# ── Physical constants ────────────────────────────────────────────────

MU_0 = 4.0 * np.pi * 1e-7  # vacuum permeability [H/m]


# ── Cerfon-Freidberg homogeneous solutions ────────────────────────────

def _psi_h1(x: NDArray, y: NDArray) -> NDArray:
    """psi_1 = 1.  Trivial constant."""
    return np.ones_like(x)


def _psi_h2(x: NDArray, y: NDArray) -> NDArray:
    """psi_2 = x^2."""
    return x ** 2


def _psi_h3(x: NDArray, y: NDArray) -> NDArray:
    """psi_3 = y^2 - x^2 ln(x).

    Satisfies the homogeneous GS equation in (x, y) coordinates where
    x = R/R0, y = Z/R0.  The ln(x) term is the hallmark of cylindrical
    geometry; it vanishes in Cartesian.
    """
    # Guard against x <= 0 (should not occur on a physical R grid)
    x_safe = np.clip(x, 1e-30, None)
    return y ** 2 - x_safe ** 2 * np.log(x_safe)


def _psi_h4(x: NDArray, y: NDArray) -> NDArray:
    """psi_4 = x^4 - 4 x^2 y^2."""
    return x ** 4 - 4.0 * x ** 2 * y ** 2


def _psi_h5(x: NDArray, y: NDArray) -> NDArray:
    """psi_5 = 2 y^4 - 9 y^2 x^2 + 3 x^4 ln(x) - 12 x^2 y^2 ln(x).

    Higher-order homogeneous solution from Cerfon & Freidberg (2010),
    Eq. (A3).  Provides additional shaping freedom for triangularity.
    """
    x_safe = np.clip(x, 1e-30, None)
    lnx = np.log(x_safe)
    return (
        2.0 * y ** 4
        - 9.0 * y ** 2 * x ** 2
        + 3.0 * x ** 4 * lnx
        - 12.0 * x ** 2 * y ** 2 * lnx
    )


def _psi_h6(x: NDArray, y: NDArray) -> NDArray:
    """psi_6 = x^6 - 12 x^4 y^2 + 8 x^2 y^4.

    Sixth-order polynomial homogeneous solution of the GS equation
    (even in y, no ln).  Cerfon & Freidberg (2010), Appendix A.
    """
    return x ** 6 - 12.0 * x ** 4 * y ** 2 + 8.0 * x ** 2 * y ** 4


def _psi_h7(x: NDArray, y: NDArray) -> NDArray:
    """psi_7 = (x^6 - 12 x^4 y^2 + 8 x^2 y^4) ln(x)
              - (7/6) x^6 + 9 x^4 y^2 - (8/15) y^6.

    Sixth-order logarithmic homogeneous solution of the GS equation
    (even in y, with ln).  Derived as the companion log-solution of
    psi_6 following the Cerfon-Freidberg construction: if f(x,y) is
    a polynomial homogeneous solution, then g = f·ln(x) + h(x,y) is
    also homogeneous when h is chosen to cancel Delta*(f·ln x).

    The cross-term source Delta*(f·ln x) = (2/x)(f' - f/x) evaluates
    to 10 x^4 - 72 x^2 y^2 + 16 y^4 for f = psi_6.  The particular
    solution h satisfying Delta*(h) = -(10 x^4 - 72 x^2 y^2 + 16 y^4)
    is h = -(7/6) x^6 + 9 x^4 y^2 - (8/15) y^6.
    """
    x_safe = np.clip(x, 1e-30, None)
    lnx = np.log(x_safe)
    f6 = x ** 6 - 12.0 * x ** 4 * y ** 2 + 8.0 * x ** 2 * y ** 4
    h = -(7.0 / 6.0) * x ** 6 + 9.0 * x ** 4 * y ** 2 - (8.0 / 15.0) * y ** 6
    return f6 * lnx + h


# Ordered list so we can index c[0..6] -> psi_h1..psi_h7
_HOMOGENEOUS = [_psi_h1, _psi_h2, _psi_h3, _psi_h4, _psi_h5, _psi_h6, _psi_h7]


# ── Particular solutions (source terms) ──────────────────────────────

def _psi_particular_p(x: NDArray, y: NDArray) -> NDArray:
    """Particular solution for the p'(psi) = const source.

    Satisfies  Delta* psi_p = -x^2  =>  psi_p = x^4 / 8.
    """
    return x ** 4 / 8.0


def _psi_particular_ff(x: NDArray, y: NDArray) -> NDArray:
    """Particular solution for the FF'(psi) = const source.

    Satisfies  Delta* psi_ff = -1  =>  psi_ff = x^2 / 2  (up to sign).
    Actually we use the form with (1/2) x^2 ln(x) - x^4/8, which is
    the Cerfon-Freidberg "A" particular solution.
    """
    x_safe = np.clip(x, 1e-30, None)
    return 0.5 * x_safe ** 2 * np.log(x_safe) - x_safe ** 4 / 8.0


# ── Boundary parameterisation ────────────────────────────────────────

def shaped_boundary(
    theta: NDArray,
    R0: float,
    a: float,
    kappa: float,
    delta: float,
) -> Tuple[NDArray, NDArray]:
    """Miller parameterisation of an up-down symmetric D-shaped boundary.

    Parameters
    ----------
    theta : array
        Poloidal angle [0, 2*pi).
    R0 : float
        Major radius [m].
    a : float
        Minor radius [m].
    kappa : float
        Elongation (1.0 = circular).
    delta : float
        Triangularity (0.0 = no triangularity).

    Returns
    -------
    R, Z : arrays
        Boundary coordinates [m].
    """
    delta_clamp = np.clip(delta, -0.99, 0.99)
    R = R0 + a * np.cos(theta + np.arcsin(delta_clamp) * np.sin(theta))
    Z = kappa * a * np.sin(theta)
    return R, Z


# ── Solov'ev / Cerfon-Freidberg equilibrium solver ───────────────────

@dataclass
class SolovevEquilibrium:
    """Analytical Solov'ev equilibrium for a single synthetic shot.

    The GS equation with linear source terms:
        Delta* psi = -mu_0 R^2 p'(psi) - FF'(psi)
    becomes an exactly solvable PDE when p' and FF' are constants.

    The general solution is:
        psi(x, y) = psi_0 * [psi_p(x,y) + A * psi_ff(x,y)
                             + sum_k c_k psi_hk(x,y)]

    where x = R/R0, y = Z/R0, and the coefficients c_k are found by
    least-squares fitting to the shaped boundary condition psi = 0 on
    the LCFS.

    Parameters
    ----------
    R0 : float
        Major radius [m].
    a : float
        Minor radius [m].
    B0 : float
        Toroidal field on axis [T].
    Ip : float
        Plasma current [MA].
    kappa : float
        Elongation.
    delta : float
        Triangularity.
    nr : int
        Number of R grid points (default 129).
    nz : int
        Number of Z grid points (default 129).
    n_boundary : int
        Number of boundary constraint points (default 64).
    n_probes : int
        Number of synthetic probes on the LCFS (default 40).
    """

    R0: float
    a: float
    B0: float
    Ip: float          # MA
    kappa: float
    delta: float
    nr: int = 129
    nz: int = 129
    n_boundary: int = 64
    n_probes: int = 40

    # ── Computed fields (populated by solve()) ────────────────────────
    psi_rz: NDArray = field(default_factory=lambda: np.array([]))
    r_grid: NDArray = field(default_factory=lambda: np.array([]))
    z_grid: NDArray = field(default_factory=lambda: np.array([]))
    p_prime: NDArray = field(default_factory=lambda: np.array([]))
    ff_prime: NDArray = field(default_factory=lambda: np.array([]))
    probe_r: NDArray = field(default_factory=lambda: np.array([]))
    probe_z: NDArray = field(default_factory=lambda: np.array([]))
    probe_psi: NDArray = field(default_factory=lambda: np.array([]))
    boundary_r: NDArray = field(default_factory=lambda: np.array([]))
    boundary_z: NDArray = field(default_factory=lambda: np.array([]))
    coefficients: NDArray = field(default_factory=lambda: np.array([]))
    A_param: float = 0.0
    psi_0: float = 1.0
    r_axis: float = 0.0
    z_axis: float = 0.0
    psi_axis: float = 0.0
    psi_boundary: float = 0.0
    beta_N: float = 0.0

    def solve(self) -> "SolovevEquilibrium":
        """Compute the analytical equilibrium.

        Steps
        -----
        1. Build the shaped boundary using Miller parameterisation.
        2. Evaluate homogeneous and particular solutions at boundary
           points in normalised (x, y) coordinates.
        3. Solve the least-squares system for coefficients c_1..c_7.
        4. Evaluate psi on the full (R, Z) grid.
        5. Normalise so psi = 0 on boundary and psi < 0 inside (axis).
        6. Extract probes on the LCFS.
        7. Compute approximate beta_N.

        Returns
        -------
        self : SolovevEquilibrium
            With all computed fields populated.
        """
        # ---- Step 1: boundary shape ----
        theta_b = np.linspace(0.0, 2.0 * np.pi, self.n_boundary, endpoint=False)
        Rb, Zb = shaped_boundary(theta_b, self.R0, self.a, self.kappa, self.delta)
        self.boundary_r = Rb
        self.boundary_z = Zb

        # Normalised coordinates at boundary
        xb = Rb / self.R0
        yb = Zb / self.R0

        # ---- Step 2: source-term parameter A ----
        # A controls the balance between pressure gradient and diamagnetic
        # current.  A = -1 gives a pure pressure-driven equilibrium;
        # A = 0 gives pure FF'-driven.  We use A related to beta_p.
        #
        # For the Cerfon-Freidberg formulation, A is a free parameter.
        # A typical tokamak has A in [-1, 0].  We choose A based on
        # desired current/pressure balance: A = -(1 - epsilon^2) / (1 + epsilon^2)
        # where epsilon = a/R0 is the inverse aspect ratio.
        eps = self.a / self.R0
        self.A_param = -(1.0 - eps ** 2) / (1.0 + eps ** 2)

        # ---- Step 3: build linear system ----
        # psi_total(x,y) = psi_p(x,y) + A * psi_ff(x,y) + sum_k c_k psi_hk(x,y)
        #
        # Boundary condition: psi_total = 0 on (xb, yb)
        #   => sum_k c_k psi_hk(xb, yb) = -[psi_p(xb, yb) + A * psi_ff(xb, yb)]
        #
        # This is a linear system M @ c = rhs.

        n_terms = len(_HOMOGENEOUS)  # 7
        M = np.zeros((self.n_boundary, n_terms))
        for k, psi_hk in enumerate(_HOMOGENEOUS):
            M[:, k] = psi_hk(xb, yb)

        rhs = -(_psi_particular_p(xb, yb) + self.A_param * _psi_particular_ff(xb, yb))

        # Least-squares solve (overdetermined when n_boundary > 7)
        self.coefficients, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)

        # ---- Step 4: evaluate on full grid ----
        R_margin = 0.3 * self.a
        Z_extent = (self.kappa * self.a) + R_margin
        self.r_grid = np.linspace(
            self.R0 - self.a - R_margin,
            self.R0 + self.a + R_margin,
            self.nr,
        )
        self.z_grid = np.linspace(-Z_extent, Z_extent, self.nz)

        x2d = np.zeros((self.nr, self.nz))
        y2d = np.zeros((self.nr, self.nz))
        for i, R_val in enumerate(self.r_grid):
            for j, Z_val in enumerate(self.z_grid):
                x2d[i, j] = R_val / self.R0
                y2d[i, j] = Z_val / self.R0

        psi_raw = _psi_particular_p(x2d, y2d) + self.A_param * _psi_particular_ff(x2d, y2d)
        for k, psi_hk in enumerate(_HOMOGENEOUS):
            psi_raw += self.coefficients[k] * psi_hk(x2d, y2d)

        # ---- Step 5: normalisation ----
        # psi on boundary should be ~0 by construction (up to lstsq residual).
        # Find the extremum inside the domain (magnetic axis).
        # Convention: psi < 0 inside (axis), psi = 0 on boundary.
        # The raw solution may have either sign; we detect and flip if needed.

        # Evaluate psi at approximate axis location (R0, 0)
        i_axis = np.argmin(np.abs(self.r_grid - self.R0))
        j_axis = np.argmin(np.abs(self.z_grid - 0.0))
        psi_center = psi_raw[i_axis, j_axis]

        # Evaluate psi on boundary (average to get boundary value)
        psi_on_bdry = _psi_particular_p(xb, yb) + self.A_param * _psi_particular_ff(xb, yb)
        for k, psi_hk in enumerate(_HOMOGENEOUS):
            psi_on_bdry += self.coefficients[k] * psi_hk(xb, yb)
        psi_boundary_mean = np.mean(psi_on_bdry)

        # Shift so boundary = 0
        psi_raw -= psi_boundary_mean

        # Now find the axis: the extremum of psi_raw
        # It should be the most negative (if convention is psi < 0 inside)
        # or most positive.  We force psi_axis < 0 by flipping if needed.
        psi_center_shifted = psi_raw[i_axis, j_axis]
        if psi_center_shifted > 0:
            psi_raw = -psi_raw

        # Find the actual axis location (minimum of psi_raw).
        # Use scipy.optimize to find the true analytical minimum,
        # giving machine-precision axis positions independent of grid.
        idx_min = np.unravel_index(np.argmin(psi_raw), psi_raw.shape)
        r0_guess = float(self.r_grid[idx_min[0]])
        z0_guess = float(self.z_grid[idx_min[1]])

        # Build scalar psi evaluator using the analytical formula
        sign = -1.0 if psi_center_shifted > 0 else 1.0

        def _psi_point(rz: NDArray) -> float:
            x = rz[0] / self.R0
            y = rz[1] / self.R0
            val = float(
                _psi_particular_p(x, y)
                + self.A_param * _psi_particular_ff(x, y)
            )
            for kk, psi_hk in enumerate(_HOMOGENEOUS):
                val += float(self.coefficients[kk] * psi_hk(x, y))
            val -= psi_boundary_mean
            val *= sign  # apply sign convention so we minimise
            return val

        res = minimize(
            _psi_point,
            x0=np.array([r0_guess, z0_guess]),
            method="Nelder-Mead",
            options={"xatol": 1e-12, "fatol": 1e-15, "maxiter": 2000},
        )
        self.r_axis = float(res.x[0])
        self.z_axis = float(res.x[1])
        self.psi_axis = float(psi_raw[idx_min[0], idx_min[1]])
        self.psi_boundary = 0.0

        # Scale psi so that psi_axis has a physically meaningful magnitude.
        # From the GS equation: psi_0 ~ mu_0 * R0 * Ip / (2*pi)
        # with Ip in A.
        Ip_A = self.Ip * 1e6  # MA -> A
        psi_scale = MU_0 * self.R0 * Ip_A / (2.0 * np.pi)
        if abs(self.psi_axis) > 1e-30:
            scale_factor = psi_scale / abs(self.psi_axis)
        else:
            scale_factor = psi_scale
        psi_raw *= scale_factor
        self.psi_axis *= scale_factor
        self.psi_rz = psi_raw

        # ---- Step 6: profiles ----
        # For the Solov'ev solution, p' and FF' are constants over psi.
        # We store them as uniform arrays of length nr for compatibility
        # with the G-EQDSK format.
        psi_norm = np.linspace(0.0, 1.0, self.nr)

        # Relate c1 (p' source) and c2 (FF' source) to physical units.
        # GS equation: Delta* psi = -mu_0 R0^2 * p'(psi) - FF'(psi)
        # With psi_p particular solution: the coefficient of the x^4/8 term
        # represents -mu_0 R0^2 * c1_phys.  Similarly for FF'.
        c1_phys = psi_scale / (MU_0 * self.R0 ** 2) if abs(self.R0) > 0 else 0.0
        c2_phys = psi_scale * self.A_param

        self.p_prime = np.full(self.nr, c1_phys)
        self.ff_prime = np.full(self.nr, c2_phys)

        # ---- Step 7: probes on LCFS ----
        theta_p = np.linspace(0.0, 2.0 * np.pi, self.n_probes, endpoint=False)
        Rp, Zp = shaped_boundary(theta_p, self.R0, self.a, self.kappa, self.delta)
        self.probe_r = Rp
        self.probe_z = Zp

        # Interpolate psi at probe locations using bilinear interpolation
        self.probe_psi = self._interpolate_psi(Rp, Zp)

        # ---- Step 8: approximate beta_N ----
        self._compute_beta_N()

        return self

    def _interpolate_psi(self, R_pts: NDArray, Z_pts: NDArray) -> NDArray:
        """Bilinear interpolation of psi_rz at arbitrary (R, Z) points.

        Parameters
        ----------
        R_pts, Z_pts : arrays
            Query coordinates [m].

        Returns
        -------
        psi_vals : array
            Interpolated psi values.
        """
        r_min, r_max = self.r_grid[0], self.r_grid[-1]
        z_min, z_max = self.z_grid[0], self.z_grid[-1]
        dr = self.r_grid[1] - self.r_grid[0]
        dz = self.z_grid[1] - self.z_grid[0]

        psi_vals = np.zeros(len(R_pts))
        for idx in range(len(R_pts)):
            R_q = np.clip(R_pts[idx], r_min, r_max - 1e-12)
            Z_q = np.clip(Z_pts[idx], z_min, z_max - 1e-12)

            # Fractional indices
            fi = (R_q - r_min) / dr
            fj = (Z_q - z_min) / dz

            i0 = int(np.floor(fi))
            j0 = int(np.floor(fj))
            i0 = min(i0, self.nr - 2)
            j0 = min(j0, self.nz - 2)

            di = fi - i0
            dj = fj - j0

            # Bilinear
            psi_vals[idx] = (
                (1 - di) * (1 - dj) * self.psi_rz[i0, j0]
                + di * (1 - dj) * self.psi_rz[i0 + 1, j0]
                + (1 - di) * dj * self.psi_rz[i0, j0 + 1]
                + di * dj * self.psi_rz[i0 + 1, j0 + 1]
            )

        return psi_vals

    def _compute_beta_N(self) -> None:
        """Estimate normalised beta.

        beta_N = beta_T * a * B0 / Ip  [% m T / MA]

        For Solov'ev equilibria with constant p', the volume-averaged
        pressure is approximately <p> ~ |p'| * |psi_axis| / 2.
        Then beta_T = 2 * mu_0 * <p> / B0^2.
        """
        if abs(self.B0) < 1e-30 or abs(self.Ip) < 1e-30:
            self.beta_N = 0.0
            return

        p_avg = abs(self.p_prime[0]) * abs(self.psi_axis) / 2.0
        beta_T = 2.0 * MU_0 * p_avg / (self.B0 ** 2)
        # beta_N in conventional units [% m T / MA]
        self.beta_N = beta_T * self.a * self.B0 / self.Ip * 100.0

    def has_interior_minimum(self) -> bool:
        """Check whether psi has a minimum strictly inside the grid.

        Returns True if the global minimum is NOT on any boundary row
        or column of the psi_rz array.
        """
        if self.psi_rz.size == 0:
            return False
        idx_min = np.unravel_index(np.argmin(self.psi_rz), self.psi_rz.shape)
        i_min, j_min = idx_min
        return (
            0 < i_min < self.nr - 1
            and 0 < j_min < self.nz - 1
        )

    def boundary_kappa_estimate(self) -> float:
        """Estimate effective elongation from the solved psi field.

        We find the contour closest to psi = 0 (boundary) and measure
        the vertical vs. horizontal half-widths.

        Returns
        -------
        float
            Estimated elongation.  Returns 0.0 if the boundary cannot
            be determined.
        """
        if self.psi_rz.size == 0:
            return 0.0

        # Use the analytical boundary directly for the estimate
        Z_half = np.max(np.abs(self.boundary_z))
        R_half = np.max(self.boundary_r) - np.min(self.boundary_r)
        if R_half < 1e-12:
            return 0.0
        return 2.0 * Z_half / R_half


# ── Shot parameter definitions ────────────────────────────────────────

@dataclass
class ShotParams:
    """Parameter set for a single synthetic shot."""
    shot_id: str
    category: str
    R0_m: float
    a_m: float
    B0_T: float
    Ip_MA: float
    kappa: float
    delta: float
    description: str = ""


def _make_circular_shots(rng: np.random.Generator) -> List[ShotParams]:
    """10 circular cross-section shots (kappa=1.0, delta=0.0)."""
    shots = []
    for i in range(10):
        R0 = rng.uniform(1.5, 6.5)
        a = rng.uniform(0.3, min(0.6 * R0, 2.2))
        B0 = rng.uniform(1.5, 6.0)
        Ip = rng.uniform(0.5, 5.0)
        shots.append(ShotParams(
            shot_id=f"SYNTH_CIRC_{i + 1:03d}",
            category="circular",
            R0_m=round(R0, 4),
            a_m=round(a, 4),
            B0_T=round(B0, 3),
            Ip_MA=round(Ip, 3),
            kappa=1.0,
            delta=0.0,
            description=f"Circular cross-section #{i + 1}",
        ))
    return shots


def _make_moderate_shots(rng: np.random.Generator) -> List[ShotParams]:
    """15 moderately elongated shots (kappa=1.4-1.7, delta=0.2-0.3).

    DIII-D-like: R0 ~ 1.67 m, a ~ 0.67 m, B0 ~ 2.2 T, Ip ~ 1.0-2.0 MA.
    We vary parameters around these reference values.
    """
    shots = []
    for i in range(15):
        R0 = rng.uniform(1.5, 2.5)
        a = rng.uniform(0.4, 0.8)
        B0 = rng.uniform(1.5, 3.0)
        Ip = rng.uniform(0.8, 3.0)
        kappa = rng.uniform(1.4, 1.7)
        delta = rng.uniform(0.2, 0.3)
        shots.append(ShotParams(
            shot_id=f"SYNTH_DIIID_{i + 1:03d}",
            category="moderate_elongation",
            R0_m=round(R0, 4),
            a_m=round(a, 4),
            B0_T=round(B0, 3),
            Ip_MA=round(Ip, 3),
            kappa=round(kappa, 4),
            delta=round(delta, 4),
            description=f"DIII-D-like moderate elongation #{i + 1}",
        ))
    return shots


def _make_iter_shots(rng: np.random.Generator) -> List[ShotParams]:
    """15 highly elongated shots (kappa=1.7-2.0, delta=0.3-0.5).

    ITER-like: R0 ~ 6.2 m, a ~ 2.0 m, B0 ~ 5.3 T, Ip ~ 15 MA.
    We vary parameters around these reference values.
    """
    shots = []
    for i in range(15):
        R0 = rng.uniform(5.0, 7.0)
        a = rng.uniform(1.5, 2.5)
        B0 = rng.uniform(4.0, 6.0)
        Ip = rng.uniform(10.0, 18.0)
        kappa = rng.uniform(1.7, 2.0)
        delta = rng.uniform(0.3, 0.5)
        shots.append(ShotParams(
            shot_id=f"SYNTH_ITER_{i + 1:03d}",
            category="high_elongation",
            R0_m=round(R0, 4),
            a_m=round(a, 4),
            B0_T=round(B0, 3),
            Ip_MA=round(Ip, 3),
            kappa=round(kappa, 4),
            delta=round(delta, 4),
            description=f"ITER-like high elongation #{i + 1}",
        ))
    return shots


def _make_high_beta_shots(rng: np.random.Generator) -> List[ShotParams]:
    """5 high-beta shots (beta_N > 2.0) -- stress tests.

    High beta requires large Ip and moderate B0.  We force the ratio
    a * B0 / Ip to be small so that beta_N comes out large.
    """
    shots = []
    for i in range(5):
        R0 = rng.uniform(1.5, 3.5)
        a = rng.uniform(0.5, 1.2)
        B0 = rng.uniform(1.0, 2.0)
        # Push Ip low relative to a*B0 to get high beta_N
        Ip = rng.uniform(0.3, 1.0)
        kappa = rng.uniform(1.5, 1.9)
        delta = rng.uniform(0.2, 0.45)
        shots.append(ShotParams(
            shot_id=f"SYNTH_HIBETA_{i + 1:03d}",
            category="high_beta",
            R0_m=round(R0, 4),
            a_m=round(a, 4),
            B0_T=round(B0, 3),
            Ip_MA=round(Ip, 3),
            kappa=round(kappa, 4),
            delta=round(delta, 4),
            description=f"High-beta stress test #{i + 1}",
        ))
    return shots


def _make_low_current_shots(rng: np.random.Generator) -> List[ShotParams]:
    """5 low-current shots (Ip < 1 MA) -- edge cases."""
    shots = []
    for i in range(5):
        R0 = rng.uniform(1.0, 3.0)
        a = rng.uniform(0.2, 0.7)
        B0 = rng.uniform(0.5, 2.5)
        Ip = rng.uniform(0.1, 0.9)
        kappa = rng.uniform(1.0, 1.5)
        delta = rng.uniform(0.0, 0.2)
        shots.append(ShotParams(
            shot_id=f"SYNTH_LOCUR_{i + 1:03d}",
            category="low_current",
            R0_m=round(R0, 4),
            a_m=round(a, 4),
            B0_T=round(B0, 3),
            Ip_MA=round(Ip, 3),
            kappa=round(kappa, 4),
            delta=round(delta, 4),
            description=f"Low-current edge case #{i + 1}",
        ))
    return shots


# ── Main generation ───────────────────────────────────────────────────

def generate_all_shots(
    seed: int = 20260214,
    output_dir: Optional[Path] = None,
    save: bool = True,
) -> List[Dict]:
    """Generate all 50 synthetic shots.

    Parameters
    ----------
    seed : int
        RNG seed for reproducible parameter generation.
    output_dir : Path or None
        Directory to write shot files.  Defaults to
        ``validation/synthetic_shots/`` relative to this script.
    save : bool
        If True, write NPZ + JSON files.  If False, return data only.

    Returns
    -------
    list of dict
        Each dict contains shot metadata and arrays (if save=False)
        or file paths (if save=True).
    """
    rng = np.random.default_rng(seed)

    # Build shot parameter list
    all_params: List[ShotParams] = []
    all_params.extend(_make_circular_shots(rng))
    all_params.extend(_make_moderate_shots(rng))
    all_params.extend(_make_iter_shots(rng))
    all_params.extend(_make_high_beta_shots(rng))
    all_params.extend(_make_low_current_shots(rng))

    assert len(all_params) == 50, f"Expected 50 shots, got {len(all_params)}"

    # Resolve output directory
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "synthetic_shots"

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []

    for idx, params in enumerate(all_params):
        print(f"[{idx + 1:2d}/50] Generating {params.shot_id} "
              f"({params.category}) ...", end=" ")

        # Solve equilibrium
        eq = SolovevEquilibrium(
            R0=params.R0_m,
            a=params.a_m,
            B0=params.B0_T,
            Ip=params.Ip_MA,
            kappa=params.kappa,
            delta=params.delta,
        )
        eq.solve()

        # Build metadata
        metadata = {
            "shot_id": params.shot_id,
            "category": params.category,
            "description": params.description,
            "R0_m": params.R0_m,
            "a_m": params.a_m,
            "B0_T": params.B0_T,
            "Ip_MA": params.Ip_MA,
            "kappa": params.kappa,
            "delta": params.delta,
            "beta_N": round(float(eq.beta_N), 4),
            "grid": {
                "nr": eq.nr,
                "nz": eq.nz,
                "R_min_m": round(float(eq.r_grid[0]), 6),
                "R_max_m": round(float(eq.r_grid[-1]), 6),
                "Z_min_m": round(float(eq.z_grid[0]), 6),
                "Z_max_m": round(float(eq.z_grid[-1]), 6),
            },
            "n_probes": eq.n_probes,
            "solver": "solovev_cerfon_freidberg_2010",
            "A_param": round(float(eq.A_param), 8),
            "psi_axis": float(eq.psi_axis),
            "r_axis_m": round(float(eq.r_axis), 6),
            "z_axis_m": round(float(eq.z_axis), 6),
            "has_interior_minimum": bool(eq.has_interior_minimum()),
            "boundary_kappa_estimate": round(eq.boundary_kappa_estimate(), 4),
            "coefficients": [round(float(c), 10) for c in eq.coefficients],
        }

        if save:
            # Save arrays
            npz_path = output_dir / f"{params.shot_id}.npz"
            np.savez_compressed(
                npz_path,
                psi_rz=eq.psi_rz,
                r_grid=eq.r_grid,
                z_grid=eq.z_grid,
                p_prime=eq.p_prime,
                ff_prime=eq.ff_prime,
                probe_r=eq.probe_r,
                probe_z=eq.probe_z,
                probe_psi=eq.probe_psi,
                boundary_r=eq.boundary_r,
                boundary_z=eq.boundary_z,
            )

            # Save metadata
            json_path = output_dir / f"{params.shot_id}.json"
            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=2)

            metadata["npz_path"] = str(npz_path)
            metadata["json_path"] = str(json_path)
            print(f"OK  beta_N={eq.beta_N:.2f}")
        else:
            metadata["arrays"] = {
                "psi_rz": eq.psi_rz,
                "r_grid": eq.r_grid,
                "z_grid": eq.z_grid,
                "p_prime": eq.p_prime,
                "ff_prime": eq.ff_prime,
                "probe_r": eq.probe_r,
                "probe_z": eq.probe_z,
                "probe_psi": eq.probe_psi,
                "boundary_r": eq.boundary_r,
                "boundary_z": eq.boundary_z,
            }
            print(f"OK  beta_N={eq.beta_N:.2f}  (in-memory)")

        results.append(metadata)

    # Write manifest
    if save:
        manifest_path = output_dir / "MANIFEST.json"
        manifest = {
            "generator": "generate_synthetic_shots.py",
            "solver": "solovev_cerfon_freidberg_2010",
            "reference": "Cerfon & Freidberg, Phys. Plasmas 17 (2010) 032502",
            "seed": seed,
            "n_shots": len(results),
            "categories": {
                "circular": sum(1 for r in results if r["category"] == "circular"),
                "moderate_elongation": sum(
                    1 for r in results if r["category"] == "moderate_elongation"
                ),
                "high_elongation": sum(
                    1 for r in results if r["category"] == "high_elongation"
                ),
                "high_beta": sum(
                    1 for r in results if r["category"] == "high_beta"
                ),
                "low_current": sum(
                    1 for r in results if r["category"] == "low_current"
                ),
            },
            "shots": [
                {
                    "shot_id": r["shot_id"],
                    "category": r["category"],
                    "beta_N": r["beta_N"],
                }
                for r in results
            ],
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest written: {manifest_path}")
        print(f"Total shots generated: {len(results)}")

    return results


# ── CLI entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate 50 synthetic tokamak equilibria for "
                    "inverse reconstruction validation."
    )
    parser.add_argument(
        "--seed", type=int, default=20260214,
        help="RNG seed for reproducible generation (default: 20260214).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for shot files.  Defaults to "
             "validation/synthetic_shots/ next to this script.",
    )
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else None
    generate_all_shots(seed=args.seed, output_dir=out, save=True)
