# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Linear Resistive Reduced MHD Eigenvalue Solver (1-D)
r"""Linear resistive reduced-MHD eigenvalue solver in one radial dimension.

The two-field model (flux :math:`\psi`, stream function :math:`\phi`, time
dependence :math:`e^{\lambda t}`) is

.. math::
    \lambda\psi = iF\phi + \eta L\psi, \qquad
    \lambda L\phi = iF L\psi - i\kappa_b J_0'\psi,

with :math:`Lf = h^{-1}(hf')' - \kappa^2 f`.  A Harris sheet uses
:math:`h=1`, :math:`\kappa=\kappa_b=k`, :math:`F=kB_y`; a periodic cylinder
uses :math:`h=r`, :math:`\kappa=\kappa_b=m/r`,
:math:`F=mB_\theta/r-n/R_0` with :math:`B_z=1`.  Units are :math:`a=1`,
:math:`v_A=1`, so :math:`\tau_A=1` and :math:`S=1/\eta`.

The Galerkin discretisation uses C1 cubic Hermite elements, following the
radial treatment of the CASTOR family (Kerner et al. 1998; Strumberger et al.,
J. Plasma Phys. 92, E113, 2026, Appendix A), and gives the generalised problem
:math:`\lambda Bx = Ax`.  Weak forms (test function :math:`w`, every test
function vanishes at the walls and :math:`h(0)=0` on the cylinder axis):

.. math::
    \int wLf\,h = -\int (w'f' + \kappa^2 wf)\,h, \qquad
    \int wFLf\,h = -\int ((w'F + wF')f' + \kappa^2 wFf)\,h.

Unstable eigenvalues are searched by :func:`unstable_search`.  Problems with
at most ``dense_limit`` unknowns are solved completely by dense QZ.  Larger
problems use shift-and-invert Arnoldi over real positive shifts
:math:`\sigma_{\max}f^{k}`, lowered to ``sigma_min`` and raised to
``sigma_ceil``; every root with positive real part inside its disk
:math:`|z-\sigma|<\sigma` is collected over all shifts (the ideal Alfvén
continuum lies on or left of the imaginary axis, so such a root is a discrete
unstable mode, not a continuum point).  Every candidate is verified by its
eigenvector residual.  The iterative path cannot prove that no stronger root
exists outside the searched disks, so its result is marked incomplete and
"no root" means "none found in the searched domain", not stability.

``cylinder_delta_prime`` solves the ideal outer equation independently by
shooting; it shares no code with the eigenvalue solver and is the reference
for the cylindrical FKR check.  Near the resonant surface the nonzero outer
solution is :math:`\psi = C[1 + a x\ln|x| + b_\pm x + \dots]`, with
:math:`x = r - r_s` and :math:`a = (m/r_s)J_0'(r_s)/F'(r_s)`, so
:math:`\psi'/\psi` holds :math:`a\ln|x|` (symmetric, cancels in the jump) and
:math:`-a^2x\ln^2|x|`, which leaves :math:`-2a^2\epsilon\ln^2\epsilon` in
the jump; the extrapolation basis is therefore
:math:`\{1, \epsilon\ln\epsilon, \epsilon, \epsilon\ln^2\epsilon\}`.

References
----------
Furth, Killeen & Rosenbluth, Phys. Fluids 6, 459 (1963).
Coppi, Galvão, Pellat, Rosenbluth & Rutherford, Sov. J. Plasma Phys. 2, 533 (1976).
Kerner et al., J. Comput. Phys. 142, 271 (1998).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
Profile = Callable[[FloatArray], FloatArray]

_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(6)  # type: ignore[no-untyped-call]  # numpy.polynomial is untyped
_GAUSS_T: FloatArray = 0.5 * (_GAUSS_NODES + 1.0)
_GAUSS_W: FloatArray = 0.5 * _GAUSS_WEIGHTS
_SMALL_R = 1e-6


def _finite_positive(name: str, value: float) -> float:
    """Return ``value`` as a float or reject a non-finite or non-positive input."""
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


@dataclass(frozen=True)
class Geometry1D:
    """Coefficient profiles of the 1-D linear reduced-MHD operator.

    ``axis_value_only`` is ``None`` for a slab (both ends are walls), ``True``
    for a cylinder with :math:`m=1` (only :math:`f(0)=0`), ``False`` for a
    cylinder with :math:`m\\ge 2` (also :math:`f'(0)=0`).
    """

    weight: Profile
    kappa_sq: Profile
    kappa_b: Profile
    f_parallel: Profile
    f_parallel_prime: Profile
    current_prime: Profile
    x_min: float
    x_max: float
    resonant_x: float
    axis_value_only: bool | None = None


def harris_slab(ka: float, half_width: float) -> Geometry1D:
    """Return the Harris sheet :math:`B_y=\\tanh x` on :math:`[-L, L]` with wave number ``ka``."""
    k = _finite_positive("ka", ka)
    width = _finite_positive("half_width", half_width)

    def ones(x: FloatArray) -> FloatArray:
        return np.ones_like(x)

    def kappa_sq(x: FloatArray) -> FloatArray:
        return np.full_like(x, k * k)

    def kappa(x: FloatArray) -> FloatArray:
        return np.full_like(x, k)

    def f_par(x: FloatArray) -> FloatArray:
        return k * np.tanh(x)

    def f_par_prime(x: FloatArray) -> FloatArray:
        return k / np.cosh(x) ** 2

    def current_prime(x: FloatArray) -> FloatArray:
        return -2.0 * np.tanh(x) / np.cosh(x) ** 2

    return Geometry1D(ones, kappa_sq, kappa, f_par, f_par_prime, current_prime, -width, width, 0.0)


@dataclass(frozen=True)
class WessonCylinder:
    """Periodic cylinder with :math:`B_z=1`, :math:`J_0=j_0(1-r^2)^\\nu`, :math:`q(0)=q_0`."""

    q0: float
    nu: float
    major_radius: float

    def __post_init__(self) -> None:
        """Reject non-physical profile parameters."""
        _finite_positive("q0", self.q0)
        _finite_positive("nu", self.nu)
        _finite_positive("major_radius", self.major_radius)

    @property
    def j0(self) -> float:
        """Return the on-axis current density that gives :math:`q(0)=q_0`."""
        return 2.0 / (self.major_radius * self.q0)

    def current(self, r: FloatArray) -> FloatArray:
        """Return :math:`J_0(r)`."""
        return self.j0 * (1.0 - np.asarray(r, dtype=np.float64) ** 2) ** self.nu

    def current_prime(self, r: FloatArray) -> FloatArray:
        """Return :math:`J_0'(r)`."""
        rr = np.asarray(r, dtype=np.float64)
        return -2.0 * self.nu * self.j0 * rr * (1.0 - rr * rr) ** (self.nu - 1.0)

    def b_theta(self, r: FloatArray) -> FloatArray:
        """Return :math:`B_\\theta(r)` from :math:`(rB_\\theta)'=rJ_0`."""
        rr = np.asarray(r, dtype=np.float64)
        safe = np.maximum(rr, _SMALL_R)
        full = (
            self.j0
            * (1.0 - (1.0 - safe * safe) ** (self.nu + 1.0))
            / (2.0 * (self.nu + 1.0) * safe)
        )
        return np.where(rr < _SMALL_R, 0.5 * self.j0 * rr, full)

    def b_theta_prime(self, r: FloatArray) -> FloatArray:
        """Return :math:`B_\\theta'(r)=J_0-B_\\theta/r`."""
        rr = np.asarray(r, dtype=np.float64)
        safe = np.maximum(rr, _SMALL_R)
        return np.where(
            rr < _SMALL_R, 0.5 * self.j0, self.current(safe) - self.b_theta(safe) / safe
        )

    def q(self, r: FloatArray) -> FloatArray:
        """Return the safety factor :math:`q=r/(R_0B_\\theta)`."""
        rr = np.asarray(r, dtype=np.float64)
        safe = np.maximum(rr, _SMALL_R)
        return np.where(rr < _SMALL_R, self.q0, safe / (self.major_radius * self.b_theta(safe)))

    def resonant_radius(self, m: int, n: int) -> float:
        """Return the radius of the :math:`q=m/n` surface inside the plasma."""
        target = m / n
        lo, hi = _SMALL_R, 1.0 - 1e-12
        if not (self.q(np.array([lo]))[0] - target) * (self.q(np.array([hi]))[0] - target) < 0.0:
            raise ValueError(f"q = {m}/{n} is not inside the plasma")
        return float(brentq(lambda r: float(self.q(np.array([r]))[0]) - target, lo, hi))

    def geometry(self, m: int, n: int) -> Geometry1D:
        """Return the operator coefficients for the helicity ``(m, n)``."""
        if m < 1 or n < 1:
            raise ValueError("m and n must be >= 1")
        mm, nn, r0 = float(m), float(n), self.major_radius

        def weight(r: FloatArray) -> FloatArray:
            return np.asarray(r, dtype=np.float64)

        def kappa_sq(r: FloatArray) -> FloatArray:
            return (mm / np.asarray(r, dtype=np.float64)) ** 2

        def kappa_b(r: FloatArray) -> FloatArray:
            return mm / np.asarray(r, dtype=np.float64)

        def f_par(r: FloatArray) -> FloatArray:
            safe = np.maximum(np.asarray(r, dtype=np.float64), _SMALL_R)
            return np.asarray(mm * self.b_theta(safe) / safe - nn / r0, dtype=np.float64)

        def f_par_prime(r: FloatArray) -> FloatArray:
            safe = np.maximum(np.asarray(r, dtype=np.float64), _SMALL_R)
            return np.asarray(
                mm * (self.b_theta_prime(safe) / safe - self.b_theta(safe) / safe**2),
                dtype=np.float64,
            )

        return Geometry1D(
            weight,
            kappa_sq,
            kappa_b,
            f_par,
            f_par_prime,
            self.current_prime,
            0.0,
            1.0,
            self.resonant_radius(m, n),
            axis_value_only=(m == 1),
        )


def clustered_grid(x_min: float, x_max: float, x_c: float, h_min: float, n_side: int) -> FloatArray:
    """Return nodes whose spacing grows geometrically from ``h_min`` away from ``x_c``."""
    _finite_positive("h_min", h_min)
    if n_side < 2:
        raise ValueError("n_side must be >= 2")
    if not x_min <= x_c <= x_max or x_min >= x_max:
        raise ValueError("x_c must lie in [x_min, x_max] and x_min < x_max")

    def side(length: float) -> FloatArray:
        if length <= 0.0:
            return np.zeros(1)
        if h_min * n_side >= length:
            return np.linspace(0.0, length, n_side + 1)
        q_hi = (length / h_min) ** (1.0 / (n_side - 1)) + 1e-9
        ratio = brentq(lambda q: h_min * (q**n_side - 1.0) / (q - 1.0) - length, 1.0 + 1e-12, q_hi)
        points = np.concatenate([[0.0], np.cumsum(h_min * ratio ** np.arange(n_side))])
        points[-1] = length
        return np.asarray(points, dtype=np.float64)

    nodes = np.unique(np.concatenate([x_c - side(x_c - x_min)[::-1], x_c + side(x_max - x_c)]))
    nodes[0], nodes[-1] = x_min, x_max
    return np.asarray(nodes, dtype=np.float64)


def _hermite(h: float) -> tuple[FloatArray, FloatArray]:
    """Return values and x-derivatives of the cubic Hermite functions at the Gauss points."""
    t = _GAUSS_T
    value = np.stack(
        [1 - 3 * t**2 + 2 * t**3, h * (t - 2 * t**2 + t**3), 3 * t**2 - 2 * t**3, h * (t**3 - t**2)]
    )
    slope = np.stack(
        [(6 * t**2 - 6 * t) / h, 1 - 4 * t + 3 * t**2, (6 * t - 6 * t**2) / h, 3 * t**2 - 2 * t]
    )
    return value, slope


@dataclass(frozen=True)
class Discretisation:
    """Assembled generalised eigenproblem :math:`\\lambda Bx=Ax` on ``nodes``."""

    nodes: FloatArray
    a: sp.csc_matrix
    b: sp.csc_matrix


def assemble(geom: Geometry1D, nodes: FloatArray, eta: float) -> Discretisation:
    """Assemble ``A`` and ``B`` for the two-field resistive problem."""
    resistivity = _finite_positive("eta", eta)
    x_nodes = np.asarray(nodes, dtype=np.float64)
    if x_nodes.ndim != 1 or x_nodes.size < 3 or np.any(np.diff(x_nodes) <= 0.0):
        raise ValueError("nodes must be a strictly increasing 1-D array of at least 3 points")
    n_dof = 2 * x_nodes.size
    rows: list[NDArray[np.int64]] = []
    cols: list[NDArray[np.int64]] = []
    blocks: dict[str, list[FloatArray]] = {"M": [], "K": [], "MF": [], "G": []}
    for e in range(x_nodes.size - 1):
        h = float(x_nodes[e + 1] - x_nodes[e])
        x = x_nodes[e] + h * _GAUSS_T
        wq = h * _GAUSS_W * geom.weight(x)
        value, slope = _hermite(h)
        k2, kb = geom.kappa_sq(x), geom.kappa_b(x)
        f, fp, jp = geom.f_parallel(x), geom.f_parallel_prime(x), geom.current_prime(x)
        blocks["M"].append((value * wq) @ value.T)
        blocks["K"].append((slope * wq) @ slope.T + (value * wq * k2) @ value.T)
        blocks["MF"].append((value * wq * f) @ value.T)
        blocks["G"].append(
            (slope * wq * f) @ slope.T
            + (value * wq * fp) @ slope.T
            + (value * wq * k2 * f) @ value.T
            + (value * wq * kb * jp) @ value.T
        )
        index = np.arange(2 * e, 2 * e + 4)
        rows.append(np.repeat(index, 4))
        cols.append(np.tile(index, 4))
    row, col = np.concatenate(rows), np.concatenate(cols)
    fixed = {0, n_dof - 2}
    if geom.axis_value_only is False:
        fixed.add(1)
    free = np.array(sorted(set(range(n_dof)) - fixed), dtype=np.int64)
    mats = {
        name: sp.coo_matrix(
            (np.concatenate([b.ravel() for b in data]), (row, col)), shape=(n_dof, n_dof)
        ).tocsr()[free][:, free]
        for name, data in blocks.items()
    }
    zero = sp.csr_matrix(mats["M"].shape)
    a = sp.bmat([[-resistivity * mats["K"], 1j * mats["MF"]], [1j * mats["G"], zero]], format="csc")
    b = sp.bmat([[mats["M"], zero], [zero, mats["K"]]], format="csc").astype(np.complex128)
    return Discretisation(nodes=x_nodes, a=a, b=b)


def eigenvalues_near(
    disc: Discretisation, sigma: complex, count: int = 6, *, maxiter: int = 2000
) -> ComplexArray:
    """Return ``count`` eigenvalues nearest ``sigma`` by shift-and-invert Arnoldi."""
    factor = spla.splu((disc.a - sigma * disc.b).tocsc())
    operator = spla.LinearOperator(
        disc.a.shape, matvec=lambda v: factor.solve(disc.b @ v), dtype=np.complex128
    )
    mu = spla.eigs(
        operator, k=count, which="LM", return_eigenvectors=False, tol=1e-10, ncv=40, maxiter=maxiter
    )
    return np.asarray(sigma + 1.0 / mu, dtype=np.complex128)


def all_eigenvalues(disc: Discretisation) -> ComplexArray:
    """Return the finite part of the full dense spectrum (small problems only)."""
    from scipy.linalg import eig

    values = eig(disc.a.toarray(), disc.b.toarray(), right=False)
    return np.asarray(values[np.isfinite(values)], dtype=np.complex128)


@dataclass(frozen=True)
class SpectralSearch:
    """Verified unstable eigenvalues and the domain that was searched for them."""

    candidates: tuple[complex, ...]
    residuals: tuple[float, ...]
    method: str
    domain: str
    complete: bool

    @property
    def most_unstable(self) -> complex | None:
        """Return the candidate with the largest real part, or ``None`` if none was found."""
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda z: z.real)


def _relative_residual(
    disc: Discretisation, lam: complex, x: ComplexArray, norm_a: float, norm_b: float
) -> float:
    """Return ``|Ax - lam Bx| / ((|A|_1 + |lam| |B|_1) |x|)``."""
    vec = np.asarray(x, dtype=np.complex128)
    defect = disc.a @ vec - lam * (disc.b @ vec)
    scale = (norm_a + abs(lam) * norm_b) * float(np.linalg.norm(vec))
    return float(np.linalg.norm(defect) / scale)


def _pairs_near(
    disc: Discretisation, sigma: float, count: int, maxiter: int
) -> tuple[ComplexArray, ComplexArray]:
    """Return eigenvalues and eigenvectors near ``sigma`` (converged ones only on ARPACK failure)."""
    factor = spla.splu((disc.a - sigma * disc.b).tocsc())
    operator = spla.LinearOperator(
        disc.a.shape, matvec=lambda v: factor.solve(disc.b @ v), dtype=np.complex128
    )
    try:
        mu, vectors = spla.eigs(operator, k=count, which="LM", tol=1e-10, ncv=40, maxiter=maxiter)
    except spla.ArpackNoConvergence as exc:
        mu = np.asarray(exc.eigenvalues, dtype=np.complex128)
        vectors = np.asarray(exc.eigenvectors, dtype=np.complex128)
    return np.asarray(sigma + 1.0 / mu, dtype=np.complex128), np.asarray(
        vectors, dtype=np.complex128
    )


def unstable_search(
    disc: Discretisation,
    *,
    sigma_max: float = 0.3,
    sigma_min: float = 1e-12,
    sigma_ceil: float = 10.0,
    factor: float = 4.0,
    count: int = 6,
    dense_limit: int = 2000,
    residual_tol: float = 1e-8,
    maxiter: int = 2000,
) -> SpectralSearch:
    """Return every verified unstable eigenvalue found, with the searched domain and completeness."""
    top = _finite_positive("sigma_max", sigma_max)
    floor = _finite_positive("sigma_min", sigma_min)
    ceil = _finite_positive("sigma_ceil", sigma_ceil)
    if factor <= 1.0:
        raise ValueError("factor must be > 1")
    if floor > top or ceil < top:
        raise ValueError("sigma_min <= sigma_max <= sigma_ceil is required")
    norm_a = float(spla.norm(disc.a, 1))
    norm_b = float(spla.norm(disc.b, 1))
    found: list[tuple[complex, float]] = []

    def keep(lam: complex, x: ComplexArray) -> None:
        res = _relative_residual(disc, lam, x, norm_a, norm_b)
        if res <= residual_tol and all(abs(lam - z) > 1e-8 * max(abs(z), floor) for z, _ in found):
            found.append((complex(lam), res))

    if disc.a.shape[0] <= dense_limit:
        from scipy.linalg import eig

        values, vectors = eig(disc.a.toarray(), disc.b.toarray())
        for k, lam in enumerate(values):
            if np.isfinite(lam) and lam.real > floor:
                keep(complex(lam), vectors[:, k])
        found.sort(key=lambda item: -item[0].real)
        return SpectralSearch(
            tuple(z for z, _ in found),
            tuple(r for _, r in found),
            "dense-qz",
            f"full spectrum, Re > {floor:g}",
            True,
        )
    shifts = [top]
    while shifts[-1] / factor >= floor:
        shifts.append(shifts[-1] / factor)
    while shifts[0] * factor <= ceil:
        shifts.insert(0, shifts[0] * factor)
    for sigma in shifts:
        values, vectors = _pairs_near(disc, sigma, count, maxiter)
        for k, lam in enumerate(values):
            if lam.real > 0.0 and abs(lam - sigma) < sigma:
                keep(complex(lam), vectors[:, k])
    found.sort(key=lambda item: -item[0].real)
    return SpectralSearch(
        tuple(z for z, _ in found),
        tuple(r for _, r in found),
        "shift-invert",
        f"union of disks |z - s| < s, s = {shifts[-1]:.3g} ... {shifts[0]:.3g} (factor {factor:g}),"
        f" {count} roots per shift",
        False,
    )


def most_unstable(
    disc: Discretisation,
    *,
    sigma_max: float = 0.3,
    sigma_min: float = 1e-12,
    sigma_ceil: float = 10.0,
    factor: float = 4.0,
    dense_limit: int = 2000,
    maxiter: int = 2000,
) -> complex | None:
    """Return the most unstable verified eigenvalue found by :func:`unstable_search`, or ``None``.

    ``None`` means no unstable root was found in the searched domain; on the
    iterative path that does not establish stability (see :class:`SpectralSearch`).
    """
    return unstable_search(
        disc,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        sigma_ceil=sigma_ceil,
        factor=factor,
        dense_limit=dense_limit,
        maxiter=maxiter,
    ).most_unstable


def growth_rate(
    geom: Geometry1D,
    eta: float,
    *,
    n_side: int = 200,
    h_min: float | None = None,
    sigma_min: float = 1e-12,
) -> float:
    """Return the largest unstable :math:`\\mathrm{Re}\\,\\lambda` (0.0 when none is found in the searched domain).

    The grid is clustered at ``geom.resonant_x`` with smallest spacing
    ``h_min`` (default :math:`0.02\\,\\eta^{1/3}`, a fraction of the resistive
    kink layer, which is wider than the tearing layer only by an O(1) factor).
    """
    resistivity = _finite_positive("eta", eta)
    spacing = 0.02 * resistivity ** (1.0 / 3.0) if h_min is None else h_min
    nodes = clustered_grid(geom.x_min, geom.x_max, geom.resonant_x, spacing, n_side)
    lam = most_unstable(assemble(geom, nodes, resistivity), sigma_min=sigma_min)
    return 0.0 if lam is None else lam.real


@dataclass(frozen=True)
class DeltaPrimeFit:
    """Extrapolated :math:`\\Delta'`, the fitted jumps and the least-squares residual."""

    value: float
    excisions: FloatArray
    jumps: FloatArray
    coefficients: FloatArray
    residual: float


def cylinder_delta_prime_fit(
    cylinder: WessonCylinder,
    m: int,
    n: int,
    *,
    excisions: FloatArray | None = None,
) -> DeltaPrimeFit:
    """Return :math:`\\Delta'` of the ideal outer equation by two-sided shooting, with fit details.

    The outer equation :math:`L\\psi-\\kappa_bJ_0'\\psi/F=0` is integrated from
    the axis (:math:`\\psi\\sim r^m`) and from the wall (:math:`\\psi=0`) to
    :math:`r_s\\mp\\epsilon`; the jump of :math:`\\psi'/\\psi` is extrapolated to
    :math:`\\epsilon\\to0` with the basis
    :math:`\\{1,\\epsilon\\ln\\epsilon,\\epsilon,\\epsilon\\ln^2\\epsilon\\}` (module docstring).
    The default excisions span :math:`10^{-4}\\ldots10^{-6}`: on the reference
    case the window :math:`10^{-3}\\ldots10^{-5}` still carries higher-order
    terms (2.7e-5 relative), while :math:`10^{-4}\\ldots10^{-6}` and
    :math:`10^{-5}\\ldots10^{-7}` agree to 3e-7.
    """
    geom = cylinder.geometry(m, n)
    rs = geom.resonant_x
    eps_values = (
        np.geomspace(1e-4, 1e-6, 9)
        if excisions is None
        else np.asarray(excisions, dtype=np.float64)
    )
    if eps_values.size < 4 or np.any(eps_values <= 0.0):
        raise ValueError("excisions must hold at least four positive values")

    def rhs(r: float, y: FloatArray) -> FloatArray:
        radius = np.array([r])
        f = geom.f_parallel(radius)[0]
        jp = cylinder.current_prime(radius)[0]
        return np.array([y[1], -y[1] / r + (m * m / (r * r)) * y[0] + (m / r) * jp * y[0] / f])

    r_start = 1e-4
    jumps = []
    for eps in eps_values:
        left = solve_ivp(
            rhs,
            (r_start, rs - eps),
            [r_start**m, m * r_start ** (m - 1)],
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
        )
        right = solve_ivp(
            rhs, (1.0, rs + eps), [0.0, -1.0], method="DOP853", rtol=1e-12, atol=1e-14
        )
        if not (left.success and right.success):
            raise RuntimeError(f"outer-equation shooting failed at excision {eps:g}")
        with np.errstate(divide="ignore", invalid="ignore"):
            jump = right.y[1, -1] / right.y[0, -1] - left.y[1, -1] / left.y[0, -1]
        if not np.isfinite(jump):
            raise RuntimeError(f"non-finite logarithmic-derivative jump at excision {eps:g}")
        jumps.append(jump)
    log_eps = np.log(eps_values)
    design = np.stack(
        [np.ones_like(eps_values), eps_values * log_eps, eps_values, eps_values * log_eps**2],
        axis=1,
    )
    jump_array = np.asarray(jumps, dtype=np.float64)
    coefficients, *_ = np.linalg.lstsq(design, jump_array, rcond=None)
    residual = float(np.linalg.norm(design @ coefficients - jump_array))
    return DeltaPrimeFit(
        value=float(coefficients[0]),
        excisions=eps_values,
        jumps=jump_array,
        coefficients=np.asarray(coefficients, dtype=np.float64),
        residual=residual,
    )


def cylinder_delta_prime(
    cylinder: WessonCylinder,
    m: int,
    n: int,
    *,
    excisions: FloatArray | None = None,
) -> float:
    """Return :math:`\\Delta'` from :func:`cylinder_delta_prime_fit`."""
    return cylinder_delta_prime_fit(cylinder, m, n, excisions=excisions).value
