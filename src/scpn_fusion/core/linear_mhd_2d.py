# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Linear Resistive Reduced MHD with Coupled Poloidal Harmonics
r"""Linear resistive reduced MHD on a shaped cross-section with coupled harmonics.

Stage 2a-i of the reactor group's linear MHD plan: a periodic cylinder whose
poloidal cross-section is shaped (elongation :math:`\kappa`, triangularity
:math:`\delta`, radial shift :math:`\Delta(r)`),

.. math::
    X = \Delta(r) + r\cos(\theta_g + \delta\sin\theta_g), \qquad
    Y = \kappa r\sin\theta_g,

with a prescribed safety factor :math:`q(r)`, axial field :math:`B_z=1` and
period :math:`2\pi R_0`. In straight-field-line coordinates
:math:`(r, \theta^*)` the 2-D Jacobian :math:`\mathcal J` is a flux function
and :math:`B_0\cdot\nabla = (qR_0)^{-1}(\partial_{\theta^*} - inq)` exactly, so
poloidal harmonics couple only through the metric :math:`g^{ij}(r,\theta^*)`.

The model is the Stage 1 one, :math:`\lambda\psi = B_0\cdot\nabla\phi +
\eta\nabla_\perp^2\psi`, :math:`\lambda\nabla_\perp^2\phi = B_0\cdot\nabla J_1
+ b_1\cdot\nabla J_0`, with :math:`b_1 = \hat z\times\nabla\psi` and
:math:`J_0 = \nabla_\perp^2\psi_0`. The weak forms use only first derivatives
of the C1 Hermite radial basis and only :math:`J_0` itself, through
:math:`\int\bar w\,b_1\cdot\nabla J_0 = -\int J_0\,b_1\cdot\nabla\bar w` and
:math:`\int\bar w\,B_0\cdot\nabla J_1 = \int\overline{v}\,\nabla_\perp^2\psi`
with :math:`v = ia(m'-nq)w`. All poloidal integrals are evaluated on a uniform
:math:`\theta^*` grid, exact for the band-limited products involved when the
grid is fine enough; the matrix elements follow Strumberger et al., J. Plasma
Phys. 92, E113 (2026), Appendix A.2, with the Fourier convolution done by
quadrature.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from scpn_fusion.core.linear_mhd_1d import (
    _GAUSS_T,
    _GAUSS_W,
    Discretisation,
    _hermite,
    clustered_grid,
    most_unstable,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
_FD_STEP = 1e-5


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class ShapedCylinder:
    """Shaped periodic cylinder with prescribed :math:`q(r)` (minor radius 1, :math:`B_z=1`)."""

    q: Callable[[FloatArray], FloatArray]
    major_radius: float
    elongation: float = 1.0
    triangularity: float = 0.0
    shift: Callable[[FloatArray], FloatArray] | None = None
    shift_prime: Callable[[FloatArray], FloatArray] | None = None
    n_theta: int = 64
    n_fine: int = 512

    def __post_init__(self) -> None:
        """Reject non-physical shape parameters."""
        if _finite("major_radius", self.major_radius) <= 0.0:
            raise ValueError("major_radius must be > 0")
        if _finite("elongation", self.elongation) <= 0.0:
            raise ValueError("elongation must be > 0")
        if abs(_finite("triangularity", self.triangularity)) >= 1.0:
            raise ValueError("|triangularity| must be < 1")
        if (self.shift is None) != (self.shift_prime is None):
            raise ValueError("shift and shift_prime must be given together")
        if self.n_theta < 8 or self.n_fine < 4 * self.n_theta:
            raise ValueError("n_theta must be >= 8 and n_fine >= 4 n_theta")

    def _xy_derivatives(self, r: FloatArray, tg: FloatArray) -> tuple[FloatArray, ...]:
        """Return X_r, X_t, Y_r, Y_t of the geometric parameterisation (broadcast r, tg)."""
        d, k = self.triangularity, self.elongation
        ang = tg + d * np.sin(tg)
        dshift = np.zeros_like(r) if self.shift_prime is None else self.shift_prime(r)
        x_r = dshift + np.cos(ang)
        x_t = -r * np.sin(ang) * (1.0 + d * np.cos(tg))
        y_r = k * np.sin(tg) + 0.0 * r
        y_t = k * r * np.cos(tg)
        return x_r, x_t, y_r, y_t

    def _theta_star(self, r: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return the fine geometric grid, theta*(r, tg) and the flux-surface Jacobian (1/2pi) loop J_g."""
        tg = np.linspace(0.0, 2.0 * np.pi, self.n_fine, endpoint=False)
        rr = r[:, None]
        x_r, x_t, y_r, y_t = self._xy_derivatives(rr, tg[None, :])
        jac = x_r * y_t - x_t * y_r
        if np.any(jac <= 0.0):
            raise ValueError(
                "the shaped cross-section has a non-positive Jacobian (overlapping surfaces)"
            )
        # spectral cumulative integral of the periodic jac over tg, O(N log N) per radius
        coef = np.fft.fft(jac, axis=1) / self.n_fine
        k = np.fft.fftfreq(self.n_fine, d=1.0 / self.n_fine)
        mean = coef[:, 0].real
        anti = np.zeros_like(coef)
        anti[:, 1:] = coef[:, 1:] / (1j * k[None, 1:])
        osc = np.real(np.fft.ifft(anti, axis=1) * self.n_fine - anti.sum(axis=1)[:, None])
        integral = mean[:, None] * tg[None, :] + osc
        theta_star = integral / mean[:, None]
        return tg, theta_star, mean

    def metric(self, r: FloatArray) -> dict[str, FloatArray]:
        """Return the metric of (r, theta*) on a uniform theta* grid at radii ``r``.

        Keys: ``jac`` (2-D Jacobian, flux function), ``grr``, ``grt``, ``gtt`` (contravariant
        metric), all shaped (len(r), n_theta).
        """
        r = np.asarray(r, dtype=np.float64)
        tstar_grid = np.linspace(0.0, 2.0 * np.pi, self.n_theta, endpoint=False)
        tg_fine, ts, mean = self._theta_star(r)
        # theta*(r +- h, tg) for the radial derivative at fixed tg
        _, ts_p, _ = self._theta_star(r + _FD_STEP)
        _, ts_m, _ = self._theta_star(r - _FD_STEP)
        dts_dr_fine = (ts_p - ts_m) / (2.0 * _FD_STEP)
        tg_on = np.empty((r.size, self.n_theta))
        dts_dr = np.empty_like(tg_on)
        for i in range(r.size):
            ext_ts = np.concatenate([ts[i] - 2 * np.pi, ts[i], ts[i] + 2 * np.pi])
            ext_tg = np.concatenate([tg_fine - 2 * np.pi, tg_fine, tg_fine + 2 * np.pi])
            tg_on[i] = np.interp(tstar_grid, ext_ts, ext_tg)
            ext_d = np.concatenate([dts_dr_fine[i]] * 3)
            dts_dr[i] = np.interp(tstar_grid, ext_ts, ext_d)
        rr = np.repeat(r[:, None], self.n_theta, axis=1)
        x_r, x_t, y_r, y_t = self._xy_derivatives(rr, tg_on)
        jac_g = x_r * y_t - x_t * y_r
        # grad r and grad tg in Cartesian components
        grad_r = np.stack([y_t, -x_t]) / jac_g
        grad_tg = np.stack([-y_r, x_r]) / jac_g
        dts_dtg = jac_g / mean[:, None]
        grad_ts = dts_dr * grad_r + dts_dtg * grad_tg
        grr = np.sum(grad_r * grad_r, axis=0)
        grt = np.sum(grad_r * grad_ts, axis=0)
        gtt = np.sum(grad_ts * grad_ts, axis=0)
        jac = 1.0 / (grad_r[0] * grad_ts[1] - grad_r[1] * grad_ts[0])
        return {"jac": jac, "grr": grr, "grt": grt, "gtt": gtt}

    def current(self, r: FloatArray) -> FloatArray:
        """Return J0 = laplacian(psi0) on the theta* grid, psi0' = J / (q R0)."""
        r = np.asarray(r, dtype=np.float64)

        def radial_flux(rad: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
            g = self.metric(rad)
            dpsi = g["jac"] / (self.q(rad)[:, None] * self.major_radius)
            return g["jac"] * g["grr"] * dpsi, g["jac"] * g["grt"] * dpsi, g["jac"]

        fp, _, _ = radial_flux(r + _FD_STEP)
        fm, _, _ = radial_flux(r - _FD_STEP)
        _, ftheta, jac = radial_flux(r)
        d_r = (fp - fm) / (2.0 * _FD_STEP)
        k = np.fft.fftfreq(self.n_theta, d=1.0 / self.n_theta)
        d_t = np.real(np.fft.ifft(1j * k * np.fft.fft(ftheta, axis=1), axis=1))
        return np.asarray((d_r + d_t) / jac, dtype=np.float64)


@dataclass(frozen=True)
class CoupledDiscretisation(Discretisation):
    """Coupled-harmonic eigenproblem with the map from free dofs to (node, slot, harmonic)."""

    m_values: tuple[int, ...] = ()
    free: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


def assemble_coupled(
    cyl: ShapedCylinder,
    nodes: FloatArray,
    eta: float,
    n: int,
    m_values: list[int],
) -> CoupledDiscretisation:
    """Assemble ``A`` and ``B`` for the coupled-harmonic two-field problem."""
    if eta <= 0.0 or not math.isfinite(eta):
        raise ValueError("eta must be finite and > 0")
    if not m_values or any(m < 1 for m in m_values) or len(set(m_values)) != len(m_values):
        raise ValueError("m_values must be distinct integers >= 1")
    x_nodes = np.asarray(nodes, dtype=np.float64)
    if x_nodes[0] != 0.0 or x_nodes[-1] != 1.0 or np.any(np.diff(x_nodes) <= 0.0):
        raise ValueError("nodes must increase strictly from 0 to 1")
    ms = np.asarray(m_values, dtype=np.float64)
    n_m = ms.size
    theta = np.linspace(0.0, 2.0 * np.pi, cyl.n_theta, endpoint=False)
    phase = np.exp(1j * ms[:, None] * theta[None, :])  # (m, t)
    dtheta = 2.0 * np.pi / cyl.n_theta
    n_field = 2 * x_nodes.size * n_m  # value and slope per node, per harmonic
    names = ("M", "K", "F", "KV", "Q")
    data: dict[str, list[ComplexArray]] = {k: [] for k in names}
    rows: list[NDArray[np.int64]] = []
    cols: list[NDArray[np.int64]] = []
    widths = np.diff(x_nodes)
    r_all = (x_nodes[:-1, None] + widths[:, None] * _GAUSS_T[None, :]).ravel()
    g_all = cyl.metric(r_all)
    j0_all = cyl.current(r_all)
    q_all = cyl.q(r_all)
    dq_all = (cyl.q(r_all + _FD_STEP) - cyl.q(r_all - _FD_STEP)) / (2.0 * _FD_STEP)
    n_g = _GAUSS_T.size
    for e in range(x_nodes.size - 1):
        h = float(widths[e])
        sl = slice(e * n_g, (e + 1) * n_g)
        r = r_all[sl]
        wq = h * _GAUSS_W * dtheta  # (g,)
        g = {key: value[sl] for key, value in g_all.items()}
        j0 = j0_all[sl]
        q = q_all[sl]
        dq = dq_all[sl]
        a = 1.0 / (q * cyl.major_radius)
        da = -dq / (q * q * cyl.major_radius)
        val, der = _hermite(h)  # (4, g)
        # basis arrays: index (dof_local = 4 * n_m), g, t
        f = (val[:, None, :, None] * phase[None, :, None, :]).reshape(4 * n_m, r.size, cyl.n_theta)
        fr = (der[:, None, :, None] * phase[None, :, None, :]).reshape(4 * n_m, r.size, cyl.n_theta)
        ft = (
            1j * ms[None, :, None, None] * val[:, None, :, None] * phase[None, :, None, :]
        ).reshape(4 * n_m, r.size, cyl.n_theta)
        mq = np.tile(ms, 4)[:, None] - n * q[None, :]  # (dof, g)
        w = wq[None, :, None] * g["jac"][None]
        jac_w = w[0]

        def bilinear_k(
            tr: ComplexArray, tt: ComplexArray, sr: ComplexArray, st: ComplexArray
        ) -> ComplexArray:
            return np.asarray(
                np.einsum("igt,jgt,gt->ij", np.conj(tr), sr, g["grr"] * jac_w)
                + np.einsum("igt,jgt,gt->ij", np.conj(tr), st, g["grt"] * jac_w)
                + np.einsum("igt,jgt,gt->ij", np.conj(tt), sr, g["grt"] * jac_w)
                + np.einsum("igt,jgt,gt->ij", np.conj(tt), st, g["gtt"] * jac_w),
                dtype=np.complex128,
            )

        data["M"].append(np.einsum("igt,jgt,gt->ij", np.conj(f), f, jac_w))
        data["K"].append(bilinear_k(fr, ft, fr, ft))
        data["F"].append(
            np.einsum("igt,jgt,gt->ij", np.conj(f), f * (1j * a[None, :] * mq)[:, :, None], jac_w)
        )
        coef: ComplexArray = np.asarray(1j * a[None, :] * mq, dtype=np.complex128)
        dcoef: ComplexArray = np.asarray(
            1j * (da[None, :] * mq - a[None, :] * n * dq[None, :]), dtype=np.complex128
        )
        vr = dcoef[:, :, None] * f + coef[:, :, None] * fr
        vt = coef[:, :, None] * ft
        data["KV"].append(bilinear_k(vr, vt, fr, ft))
        # Q(w, psi) = -int J0 (psi_r conj(w)_t - psi_t conj(w)_r) dr dtheta  (J cancels the 1/J of b1)
        wj = wq[:, None] * j0
        data["Q"].append(
            -(
                np.einsum("igt,jgt,gt->ij", np.conj(ft), fr, wj)
                - np.einsum("igt,jgt,gt->ij", np.conj(fr), ft, wj)
            )
        )
        local = np.array(
            [(2 * e + slot) * n_m + mi for slot in range(4) for mi in range(n_m)], dtype=np.int64
        )
        rows.append(np.repeat(local, local.size))
        cols.append(np.tile(local, local.size))
    row, col = np.concatenate(rows), np.concatenate(cols)
    fixed: set[int] = set()
    last = x_nodes.size - 1
    for mi, m in enumerate(m_values):
        fixed.add((2 * last) * n_m + mi)  # wall value
        fixed.add(0 * n_m + mi)  # axis value
        if m >= 2:
            fixed.add(1 * n_m + mi)  # axis slope
    free = np.array(sorted(set(range(n_field)) - fixed), dtype=np.int64)
    mats = {
        k: sp.coo_matrix(
            (np.concatenate([b.ravel() for b in v]), (row, col)), shape=(n_field, n_field)
        ).tocsr()[free][:, free]
        for k, v in data.items()
    }
    zero = sp.csr_matrix(mats["M"].shape)
    a_mat = sp.bmat(
        [[-eta * mats["K"], mats["F"]], [-(mats["KV"] + mats["Q"]), zero]], format="csc"
    )
    b_mat = sp.bmat([[mats["M"], zero], [zero, mats["K"]]], format="csc")
    return CoupledDiscretisation(
        nodes=x_nodes, a=a_mat, b=b_mat, m_values=tuple(m_values), free=free
    )


def harmonic_fractions(disc: CoupledDiscretisation, eigenvalue: complex) -> dict[int, float]:
    r"""Return the share of the flux-perturbation nodal values carried by each harmonic.

    The eigenvector nearest ``eigenvalue`` is computed by one shift-and-invert
    Arnoldi solve; shares are the squared moduli of the nodal values of
    :math:`\psi_m` (slope dofs excluded), normalised to sum to one.
    """
    shift = complex(eigenvalue) * (1.0 + 1e-7)
    factor = spla.splu((disc.a - shift * disc.b).tocsc())
    operator = spla.LinearOperator(
        disc.a.shape, matvec=lambda v: factor.solve(disc.b @ v), dtype=np.complex128
    )
    _, vectors = spla.eigs(operator, k=1, which="LM", tol=1e-10, ncv=40, maxiter=3000)
    psi = vectors[: disc.free.size, 0]
    n_m = len(disc.m_values)
    energy = np.zeros(n_m)
    for index, value in zip(disc.free, psi, strict=True):
        node_slot, mi = divmod(int(index), n_m)
        if node_slot % 2 == 0:
            energy[mi] += abs(value) ** 2
    total = float(energy.sum())
    return {m: float(e / total) for m, e in zip(disc.m_values, energy, strict=True)}


def coupled_growth_rate(
    cyl: ShapedCylinder,
    eta: float,
    n: int,
    m_values: list[int],
    *,
    resonant_r: float,
    n_side: int = 120,
    sigma_min: float = 1e-12,
) -> float:
    """Largest unstable Re(lambda) for toroidal number ``n`` and harmonics ``m_values``."""
    nodes = clustered_grid(0.0, 1.0, resonant_r, 0.02 * eta ** (1.0 / 3.0), n_side)
    lam = most_unstable(assemble_coupled(cyl, nodes, eta, n, m_values), sigma_min=sigma_min)
    return 0.0 if lam is None else lam.real
