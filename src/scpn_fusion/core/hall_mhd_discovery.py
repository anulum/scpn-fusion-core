# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Hall MHD Discovery
"""Reduced Hall-MHD discovery workflow for zonal-flow and tearing diagnostics.

The implementation is intentionally lightweight and deterministic so that CI and
local benchmark tasks can exercise a full discovery-style simulation path:

* semi-spectral Hall-MHD dynamics in periodic geometry,
* short parameter sweeps for ``eta``/``nu`` response grids,
* bisection-style tearing threshold search, and
* automated visual output for inspection.
"""

from __future__ import annotations

import logging
from typing import Protocol, Sequence, cast

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

GRID = 64
L = 2 * np.pi
DT = 0.005
STEPS = 2000


class HallMHDSimulator(Protocol):
    """Sim-loop protocol shared by the NumPy and Rust Hall-MHD backends."""

    def step(self) -> tuple[float, float]:
        """Advance one step, returning ``(total_energy, zonal_energy)``."""
        ...

    @property
    def energy_history(self) -> Sequence[float]:
        """Per-step total-energy history."""
        ...


def create_hall_mhd(
    N: int = GRID,
    eta: float | None = None,
    nu: float | None = None,
    *,
    seed: int | None = None,
    background_amplitude: float = 0.0,
) -> HallMHDSimulator:
    """Return the fastest available Hall-MHD discovery simulator.

    Dispatches Rust → NumPy through the class-kernel registry. Both tiers
    implement the same reconciled reduced Hall-MHD model; trajectories are
    statistically equivalent (language-native seeded RNG streams), and the
    returned object satisfies the :class:`HallMHDSimulator` sim-loop protocol.

    Parameters
    ----------
    N : int
        Grid size per dimension.
    eta : float, optional
        Resistivity; backend default ``1e-4`` when omitted.
    nu : float, optional
        Hyper-viscosity; backend default ``1e-4`` when omitted.
    seed : int, optional
        Per-backend deterministic RNG seed for reproducible replay.
    background_amplitude : float
        Static current-sheet amplitude ``A`` in ``psi_0 = A cos(x)``; zero
        keeps the unforced decay sandbox.

    Returns
    -------
    HallMHDSimulator
        The fastest available backend instance.
    """
    from scpn_fusion.core._multi_compat import dispatch_kernel_class

    simulator_cls = dispatch_kernel_class("hall_mhd_discovery")
    return cast(
        HallMHDSimulator,
        simulator_cls(N, eta, nu, seed=seed, background_amplitude=background_amplitude),
    )


def spitzer_resistivity(T_e_eV: float, Z_eff: float = 1.0, ln_lambda: float = 17.0) -> float:
    """Spitzer resistivity [Ohm*m]. eta = 1.65e-9 * Z_eff * ln_lambda / T_e^1.5."""
    if T_e_eV <= 0:
        return 1e-4
    return float(1.65e-9 * Z_eff * ln_lambda / (T_e_eV**1.5))


class HallMHD:
    """3D-like Reduced Hall-MHD with magnetic flutter and shear flows.

    Fields: phi (stream function), psi (magnetic flux),
    U (vorticity), J (current density).

    An optional static background flux ``psi_0 = background_amplitude * cos(x)``
    (a doubly-periodic current sheet) provides the classic reduced tearing-mode
    drive: perturbations grow by reconnection at the resonant surfaces while
    the background is treated as externally sustained (only the perturbation is
    resistively dissipated). With ``background_amplitude = 0`` the model is the
    unforced decaying discovery sandbox.
    """

    def __init__(
        self,
        N: int = GRID,
        eta: float | None = None,
        nu: float | None = None,
        *,
        seed: int | None = None,
        background_amplitude: float = 0.0,
    ) -> None:
        self.N = N
        k = np.fft.fftfreq(N, d=L / (2 * np.pi * N))
        self.kx, self.ky = np.meshgrid(k, k)
        self.k2 = self.kx**2 + self.ky**2
        self.k2[0, 0] = 1.0  # Avoid singularity

        # Hyper-viscosity mask
        kmax = np.max(k)
        self.mask = np.where(self.k2 < (2 / 3 * kmax) ** 2, 1.0, 0.0)

        # Init Random Fields (seedable for per-backend deterministic replay)
        noise = 1e-3
        rng = np.random.default_rng(seed)
        self.phi_k: ComplexArray = np.asarray(
            fft2(rng.standard_normal((N, N)) * noise) * self.mask, dtype=np.complex128
        )
        self.psi_k: ComplexArray = np.asarray(
            fft2(rng.standard_normal((N, N)) * noise) * self.mask, dtype=np.complex128
        )

        # Static background flux psi_0 = A cos(x): the tearing-mode drive.
        self.background_amplitude = float(background_amplitude)
        x = np.linspace(0.0, L, N, endpoint=False)
        xx, _ = np.meshgrid(x, x)
        self.psi0_k: ComplexArray = np.asarray(
            fft2(self.background_amplitude * np.cos(xx)) * self.mask,
            dtype=np.complex128,
        )

        # Physics Constants
        self.rho_s = 0.1  # Larmor radius (Hall scale)
        self.beta = 0.01  # Plasma Beta
        self.nu = 1e-4  # Hyper-viscosity
        if nu is not None:
            self.nu = nu
        self.eta = 1e-4  # Resistivity
        if eta is not None:
            self.eta = eta
        self.energy_history: list[float] = []

    def poisson_bracket(self, A_k: ComplexArray, B_k: ComplexArray) -> ComplexArray:
        """Compute the 2D Poisson bracket ``[A, B]`` in spectral space.

        The bracket uses FFT-domain derivatives and returns the transformed
        commutator ``dxA * dyB - dyA * dxB``.

        Parameters
        ----------
        A_k : ComplexArray
            Real- or complex-valued spectral field ``A(kx, ky)``.
        B_k : ComplexArray
            Real- or complex-valued spectral field ``B(kx, ky)``.

        Returns
        -------
        ComplexArray
            Fourier representation of the Poisson bracket ``[A, B]``.
        """
        # [A, B] = dxA dyB - dyA dxB
        dxA = ifft2(1j * self.kx * A_k)
        dyA = ifft2(1j * self.ky * A_k)
        dxB = ifft2(1j * self.kx * B_k)
        dyB = ifft2(1j * self.ky * B_k)
        return np.asarray(fft2(dxA * dyB - dyA * dxB) * self.mask, dtype=np.complex128)

    def dynamics(self, phi: ComplexArray, psi: ComplexArray) -> tuple[ComplexArray, ComplexArray]:
        """Evaluate the reduced Hall-MHD right-hand side for (phi, psi).

        dU/dt = -[phi, U] + beta*[J_tot, psi_tot] - nu*k^4*U and
        dpsi/dt = -[phi, psi_tot] + rho_s^2*[J_tot, psi_tot] - eta*k^2*psi,
        where U = del^2 phi, psi_tot = psi_0 + psi includes the optional static
        background current sheet, J_tot = del^2 psi_tot, and only the
        perturbation psi is resistively dissipated (the background is treated
        as externally sustained).
        """
        # Derivatives (totals include the optional tearing background)
        psi_tot = psi + self.psi0_k
        U = -self.k2 * phi
        J_tot = -self.k2 * psi_tot

        # Nonlinear terms
        comm_phi_U = self.poisson_bracket(phi, U)
        comm_J_psi = self.poisson_bracket(J_tot, psi_tot)
        comm_phi_psi = self.poisson_bracket(phi, psi_tot)

        # Hall term (makes it Hall-MHD)
        comm_J_psi_hall = comm_J_psi * (self.rho_s**2)

        # Vorticity Equation
        # dU/dt = -[phi, U] + beta * [J_tot, psi_tot] - nu*k^4*U
        dU_dt = -comm_phi_U + (self.beta * comm_J_psi) - (self.nu * self.k2**2 * U)

        # Ohm's Law (Magnetic Flux)
        # dpsi/dt = -[phi, psi_tot] + Hall_Term + eta*del^2 psi
        # (resistive dissipation of the perturbation: eta*del^2 -> -eta*k^2)
        dpsi_dt = -comm_phi_psi + comm_J_psi_hall - (self.eta * self.k2 * psi)

        # Invert Vorticity to get dphi/dt
        dphi_dt = -dU_dt / self.k2
        dphi_dt[0, 0] = 0.0

        return (
            np.asarray(dphi_dt, dtype=np.complex128),
            np.asarray(dpsi_dt, dtype=np.complex128),
        )

    def step(self) -> tuple[float, float]:
        """Advance one reduced Hall-MHD pseudo-time step (RK2).

        Returns
        -------
        tuple
            ``(total_energy, zonal_energy)`` for the post-step state in spectral
            coordinates. ``total_energy`` is the total potential energy proxy
            from spectral coefficients; ``zonal_energy`` accumulates non-zero
            ky=0 modes as a zonal-flow metric.
        """
        # RK2 Time stepping
        dp1, ds1 = self.dynamics(self.phi_k, self.psi_k)

        p_mid = self.phi_k + 0.5 * DT * dp1
        s_mid = self.psi_k + 0.5 * DT * ds1

        dp2, ds2 = self.dynamics(p_mid, s_mid)

        self.phi_k += DT * dp2
        self.psi_k += DT * ds2

        # Zonal Flow Energy (ky=0 modes)
        # These are the flows that kill turbulence
        # Filter where ky=0 and kx!=0
        zonal_mask = (np.abs(self.ky) < 1e-9) & (np.abs(self.kx) > 1e-9)
        zonal_energy = float(np.sum(np.abs(self.phi_k[zonal_mask]) ** 2))

        total_energy = float(np.sum(np.abs(self.phi_k) ** 2))
        self.energy_history.append(total_energy)

        return total_energy, zonal_energy

    def parameter_sweep(
        self,
        eta_range: tuple[float, float],
        nu_range: tuple[float, float],
        n_steps: int = 5,
        sim_steps: int = 200,
        *,
        seed: int | None = 0,
        background_amplitude: float = 1.0,
    ) -> dict[str, list[float]]:
        """Run a grid of driven simulations varying eta and nu, returning growth rates.

        Each grid point evolves a fresh current-sheet-driven simulation (see the
        class docstring) from a seeded initial condition, so the sweep is
        reproducible per backend. The growth rate is the mean log-slope of the
        late-time energy history.
        """
        results: dict[str, list[float]] = {"eta": [], "nu": [], "growth_rate": []}
        for eta_val in np.linspace(eta_range[0], eta_range[1], n_steps):
            for nu_val in np.linspace(nu_range[0], nu_range[1], n_steps):
                sim = create_hall_mhd(
                    self.N,
                    eta=float(eta_val),
                    nu=float(nu_val),
                    seed=seed,
                    background_amplitude=background_amplitude,
                )
                for _ in range(sim_steps):
                    sim.step()
                history = np.asarray(sim.energy_history, dtype=np.float64)
                if history.size > 10:
                    e = history[-10:]
                    growth = float(np.mean(np.diff(np.log(np.maximum(e, 1e-30)))))
                else:
                    growth = 0.0
                results["eta"].append(float(eta_val))
                results["nu"].append(float(nu_val))
                results["growth_rate"].append(growth)
        return results

    def find_tearing_threshold(
        self,
        eta_range: tuple[float, float] = (1e-6, 1e-2),
        n_bisect: int = 10,
        sim_steps: int = 500,
        *,
        seed: int | None = 0,
        background_amplitude: float = 1.0,
    ) -> dict[str, float]:
        """Bisection search for the marginal resistivity of the driven sheet.

        With the static current-sheet drive enabled, perturbation growth is
        sustained at low resistivity and suppressed once resistive dissipation
        of the perturbation dominates; the bisection brackets the empirical
        marginal ``eta`` where the late-time log-slope of the energy history
        changes sign in this box. This is an empirical sandbox threshold, not a
        literature-parity tearing growth-rate claim.
        """
        lo, hi = eta_range
        for _ in range(n_bisect):
            mid = float(np.sqrt(lo * hi))  # geometric mean
            sim = create_hall_mhd(
                self.N,
                eta=mid,
                seed=seed,
                background_amplitude=background_amplitude,
            )
            for _ in range(sim_steps):
                sim.step()
            history = np.asarray(sim.energy_history, dtype=np.float64)
            if history.size > 20:
                e = history[-20:]
                growth = float(np.mean(np.diff(np.log(np.maximum(e, 1e-30)))))
            else:
                growth = 0.0
            if growth > 0:
                lo = mid  # still growing: marginal eta lies above mid
            else:
                hi = mid  # decaying: marginal eta lies below mid
        return {"threshold_eta": float(np.sqrt(lo * hi)), "lo": lo, "hi": hi}


def run_discovery_sim() -> None:
    """Run the standalone Hall-MHD discovery demo and emit figure artifacts.

    Writes ``Hall_MHD_Discovery.png`` and ``Hall_MHD_Structure.png`` and logs
    periodic progress snapshots.
    """
    logger.info("SCPN Hall-MHD zonal-flow discovery start")
    logger.info("Searching for spontaneous H-mode transition")

    sim = HallMHD()

    history_E = []
    history_Z = []

    logger.info("Hall-MHD discovery timesteps: %d", STEPS)

    for t in range(STEPS):
        E_tot, E_zonal = sim.step()
        history_E.append(E_tot)
        history_Z.append(E_zonal)

        if t % 100 == 0:
            ratio = E_zonal / E_tot if E_tot > 0 else 0
            logger.info(
                "Hall-MHD step: step=%d total_energy=%.2e zonal_energy=%.2e zonal_percent=%.1f",
                t,
                E_tot,
                E_zonal,
                ratio * 100,
            )

    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history_E, label="Total Turbulence Energy")
    ax.plot(history_Z, label="Zonal Flow Energy")
    ax.set_yscale("log")
    ax.set_title("Spontaneous Generation of Zonal Flows (Self-Organization)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy (Spectral)")
    ax.legend()
    plt.savefig("Hall_MHD_Discovery.png")
    logger.info("Hall-MHD discovery figure saved: Hall_MHD_Discovery.png")

    # Final State
    phi_real = np.real(ifft2(sim.phi_k))
    plt.figure()
    plt.imshow(phi_real, cmap="RdBu")
    plt.title("Turbulence Potential Structure")
    plt.colorbar()
    plt.savefig("Hall_MHD_Structure.png")


if __name__ == "__main__":
    run_discovery_sim()
