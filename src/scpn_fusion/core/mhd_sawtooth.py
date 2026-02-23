# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — MHD Sawtooth
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

logger = logging.getLogger(__name__)

# Kadomtsev (1975) sawtooth crash model parameters
_CRASH_THRESHOLD = 0.1     # psi_11 amplitude triggering reconnection
_CRASH_REDUCTION = 0.01    # post-crash amplitude multiplier (99% reduction, inside q<1 only)
_Q_RECOVERY_RATE = 0.05    # exponential relaxation rate toward equilibrium q-profile


class ReducedMHD:
    """
    Simulates the internal m=1, n=1 Kink Mode (Sawtooth Instability).
    Solves Reduced MHD equations in cylindrical geometry.
    Demonstrates Magnetic Reconnection (Kadomtsev Model).
    """
    def __init__(self, nr=100):
        self.nr = nr
        self.r = np.linspace(0, 1, nr)
        self.dr = self.r[1] - self.r[0]
        
        self.psi_11 = np.zeros(nr, dtype=complex)  # m=1,n=1 flux perturbation
        self.phi_11 = np.zeros(nr, dtype=complex)  # stream function perturbation

        # q(0) < 1 required for internal kink instability
        self.q = 0.8 + 2.0 * self.r**2
        self.S = 1e4       # Lundquist number (real tokamak ~1e8)
        self.eta = 1.0 / self.S
        self.nu = 1e-4     # viscosity

        self.psi_11 = 1e-4 * self.r * (1 - self.r) * (1+1j)

    def laplacian(self, f, m=1):
        """Radial Laplacian: 1/r d/dr (r df/dr) - m^2/r^2 f"""
        # Finite difference
        d2f = np.zeros_like(f)
        d2f[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / self.dr**2
        
        # 1/r df/dr term
        df = np.zeros_like(f)
        df[1:-1] = (f[2:] - f[:-2]) / (2*self.dr)
        
        term1 = d2f
        term2 = (1.0 / self.r[1:-1]) * df[1:-1]
        term3 = -(m**2 / self.r[1:-1]**2) * f[1:-1]
        
        res = np.zeros_like(f)
        res[1:-1] = term1[1:-1] + term2 + term3
        return res

    def step(self, dt=0.01):
        """
        Time integration (Semi-Implicit)
        Equations:
        1. d(W_11)/dt = [J_eq, psi_11] + [J_11, psi_eq] + Dissipation
        2. d(psi_11)/dt = B_parallel * phi_11 + eta * J_11
        """
        # Linear growth drive: γ ~ (1/q - 1)
        growth_drive = (1.0 / self.q - 1.0)

        # k_∥ ≈ (1/q - 1)(m/R)
        k_par = (1.0 / self.q - 1.0)

        J_11 = -self.laplacian(self.psi_11)  # current perturbation
        dpsi_dt = (k_par * self.phi_11) + (self.eta * J_11)  # Ohm's law

        # Vorticity: dU/dt = k_∥ J + instability source
        U_11 = self.laplacian(self.phi_11)
        
        dU_dt = (k_par * J_11) + (growth_drive * self.psi_11)

        self.psi_11 += dpsi_dt * dt
        U_11 += dU_dt * dt
        self.phi_11 = self.solve_poisson(U_11)  # Del^2 phi = U (Thomas O(N))

        # Kadomtsev crash: reduce perturbation inside q=1 surface, flatten q-profile
        amplitude = np.max(np.abs(self.psi_11))
        crash = False
        if amplitude > _CRASH_THRESHOLD:
            logger.warning("SAWTOOTH CRASH")
            inside_q1 = self.q < 1.0
            self.psi_11[inside_q1] *= _CRASH_REDUCTION
            self.phi_11[inside_q1] *= _CRASH_REDUCTION
            self.q[self.r < 0.4] = 1.05
            crash = True

        # q-profile recovery toward equilibrium
        self.q = self.q - (self.q - (0.8 + 2.0 * self.r ** 2)) * _Q_RECOVERY_RATE
        
        return amplitude, crash

    def solve_poisson(self, U):
        """
        Solves Del^2 phi = U for phi using the Thomas Algorithm (O(N)).
        Optimized for tridiagonal Laplacian in cylindrical coordinates.
        """
        N = self.nr
        # A_i * phi_{i-1} + B_i * phi_i + C_i * phi_{i+1} = U_i
        # Using the same discretization as in the manual setup:
        # B_i = -2*coeff - 1.0/(r^2)
        # A_i = coeff - 1.0/(2*r*dr)
        # C_i = coeff + 1.0/(2*r*dr)
        
        dr = self.dr
        coeff = 1.0 / dr**2
        
        a = np.zeros(N, dtype=complex)
        b = np.ones(N, dtype=complex) # Boundary B_0=1
        c = np.zeros(N, dtype=complex)
        d = U.copy()
        
        # Dirichlet at r=0 and r=1
        d[0] = 0.0
        d[-1] = 0.0
        
        for i in range(1, N-1):
            r = max(self.r[i], 1e-10)
            a[i] = coeff - 1.0/(2*r*dr)
            b[i] = -2*coeff - 1.0/(r**2)
            c[i] = coeff + 1.0/(2*r*dr)
            
        # Thomas Algorithm
        c_prime = np.zeros(N, dtype=complex)
        d_prime = np.zeros(N, dtype=complex)
        
        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]
        
        for i in range(1, N):
            m = b[i] - a[i] * c_prime[i-1]
            if i < N-1:
                c_prime[i] = c[i] / m
            d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / m
            
        # Back substitution
        res = np.zeros(N, dtype=complex)
        res[N-1] = d_prime[N-1]
        for i in range(N-2, -1, -1):
            res[i] = d_prime[i] - c_prime[i] * res[i+1]
            
        return res

def run_sawtooth_sim():
    logger.info("--- SCPN 3D MHD: SAWTOOTH INSTABILITY ---")
    
    sim = ReducedMHD()
    
    history_amp = []
    frames = []
    
    fig, ax = plt.subplots()
    ax.set_title("Mode Amplitude (m=1, n=1)")
    line, = ax.plot([], [], 'r-')
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 0.15)
    
    # Run
    for t in range(500):
        amp, crash = sim.step()
        history_amp.append(amp)
        
        if crash:
            logger.warning("Time %d: RECONNECTION EVENT", t)
            
        if t % 5 == 0:
            line.set_data(range(len(history_amp)), history_amp)
            # Capture frame logic omitted for speed in CLI, we save plot at end
    
    plt.plot(history_amp)
    plt.xlabel("Time steps")
    plt.ylabel("Perturbation Amplitude")
    plt.title("Sawtooth Cycles (Growth -> Crash -> Recovery)")
    plt.grid(True)
    plt.savefig("MHD_Sawtooth.png")
    logger.info("Saved: MHD_Sawtooth.png")
    
    # 2D Reconstruction of the Island
    # Psi_total = Psi_eq(r) + Re[ Psi_11(r) * exp(i(theta - phi)) ]
    theta = np.linspace(0, 2*np.pi, 100)
    R_grid, Theta_grid = np.meshgrid(sim.r, theta)
    
    # Perturbation map
    Psi_pert = np.real(sim.psi_11.reshape(1, -1) * np.exp(1j * Theta_grid))
    
    plt.figure()
    plt.contourf(R_grid * np.cos(Theta_grid), R_grid * np.sin(Theta_grid), Psi_pert, cmap='RdBu')
    plt.title("m=1 Magnetic Island Structure")
    plt.colorbar()
    plt.savefig("Magnetic_Island_2D.png")
    logger.info("Saved: Magnetic_Island_2D.png")

if __name__ == "__main__":
    run_sawtooth_sim()
