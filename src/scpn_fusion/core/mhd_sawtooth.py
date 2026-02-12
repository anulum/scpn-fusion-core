# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — MHD Sawtooth
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import sys
import os

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
        
        # State Variables (Perturbations)
        # psi_11: Magnetic Flux perturbation (complex, for m=1, n=1 mode)
        # phi_11: Stream Function perturbation (velocity)
        self.psi_11 = np.zeros(nr, dtype=complex)
        self.phi_11 = np.zeros(nr, dtype=complex)
        
        # Equilibrium Profiles (Background)
        # q-profile: Safety factor. Instability requires q(0) < 1.
        self.q = 0.8 + 2.0 * self.r**2 
        
        # Physics Parameters
        self.S = 1e4  # Lundquist Number (Resistivity^-1). Real plasma S ~ 10^8 (too slow for demo)
        self.eta = 1.0 / self.S # Resistivity
        self.nu = 1e-4 # Viscosity
        
        # Initialize small perturbation
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
        # 1. Equilibrium Quantities
        # J_eq_z ~ 1/r d/dr (r B_theta) ~ 1/r d/dr (r^2/q)
        # Simplified linear growth rate drive:
        # gamma ~ (1/q - 1)
        
        growth_drive = (1.0 / self.q - 1.0)
        
        # 2. Field Evolution (Ohm's Law)
        # dpsi/dt = k_parallel * phi - eta * J
        # k_parallel = (1/q - 1) * (m/R) approx
        
        k_par = (1.0 / self.q - 1.0)
        
        # Current perturbation J = -Del^2 psi
        J_11 = -self.laplacian(self.psi_11)
        
        dpsi_dt = (k_par * self.phi_11) + (self.eta * J_11)
        
        # 3. Vorticity Evolution (Equation of Motion)
        # U = Del^2 phi
        # dU/dt = k_par * J + Viscosity
        U_11 = self.laplacian(self.phi_11)
        
        dU_dt = (k_par * J_11) + (growth_drive * self.psi_11) # Instability source
        
        # Update
        self.psi_11 += dpsi_dt * dt
        U_11 += dU_dt * dt
        
        # Invert U -> Phi (Poisson Solver)
        # Del^2 phi = U.  Using simplified relaxation or direct solve.
        # Since Laplacian matrix is Tridiagonal, we can solve fast.
        # Here we use spectral relaxation for code compactness
        self.phi_11 = self.solve_poisson(U_11)
        
        # Nonlinear Saturation (The Crash)
        # If amplitude gets too large, it flattens the profile (simulated)
        amplitude = np.max(np.abs(self.psi_11))
        crash = False
        if amplitude > 0.1:
            print("  >>> SAWTOOTH CRASH! <<<")
            self.psi_11 *= 0.1 # Collapse
            self.q[self.r < 0.4] = 1.05 # Flatten q-profile (Reconnection)
            crash = True
            
        # q-profile recovery (Heating)
        self.q = self.q - (self.q - (0.8 + 2.0*self.r**2)) * 0.05
        
        return amplitude, crash

    def solve_poisson(self, U):
        # Solves Del^2 phi = U for phi
        # Trivial approx: phi ~ -U * r^2 (very rough)
        # Better: Tridiagonal solve.
        # Construct Matrix A
        N = self.nr
        A = np.zeros((N, N), dtype=complex)
        
        coeff = 1.0 / self.dr**2
        
        for i in range(1, N-1):
            r = self.r[i]
            A[i, i-1] = coeff - 1.0/(2*r*self.dr)
            A[i, i]   = -2*coeff - 1.0/(r**2)
            A[i, i+1] = coeff + 1.0/(2*r*self.dr)
            
        # Boundary
        A[0,0] = 1; A[-1,-1] = 1
        
        res = np.linalg.solve(A, U)
        return res

def run_sawtooth_sim():
    print("--- SCPN 3D MHD: SAWTOOTH INSTABILITY ---")
    
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
            print(f"Time {t}: RECONNECTION EVENT")
            
        if t % 5 == 0:
            line.set_data(range(len(history_amp)), history_amp)
            # Capture frame logic omitted for speed in CLI, we save plot at end
    
    plt.plot(history_amp)
    plt.xlabel("Time steps")
    plt.ylabel("Perturbation Amplitude")
    plt.title("Sawtooth Cycles (Growth -> Crash -> Recovery)")
    plt.grid(True)
    plt.savefig("MHD_Sawtooth.png")
    print("Saved: MHD_Sawtooth.png")
    
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
    print("Saved: Magnetic_Island_2D.png")

if __name__ == "__main__":
    run_sawtooth_sim()
