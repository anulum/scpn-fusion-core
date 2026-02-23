# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Hall MHD Discovery
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
import sys

GRID = 64
L = 2 * np.pi
DT = 0.005
STEPS = 2000

def spitzer_resistivity(T_e_eV, Z_eff=1.0, ln_lambda=17.0):
    """Spitzer resistivity [Ohm*m]. eta = 1.65e-9 * Z_eff * ln_lambda / T_e^1.5"""
    if T_e_eV <= 0:
        return 1e-4
    return 1.65e-9 * Z_eff * ln_lambda / (T_e_eV ** 1.5)


class HallMHD:
    """
    3D-like Reduced Hall-MHD.
    Includes Magnetic Flutter (Psi perturbation) and Shear Flows.
    Fields:
      phi (Stream function / Potential)
      psi (Magnetic Flux)
      U   (Vorticity)
      J   (Current Density)
    """
    def __init__(self, N=GRID, eta=None, nu=None):
        self.N = N
        k = np.fft.fftfreq(N, d=L/(2*np.pi*N))
        self.kx, self.ky = np.meshgrid(k, k)
        self.k2 = self.kx**2 + self.ky**2
        self.k2[0,0] = 1.0 # Avoid singularity
        
        # Hyper-viscosity mask
        kmax = np.max(k)
        self.mask = np.where(self.k2 < (2/3 * kmax)**2, 1.0, 0.0)
        
        # Init Random Fields
        noise = 1e-3
        self.phi_k = fft2(np.random.randn(N,N) * noise) * self.mask
        self.psi_k = fft2(np.random.randn(N,N) * noise) * self.mask
        
        # Physics Constants
        self.rho_s = 0.1  # Larmor radius (Hall scale)
        self.beta = 0.01  # Plasma Beta
        self.nu = 1e-4    # Viscosity
        if nu is not None: self.nu = nu
        self.eta = 1e-4   # Resistivity
        if eta is not None: self.eta = eta
        self.energy_history = []

    def poisson_bracket(self, A_k, B_k):
        # [A, B] = dxA dyB - dyA dxB
        dxA = ifft2(1j * self.kx * A_k)
        dyA = ifft2(1j * self.ky * A_k)
        dxB = ifft2(1j * self.kx * B_k)
        dyB = ifft2(1j * self.ky * B_k)
        return fft2(dxA * dyB - dyA * dxB) * self.mask

    def dynamics(self, phi, psi):
        """
        Reduced MHD Equations:
        d(U)/dt = [phi, U] + [J, psi] + nu*del^4 U
        d(psi)/dt = [phi, psi] + rho_s^2 [J, psi] - eta*J
        where U = del^2 phi, J = del^2 psi
        """
        # Derivatives
        U = -self.k2 * phi
        J = -self.k2 * psi
        
        # Nonlinear terms
        comm_phi_U = self.poisson_bracket(phi, U)
        comm_J_psi = self.poisson_bracket(J, psi)
        comm_phi_psi = self.poisson_bracket(phi, psi)
        
        # Hall term (makes it Hall-MHD)
        comm_J_psi_hall = comm_J_psi * (self.rho_s**2)
        
        # Vorticity Equation
        # dU/dt = -[phi, U] + beta * [J, psi] + dissipation
        dU_dt = -comm_phi_U + (self.beta * comm_J_psi) - (self.nu * self.k2**2 * U)
        
        # Ohm's Law (Magnetic Flux)
        # dpsi/dt = -[phi, psi] + Hall_Term - eta*J
        dpsi_dt = -comm_phi_psi + comm_J_psi_hall + (self.eta * self.k2 * psi) # + eta*J means -eta*del2psi
        
        # Invert Vorticity to get dphi/dt
        dphi_dt = -dU_dt / self.k2
        dphi_dt[0,0] = 0.0
        
        return dphi_dt, dpsi_dt

    def step(self):
        # RK2 Time stepping
        dp1, ds1 = self.dynamics(self.phi_k, self.psi_k)
        
        p_mid = self.phi_k + 0.5*DT*dp1
        s_mid = self.psi_k + 0.5*DT*ds1
        
        dp2, ds2 = self.dynamics(p_mid, s_mid)
        
        self.phi_k += DT * dp2
        self.psi_k += DT * ds2
        
        # Zonal Flow Energy (ky=0 modes)
        # These are the flows that kill turbulence
        # Filter where ky=0 and kx!=0
        zonal_mask = (np.abs(self.ky) < 1e-9) & (np.abs(self.kx) > 1e-9)
        zonal_energy = np.sum(np.abs(self.phi_k[zonal_mask])**2)
        
        total_energy = np.sum(np.abs(self.phi_k)**2)
        self.energy_history.append(total_energy)

        return total_energy, zonal_energy

    def parameter_sweep(self, eta_range, nu_range, n_steps=5, sim_steps=200):
        """Run grid of simulations varying eta and nu. Returns dict with growth rates."""
        results = {'eta': [], 'nu': [], 'growth_rate': []}
        for eta_val in np.linspace(eta_range[0], eta_range[1], n_steps):
            for nu_val in np.linspace(nu_range[0], nu_range[1], n_steps):
                sim = HallMHD(self.N, eta=eta_val, nu=nu_val)
                for _ in range(sim_steps):
                    sim.step()
                if len(sim.energy_history) > 10:
                    e = np.array(sim.energy_history[-10:])
                    growth = np.mean(np.diff(np.log(np.maximum(e, 1e-30))))
                else:
                    growth = 0.0
                results['eta'].append(eta_val)
                results['nu'].append(nu_val)
                results['growth_rate'].append(growth)
        return results

    def find_tearing_threshold(self, eta_range=(1e-6, 1e-2), n_bisect=10, sim_steps=500):
        """Bisection search for marginal tearing stability threshold."""
        lo, hi = eta_range
        for _ in range(n_bisect):
            mid = np.sqrt(lo * hi)  # geometric mean
            sim = HallMHD(self.N, eta=mid)
            for _ in range(sim_steps):
                sim.step()
            if len(sim.energy_history) > 20:
                e = np.array(sim.energy_history[-20:])
                growth = np.mean(np.diff(np.log(np.maximum(e, 1e-30))))
            else:
                growth = 0.0
            if growth > 0:
                hi = mid
            else:
                lo = mid
        return {'threshold_eta': np.sqrt(lo * hi), 'lo': lo, 'hi': hi}

def run_discovery_sim():
    print("--- SCPN HALL-MHD: ZONAL FLOW DISCOVERY ---")
    print("Searching for spontaneous H-Mode transition...")
    
    sim = HallMHD()
    
    history_E = []
    history_Z = []
    
    print(f"Running {STEPS} timesteps...")
    
    for t in range(STEPS):
        E_tot, E_zonal = sim.step()
        history_E.append(E_tot)
        history_Z.append(E_zonal)
        
        if t % 100 == 0:
            ratio = E_zonal / E_tot if E_tot > 0 else 0
            print(f"Step {t}: Total E={E_tot:.2e} | Zonal E={E_zonal:.2e} ({ratio*100:.1f}%)")
            
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history_E, label='Total Turbulence Energy')
    ax.plot(history_Z, label='Zonal Flow Energy')
    ax.set_yscale('log')
    ax.set_title("Spontaneous Generation of Zonal Flows (Self-Organization)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy (Spectral)")
    ax.legend()
    plt.savefig("Hall_MHD_Discovery.png")
    print("Saved: Hall_MHD_Discovery.png")
    
    # Final State
    phi_real = np.real(ifft2(sim.phi_k))
    plt.figure()
    plt.imshow(phi_real, cmap='RdBu')
    plt.title("Turbulence Potential Structure")
    plt.colorbar()
    plt.savefig("Hall_MHD_Structure.png")

if __name__ == "__main__":
    run_discovery_sim()
