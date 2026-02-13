# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — TEMHD Peltier
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt

class TEMHD_Stabilizer:
    """
    Implicit 1D Heat Solver for TEMHD-stabilized Liquid Metal Divertors.
    """
    def __init__(self, layer_thickness_mm=5.0, B_field=10.0):
        self.L = layer_thickness_mm / 1000.0
        self.B0 = B_field
        self.N = 50
        self.z = np.linspace(0, self.L, self.N)
        self.dz = self.z[1] - self.z[0]
        self.rho = 500.0
        self.cp = 4200.0
        self.k_thermal = 50.0
        self.Seebeck = 20e-6
        self.sigma = 3e6
        self.viscosity = 1e-3
        self.T = np.ones(self.N) * 300.0
        self.T_wall = 300.0

    def solve_tridiagonal(self, a, b, c, d):
        """Solve tridiagonal system Ax=d via Thomas algorithm.

        Parameters
        ----------
        a : sub-diagonal (n-1)
        b : diagonal (n)
        c : super-diagonal (n-1)
        d : rhs (n)
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        c = np.asarray(c, dtype=float)
        d = np.asarray(d, dtype=float)

        n = int(d.size)
        if b.size != n:
            raise ValueError(f"b length {b.size} must equal d length {n}")
        if n == 0:
            return np.array([], dtype=float)
        if a.size != max(n - 1, 0) or c.size != max(n - 1, 0):
            raise ValueError(
                f"Invalid tridiagonal sizes: len(a)={a.size}, len(b)={b.size}, "
                f"len(c)={c.size}, len(d)={n}"
            )

        if n == 1:
            if abs(b[0]) < 1e-14:
                raise ValueError("Singular diagonal encountered in tridiagonal solve.")
            return np.array([d[0] / b[0]], dtype=float)

        c_prime = np.zeros(n - 1, dtype=float)
        d_prime = np.zeros(n, dtype=float)

        den = b[0]
        if abs(den) < 1e-14:
            raise ValueError("Singular diagonal encountered in tridiagonal solve.")
        c_prime[0] = c[0] / den
        d_prime[0] = d[0] / den

        for i in range(1, n):
            den = b[i] - a[i - 1] * c_prime[i - 1]
            if abs(den) < 1e-14:
                raise ValueError("Singular diagonal encountered in tridiagonal solve.")
            if i < n - 1:
                c_prime[i] = c[i] / den
            d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / den

        res = np.zeros(n, dtype=float)
        res[-1] = d_prime[-1]
        for i in range(n - 2, -1, -1):
            res[i] = d_prime[i] - c_prime[i] * res[i + 1]
        return res

    def step(self, heat_flux_MW_m2, dt=0.1):
        dt = float(dt)
        heat_flux_MW_m2 = float(heat_flux_MW_m2)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be a finite positive value.")
        if not np.isfinite(heat_flux_MW_m2) or heat_flux_MW_m2 < 0.0:
            raise ValueError("heat_flux_MW_m2 must be a finite non-negative value.")
        if not np.isfinite(self.dz) or self.dz <= 0.0:
            raise ValueError("Invalid grid spacing dz in TEMHD solver.")
        if not np.all(np.isfinite(self.T)):
            raise ValueError("Temperature state contains non-finite values.")

        grad_T = np.gradient(self.T, self.dz)
        J_te = -self.sigma * self.Seebeck * grad_T
        F_lorentz = np.abs(J_te * self.B0)
        # Empirical scaling for TEMHD convection: k_eff = k * (1 + C * gradT * B)
        # Based on Jaworski et al. (2013)
        v_conv = (F_lorentz * self.dz**2) / (self.viscosity + 1e-9)
        alpha = self.k_thermal / (self.rho * self.cp)
        Pe = np.clip(v_conv * self.dz / alpha, 0, 200.0)
        k_eff = np.maximum(self.k_thermal * (1.0 + 0.2 * Pe), 1e-9)
        
        r = (k_eff * dt) / (self.rho * self.cp * self.dz**2)
        if not np.all(np.isfinite(r)):
            raise ValueError("Non-finite diffusion coefficients encountered.")
        # Matrix diagonals
        b = 1.0 + 2.0 * r[1:]
        a = -r[2:] # Sub
        c = -r[1:-1] # Super
        d = self.T[1:].copy()
        
        # BCs
        d[0] += r[1] * self.T_wall
        b[-1] = 1.0 + r[-1]
        q_in = heat_flux_MW_m2 * 1e6
        d[-1] += r[-1] * (q_in * self.dz / k_eff[-1])
        
        self.T[1:] = self.solve_tridiagonal(a, b, c, d)
        return self.T[-1], np.max(k_eff)

def run_temhd_experiment():
    sim_pel = TEMHD_Stabilizer(B_field=10.0)
    flux_ramp = np.linspace(0, 100, 20)
    
    res_T = []
    print(f"{'Flux':<10} | {'T_surf':<10}")
    for q in flux_ramp:
        for _ in range(20): T, _ = sim_pel.step(q, dt=0.5)
        res_T.append(T)
        if int(q) % 20 == 0: print(f"{q:<10.1f} | {T:<10.1f}")
    
    plt.plot(flux_ramp, res_T, 'r-')
    plt.axhline(1342, color='k', ls='--')
    plt.savefig("TEMHD_Corrected.png")

if __name__ == "__main__":
    run_temhd_experiment()
