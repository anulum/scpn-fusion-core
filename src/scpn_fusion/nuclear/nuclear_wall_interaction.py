# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Nuclear Wall Interaction
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Callable, Optional
from scpn_fusion.core.fusion_ignition_sim import FusionBurnPhysics
from scpn_fusion.engineering.cad_raytrace import CADLoadReport, estimate_surface_loading

# --- MATERIALS DATABASE ---
# Thresholds for neutron damage before replacement is needed
MATERIALS = {
    'Tungsten (W)': {'dpa_limit': 50.0, 'sigma_dpa': 1000}, # Divertor armor
    'Eurofer (Steel)': {'dpa_limit': 150.0, 'sigma_dpa': 500}, # Structural blanket
    'Beryllium (Be)': {'dpa_limit': 10.0, 'sigma_dpa': 200}, # First wall (old design)
}


def default_iter_config_path() -> str:
    """Resolve repository-local default ITER configuration path."""
    return str(Path(__file__).resolve().parents[3] / "iter_config.json")

class NuclearEngineeringLab(FusionBurnPhysics):
    """
    Simulates the nuclear interaction between Plasma and the Reactor Vessel.
    1. Helium Ash accumulation.
    2. Neutron Flux distribution on the First Wall.
    3. Material Damage (DPA).
    """
    def __init__(self, config_path):
        super().__init__(config_path)

    def _build_neutron_source_map(self):
        idx_max = np.argmax(self.Psi)
        iz_ax, ir_ax = np.unravel_index(idx_max, self.Psi.shape)
        psi_axis = self.Psi[iz_ax, ir_ax]
        xp, psi_x = self.find_x_point(self.Psi)
        _ = xp
        psi_edge = psi_x
        if abs(psi_edge - psi_axis) < 1.0:
            psi_edge = np.min(self.Psi)

        psi_norm = (self.Psi - psi_axis) / (psi_edge - psi_axis)
        psi_norm = np.clip(psi_norm, 0, 1)
        mask = psi_norm < 1.0

        s_peak = 1e18
        source_map = np.zeros_like(self.Psi)
        source_map[mask] = s_peak * (1.0 - psi_norm[mask]) ** 2
        return source_map
        
    def generate_first_wall(self):
        """
        Defines the geometry of the reactor wall (Vacuum Vessel).
        Approximated as a D-shaped contour surrounding the plasma.
        """
        theta = np.linspace(0, 2*np.pi, 200)
        # Wall Parameters
        R0, a, kappa, delta = 5.0, 3.0, 1.9, 0.4 
        
        # Parametric Wall
        R_wall = R0 + a * np.cos(theta + np.arcsin(delta)*np.sin(theta))
        Z_wall = kappa * a * np.sin(theta)
        
        return R_wall, Z_wall

    def simulate_ash_poisoning(
        self,
        burn_time_sec: int = 1000,
        tau_He_ratio: float = 5.0,
        pumping_efficiency: float = 1.0,
    ):
        """
        Simulates the drop in fusion power due to Helium buildup.
        tau_He_ratio: Ratio of Helium particle confinement to Energy confinement (tau_He / tau_E).
        If ratio > 10, the reactor chokes.
        """
        burn_steps = int(burn_time_sec)
        if burn_steps < 1:
            raise ValueError("burn_time_sec must be >= 1.")
        tau_He_ratio = float(tau_He_ratio)
        if not np.isfinite(tau_He_ratio) or tau_He_ratio <= 0.0:
            raise ValueError("tau_He_ratio must be finite and > 0.")
        pumping_efficiency = float(pumping_efficiency)
        if not np.isfinite(pumping_efficiency) or not (0.0 <= pumping_efficiency <= 1.0):
            raise ValueError("pumping_efficiency must be finite and in [0, 1].")

        print(f"[Nuclear] Simulating Ash Poisoning (tau_He*/tau_E = {tau_He_ratio})...")
        
        # 1. Get Base Plasma State
        self.solve_equilibrium()
        
        # Initial Conditions
        n_e_target = 1.0e20 # Electron density (Greenwald Limit constant)
        f_He = 0.0 # Helium fraction
        dt = 1.0 # Second
        
        history = {'time': [], 'P_fus': [], 'f_He': [], 'Q': []}
        
        # Volume (Approximation)
        Vol = 800.0 # m^3
        
        for t in range(burn_steps):
            # A. Composition (Quasi-neutrality constraint)
            # n_e = n_D + n_T + 2*n_He + Z_imp*n_imp
            # Assume n_D = n_T
            # n_fuel = n_e - 2*n_He
            
            n_He = f_He * n_e_target
            n_fuel = n_e_target - 2*n_He
            n_D = 0.5 * n_fuel
            n_T = 0.5 * n_fuel
            
            if n_fuel < 0:
                print("  -> Plasma Quenched (Dilution Limit)")
                break
                
            # B. Reaction Rate
            T_keV = 20.0 # Keep temp constant for this isolation study
            sigmav = self.bosch_hale_dt(T_keV)
            
            # Reaction rate per volume
            R_fus = n_D * n_T * sigmav
            
            # C. Ash Dynamics (0D equation)
            # dn_He/dt = Source(Fusion) - Sink(Transport/Pump)
            tau_E = 3.0
            tau_He = tau_He_ratio * tau_E
            
            dn_He = R_fus - (pumping_efficiency * n_He / tau_He)
            
            # Update State
            n_He += dn_He * dt
            f_He = n_He / n_e_target
            
            # D. Power Output
            E_fus = 17.6 * 1.602e-13
            P_fus_MW = (R_fus * E_fus * Vol) / 1e6
            
            history['time'].append(t)
            history['P_fus'].append(P_fus_MW)
            history['f_He'].append(f_He)
            
        return history

    def calculate_neutron_wall_loading(self):
        """
        Ray-Tracing calculation of 14.1 MeV neutrons hitting the wall.
        """
        print("[Nuclear] Calculating Neutron Wall Loading (NWL)...")
        
        # 1. Source: Plasma Grid (Fusion Power Density)
        # We reuse the thermodynamics calculation to get local emissivity
        # Recalculate power density profile
        Source_Map = self._build_neutron_source_map()
        
        # 2. Target: First Wall Segments
        Rw, Zw = self.generate_first_wall()
        wall_flux = np.zeros(len(Rw))
        
        # 3. Ray Tracing Integration (Line-of-Sight)
        # For every point on wall, sum contrib from every point in plasma
        # Flux = Sum( Source_i * dV_i / (4*pi*r^2) ) * cos(incidence)
        
        # Optimization: Downsample plasma grid for ray tracing
        step = 4
        RR_sub = self.RR[::step, ::step]
        ZZ_sub = self.ZZ[::step, ::step]
        S_sub = Source_Map[::step, ::step]
        dV = (self.dR * step) * (self.dZ * step) * 2 * np.pi * RR_sub
        
        # Flatten sources
        src_r = RR_sub.flatten()
        src_z = ZZ_sub.flatten()
        src_S = S_sub.flatten() * dV.flatten()
        
        # Filter only active plasma points
        active_idx = src_S > 0
        src_r = src_r[active_idx]
        src_z = src_z[active_idx]
        src_S = src_S[active_idx]
        
        print(f"  Ray-tracing from {len(src_r)} plasma elements to {len(Rw)} wall segments...")
        
        for i in range(len(Rw)):
            # Target point
            wx, wz = Rw[i], Zw[i]
            
            # Normal vector of wall (approx)
            if i < len(Rw)-1:
                dx, dz = Rw[i+1]-Rw[i], Zw[i+1]-Zw[i]
            else:
                dx, dz = Rw[0]-Rw[i], Zw[0]-Zw[i]
            normal = np.array([-dz, dx])
            normal /= np.linalg.norm(normal)
            
            # Vector from source to target
            vec_r = wx - src_r
            vec_z = wz - src_z
            dist_sq = vec_r**2 + vec_z**2
            dist = np.sqrt(dist_sq)
            
            # Toroidal View-Factor Correction (P1.1)
            # Neutrons from a toroidal source ring at (src_r, src_z)
            # The flux at (wx, wz) scales with 1/dist but also depends on 
            # the toroidal integral. For large R, it approaches spherical 1/r2.
            # For small R, the "inner" wall sees more flux due to curvature.
            toroidal_correction = src_r / wx # Flux enhancement on inner wall
            
            # Cosine incidence (Dot product with normal)
            # normal is [nr, nz], vec is [vec_r, vec_z]
            # normalize vec
            unit_vec_r = vec_r / dist
            unit_vec_z = vec_z / dist
            cos_theta = np.maximum(0.0, (unit_vec_r * normal[0] + unit_vec_z * normal[1]))
            
            flux_contrib = (src_S * toroidal_correction * cos_theta) / (4.0 * np.pi * dist_sq)
            
            # Sum up
            wall_flux[i] = np.sum(flux_contrib)
            
        return Rw, Zw, wall_flux

    def calculate_cad_wall_loading(
        self,
        vertices,
        faces,
        source_points_xyz=None,
        source_strength_w=None,
    ) -> CADLoadReport:
        """
        Reduced CAD loading estimate on imported STEP/STL meshes.
        """
        self.solve_equilibrium()
        if source_points_xyz is None or source_strength_w is None:
            source_map = self._build_neutron_source_map()
            # Downsample source grid to keep raytrace cheap.
            step = 5
            rr = self.RR[::step, ::step].ravel()
            zz = self.ZZ[::step, ::step].ravel()
            ss = source_map[::step, ::step].ravel()
            keep = ss > 0.0
            source_points_xyz = np.stack(
                [rr[keep], np.zeros(np.count_nonzero(keep)), zz[keep]], axis=1
            )
            # Convert neutron source proxy to Watts with 14.1 MeV per neutron.
            e_j = 14.1e6 * 1.602e-19
            cell_volume = self.dR * self.dZ * 2 * np.pi * np.maximum(rr[keep], 1e-3)
            source_strength_w = ss[keep] * cell_volume * e_j
        return estimate_surface_loading(
            np.asarray(vertices, dtype=np.float64),
            np.asarray(faces, dtype=np.int64),
            np.asarray(source_points_xyz, dtype=np.float64),
            np.asarray(source_strength_w, dtype=np.float64),
        )

    def analyze_materials(self, wall_flux):
        """
        Calculates lifespan for different materials.
        """
        # Convert Flux (n/m2/s) to MW/m2 (14 MeV per neutron)
        # 1 MeV = 1.6e-13 J
        # Flux * 14 * 1.6e-13 = Watts/m2
        MW_m2 = wall_flux * 14.1 * 1.602e-13 / 1e6 # MW/m2
        
        peak_load = np.max(MW_m2)
        avg_load = np.mean(MW_m2)
        
        print(f"[Nuclear] Wall Loading: Peak={peak_load:.2f} MW/m2, Avg={avg_load:.2f} MW/m2")
        
        results = {}
        for mat_name, props in MATERIALS.items():
            # DPA accumulation per year
            # Rule of thumb: 1 MW/m2 ~ 10 DPA/fpy (Full Power Year)
            dpa_per_year = peak_load * 10.0 
            
            lifespan = props['dpa_limit'] / dpa_per_year if dpa_per_year > 0 else 999.0
            results[mat_name] = lifespan
            
        return results, MW_m2

def run_nuclear_sim(
    config_path: Optional[str] = None,
    *,
    save_plot: bool = True,
    output_path: str = "Nuclear_Engineering_Report.png",
    verbose: bool = True,
    lab_factory: Callable[[str], Any] = NuclearEngineeringLab,
) -> dict[str, Any]:
    if config_path is None:
        config_path = default_iter_config_path()
    config_path = str(config_path)

    if verbose:
        print("--- SCPN NUCLEAR ENGINEERING: Ash & Materials ---")
        print(f"[Nuclear] Config: {config_path}")
    lab = lab_factory(config_path)
    
    # 1. Ash Simulation
    # Simulate two scenarios: Good Pumping (tau=5) vs Bad Pumping (tau=15)
    hist_good = lab.simulate_ash_poisoning(tau_He_ratio=5.0, pumping_efficiency=1.0)
    hist_bad = lab.simulate_ash_poisoning(tau_He_ratio=15.0, pumping_efficiency=0.5)
    
    # 2. Neutron Wall Load
    Rw, Zw, neutron_flux = lab.calculate_neutron_wall_loading()
    
    # 3. Material Analysis
    lifespans, mw_load = lab.analyze_materials(neutron_flux)
    
    plot_saved = False
    plot_error: Optional[str] = None
    if save_plot:
        try:
            fig = plt.figure(figsize=(15, 10))
            
            # Plot A: Ash Poisoning
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.plot(hist_good['time'], hist_good['P_fus'], 'g-', label='Good Pumping (Tau*=5)')
            ax1.plot(hist_bad['time'], hist_bad['P_fus'], 'r--', label='Bad Pumping (Tau*=15)')
            ax1.set_title("Fusion Power Evolution (Helium Poisoning)")
            ax1.set_xlabel("Burn Time (s)")
            ax1.set_ylabel("Power (MW)")
            ax1.legend()
            ax1.grid(True)
            
            # Plot B: Helium Fraction
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.plot(hist_good['time'], np.array(hist_good['f_He'])*100, 'g-', label='He % (Good)')
            ax2.plot(hist_bad['time'], np.array(hist_bad['f_He'])*100, 'r--', label='He % (Bad)')
            ax2.axhline(10.0, color='k', linestyle=':', label='Dilution Limit (10%)')
            ax2.set_title("Helium Ash Accumulation")
            ax2.set_ylabel("He Concentration (%)")
            ax2.legend()
            ax2.grid(True)
            
            # Plot C: Neutron Wall Load (Heatmap on Wall)
            ax3 = fig.add_subplot(2, 2, 3)
            # Plot Plasma Core
            ax3.contour(lab.RR, lab.ZZ, lab.Psi, levels=10, colors='gray', alpha=0.3)
            # Plot Wall colored by Load
            sc = ax3.scatter(Rw, Zw, c=mw_load, cmap='inferno', s=20)
            plt.colorbar(sc, ax=ax3, label='Neutron Load (MW/m2)')
            ax3.set_title("Neutron Flux Distribution (2D)")
            ax3.axis('equal')
            
            # Plot D: Component Lifespan
            ax4 = fig.add_subplot(2, 2, 4)
            mats = list(lifespans.keys())
            years = list(lifespans.values())
            colors = ['gray', 'orange', 'green']
            ax4.bar(mats, years, color=colors)
            ax4.set_title("First Wall Component Lifespan (at Peak Flux)")
            ax4.set_ylabel("Full Power Years (FPY)")
            ax4.axhline(5.0, color='r', linestyle='--', label='Maintenance Cycle (5y)')
            ax4.legend()
            
            plt.tight_layout()
            plt.savefig(output_path)
            plot_saved = True
        except Exception as exc:  # pragma: no cover - backend-dependent
            plot_error = f"{exc.__class__.__name__}: {exc}"

    if verbose:
        if plot_saved:
            print(f"\nResults saved: {output_path}")
        print("Material Lifespan Estimates:")
        for m, y in lifespans.items():
            print(f"  {m}: {y:.1f} years")

    good_f_he = float(hist_good["f_He"][-1]) if hist_good["f_He"] else 0.0
    bad_f_he = float(hist_bad["f_He"][-1]) if hist_bad["f_He"] else 0.0
    peak_load = float(np.max(mw_load)) if np.size(mw_load) else 0.0
    avg_load = float(np.mean(mw_load)) if np.size(mw_load) else 0.0
    lifespan_years = np.asarray(list(lifespans.values()), dtype=np.float64)
    if lifespan_years.size == 0:
        min_lifespan = 0.0
        max_lifespan = 0.0
    else:
        min_lifespan = float(np.min(lifespan_years))
        max_lifespan = float(np.max(lifespan_years))
    return {
        "config_path": config_path,
        "good_final_f_he": good_f_he,
        "bad_final_f_he": bad_f_he,
        "peak_wall_load_mw_m2": peak_load,
        "avg_wall_load_mw_m2": avg_load,
        "min_material_lifespan_years": min_lifespan,
        "max_material_lifespan_years": max_lifespan,
        "plot_saved": bool(plot_saved),
        "plot_error": plot_error,
    }

if __name__ == "__main__":
    run_nuclear_sim()
