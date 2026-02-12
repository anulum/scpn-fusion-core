import numpy as np
import matplotlib.pyplot as plt
try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel
import sys
import os

# --- FLIGHT PARAMETERS ---
SHOT_DURATION = 50 # Time steps
TARGET_R = 6.2     # Target Major Radius
TARGET_Z = 0.0     # Target Vertical Position
TARGET_ELONGATION = 1.7 

class IsoFluxController:
    """
    Simulates the Plasma Control System (PCS).
    Uses PID loops to adjust Coil Currents to maintain plasma shape.
    """
    def __init__(self, config_file):
        self.kernel = FusionKernel(config_file)
        self.history = {'t': [], 'Ip': [], 'R_axis': [], 'Z_axis': [], 'X_point': []}
        
        # PID Gains for Position Control
        # Radial Control (Horizontal) -> Controlled by Outer Coils (PF2, PF3, PF4)
        self.pid_R = {'Kp': 2.0, 'Ki': 0.1, 'Kd': 0.5, 'err_sum': 0, 'last_err': 0}
        
        # Vertical Control (Z-pos) -> Controlled by Top/Bottom diff (PF1 vs PF5)
        self.pid_Z = {'Kp': 5.0, 'Ki': 0.2, 'Kd': 2.0, 'err_sum': 0, 'last_err': 0}

    def pid_step(self, pid, error):
        pid['err_sum'] += error
        d_err = error - pid['last_err']
        pid['last_err'] = error
        return (pid['Kp'] * error) + (pid['Ki'] * pid['err_sum']) + (pid['Kd'] * d_err)

    def run_shot(self):
        print("--- INITIATING TOKAMAK FLIGHT SIMULATOR ---")
        print("Scenario: Current Ramp-Up & Divertor Formation")
        
        # Initial Solve
        self.kernel.solve_equilibrium()
        
        # Physics Evolution Loop
        for t in range(SHOT_DURATION):
            # 1. EVOLVE PHYSICS (Scenario)
            # Ramp up plasma current
            target_Ip = 5.0 + (10.0 * t / SHOT_DURATION) # 5MA -> 15MA
            self.kernel.cfg['physics']['plasma_current_target'] = target_Ip
            
            # Increase Pressure (Heating) -> This pushes plasma outward (Shafranov Shift)
            # The controller must fight this drift!
            beta_increase = 1.0 + (0.05 * t)
            
            # 2. MEASURE STATE (Diagnostics)
            # Find current axis
            idx_max = np.argmax(self.kernel.Psi)
            iz, ir = np.unravel_index(idx_max, self.kernel.Psi.shape)
            curr_R = self.kernel.R[ir]
            curr_Z = self.kernel.Z[iz]
            
            # Find X-point (Divertor)
            xp_pos, _ = self.kernel.find_x_point(self.kernel.Psi)
            
            # 3. COMPUTE CONTROL (Iso-Flux)
            err_R = TARGET_R - curr_R
            err_Z = TARGET_Z - curr_Z
            
            # Control Actions (Current Deltas)
            ctrl_radial = self.pid_step(self.pid_R, err_R)
            ctrl_vertical = self.pid_step(self.pid_Z, err_Z)
            
            # 4. ACTUATE COILS (Map Control -> Coils)
            # Radial Correction: If R is too small (Inner), Push with Outer Coils
            # PF3 is the main pusher
            self.kernel.cfg['coils'][2]['current'] += ctrl_radial 
            
            # Vertical Correction: Differential pull
            self.kernel.cfg['coils'][0]['current'] -= ctrl_vertical # Top
            self.kernel.cfg['coils'][4]['current'] += ctrl_vertical # Bottom
            
            # 5. SOLVE NEW EQUILIBRIUM
            # We use the previous solution as guess (hot start) -> Faster
            self.kernel.solve_equilibrium()
            
            # Log
            self.history['t'].append(t)
            self.history['Ip'].append(target_Ip)
            self.history['R_axis'].append(curr_R)
            self.history['Z_axis'].append(curr_Z)
            self.history['X_point'].append(xp_pos)
            
            print(f"Time {t}: Ip={target_Ip:.1f}MA | Axis=({curr_R:.2f}, {curr_Z:.2f}) | Ctrl_R={ctrl_radial:.2f}")

        self.visualize_flight()

    def visualize_flight(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Trajectory Plot
        ax1.set_title("Plasma Trajectory Control")
        ax1.plot(self.history['t'], self.history['R_axis'], 'b-', label='R Axis (Radial)')
        ax1.plot(self.history['t'], self.history['Z_axis'], 'r-', label='Z Axis (Vertical)')
        
        # Targets
        ax1.axhline(TARGET_R, color='b', linestyle='--', alpha=0.5, label='Target R')
        ax1.axhline(TARGET_Z, color='r', linestyle='--', alpha=0.5, label='Target Z')
        
        ax1.set_xlabel("Shot Time (a.u.)")
        ax1.set_ylabel("Position (m)")
        ax1.legend()
        ax1.grid(True)
        
        # 2. X-Point Evolution (Divertor Stability)
        rx = [p[0] for p in self.history['X_point']]
        rz = [p[1] for p in self.history['X_point']]
        
        # Filter out 0,0 (Limiter phase)
        valid_idx = [i for i, x in enumerate(rx) if x > 1.0]
        if valid_idx:
            ax2.plot([rx[i] for i in valid_idx], [rz[i] for i in valid_idx], 'g-o', markersize=4)
            ax2.set_title("Divertor X-Point Movement")
            ax2.set_xlabel("R (m)")
            ax2.set_ylabel("Z (m)")
            ax2.grid(True)
            
            # Draw final shape
            ax2.contour(self.kernel.RR, self.kernel.ZZ, self.kernel.Psi, levels=10, colors='k', alpha=0.2)
        else:
            ax2.text(0.5, 0.5, "Plasma Remained Limited (No Divertor)", ha='center')

        plt.tight_layout()
        plt.savefig("Tokamak_Flight_Report.png")
        print("Flight Sim Complete. Report: Tokamak_Flight_Report.png")

if __name__ == "__main__":
    # Path to existing config
    cfg_path = "03_CODE/SCPN-Fusion-Core/iter_config.json"
    sim = IsoFluxController(cfg_path)
    sim.run_shot()
