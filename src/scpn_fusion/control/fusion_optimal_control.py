import numpy as np
import matplotlib.pyplot as plt
try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel
import sys
import os

# --- MISSION PARAMETERS ---
TARGET_R = 6.0
TARGET_Z = 0.0
SHOT_STEPS = 50

class OptimalController:
    """
    MIMO (Multiple Input Multiple Output) Controller using Linear Response Matrix.
    Standard technique for ITER Shape Control.
    """
    def __init__(self, config_file):
        self.kernel = FusionKernel(config_file)
        self.n_coils = len(self.kernel.cfg['coils'])
        self.coil_names = [c['name'] for c in self.kernel.cfg['coils']]
        
        # Jacobian Matrix: d(PlasmaState)/d(CoilCurrents)
        # State Vector = [R_axis, Z_axis]
        self.response_matrix = np.zeros((2, self.n_coils)) 
        
    def identify_system(self):
        """
        Perturbs each coil individually to measure plasma response.
        Builds the linearized physics model (The Jacobian).
        """
        print("[OptControl] Identifying System Response Matrix...")
        
        # 1. Base Equilibrium
        self.kernel.solve_equilibrium()
        base_R, base_Z = self.get_plasma_pos()
        print(f"  Base Position: R={base_R:.3f}, Z={base_Z:.3f}")
        
        perturbation = 0.5 # Amps
        
        for i in range(self.n_coils):
            # Save original current
            orig_I = self.kernel.cfg['coils'][i]['current']
            
            # Perturb +
            self.kernel.cfg['coils'][i]['current'] += perturbation
            self.kernel.solve_equilibrium()
            pos_plus = self.get_plasma_pos()
            
            # Perturb - (Central Difference for better accuracy)
            self.kernel.cfg['coils'][i]['current'] = orig_I - perturbation
            self.kernel.solve_equilibrium()
            pos_minus = self.get_plasma_pos()
            
            # Restore
            self.kernel.cfg['coils'][i]['current'] = orig_I
            self.kernel.solve_equilibrium() # Reset physics state
            
            # Calculate Gradients (Sensitivity)
            dR_dI = (pos_plus[0] - pos_minus[0]) / (2 * perturbation)
            dZ_dI = (pos_plus[1] - pos_minus[1]) / (2 * perturbation)
            
            self.response_matrix[0, i] = dR_dI
            self.response_matrix[1, i] = dZ_dI
            
            print(f"  Coil {self.coil_names[i]}: dR/dI={dR_dI:.4f}, dZ/dI={dZ_dI:.4f}")
            
        print("[OptControl] System Identification Complete.")
        
    def get_plasma_pos(self):
        """Returns (R_axis, Z_axis)"""
        idx_max = np.argmax(self.kernel.Psi)
        iz, ir = np.unravel_index(idx_max, self.kernel.Psi.shape)
        return np.array([self.kernel.R[ir], self.kernel.Z[iz]])

    def compute_optimal_correction(self, current_pos, target_pos):
        """
        Solves: Error = J * Delta_I
        Delta_I = pinv(J) * Error
        Uses SVD for stable inversion.
        """
        error = target_pos - current_pos
        
        # SVD Inverse (Pseudoinverse)
        # We limit the magnitude of correction to avoid coil stress
        U, S, Vt = np.linalg.svd(self.response_matrix, full_matrices=False)
        
        # Tikhonov Regularization (Damping small singular values)
        limit = 1e-2
        S_inv = np.array([1/s if s > limit else 0 for s in S])
        
        # J_inv = V * S_inv * U_transpose
        J_inv = np.dot(Vt.T, np.dot(np.diag(S_inv), U.T))
        
        delta_currents = np.dot(J_inv, error)
        
        # Gain clamping (Safety)
        delta_currents = np.clip(delta_currents, -5.0, 5.0)
        
        return delta_currents

    def run_optimal_shot(self):
        print("\n--- INITIATING OPTIMAL CONTROL SHOT ---")
        
        # History
        h_t, h_R, h_Z, h_Ip = [], [], [], []
        
        # Run Shot
        for t in range(SHOT_STEPS):
            # 1. Scenario: Ramp Pressure & Current
            # We aggressively increase Beta (Pressure) to provoke Shafranov Shift
            target_Ip = 10.0 + (5.0 * t / SHOT_STEPS)
            self.kernel.cfg['physics']['plasma_current_target'] = target_Ip
            
            # 2. Measure
            curr_pos = self.get_plasma_pos()
            target_vec = np.array([TARGET_R, TARGET_Z])
            
            # 3. Solve Optimal Control
            # "How do I fix this error instantly?"
            dI = self.compute_optimal_correction(curr_pos, target_vec)
            
            # Apply (Proportional Feedback)
            gain = 0.8 # Learning rate
            for i in range(self.n_coils):
                self.kernel.cfg['coils'][i]['current'] += dI[i] * gain
                
            # 4. Evolve Physics
            self.kernel.solve_equilibrium()
            
            # Log
            h_t.append(t)
            h_R.append(curr_pos[0])
            h_Z.append(curr_pos[1])
            h_Ip.append(target_Ip)
            
            err = np.linalg.norm(target_vec - curr_pos)
            print(f"Step {t}: R={curr_pos[0]:.3f} (Tgt {TARGET_R}), Z={curr_pos[1]:.3f} | Err={err:.4f} | Max dI={np.max(np.abs(dI)):.2f}")

        self.plot_telemetry(h_t, h_R, h_Z, h_Ip)

    def plot_telemetry(self, t, R, Z, Ip):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Position Control Quality
        ax1.set_title("Optimal Position Control (SVD-MIMO)")
        ax1.plot(t, R, 'b-o', label='R Axis')
        ax1.plot(t, Z, 'r-s', label='Z Axis')
        ax1.axhline(TARGET_R, color='b', linestyle='--', alpha=0.5)
        ax1.axhline(TARGET_Z, color='r', linestyle='--', alpha=0.5)
        ax1.set_ylim(TARGET_Z - 3, TARGET_R + 3)
        ax1.legend()
        ax1.grid(True)
        
        # Final Equilibrium Shape
        ax2.set_title("Final Plasma State")
        ax2.contour(self.kernel.RR, self.kernel.ZZ, self.kernel.Psi, levels=20, colors='k')
        ax2.imshow(self.kernel.J_phi, extent=[1,9,-5,5], origin='lower', cmap='hot', alpha=0.5)
        # Plot Coils
        for c in self.kernel.cfg['coils']:
             ax2.plot(c['r'], c['z'], 'rx' if c['current']>0 else 'bx', markersize=10)
             
        plt.tight_layout()
        plt.savefig("Optimal_Control_Result.png")
        print("Analysis saved: Optimal_Control_Result.png")

if __name__ == "__main__":
    # Path to config
    cfg = "03_CODE/SCPN-Fusion-Core/iter_config.json"
    
    # Initialize and Identify
    pilot = OptimalController(cfg)
    pilot.identify_system()
    
    # Run
    pilot.run_optimal_shot()
