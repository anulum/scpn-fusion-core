import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import sys
import os

# --- CONTROL ROOM PARAMETERS ---
RESOLUTION = 60
SIM_DURATION = 200
FPS = 10

class TokamakPhysicsEngine:
    """
    Simulates the Grad-Shafranov equilibrium geometry.
    Models Plasma Shape (Kappa, Delta) and Vertical Stability.
    """
    def __init__(self):
        # Grid R (Radius) vs Z (Vertical)
        self.R = np.linspace(1.0, 5.0, RESOLUTION)
        self.Z = np.linspace(-3.0, 3.0, RESOLUTION)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        
        # Plasma Parameters (ITER-like)
        self.R0 = 3.0   # Major Radius
        self.a = 1.0    # Minor Radius
        self.kappa = 1.7 # Elongation (Oval shape)
        self.delta = 0.33 # Triangularity (D-shape)
        
        # State
        self.z_pos = 0.0 # Vertical Position (Instability variable)
        self.v_drift = 0.0 # Vertical Velocity
        self.density = np.zeros((RESOLUTION, RESOLUTION))
        
    def solve_flux_surfaces(self):
        """
        Analytic approximation of Grad-Shafranov flux surfaces.
        Miller parameterization for D-shaped plasma.
        """
        # Coordinate transform to flux coordinates
        # R = R0 + r*cos(theta + arcsin(delta)*sin(theta))
        # Z = kappa*r*sin(theta)
        
        # Inverse mapping for the grid:
        # Distance from magnetic axis with shaping corrections
        rho_sq = ((self.RR - self.R0)**2 + 
                 ((self.ZZ - self.z_pos) / self.kappa)**2 - 
                 2 * self.delta * (self.RR - self.R0) * ((self.RR - self.R0)**2) ) # Simplified Triangularity term
        
        # Normalized Flux (0 at center, 1 at edge)
        psi = rho_sq / (self.a**2)
        
        # Plasma Density Profile (Parabolic inside LCFS - Last Closed Flux Surface)
        self.density = np.where(psi < 1.0, (1.0 - psi)**1.5, 0.0)
        
        # Add Turbulence (Perturbations on flux surfaces)
        noise = np.random.normal(0, 0.05, self.density.shape) * self.density
        self.density = np.clip(self.density + noise, 0, None)
        
        return self.density, psi

    def step_dynamics(self, coil_action_top, coil_action_bottom):
        """
        Simulates Vertical Displacement Event (VDE).
        Unstable equilibrium: Plasma naturally wants to drift Up or Down.
        """
        # Physics of VDE: d2z/dt2 = Gamma*z + Control_Force
        instability_growth = 0.1 # Unstable index
        
        # Control Force (Push/Pull from Poloidal Field Coils)
        # Top coil pushes down (negative force), Bottom pushes up
        control_force = (coil_action_bottom - coil_action_top) * 0.2
        
        # Random kick (ELM or disruption precursor)
        disturbance = np.random.normal(0, 0.01)
        if np.random.rand() < 0.05: disturbance += 0.2 # Big kick!
        
        # Integrate Motion
        accel = (instability_growth * self.z_pos) + control_force + disturbance
        self.v_drift += accel
        self.z_pos += self.v_drift
        
        # Friction/Damping (Lenz law in wall)
        self.v_drift *= 0.9 

        return self.z_pos

class DiagnosticSystem:
    """
    Simulates Noisy Sensors (Magnetic Probes).
    """
    def measure_position(self, true_z):
        noise = np.random.normal(0, 0.05)
        return true_z + noise

class NeuralController:
    """
    PID-Neural Hybrid for Vertical Stability.
    """
    def __init__(self):
        self.integral_error = 0.0
        self.prev_error = 0.0
        
    def compute_action(self, measured_z):
        # PID Part (Fast Reaction)
        Kp = 5.0
        Ki = 0.1
        Kd = 8.0
        
        error = 0.0 - measured_z # Target is Z=0
        self.integral_error += error
        derivative = error - self.prev_error
        
        pid_out = (Kp * error) + (Ki * self.integral_error) + (Kd * derivative)
        self.prev_error = error
        
        # Action Splitting (Top vs Bottom Coils)
        # If we need to go Down (pid_out < 0), Top pushes (Action > 0)
        # If we need to go Up (pid_out > 0), Bottom pushes
        
        # Normalize to 0..1
        force = np.tanh(pid_out) 
        
        if force > 0: # Need to move UP
            return 0.0, abs(force) # Top=0, Bottom=Push
        else: # Need to move DOWN
            return abs(force), 0.0 # Top=Push, Bottom=0

def run_control_room():
    print("--- SCPN FUSION CONTROL ROOM: Grad-Shafranov VDE Simulation ---")
    
    reactor = TokamakPhysicsEngine()
    sensors = DiagnosticSystem()
    ai = NeuralController()
    
    # Setup Animation
    fig = plt.figure(figsize=(12, 8), facecolor='#1e1e1e')
    gs = fig.add_gridspec(2, 2)
    
    # Ax1: Main Reactor View (Cross Section)
    ax_plasma = fig.add_subplot(gs[:, 0])
    ax_plasma.set_facecolor('black')
    ax_plasma.set_title("Tokamak Cross-Section (Live)", color='white')
    
    # Ax2: Vertical Position Trace
    ax_trace = fig.add_subplot(gs[0, 1])
    ax_trace.set_facecolor('#2e2e2e')
    ax_trace.set_title("Vertical Displacement (Z-Pos)", color='white')
    ax_trace.set_ylim(-1.5, 1.5)
    ax_trace.grid(True, color='#444')
    line_z, = ax_trace.plot([], [], 'cyan', lw=2)
    line_setpoint, = ax_trace.plot([], [], 'r--', alpha=0.5)
    
    # Ax3: Coil Currents
    ax_coils = fig.add_subplot(gs[1, 1])
    ax_coils.set_facecolor('#2e2e2e')
    ax_coils.set_title("Poloidal Field Coil Currents", color='white')
    ax_coils.set_ylim(0, 1.1)
    bar_top = ax_coils.bar([0], [0], color='red', label='Top Coil')
    bar_bot = ax_coils.bar([1], [0], color='blue', label='Bottom Coil')
    ax_coils.set_xticks([0, 1])
    ax_coils.set_xticklabels(['Top', 'Bottom'], color='white')
    ax_coils.legend()

    # Data Storage
    history_z = []
    history_t = []
    
    def update(frame):
        # 1. Physics Step
        true_z = reactor.step_dynamics(bar_top[0].get_height(), bar_bot[0].get_height())
        density, psi = reactor.solve_flux_surfaces()
        
        # 2. Sensing & Control
        measured_z = sensors.measure_position(true_z)
        act_top, act_bot = ai.compute_action(measured_z)
        
        # 3. Update Visuals
        
        # Plasma Shape
        ax_plasma.clear()
        ax_plasma.set_facecolor('black')
        ax_plasma.set_title(f"Plasma Shape (t={frame})", color='white')
        # Plot Density Heatmap
        ax_plasma.imshow(density, extent=[1, 5, -3, 3], origin='lower', cmap='plasma', vmin=0, vmax=1)
        # Plot Flux Surfaces (Contours)
        ax_plasma.contour(reactor.R, reactor.Z, psi, levels=[0.2, 0.4, 0.6, 0.8, 1.0], colors='white', alpha=0.3, linewidths=0.5)
        # Draw Wall
        rect = plt.Rectangle((1.0, -2.8), 4.0, 5.6, linewidth=2, edgecolor='gray', facecolor='none')
        ax_plasma.add_patch(rect)
        
        # Coils Indicators
        # Visual feedback of coils firing
        t_alpha = 0.3 + (act_top * 0.7)
        b_alpha = 0.3 + (act_bot * 0.7)
        ax_plasma.plot(3.0, 2.9, 's', color='red', markersize=20, alpha=t_alpha) # Top Coil
        ax_plasma.plot(3.0, -2.9, 's', color='blue', markersize=20, alpha=b_alpha) # Bottom Coil
        
        # Trace
        history_z.append(true_z)
        history_t.append(frame)
        if len(history_z) > 50: 
            history_z.pop(0)
            history_t.pop(0)
            
        line_z.set_data(range(len(history_z)), history_z)
        line_setpoint.set_data(range(len(history_z)), [0]*len(history_z))
        ax_trace.set_xlim(0, 50)
        
        # Bars
        bar_top[0].set_height(act_top)
        bar_bot[0].set_height(act_bot)
        
        return ax_plasma, line_z, bar_top

    print("Rendering Control Room Animation...")
    ani = FuncAnimation(fig, update, frames=SIM_DURATION, interval=100, blit=False)
    
    # Save as GIF
    ani.save("SCPN_Fusion_Control_Room.gif", writer=PillowWriter(fps=15))
    print("\nSimulation Complete. Animation saved: SCPN_Fusion_Control_Room.gif")
    
    # Save final frame as static report too
    plt.savefig("SCPN_Fusion_Status_Report.png")

if __name__ == "__main__":
    run_control_room()
