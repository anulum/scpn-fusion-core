# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Tokamak Digital Twin
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os

# --- HYPER-PARAMETERS ---
GRID_SIZE = 40        # 40x40 Poloidal Cross-section
TIME_STEPS = 10000    # Training duration (Increased)
LEARNING_RATE = 0.0001 # Reduced for stability
HIDDEN_SIZE = 64
BATCH_SIZE = 32
MEMORY_SIZE = 1000

# Physics Constants
R_MAJ = 2.0  # Major Radius
R_MIN = 0.8  # Minor Radius

class TokamakTopoloy:
    """
    Handles the magnetic geometry (Safety Factor q-profile).
    Instabilities occur at 'Rational Surfaces' (q = 2, q = 3, q = 1.5).
    """
    def __init__(self, size=GRID_SIZE):
        self.size = size
        # Create coordinate grid (centered)
        y, x = np.ogrid[-size/2:size/2, -size/2:size/2]
        self.r_map = np.sqrt(x**2 + y**2) / (size/2) # Normalized radius 0.0-1.0
        self.mask = self.r_map <= 1.0 # Plasma is only inside the circle
        
        # Initial q-profile (Safety Factor)
        # q(r) usually rises from ~1.0 at core to ~4.0 at edge
        self.q0 = 1.0
        self.qa = 3.0
        self.update_q_profile(0.0) # 0.0 = no extra modification
        
    def update_q_profile(self, current_drive_action):
        """
        Action modifies the magnetic shear (twisting of lines).
        current_drive: -1.0 to 1.0 (Non-inductive current drive)
        """
        # Physical effect: Current drive changes q at the center/edge
        mod_q0 = self.q0 - (0.2 * current_drive_action)
        mod_qa = self.qa + (0.5 * current_drive_action)
        
        # Parabolic q-profile: q(r) = q0 + (qa-q0)*r^2
        self.q_map = mod_q0 + (mod_qa - mod_q0) * (self.r_map**2)
        
    def get_rational_surfaces(self):
        """
        Returns a boolean map of where Magnetic Islands are likely to form.
        Resonances at q = 1.5, 2.0, 2.5, 3.0
        """
        resonances = [1.5, 2.0, 2.5, 3.0]
        danger_map = np.zeros((self.size, self.size), dtype=bool)
        
        for res in resonances:
            # Bandwidth of resonance (island width)
            island_width = 0.05 
            mask = (np.abs(self.q_map - res) < island_width) & self.mask
            danger_map = np.logical_or(danger_map, mask)
            
        return danger_map

class Plasma2D:
    """
    2D Diffusive-Reaction Model on a Poloidal Cross-section.
    """
    def __init__(self, topology):
        self.topo = topology
        self.T = np.zeros((GRID_SIZE, GRID_SIZE)) # Temperature Map
        self.T_core_hist = []
        
    def step(self, action):
        """
        Evolves plasma for one time step.
        action: Control signal for current drive (modifies q-profile).
        """
        # 1. Update Topology (Magnetic Geometry)
        self.topo.update_q_profile(action)
        danger_zones = self.topo.get_rational_surfaces()
        
        # 2. Source Term (Core Heating)
        # Gaussian heating at center
        center = GRID_SIZE // 2
        self.T[center, center] += 5.0
        
        # 3. Diffusion with Topological Instability
        # D = D_base + D_turb (where q is rational)
        D_base = 0.01
        D_turb = 0.5 # High diffusion in islands
        
        diffusivity = np.ones_like(self.T) * D_base
        diffusivity[danger_zones] = D_turb # Islands are leaky!
        
        # Laplacian (Finite Difference)
        T_up = np.roll(self.T, -1, axis=0)
        T_down = np.roll(self.T, 1, axis=0)
        T_left = np.roll(self.T, -1, axis=1)
        T_right = np.roll(self.T, 1, axis=1)
        
        laplacian = (T_up + T_down + T_left + T_right - 4*self.T)
        
        # Update T
        # Radiation Loss (Stabilization) - Stefan-Boltzmann-like cooling
        radiation = 0.0001 * (self.T**2) # Simplified T^2 for numeric stability
        self.T += diffusivity * laplacian - radiation
        
        # Boundary Condition (Cold Walls)
        self.T[~self.topo.mask] = 0.0
        self.T = np.clip(self.T, 0, 100.0) # Saturation limit (Physical Ceiling)
        
        # Metrics
        core_temp = self.T[center, center]
        avg_temp = np.mean(self.T[self.topo.mask])
        self.T_core_hist.append(core_temp)
        
        return self.T.flatten(), avg_temp

class SimpleNeuralNet:
    """
    A lightweight Multi-Layer Perceptron (MLP) written in numpy.
    Implements a Policy Network for Continuous Control.
    """
    def __init__(self, input_size, hidden_size, output_size):
        # Xavier Initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1/input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1/hidden_size)
        self.b2 = np.zeros((1, output_size))
        
    def forward(self, x):
        # x shape: (batch, input)
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = np.tanh(self.z1) # Activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.out = np.tanh(self.z2) # Output -1 to 1 (Action)
        return self.out
    
    def train_step(self, x, target_action, advantage):
        """
        Simplified Policy Gradient update (REINFORCE-like).
        We want to move the output closer to 'target_action' scaled by 'advantage'.
        """
        # Forward pass
        pred = self.forward(x)
        
        # Gradient of MSE loss: L = (pred - target)^2
        # But for RL, we treat 'target' as (pred + noise * advantage)
        # Simplified: We want to push 'pred' in direction of Advantage
        
        grad_out = -(advantage) # If advantage > 0, increase output
        
        # Backprop through tanh output
        d_z2 = grad_out * (1 - self.out**2)
        d_W2 = np.dot(self.a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)
        
        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * (1 - self.a1**2)
        d_W1 = np.dot(x.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)
        
        # Update Weights
        self.W1 -= LEARNING_RATE * d_W1
        self.b1 -= LEARNING_RATE * d_b1
        self.W2 -= LEARNING_RATE * d_W2
        self.b2 -= LEARNING_RATE * d_b2
        
        return np.mean(np.abs(grad_out))

def run_digital_twin():
    print("--- SCPN 2D TOKAMAK DIGITAL TWIN + NEURAL CONTROL ---")
    
    topo = TokamakTopoloy()
    plasma = Plasma2D(topo)
    
    # State: Simplified to radial profile samples (to keep NN small)
    # We take 40 points along the midplane
    state_dim = GRID_SIZE 
    brain = SimpleNeuralNet(state_dim, HIDDEN_SIZE, 1)
    
    history_rewards = []
    history_actions = []
    
    print(f"Training Neural Network for {TIME_STEPS} steps...")
    
    for t in range(TIME_STEPS):
        # 1. Observe State (Midplane Profile)
        midplane_idx = GRID_SIZE // 2
        state_vector = plasma.T[midplane_idx, :].reshape(1, -1)
        
        # 2. Action (Explore vs Exploit)
        # Add exploration noise
        noise = np.random.normal(0, 0.2)
        raw_action = brain.forward(state_vector)
        action = np.clip(raw_action + noise, -1.0, 1.0)[0,0]
        
        # 3. Physics Step
        _, avg_temp = plasma.step(action)
        
        # 4. Reward
        # We want High Temp, but we lose points for instability (implicit in low temp)
        islands_area = np.sum(topo.get_rational_surfaces())
        reward = avg_temp - (islands_area * 0.05)
        
        # 5. Learn (On-Policy / Immediate)
        # If reward is better than recent average, encourage this action direction
        baseline = np.mean(history_rewards[-50:]) if len(history_rewards) > 50 else 0
        advantage = (reward - baseline) * noise # Simple derivative-free estimator trick
        
        loss = brain.train_step(state_vector, None, advantage)
        
        history_rewards.append(reward)
        history_actions.append(action)
        
        if t % 500 == 0:
            print(f"Step {t}: AvgTemp={avg_temp:.2f} | Action={action:.2f} | Loss={loss:.4f} | Islands Detected={np.sum(topo.get_rational_surfaces())} px")

    # --- VISUALIZATION ---
    fig = plt.figure(figsize=(15, 6))
    
    # 1. 2D Plasma Cross-Section (Heatmap)
    ax1 = fig.add_subplot(1, 3, 1)
    im = ax1.imshow(plasma.T, cmap='inferno', origin='lower')
    ax1.set_title("Final Plasma Cross-Section (2D)")
    plt.colorbar(im, ax=ax1, label='Temperature (keV)')
    
    # Overlay Magnetic Islands
    islands = topo.get_rational_surfaces()
    ax1.contour(islands, colors='cyan', levels=[0.5], linewidths=1, alpha=0.5)
    ax1.text(2, 2, "Cyan = q-Resonance (Islands)", color='cyan', fontsize=8)

    # 2. Learning Curve
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(history_rewards, color='orange', alpha=0.6)
    # Moving average
    if len(history_rewards) > 50:
        mov_avg = np.convolve(history_rewards, np.ones(50)/50, mode='valid')
        ax2.plot(range(len(mov_avg)), mov_avg, 'r-', linewidth=2, label='Moving Avg')
    ax2.set_title("Neural Network Learning Curve")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Reward (Confinement)")
    ax2.legend()
    
    # 3. q-Profile and Stability
    ax3 = fig.add_subplot(1, 3, 3)
    r_axis = np.linspace(0, 1, GRID_SIZE//2) # Fix: Match slice size
    q_axis = topo.q_map[GRID_SIZE//2, GRID_SIZE//2:] # Radial slice
    ax3.plot(r_axis, q_axis, 'b-', linewidth=2, label='Safety Factor q(r)')
    
    # Draw Danger Zones
    for q_res in [1.5, 2.0, 2.5, 3.0]:
        ax3.axhline(q_res, color='red', linestyle='--', alpha=0.3)
        ax3.text(0.1, q_res, f"q={q_res}", color='red', fontsize=8)
        
    ax3.set_title("Final Safety Factor Profile")
    ax3.set_xlabel("Normalized Radius r/a")
    ax3.set_ylabel("q")
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig("Tokamak_Digital_Twin.png")
    print("\nDigital Twin Simulation Complete. Snapshot saved: Tokamak_Digital_Twin.png")

if __name__ == "__main__":
    run_digital_twin()
