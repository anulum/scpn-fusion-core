# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Sandpile Fusion Reactor
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys
import os

L = 100            # System size (Radial points from Core to Edge)
Z_CRIT_BASE = 4.0  # Base Critical Gradient (Stability threshold)
DRIVE_RATE = 1     # Grains added per step
TIME_STEPS = 5000  # Total simulation time

class TokamakSandpile:
    """
    Implements a modified Bak-Tang-Wiesenfeld (BTW) sandpile model 
    to simulate self-organized criticality (SOC) in plasma turbulence. 
    
    Reference: 'Running Sandpile' models for L-H transition.
    """
    def __init__(self, size=L):
        if isinstance(size, bool) or int(size) < 3:
            raise ValueError("size must be an integer >= 3.")
        self.size = int(size)
        self.Z = np.zeros(self.size, dtype=int) # Local gradients (slope)
        self.h = np.zeros(self.size, dtype=int) # Height (Temperature/Density)
        self.avalanche_history = []
        self.confinement_history = []
        self.edge_loss_events = 0
        self.last_edge_loss_events = 0
        
        # Magnetic Control (Z_crit modification)
        self.control_profile = np.zeros(size) 

    def drive(self):
        """Adds energy (sand) to the core."""
        # Core heating: Add grain to center
        self.Z[0] += 1
        
    def relax(self, suppression_strength=0.0):
        """
        The Avalanche Step.
        If slope > critical, redistribute energy.
        suppression_strength: 0.0 to 1.0 (Effect of HJB control)
        """
        suppression_strength = float(suppression_strength)
        if not np.isfinite(suppression_strength):
            raise ValueError("suppression_strength must be finite.")
        if suppression_strength < 0.0 or suppression_strength > 1.0:
            raise ValueError("suppression_strength must be in [0, 1].")

        # Dynamic Critical Gradient: 
        # Stronger magnetic field (control) = Higher threshold before collapse
        current_Z_crit = Z_CRIT_BASE + (2.0 * suppression_strength)
        
        sites_active = np.where(self.Z >= current_Z_crit)[0]
        avalanche_size = 0
        edge_loss_events = 0
        
        # While there are unstable sites, relax them
        # (In a real sandpile this cascades instantenously, here we iterate)
        sub_steps = 0
        while len(sites_active) > 0 and sub_steps < 100:
            for i in sites_active:
                # Topple rule: Z_i reduces, Z_i+1 increases
                # This conserves mass internally, loses at edge
                
                amount = 2 # Standard BTW transfer
                self.Z[i] -= amount
                
                if i + 1 < self.size:
                    self.Z[i+1] += 1
                else:
                    # Edge loss (Energy leaves reactor)
                    edge_loss_events += 1
                     
                if i - 1 >= 0:
                    self.Z[i-1] += 1
                
                avalanche_size += 1
            
            # Re-check instability
            sites_active = np.where(self.Z >= current_Z_crit)[0]
            sub_steps += 1
        self.edge_loss_events += edge_loss_events
        self.last_edge_loss_events = edge_loss_events
        return avalanche_size

    def calculate_profile(self):
        """Reconstructs the Height profile (Temperature) from gradients."""
        # H[i] = Sum(Z[j]) for j from i to End
        # We integrate backwards from edge (0) to core
        self.h = np.cumsum(self.Z[::-1])[::-1]
        return self.h

class HJB_Avalanche_Controller:
    """
    A heuristic closed-loop controller for avalanche suppression.
    Observed State: Current Avalanche Size.
    Action: Increase/Decrease Magnetic Shear (Z_crit).
    Reward: Maximize Core Height (Confinement) - Action Cost.
    """
    def __init__(self):
        self.shear = 0.0
        self.alpha = 0.1 # Learning rate / Reaction speed

    def act(self, last_avalanche_size, current_core_temp):
        # Heuristic Policy derived from Optimal Control:
        # If avalanche is big (instability), clamp down (increase shear).
        # If stable but low temp, relax shear (to allow some transport/fueling).
        
        target_shear = 0.0
        
        if last_avalanche_size > 5:
            # Active turbulence detected! Suppress!
            target_shear = 1.0
        elif current_core_temp < 100:
             # Need better confinement
             target_shear = 0.5
        else:
            # Stable, relax to save energy
            target_shear = 0.0
            
        # Smooth transition
        self.shear += self.alpha * (target_shear - self.shear)
        self.shear = np.clip(self.shear, 0.0, 1.0)
        return self.shear

def run_sandpile_simulation():
    print("--- SCPN FUSION: Self-Organized Criticality (Sandpile) Model ---")
    
    reactor = TokamakSandpile(L)
    controller = HJB_Avalanche_Controller()
    
    # Storage for Visualization
    history_map = np.zeros((TIME_STEPS, L))
    history_avalanches = []
    history_control = []
    history_core_E = []
    
    print(f"Simulating {TIME_STEPS} transport steps...")
    
    for t in range(TIME_STEPS):
        reactor.drive()

        last_av = history_avalanches[-1] if len(history_avalanches) > 0 else 0
        core_temp = reactor.h[0] if t > 0 else 0
        
        action_shear = controller.act(last_av, core_temp)
        
        av_size = reactor.relax(suppression_strength=action_shear)
        profile = reactor.calculate_profile()
        
        history_map[t, :] = reactor.Z # Store gradients (instability map)
        history_avalanches.append(av_size)
        history_control.append(action_shear)
        history_core_E.append(profile[0])
        
        if t % 500 == 0:
            print(f"Step {t}: Core Temp={profile[0]}, Avalanche={av_size}, Control={action_shear:.2f}")

    fig = plt.figure(figsize=(12, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    im = ax1.imshow(history_map.T, aspect='auto', cmap='magma', origin='lower',
                   norm=LogNorm(vmin=1, vmax=np.max(history_map)))
    ax1.set_title("Turbulence Evolution (Gradients)")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Radial Position (Core -> Edge)")
    plt.colorbar(im, ax=ax1, label='Gradient Z (Log Scale)')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(history_core_E, color='orange', label='Core Energy (Temperature)')
    ax2.set_title("Confinement Performance")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Arbitrary Units")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3)
    counts, bins = np.histogram(history_avalanches, bins=50)
    centers = (bins[:-1] + bins[1:]) / 2
    ax3.loglog(centers, counts, 'x', color='red')
    ax3.set_title("Avalanche Size Distribution (SOC Check)")
    ax3.set_xlabel("Size (S)")
    ax3.set_ylabel("Count N(S)")
    ax3.grid(True, which="both", ls="-", alpha=0.2)
    
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(history_control, color='blue', alpha=0.7)
    ax4.set_title("HJB Controller Action (Magnetic Shear)")
    ax4.set_xlabel("Time Step")
    ax4.set_ylabel("Suppression Strength (0-1)")
    ax4.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig("Sandpile_Fusion_Report.png")
    print("\nSimulation Complete. Report saved: Sandpile_Fusion_Report.png")

if __name__ == "__main__":
    run_sandpile_simulation()
