# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Advanced SOC Fusion Learning
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

# --- ADVANCED PHYSICS PARAMETERS ---
L = 60              # System size
TIME_STEPS = 10000  # Longer simulation for learning
Z_CRIT_BASE = 6.0   # Base stability
FLOW_GENERATION = 0.2 # How much turbulence converts to Zonal Flow
FLOW_DAMPING = 0.05   # How fast Zonal Flow decays
SHEAR_EFFICIENCY = 3.0 # How effective Flow is at raising Z_crit

# --- Q-LEARNING PARAMETERS ---
ALPHA = 0.1  # Learning Rate
GAMMA = 0.95 # Discount Factor
EPSILON = 0.1 # Exploration Rate
N_STATES_TURB = 5
N_STATES_FLOW = 5
N_ACTIONS = 3 # Decrease, Maintain, Increase Ext. Shear

class CoupledSandpileReactor:
    """
    Predator-Prey Sandpile Model.
    Simulates the interaction between Turbulence (Avalanches) and Zonal Flows.
    """
    def __init__(self, size=L):
        self.size = size
        self.Z = np.zeros(size, dtype=int) # Temperature Gradient
        self.h = np.zeros(size, dtype=int) # Temperature Profile
        self.flow = 0.0 # Zonal Flow Velocity (Global approximation)
        
    def drive(self):
        # Add heat to core
        self.Z[0] += 1
        
    def step_physics(self, external_shear):
        """
        Evolves the system one step.
        Returns: avalanche_size, flow_level
        """
        # 1. Calculate Dynamic Critical Gradient
        # Z_crit increases with Flow (Internal) and External Shear (Control)
        # This is the "Transport Barrier" mechanism.
        eff_shear = self.flow + external_shear
        current_Z_crit = Z_CRIT_BASE + (SHEAR_EFFICIENCY * eff_shear)
        
        # 2. Avalanche Dynamics (Relaxation)
        sites_active = np.where(self.Z >= current_Z_crit)[0]
        avalanche_size = 0
        total_topple = 0
        
        sub_steps = 0
        while len(sites_active) > 0 and sub_steps < 50:
            for i in sites_active:
                amount = 2
                self.Z[i] -= amount
                if i + 1 < self.size:
                    self.Z[i+1] += 1
                if i - 1 >= 0:
                    self.Z[i-1] += 1
                
                total_topple += 1
            
            avalanche_size += len(sites_active)
            sites_active = np.where(self.Z >= current_Z_crit)[0]
            sub_steps += 1
            
        # 3. Predator-Prey Update
        # Turbulence drives Flow
        self.flow += (total_topple * FLOW_GENERATION / self.size) 
        # Flow decays (Damping)
        self.flow *= (1.0 - FLOW_DAMPING)
        
        # Clip flow to realistic bounds
        self.flow = np.clip(self.flow, 0, 5.0)
        
        return total_topple, self.flow, eff_shear

    def get_profile_energy(self):
        self.h = np.cumsum(self.Z[::-1])[::-1]
        return self.h[0]

class FusionAI_Agent:
    """
    Tabular Q-Learning Agent.
    Learns to maximize confinement while minimizing external power usage.
    """
    def __init__(self):
        self.q_table = np.zeros((N_STATES_TURB, N_STATES_FLOW, N_ACTIONS))
        self.last_state = (0, 0)
        self.last_action = 1
        self.total_reward = 0
        
    def discretize_state(self, turb, flow):
        # Map continuous physics to discrete grid indices
        s_turb = min(int(np.log1p(turb)), N_STATES_TURB - 1)
        s_flow = min(int(flow), N_STATES_FLOW - 1)
        return (s_turb, s_flow)
    
    def choose_action(self, state):
        # Epsilon-Greedy Strategy
        if np.random.rand() < EPSILON:
            return np.random.randint(N_ACTIONS)
        else:
            return np.argmax(self.q_table[state])
            
    def learn(self, new_state, reward):
        old_q = self.q_table[self.last_state][self.last_action]
        max_future_q = np.max(self.q_table[new_state])
        
        # Bellman Equation Update
        new_q = old_q + ALPHA * (reward + GAMMA * max_future_q - old_q)
        self.q_table[self.last_state][self.last_action] = new_q
        self.total_reward += reward

def run_advanced_learning_sim():
    print("--- SCPN MASTERPIECE: Predator-Prey Physics + Q-Learning Control ---")
    
    reactor = CoupledSandpileReactor(L)
    brain = FusionAI_Agent()
    
    # Histories
    h_turb = []
    h_flow = []
    h_temp = []
    h_shear_ctrl = []
    
    current_ext_shear = 0.0
    
    # SIMULATION LOOP
    for t in range(TIME_STEPS):
        # 1. Observe State
        # We need "last step" metrics for state
        if t == 0:
            turb, flow = 0, 0
        else:
            turb = h_turb[-1]
            flow = h_flow[-1]
            
        state = brain.discretize_state(turb, flow)
        
        # 2. AI Action
        action_idx = brain.choose_action(state)
        # Map Action: 0=Decrease, 1=Hold, 2=Increase
        if action_idx == 0: current_ext_shear -= 0.05
        if action_idx == 2: current_ext_shear += 0.05
        current_ext_shear = np.clip(current_ext_shear, 0.0, 1.0)
        
        # 3. Physics Step
        reactor.drive()
        # To make it harder, we add random noise to physics parameters occasionally
        if np.random.rand() < 0.01: reactor.drive() 
        
        av_size, flow_val, total_shear = reactor.step_physics(current_ext_shear)
        core_temp = reactor.get_profile_energy()
        
        # 4. Reward Calculation
        # Reward = High Temp - Penalty for Turbulence - Cost of Control
        reward = (core_temp * 0.1) - (av_size * 0.5) - (current_ext_shear * 2.0)
        
        # 5. Learning Step
        brain.last_state = state
        brain.last_action = action_idx
        
        # Need next state for learning (conceptually we learn from t to t+1)
        # Here we do on-policy update next loop or simplified:
        if t > 0:
            brain.learn(state, reward)
            
        # Logging
        h_turb.append(av_size)
        h_flow.append(flow_val)
        h_temp.append(core_temp)
        h_shear_ctrl.append(current_ext_shear)
        
        if t % 1000 == 0:
            print(f"Step {t}: Temp={core_temp:<4} | Flow={flow_val:<5.2f} | Turb={av_size:<4} | AI_Shear={current_ext_shear:.2f} | Q-Avg={np.mean(brain.q_table):.2f}")

    # --- VISUALIZATION ---
    fig = plt.figure(figsize=(14, 8))
    
    # 1. Phase Space: Turbulence vs Flow (The Predator-Prey Cycle)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(h_turb[-1000:], h_flow[-1000:], 'k-', alpha=0.3)
    ax1.plot(h_turb[-100:], h_flow[-100:], 'r-', linewidth=2, label='Last 100 steps')
    ax1.set_title("Phase Space: L-H Transition")
    ax1.set_xlabel("Turbulence (Avalanche)")
    ax1.set_ylabel("Zonal Flow (Shear)")
    ax1.legend()
    
    # 2. Time Series: Physics
    ax2 = fig.add_subplot(2, 3, (2, 3))
    ax2.plot(h_temp, color='orange', label='Core Temperature')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(h_flow, color='blue', alpha=0.3, label='Internal Flow')
    ax2.set_title("Reactor Evolution: Temperature Growth")
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    
    # 3. Time Series: AI Control
    ax3 = fig.add_subplot(2, 3, (4, 5))
    ax3.plot(h_turb, 'r-', alpha=0.3, label='Turbulence')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(h_shear_ctrl, 'g-', linewidth=2, label='AI Control Signal')
    ax3.set_title("AI Agent Response to Instability")
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    
    # 4. Learned Policy Map
    ax4 = fig.add_subplot(2, 3, 6)
    # Visualizing the best action for each state (Turb vs Flow)
    policy_map = np.argmax(brain.q_table, axis=2)
    im = ax4.imshow(policy_map, origin='lower', cmap='viridis')
    ax4.set_title("Learned Policy (Q-Table)")
    ax4.set_xlabel("Flow State")
    ax4.set_ylabel("Turbulence State")
    plt.colorbar(im, ax=ax4, label='Action (0=Down, 2=Up)')

    plt.tight_layout()
    plt.savefig("Advanced_SOC_Learning.png")
    print("\nSimulation Complete. Analysis saved: Advanced_SOC_Learning.png")

if __name__ == "__main__":
    run_advanced_learning_sim()
