import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from collections import deque

# Add sc-neurocore to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../sc-neurocore/src')))
from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron
from sc_neurocore.sources.quantum_entropy import QuantumEntropySource

# Add fusion core to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
try:
    from scpn_fusion.core._rust_compat import FusionKernel, RUST_BACKEND
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel
    RUST_BACKEND = False

# --- CONTROL PARAMETERS ---
SHOT_DURATION = 100
TARGET_R = 6.2
TARGET_Z = 0.0

class SpikingControllerPool:
    """
    A population of Stochastic LIF neurons acting as a controller.
    Concept: Rate Coding.
    - Positive Error -> Excitatory Input -> High Firing Rate -> Positive Output
    - Negative Error -> Inhibitory Input -> Low Firing Rate -> Negative Output
    
    To handle bipolar output, we use a "Push-Pull" pair of populations.
    - Pop A: Excited by +Error
    - Pop B: Excited by -Error
    Output = (Rate_A - Rate_B) * Gain
    """
    def __init__(self, n_neurons=20, gain=1.0, tau_window=10, use_quantum=False):
        self.n_neurons = n_neurons
        self.gain = gain
        self.window_size = tau_window
        self.use_quantum = use_quantum
        
        # Initialize Entropy Source (Quantum or None)
        self.q_source = None
        if use_quantum:
            self.q_source = QuantumEntropySource(n_qubits=4) # 16-dim Hilbert Space
        
        # Two populations: Agonist (Push) and Antagonist (Pull)
        # Pass entropy_source to neurons
        self.pop_pos = [StochasticLIFNeuron(seed=i, entropy_source=self.q_source) for i in range(n_neurons)]
        self.pop_neg = [StochasticLIFNeuron(seed=i+1000, entropy_source=self.q_source) for i in range(n_neurons)]
        
        # Spike History (for Rate decoding)
        self.history_pos = deque(maxlen=tau_window)
        self.history_neg = deque(maxlen=tau_window)
        
        # Init history with zeros
        for _ in range(tau_window):
            self.history_pos.append(0)
            self.history_neg.append(0)
            
    def step(self, error_signal):
        # 1. Encode Error to Current
        # Scaling: Error 0.1m -> Current 0.5nA (approx)
        i_scale = 5.0
        
        # Rectified inputs
        input_pos = max(0, error_signal) * i_scale
        input_neg = max(0, -error_signal) * i_scale
        
        # Base current (spontaneous activity)
        i_bias = 0.1
        
        # 2. Update Neurons
        spikes_pos = 0
        for n in self.pop_pos:
            if n.step(i_bias + input_pos): spikes_pos += 1
            
        spikes_neg = 0
        for n in self.pop_neg:
            if n.step(i_bias + input_neg): spikes_neg += 1
            
        # 3. Record History
        self.history_pos.append(spikes_pos)
        self.history_neg.append(spikes_neg)
        
        # 4. Decode Rate
        # Average spikes per step
        rate_pos = sum(self.history_pos) / (self.window_size * self.n_neurons)
        rate_neg = sum(self.history_neg) / (self.window_size * self.n_neurons)
        
        # Net Output
        output = (rate_pos - rate_neg) * self.gain
        return output

class NeuroCyberneticController:
    """
    Replaces the PID controller with a Biologically Plausible SNN.
    Connects Layer 3 (Fusion) with Layer 2 (Neurocore).
    """
    def __init__(self, config_file):
        self.kernel = FusionKernel(config_file)
        self.history = {'t': [], 'Ip': [], 'R_axis': [], 'Z_axis': [], 'Control_R': [], 'Spike_Rates': []}
        self.brain_R = None
        self.brain_Z = None

    def initialize_brains(self, use_quantum=False):
        # Neural Controllers
        # Radial: Controls horizontal position
        self.brain_R = SpikingControllerPool(n_neurons=50, gain=10.0, tau_window=20, use_quantum=use_quantum)
        
        # Vertical: Controls vertical position
        self.brain_Z = SpikingControllerPool(n_neurons=50, gain=20.0, tau_window=20, use_quantum=use_quantum)

    def run_shot(self):
        self.initialize_brains(use_quantum=False)
        self._execute_simulation("Neuro-Cybernetic (Classical SNN)")

    def run_quantum_shot(self):
        self.initialize_brains(use_quantum=True)
        self._execute_simulation("Quantum-Neuro Hybrid (QNN)")

    def _execute_simulation(self, title):
        print(f"--- {title.upper()} PLASMA INTERFACE ---")
        print(f"Initializing Stochastic Neural Network (SNN)...")
        print(f"Neurons: {self.brain_R.n_neurons * 4} (Push-Pull Configuration)")
        
        # Reset History
        self.history = {'t': [], 'Ip': [], 'R_axis': [], 'Z_axis': [], 'Control_R': [], 'Spike_Rates': []}
        
        # Initial Physics
        self.kernel.solve_equilibrium()
        
        for t in range(SHOT_DURATION):
            # 1. Physics Disturbance (Shafranov Shift scenario)
            target_Ip = 5.0 + (10.0 * t / SHOT_DURATION)
            self.kernel.cfg['physics']['plasma_current_target'] = target_Ip
            
            # Artificial Drift (Simulating instability)
            # drift = 0.01 * t
            
            # 2. Measure State (Sensory Input)
            idx_max = np.argmax(self.kernel.Psi)
            iz, ir = np.unravel_index(idx_max, self.kernel.Psi.shape)
            curr_R = self.kernel.R[ir]
            curr_Z = self.kernel.Z[iz]
            
            err_R = TARGET_R - curr_R 
            err_Z = TARGET_Z - curr_Z
            
            # 3. Neural Processing (The Brain)
            ctrl_R = self.brain_R.step(err_R)
            ctrl_Z = self.brain_Z.step(err_Z)
            
            # 4. Actuation (The Body)
            # PF3 (Outer Coil) pushes Radial
            self.kernel.cfg['coils'][2]['current'] += ctrl_R
            
            # PF1/PF5 (Top/Bottom) pull Vertical
            self.kernel.cfg['coils'][0]['current'] -= ctrl_Z
            self.kernel.cfg['coils'][4]['current'] += ctrl_Z
            
            # 5. Evolve
            self.kernel.solve_equilibrium()
            
            # Log
            self.history['t'].append(t)
            self.history['R_axis'].append(curr_R)
            self.history['Z_axis'].append(curr_Z)
            self.history['Control_R'].append(ctrl_R)
            
            print(f"T={t}: Pos=({curr_R:.2f}, {curr_Z:.2f}) | Err=({err_R:.3f}, {err_Z:.3f}) | Brain_Out=({ctrl_R:.3f}, {ctrl_Z:.3f})")

        self.visualize(title)

    def visualize(self, title):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.set_title(f"{title} Control")
        ax1.plot(self.history['t'], self.history['R_axis'], 'b-', label='R (Radial)')
        ax1.plot(self.history['t'], self.history['Z_axis'], 'r-', label='Z (Vertical)')
        ax1.axhline(TARGET_R, color='b', linestyle='--', alpha=0.3)
        ax1.axhline(TARGET_Z, color='r', linestyle='--', alpha=0.3)
        ax1.set_ylabel("Position (m)")
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_title("Neural Control Activity")
        ax2.plot(self.history['t'], self.history['Control_R'], 'k-', label='Radial Command')
        ax2.set_ylabel("Current Delta (A)")
        ax2.set_xlabel("Time Step")
        ax2.legend()
        
        filename = f"{title.replace(' ', '_')}_Result.png"
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Analysis saved: {filename}")

if __name__ == "__main__":
    cfg = "03_CODE/SCPN-Fusion-Core/iter_config.json"
    nc = NeuroCyberneticController(cfg)
    
    # Check CLI args for Quantum Mode
    if len(sys.argv) > 1 and sys.argv[1] == "quantum":
        nc.run_quantum_shot()
    else:
        nc.run_shot()
