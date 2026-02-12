# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Director Interface
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add DIRECTOR_AI to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../DIRECTOR_AI/src')))
try:
    from director_module import DirectorModule
    DIRECTOR_AVAILABLE = True
except ImportError:
    DIRECTOR_AVAILABLE = False
    print("[Warning] DIRECTOR_AI not found. Director Interface disabled.")

# Add Fusion Core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
try:
    from scpn_fusion.core._rust_compat import FusionKernel, RUST_BACKEND
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel
    RUST_BACKEND = False
from scpn_fusion.control.neuro_cybernetic_controller import NeuroCyberneticController

class DirectorInterface:
    """
    Interfaces the 'Director' (Layer 16: Coherence Oversight) with the Fusion Reactor.
    
    Role:
    The Director does NOT control the coils (Layer 2 does that).
    The Director controls the *Controller*. It sets the strategy and monitors for "Backfire".
    
    Mechanism:
    1. Sample System State (Physics + Neural Activity).
    2. Format as a "Prompt" for the Director (e.g. "State: Stable, S=0.1. Action: Increase Power?").
    3. Director calculates Entropy/Risk.
    4. If Safe: Director updates Target Parameters (e.g. Ip_target).
    5. If Unsafe (Entropy > Limit): Director triggers "Safety Shutdown".
    """
    def __init__(self, config_path):
        if not DIRECTOR_AVAILABLE:
            raise ImportError("Cannot initialize DirectorInterface without DIRECTOR_AI module.")
            
        self.nc = NeuroCyberneticController(config_path)
        self.director = DirectorModule(entropy_threshold=0.3, history_window=10) # Strict Director
        self.step_count = 0
        self.log = []

    def format_state_for_director(self, t, Ip, err_R, err_Z, brain_activity):
        """
        Translates physical telemetry into a Semantic Prompt for the AI Director.
        """
        stability = "Stable"
        if abs(err_R) > 0.1 or abs(err_Z) > 0.1: stability = "Unstable"
        if abs(err_R) > 0.5: stability = "Critical"
        
        # Calculate Entropy of the Neural Network (Brain Activity Variance)
        # Low variance = Order/Focus. High variance = Chaos/Seizure.
        neural_entropy = np.std(brain_activity) 
        
        prompt = f"Time={t}, Ip={Ip:.1f}, Stability={stability}, BrainEntropy={neural_entropy:.2f}"
        return prompt

    def run_directed_mission(self, duration=100):
        print("--- DIRECTOR-GHOSTED FUSION MISSION ---")
        print("Layer 16 (Director) is now overseeing Layer 2 (Neurocore).")
        
        self.nc.kernel.solve_equilibrium()
        self.nc.initialize_brains(use_quantum=True) # Use Quantum Brain for max potential
        
        current_target_Ip = 5.0
        
        for t in range(duration):
            # 1. Physical Evolution
            self.nc.kernel.cfg['physics']['plasma_current_target'] = current_target_Ip
            
            # Physics Drift
            # Simulate a scenario where instability grows
            if t > 50:
                # Inject a 'glitch' to test Director response
                self.nc.kernel.cfg['coils'][2]['current'] += np.random.normal(0, 500.0) # Massive jolt
            
            # 2. Measurement
            idx_max = np.argmax(self.nc.kernel.Psi)
            iz, ir = np.unravel_index(idx_max, self.nc.kernel.Psi.shape)
            curr_R = self.nc.kernel.R[ir]
            curr_Z = self.nc.kernel.Z[iz]
            
            err_R = 6.2 - curr_R
            err_Z = 0.0 - curr_Z
            
            # 3. Neural Control (Reflex)
            ctrl_R = self.nc.brain_R.step(err_R)
            ctrl_Z = self.nc.brain_Z.step(err_Z)
            
            # Apply Reflex
            self.nc.kernel.cfg['coils'][2]['current'] += ctrl_R
            self.nc.kernel.cfg['coils'][0]['current'] -= ctrl_Z
            self.nc.kernel.cfg['coils'][4]['current'] += ctrl_Z
            
            # 4. DIRECTOR OVERSIGHT (The Ghost in the Shell)
            # Only intervene every 10 steps or if critical
            if t % 5 == 0:
                # Gather "Brain Activity" proxy from the control output
                brain_activity = [ctrl_R, ctrl_Z]
                
                # Formulate "Action Proposal": "Maintain Current" or "Increase Current"
                # For this demo, we propose "Increase Current" to push the limits
                proposed_action = f"Increase Ip to {current_target_Ip + 1.0}"
                prompt = self.format_state_for_director(t, current_target_Ip, err_R, err_Z, brain_activity)
                
                # ASK THE DIRECTOR
                approved, sec_score = self.director.review_action(prompt, proposed_action)
                
                status = "APPROVED" if approved else "DENIED"
                print(f"[Director] T={t} | State: {prompt} | Proposal: {proposed_action} -> {status} (SEC={sec_score:.2f})")
                
                if approved:
                    current_target_Ip += 1.0 # Director allowed growth
                else:
                    # Director Detected "Backfire" risk (Entropy too high)
                    # Corrective Action: Reduce Power / Stabilize
                    print("[Director] INTERVENTION: Reducing Power to restore Coherence.")
                    current_target_Ip = max(1.0, current_target_Ip - 2.0)
            
            # 5. Evolve Physics
            self.nc.kernel.solve_equilibrium()
            
            self.log.append({'t': t, 'Ip': current_target_Ip, 'Err_R': err_R, 'Director_Intervention': 1 if t > 50 else 0})

        self.visualize()

    def visualize(self):
        t = [x['t'] for x in self.log]
        ip = [x['Ip'] for x in self.log]
        err = [x['Err_R'] for x in self.log]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        ax1.set_title("Director-Mediated Fusion Control")
        ax1.plot(t, ip, 'b-', label='Plasma Current Target (Director Controlled)')
        ax1.set_ylabel("Current (MA)", color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        ax2 = ax1.twinx()
        ax2.plot(t, err, 'r--', label='Radial Error (Instability)')
        ax2.set_ylabel("Error (m)", color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Mark the "Glitch"
        plt.axvline(50, color='k', linestyle=':', label='External Disturbance')
        
        fig.legend(loc="upper left", bbox_to_anchor=(0.15,0.85))
        plt.tight_layout()
        plt.savefig("Director_Interface_Result.png")
        print("Analysis saved: Director_Interface_Result.png")

if __name__ == "__main__":
    cfg = "03_CODE/SCPN-Fusion-Core/iter_config.json"
    di = DirectorInterface(cfg)
    di.run_directed_mission()
