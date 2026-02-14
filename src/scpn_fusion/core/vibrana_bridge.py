# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Vibrana Bridge
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import sys
import os
import numpy as np
import threading
import queue

# Add VIBRANA path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../CCW_Standalone')))
try:
    from ccw_sota_engine import CCWStateOfTheArtEngine, CCWConfiguration, BiometricData, AttractorType
    VIBRANA_AVAILABLE = True
except ImportError:
    VIBRANA_AVAILABLE = False
    print("[Warning] CCW_Standalone not found. VIBRANA Bridge disabled.")

try:
    from scpn_fusion.core._rust_compat import FusionKernel, RUST_BACKEND
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel
    RUST_BACKEND = False
from scpn_fusion.control.neuro_cybernetic_controller import NeuroCyberneticController

class VibranaFusionBridge:
    """
    Layer 3 (Fusion) -> Layer 5 (VIBRANA Audio) Bridge.
    
    "The Song of the Sun"
    
    Mechanism:
    Real-time Sonification of Tokamak Physics.
    - Plasma Current (Ip) -> Carrier Frequency (Pitch)
    - Stability Error (dR) -> Chaos Intensity (Dissonance)
    - Confinement (Tau) -> Binaural Beat (Brain State)
    - Magnetic Flux (Psi) -> Harmonic Series (Timbre)
    
    This provides an intuitive "Auditory Dashboard" for the operator.
    Stable Plasma = Harmonic, Angelic Sound (Solfeggio/Golden Ratio).
    Unstable Plasma = Chaotic, Dissonant Screeching (Lorenz Attractor).
    """
    
    def __init__(self, config_path):
        if not VIBRANA_AVAILABLE:
            raise ImportError("VIBRANA Engine not available.")
            
        self.nc = NeuroCyberneticController(config_path)
        
        # Audio Config
        self.audio_config = CCWConfiguration(
            sample_rate=44100,
            duration_minutes=1.0, # Real-time chunks
            chaos_enabled=True,
            attractor_type=AttractorType.LORENZ
        )
        self.engine = CCWStateOfTheArtEngine(self.audio_config)
        
    def map_physics_to_audio(self, t, Ip, err_R, err_Z, psi_matrix):
        """
        The Transduction Function: Physics -> Acoustics
        """
        # 1. Pitch (Carrier) driven by Current
        # 5MA -> 200Hz, 15MA -> 600Hz
        carrier_freq = 200.0 + (Ip * 20.0)
        self.engine.config.carrier_frequency = np.clip(carrier_freq, 100, 800)
        
        # 2. Chaos (Dissonance) driven by Instability
        total_error = np.sqrt(err_R**2 + err_Z**2)
        # 0.0 error -> 0.0 chaos
        # 0.5 error -> 1.0 chaos (Maximum screaming)
        chaos_level = np.clip(total_error * 2.0, 0.0, 1.0)
        self.engine.config.chaos_intensity = chaos_level
        
        # 3. Binaural Beat (Brain State) driven by Stability
        # Stable -> Theta (Meditation/Flow) ~ 5Hz
        # Unstable -> Gamma (High Alert) ~ 40Hz
        if total_error < 0.05:
            beat_freq = 5.0 # Theta
        elif total_error < 0.2:
            beat_freq = 10.0 # Alpha
        else:
            beat_freq = 40.0 # Gamma (Warning)
        self.engine.config.binaural_beat_frequency = beat_freq
        
        # 4. Harmonics (Timbre) driven by Magnetic Topology
        # Complexity of flux surface shape maps to harmonic richness
        flux_complexity = np.std(psi_matrix)
        if flux_complexity > 0.5:
            self.engine.config.golden_ratio_harmonics = True # Rich sound
        else:
            self.engine.config.golden_ratio_harmonics = False # Pure tone
            
        return {
            "Carrier": carrier_freq,
            "Chaos": chaos_level,
            "Beat": beat_freq
        }

    def run_sonification_session(self, duration_steps=100):
        print("--- VIBRANA FUSION BRIDGE: SONIFICATION INITIATED ---")
        print("Listen to the Plasma...")
        
        self.nc.kernel.solve_equilibrium()
        self.nc.initialize_brains(use_quantum=True)
        
        # We will generate audio metrics log instead of real-time audio 
        # because we are in a CLI without speakers.
        audio_log = []
        
        current_target_Ip = 5.0
        
        for t in range(duration_steps):
            # 1. Evolve Physics
            self.nc.kernel.cfg['physics']['plasma_current_target'] = current_target_Ip
            current_target_Ip += 0.1 # Ramp up
            
            # Inject Instability at t=50
            if t == 50:
                print(">> INJECTING TURBULENCE <<")
                self.nc.kernel.cfg['coils'][2]['current'] += 2000.0
            
            # 2. Measure
            idx_max = np.argmax(self.nc.kernel.Psi)
            iz, ir = np.unravel_index(idx_max, self.nc.kernel.Psi.shape)
            curr_R = self.nc.kernel.R[ir]
            curr_Z = self.nc.kernel.Z[iz]
            
            err_R = 6.2 - curr_R
            err_Z = 0.0 - curr_Z
            
            # 3. Neuro Control
            ctrl_R = self.nc.brain_R.step(err_R)
            ctrl_Z = self.nc.brain_Z.step(err_Z)
            
            self.nc.kernel.cfg['coils'][2]['current'] += ctrl_R
            self.nc.kernel.cfg['coils'][0]['current'] -= ctrl_Z
            self.nc.kernel.cfg['coils'][4]['current'] += ctrl_Z
            
            self.nc.kernel.solve_equilibrium()
            
            # 4. SONIFY
            audio_params = self.map_physics_to_audio(t, current_target_Ip, err_R, err_Z, self.nc.kernel.Psi)
            
            print(f"T={t} | Err={err_R:.3f} | Audio: {audio_params['Carrier']:.0f}Hz + {audio_params['Beat']}Hz Beat | Chaos={audio_params['Chaos']:.2f}")
            
            audio_log.append({
                't': t,
                'error': np.sqrt(err_R**2 + err_Z**2),
                'carrier': audio_params['Carrier'],
                'chaos': audio_params['Chaos']
            })
            
        self.visualize_soundscape(audio_log)

    def visualize_soundscape(self, log):
        import matplotlib.pyplot as plt
        
        t = [x['t'] for x in log]
        chaos = [x['chaos'] for x in log]
        carrier = [x['carrier'] for x in log]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Spectrogram-like visualization
        ax1.set_title("The Song of the Tokamak (Sonification)")
        ax1.plot(t, carrier, 'b-', label='Pitch (Plasma Current)')
        ax1.set_ylabel("Frequency (Hz)")
        ax1.grid(True)
        ax1.legend()
        
        # Dissonance
        ax2.fill_between(t, chaos, color='r', alpha=0.5, label='Dissonance (Instability)')
        ax2.set_ylabel("Chaos Level (0-1)")
        ax2.set_xlabel("Time Step")
        ax2.set_ylim(0, 1.1)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig("Vibrana_Sonification_Result.png")
        print("Analysis saved: Vibrana_Sonification_Result.png")

if __name__ == "__main__":
    cfg = "03_CODE/SCPN-Fusion-Core/iter_config.json"
    bridge = VibranaFusionBridge(cfg)
    bridge.run_sonification_session()
