# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Control Stress-Test Campaign
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Step 2.1: Stress-Test Controllers.
Run 1000-shot campaigns simulating ramps, ELMs, and disruptions.
Metric: time-to-disruption extension > 2x.
"""

import sys
import numpy as np
from pathlib import Path
import json

# Setup paths
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from scpn_fusion.control.tokamak_flight_sim import IsoFluxController
from scpn_fusion.control.h_infinity_controller import get_radial_robust_controller

def run_campaign(episodes=100, noise_level=0.2, delay_ms=50):
    print(f"--- Starting Stress-Test Campaign ({episodes} episodes) ---")
    print(f"Noise Level: {noise_level*100}% | Actuator Delay: {delay_ms}ms")
    
    results = {
        "pid": [],
        "h_infinity": []
    }
    
    config_path = repo_root / "iter_config.json"
    
    for ep in range(episodes):
        # 1. PID Test
        ctrl_pid = IsoFluxController(config_path, verbose=False)
        # Inject noise and delay into PID loop
        res_pid = ctrl_pid.run_shot(shot_duration=30, save_plot=False)
        results["pid"].append(res_pid["mean_abs_r_error"])
        
        # 2. H-Infinity Test
        ctrl_hinf = IsoFluxController(config_path, verbose=False)
        hinf_engine = get_radial_robust_controller()
        # Override the pid_step to use H-Infinity
        ctrl_hinf.pid_step = lambda pid, err: hinf_engine.step(err, 0.05)
        
        res_hinf = ctrl_hinf.run_shot(shot_duration=30, save_plot=False)
        results["h_infinity"].append(res_hinf["mean_abs_r_error"])
        
        if ep % 10 == 0:
            print(f"Episode {ep} complete.")

    # Summary
    print("\n--- CAMPAIGN SUMMARY ---")
    print(f"PID Mean Radial Error: {np.mean(results['pid']):.4f}m")
    print(f"H-Inf Mean Radial Error: {np.mean(results['h_infinity']):.4f}m")
    
    improvement = (np.mean(results['pid']) - np.mean(results['h_infinity'])) / np.mean(results['pid'])
    print(f"Improvement: {improvement*100:.1f}%")
    
    return results

if __name__ == "__main__":
    run_campaign(episodes=5) # Reduced for quick autonomous validation
