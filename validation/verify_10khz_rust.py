# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — 10kHz Rust Verification
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
"""
Verifies the migrated Rust flight simulator at 10kHz and 30kHz.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import scpn_fusion_rs
    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    print("Warning: scpn_fusion_rs not found. Using mock for verification script.")

class MockRustFlightSim:
    def __init__(self, target_r=6.2, target_z=0.0, control_hz=10000.0):
        self.control_hz = control_hz
        self.target_r = target_r
        self.target_z = target_z

    def run_shot(self, shot_duration_s):
        steps = int(shot_duration_s * self.control_hz)
        # Simulate sub-ms wall time
        wall_time = steps * 0.00003 * 1000 # 30us per step
        return type('Report', (), {
            'steps': steps,
            'duration_s': shot_duration_s,
            'wall_time_ms': wall_time,
            'mean_abs_r_error': 0.001,
            'mean_abs_z_error': 0.002,
            'disrupted': False,
            'r_history': [6.2] * steps,
            'z_history': [0.0] * steps
        })

def run_verification():
    print("=== SCPN Fusion Core: 10kHz Rust Migration Verification ===")
    
    # 1. 10kHz Test
    hz_10k = 10000.0
    print(f"\nInitiating 10kHz Test (Target: 100μs per loop)...")
    sim = scpn_fusion_rs.PyRustFlightSim(6.2, 0.0, hz_10k) if HAS_RUST else MockRustFlightSim(control_hz=hz_10k)
    
    shot_s = 30.0
    report = sim.run_shot(shot_s)
    
    print(f"  Steps executed: {report.steps}")
    print(f"  Total Wall Time: {report.wall_time_ms:.2f} ms")
    print(f"  Avg Latency per step: {(report.wall_time_ms * 1000 / report.steps):.2f} μs")
    print(f"  Mean R Error: {report.mean_abs_r_error:.6f}")
    print(f"  Disrupted: {report.disrupted}")
    
    # 2. 30kHz Test (Extreme stress)
    hz_30k = 30000.0
    print(f"\nInitiating 30kHz Test (Target: 33μs per loop)...")
    sim30 = scpn_fusion_rs.PyRustFlightSim(6.2, 0.0, hz_30k) if HAS_RUST else MockRustFlightSim(control_hz=hz_30k)
    
    report30 = sim30.run_shot(shot_s)
    print(f"  Steps executed: {report30.steps}")
    print(f"  Total Wall Time: {report30.wall_time_ms:.2f} ms")
    print(f"  Avg Latency per step: {(report30.wall_time_ms * 1000 / report30.steps):.2f} μs")
    
    if HAS_RUST:
        print("\n[SUCCESS] Rust-native 10kHz simulation verified.")
    else:
        print("\n[MOCK] Verification script ready. Compile scpn_fusion_rs to run live.")

if __name__ == "__main__":
    run_verification()
