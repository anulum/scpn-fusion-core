# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Run Realtime Simulation
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import argparse
import multiprocessing
import time
from pathlib import Path
import numpy as np

from scpn_fusion.core.fusion_kernel import FusionKernel
from scpn_fusion.nuclear.pwi_erosion import SputteringPhysics
from scpn_fusion.control.torax_hybrid_loop import run_nstxu_torax_hybrid_campaign

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "validation" / "iter_validated_config.json"

# --- SHARED MEMORY STRUCTURE ---
# We use a Dictionary Proxy to simulate the Network Bus (Redis/Kafka)
def physics_node(bus, stop_event, cfg_path: str):
    print("[PHYSICS] Node Started. Booting Reactor...")
    
    # Init Physics
    kernel = FusionKernel(cfg_path)
    pwi = SputteringPhysics("Tungsten")
    
    t = 0
    while not stop_event.is_set():
        start_time = time.time()
        
        # 1. READ INPUTS (Action from Controller)
        action = bus.get('control_action', None)
        if action:
            # Apply control currents (simplified application)
            # action is delta_currents array
            for i, delta in enumerate(action):
                if i < len(kernel.cfg['coils']):
                    kernel.cfg['coils'][i]['current'] += delta
        
        # 2. EVOLVE PHYSICS
        # Add random noise (Real world chaos)
        kernel.cfg['physics']['plasma_current_target'] += np.random.normal(0, 0.01)
        
        kernel.solve_equilibrium()
        
        # PWI Physics
        # Assume T_div depends on heating power (simplified coupling)
        T_div = 50.0 + np.random.normal(0, 2.0)
        pwi_res = pwi.calculate_erosion_rate(1e23, T_div)
        
        # 3. PUBLISH STATE (Telemetry)
        # Extract key metrics
        idx_max = np.argmax(kernel.Psi)
        iz, ir = np.unravel_index(idx_max, kernel.Psi.shape)
        
        state = {
            'timestamp': t,
            'R_axis': kernel.R[ir],
            'Z_axis': kernel.Z[iz],
            'Ip': kernel.cfg['physics']['plasma_current_target'],
            'Erosion_Rate': pwi_res['Erosion_mm_year'],
            'T_divertor': T_div
        }
        
        bus['telemetry'] = state
        t += 1
        
        # Real-time pacing (simulate 100Hz loop)
        elapsed = time.time() - start_time
        sleep_time = max(0, 0.01 - elapsed)
        # time.sleep(sleep_time) # Uncomment for true RT, commented for fast demo

def control_node(bus, stop_event):
    print("[CONTROL] Node Started. Waiting for telemetry...")
    
    target_R = 6.2
    Kp = 0.5
    
    while not stop_event.is_set():
        # 1. READ SENSORS
        state = bus.get('telemetry', None)
        if state is None:
            time.sleep(0.1)
            continue
            
        # 2. ALGORITHM (Simple Proportional Controller for Demo)
        # Real version would use the MPC module
        error_R = target_R - state['R_axis']
        
        # Action: Adjust Outer Coils (PF3, PF4)
        # If R is too small (Inner), Push with Outer Coils? No, pull.
        # Simple logic: dI = Kp * error
        dI = Kp * error_R
        
        action = [0.0]*7
        action[2] = dI # PF3
        action[3] = dI # PF4
        
        # 3. ACTUATE
        bus['control_action'] = action
        
        time.sleep(0.05) # 20Hz Control Loop

def logger_node(bus, stop_event):
    print("[LOGGER] Node Started. Recording Stream...")
    
    history = []
    last_t = -1
    
    while not stop_event.is_set():
        state = bus.get('telemetry', None)
        if state and state['timestamp'] != last_t:
            try:
                print(f"LOG: T={state['timestamp']} | R={state['R_axis']:.2f}m | Erosion={state['Erosion_mm_year']:.1f} mm/y")
                last_t = state['timestamp']
                history.append(state)
            except KeyError:
                pass # Data not fully ready yet
            
        time.sleep(0.1)

def run_digital_twin_2_0(config_path: Path = DEFAULT_CONFIG_PATH):
    print("==================================================")
    print("   SCPN DIGITAL TWIN 2.0: REAL-TIME ENGINE        ")
    print("   Architecture: Asynchronous Multiprocessing     ")
    print("==================================================")
    
    # Shared Memory Manager
    manager = multiprocessing.Manager()
    bus = manager.dict()
    stop_event = manager.Event()
    
    # Processes
    p_phys = multiprocessing.Process(
        target=physics_node,
        args=(bus, stop_event, str(config_path)),
    )
    p_ctrl = multiprocessing.Process(target=control_node, args=(bus, stop_event))
    p_log = multiprocessing.Process(target=logger_node, args=(bus, stop_event))
    
    # Start
    p_phys.start()
    p_ctrl.start()
    p_log.start()
    
    # Run for 5 seconds
    time.sleep(5)
    
    # Shutdown
    print("\n[SYSTEM] Shutting down...")
    stop_event.set()
    p_phys.join()
    p_ctrl.join()
    p_log.join()
    print("[SYSTEM] Offline.")


def run_torax_hybrid_smoke(seed=42, episodes=8, steps_per_episode=180):
    """Run synthetic TORAX+SNN hybrid loop smoke for realtime path checks."""
    print("==================================================")
    print("   SCPN TORAX-HYBRID REALTIME LOOP (SMOKE)        ")
    print("==================================================")
    result = run_nstxu_torax_hybrid_campaign(
        seed=seed,
        episodes=episodes,
        steps_per_episode=steps_per_episode,
    )
    summary = {
        "disruption_avoidance_rate": result.disruption_avoidance_rate,
        "torax_parity_pct": result.torax_parity_pct,
        "p95_loop_latency_ms": result.p95_loop_latency_ms,
        "passes_thresholds": result.passes_thresholds,
    }
    print(
        f"[GAI-02] avoidance={summary['disruption_avoidance_rate']:.3f}, "
        f"parity={summary['torax_parity_pct']:.2f}%, "
        f"p95_latency={summary['p95_loop_latency_ms']:.4f} ms, "
        f"passes={summary['passes_thresholds']}"
    )
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SCPN realtime digital-twin or TORAX hybrid smoke."
    )
    parser.add_argument(
        "--torax-hybrid",
        action="store_true",
        help="Run TORAX-hybrid smoke benchmark instead of digital twin loop.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to reactor JSON configuration for digital twin mode.",
    )
    args = parser.parse_args()

    if args.torax_hybrid:
        run_torax_hybrid_smoke()
    else:
        run_digital_twin_2_0(config_path=args.config)
