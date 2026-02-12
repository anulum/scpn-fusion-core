import sys
import os
import subprocess

def run_quantum_suite():
    print("--- SCPN QUANTUM FUSION BRIDGE ---")
    print("Leveraging Quantum Hardware for Plasma Physics")
    
    # Path to Quantum Lab
    # Assuming standard directory structure: scpn/03_CODE/QUANTUM_LAB
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../QUANTUM_LAB'))
    
    if not os.path.exists(base_path):
        print(f"Error: Quantum Lab not found at {base_path}")
        return

    print("\n[1] Quantum Transport Simulation (Trotterization)")
    script_1 = os.path.join(base_path, "14_quantum_plasma_simulation.py")
    subprocess.run([sys.executable, script_1])
    
    print("\n[2] Quantum Equilibrium Solver (VQE)")
    script_2 = os.path.join(base_path, "15_vqe_grad_shafranov.py")
    subprocess.run([sys.executable, script_2])

    print("\n[3] Physics-Informed Knm-VQE (Topology Ansatz)")
    script_3 = os.path.join(base_path, "16_knm_vqe_fusion.py")
    subprocess.run([sys.executable, script_3])
    
    print("\n--- QUANTUM INTEGRATION COMPLETE ---")

if __name__ == "__main__":
    run_quantum_suite()
