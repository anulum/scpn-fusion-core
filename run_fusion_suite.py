import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="SCPN Fusion Core - Simulation Suite")
    parser.add_argument("mode", choices=[
        "kernel", "flight", "optimal", "learning", "digital-twin", "control-room",
        "sandpile", "nuclear", "breeding", "safety", "optimizer", "divertor",
        "diagnostics", "sawtooth", "neural", "geometry", "spi", "scanner",
        "heating", "wdm", "quantum", "q-control", "neuro-control", "neuro-quantum",
        "lazarus", "director", "vibrana"
    ], help="Simulation mode to run")

    args = parser.parse_args()

    # Map modes to module paths (relative to repo root)
    scripts = {
        "kernel": "src/scpn_fusion/core/fusion_kernel.py",
        "flight": "src/scpn_fusion/control/tokamak_flight_sim.py",
        "optimal": "src/scpn_fusion/control/fusion_optimal_control.py",
        "learning": "src/scpn_fusion/control/advanced_soc_fusion_learning.py",
        "digital-twin": "src/scpn_fusion/control/tokamak_digital_twin.py",
        "control-room": "src/scpn_fusion/control/fusion_control_room.py",
        "sandpile": "src/scpn_fusion/core/sandpile_fusion_reactor.py",
        "nuclear": "src/scpn_fusion/nuclear/nuclear_wall_interaction.py",
        "breeding": "src/scpn_fusion/nuclear/blanket_neutronics.py",
        "safety": "src/scpn_fusion/control/disruption_predictor.py",
        "optimizer": "src/scpn_fusion/core/compact_reactor_optimizer.py",
        "divertor": "src/scpn_fusion/core/divertor_thermal_sim.py",
        "diagnostics": "src/scpn_fusion/diagnostics/run_diagnostics.py",
        "sawtooth": "src/scpn_fusion/core/mhd_sawtooth.py",
        "neural": "src/scpn_fusion/core/neural_equilibrium.py",
        "geometry": "src/scpn_fusion/core/geometry_3d.py",
        "spi": "src/scpn_fusion/control/spi_mitigation.py",
        "scanner": "src/scpn_fusion/core/global_design_scanner.py",
        "heating": "src/scpn_fusion/core/rf_heating.py",
        "wdm": "src/scpn_fusion/core/wdm_engine.py",
        "quantum": "src/scpn_fusion/core/quantum_bridge.py",
        "q-control": "src/scpn_fusion/core/quantum_bridge.py",
        "neuro-control": "src/scpn_fusion/control/neuro_cybernetic_controller.py",
        "neuro-quantum": "src/scpn_fusion/control/neuro_cybernetic_controller.py",
        "lazarus": "src/scpn_fusion/core/lazarus_bridge.py",
        "director": "src/scpn_fusion/control/director_interface.py",
        "vibrana": "src/scpn_fusion/core/vibrana_bridge.py"
    }

    script_path = scripts.get(args.mode)

    if script_path:
        # Resolve relative to this script's directory
        repo_root = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(repo_root, script_path)

        if not os.path.exists(full_path):
            print(f"Error: {script_path} not found")
            sys.exit(1)

        print(f"Launching SCPN Fusion Suite: {args.mode}...")
        sys.path.insert(0, repo_root)
        os.system(f'python "{full_path}"')
    else:
        print("Invalid mode.")
        sys.exit(1)

if __name__ == "__main__":
    main()
