# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Run Fusion Suite
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import argparse
import os
import sys

# Public simulation modes — fully self-contained, tested, validated.
PUBLIC_MODES = [
    "kernel", "flight", "optimal", "learning", "digital-twin", "control-room",
    "sandpile", "nuclear", "breeding", "safety", "optimizer", "divertor",
    "diagnostics", "sawtooth", "geometry", "spi", "scanner",
    "heating", "wdm", "neuro-control",
]

# Reduced-order surrogate modes — functional but not on the critical control
# path.  Use for rapid design-space exploration and batch sweeps.
# Unlocked with --surrogate or SCPN_SURROGATE=1.
SURROGATE_MODES = [
    "neural",  # PCA+MLP equilibrium surrogate (train with --train-sparc)
]

# Experimental modes — require external SCPN components not shipped in this
# repo.  Unlocked with --experimental or SCPN_EXPERIMENTAL=1.
EXPERIMENTAL_MODES = [
    "quantum", "q-control", "neuro-quantum", "lazarus", "director", "vibrana",
]

# Map every mode to its entry-point script.
SCRIPTS = {
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
    "neuro-control": "src/scpn_fusion/control/neuro_cybernetic_controller.py",
    # Experimental — bridges to external SCPN components
    "quantum": "src/scpn_fusion/core/quantum_bridge.py",
    "q-control": "src/scpn_fusion/core/quantum_bridge.py",
    "neuro-quantum": "src/scpn_fusion/control/neuro_cybernetic_controller.py",
    "lazarus": "src/scpn_fusion/core/lazarus_bridge.py",
    "director": "src/scpn_fusion/control/director_interface.py",
    "vibrana": "src/scpn_fusion/core/vibrana_bridge.py",
}


def main():
    experimental = os.environ.get("SCPN_EXPERIMENTAL", "0") == "1"
    surrogate = os.environ.get("SCPN_SURROGATE", "0") == "1"

    available = list(PUBLIC_MODES)
    if surrogate:
        available += SURROGATE_MODES
    if experimental:
        available += EXPERIMENTAL_MODES

    parser = argparse.ArgumentParser(
        description="SCPN Fusion Core - Simulation Suite",
        epilog="Set --surrogate or SCPN_SURROGATE=1 to unlock reduced-order "
               "surrogate modes.  Set --experimental or SCPN_EXPERIMENTAL=1 "
               "to unlock experimental modes (quantum, vibrana, lazarus, director).",
    )
    parser.add_argument("mode", choices=available, help="Simulation mode to run")
    parser.add_argument(
        "--surrogate", action="store_true",
        help="Unlock reduced-order surrogate modes (neural)",
    )
    parser.add_argument(
        "--experimental", action="store_true",
        help="Unlock experimental modes that depend on external SCPN components",
    )

    # Re-parse if flags were passed (need to rebuild choices).
    argv = sys.argv[1:]
    if "--experimental" in argv or "--surrogate" in argv:
        if "--experimental" in argv:
            experimental = True
        if "--surrogate" in argv:
            surrogate = True
        available = list(PUBLIC_MODES)
        if surrogate:
            available += SURROGATE_MODES
        if experimental:
            available += EXPERIMENTAL_MODES
        parser = argparse.ArgumentParser(
            description="SCPN Fusion Core - Simulation Suite",
        )
        parser.add_argument("mode", choices=available, help="Simulation mode to run")
        parser.add_argument("--experimental", action="store_true")
        parser.add_argument("--surrogate", action="store_true")

    args = parser.parse_args(argv)

    if args.mode in EXPERIMENTAL_MODES:
        print(f"[experimental] Running {args.mode} — requires external SCPN components.")
    if args.mode in SURROGATE_MODES:
        print(f"[surrogate] Running {args.mode} — reduced-order surrogate, not on critical path.")

    script_path = SCRIPTS.get(args.mode)

    if script_path:
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
