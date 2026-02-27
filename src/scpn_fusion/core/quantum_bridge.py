# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Quantum Bridge
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import sys
import subprocess
from pathlib import Path


_QUANTUM_SCRIPT_NAMES = (
    "14_quantum_plasma_simulation.py",
    "15_vqe_grad_shafranov.py",
    "16_knm_vqe_fusion.py",
)


def _resolve_quantum_lab_path(base_path: str | Path | None = None) -> Path:
    if base_path is not None:
        return Path(base_path).expanduser().resolve()
    return Path(__file__).resolve().parents[4] / "QUANTUM_LAB"


def run_quantum_suite(*, base_path: str | Path | None = None) -> dict[str, object]:
    print("--- SCPN QUANTUM FUSION BRIDGE ---")
    print("Leveraging Quantum Hardware for Plasma Physics")
    
    lab_path = _resolve_quantum_lab_path(base_path)
    if not lab_path.is_dir():
        raise FileNotFoundError(f"Quantum Lab not found at {lab_path}")

    script_paths = [lab_path / name for name in _QUANTUM_SCRIPT_NAMES]
    missing = [p for p in script_paths if not p.is_file()]
    if missing:
        missing_text = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"Quantum Lab missing required scripts: {missing_text}"
        )

    print("\n[1] Quantum Transport Simulation (Trotterization)")
    try:
        subprocess.run([sys.executable, str(script_paths[0])], check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Quantum script failed: {script_paths[0].name} (exit={exc.returncode})"
        ) from exc
    
    print("\n[2] Quantum Equilibrium Solver (VQE)")
    try:
        subprocess.run([sys.executable, str(script_paths[1])], check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Quantum script failed: {script_paths[1].name} (exit={exc.returncode})"
        ) from exc

    print("\n[3] Physics-Informed Knm-VQE (Topology Ansatz)")
    try:
        subprocess.run([sys.executable, str(script_paths[2])], check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Quantum script failed: {script_paths[2].name} (exit={exc.returncode})"
        ) from exc
    
    print("\n--- QUANTUM INTEGRATION COMPLETE ---")
    return {
        "ok": True,
        "base_path": str(lab_path),
        "scripts": [p.name for p in script_paths],
    }

if __name__ == "__main__":
    run_quantum_suite()
