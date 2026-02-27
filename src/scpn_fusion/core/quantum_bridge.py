# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Quantum Bridge
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path


_QUANTUM_SCRIPT_NAMES = (
    "14_quantum_plasma_simulation.py",
    "15_vqe_grad_shafranov.py",
    "16_knm_vqe_fusion.py",
)
_QUANTUM_STEP_LABELS = (
    "[1] Quantum Transport Simulation (Trotterization)",
    "[2] Quantum Equilibrium Solver (VQE)",
    "[3] Physics-Informed Knm-VQE (Topology Ansatz)",
)
_QUANTUM_SCRIPT_TIMEOUT_SECONDS = 1800.0


def _resolve_quantum_lab_path(base_path: str | Path | None = None) -> Path:
    if base_path is not None:
        return Path(base_path).expanduser().resolve()
    return Path(__file__).resolve().parents[4] / "QUANTUM_LAB"


def _normalize_script_timeout_seconds(timeout_s: float) -> float:
    timeout = float(timeout_s)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("script_timeout_seconds must be finite and > 0.")
    return timeout


def run_quantum_suite(
    *,
    base_path: str | Path | None = None,
    script_timeout_seconds: float = _QUANTUM_SCRIPT_TIMEOUT_SECONDS,
) -> dict[str, object]:
    print("--- SCPN QUANTUM FUSION BRIDGE ---")
    print("Leveraging Quantum Hardware for Plasma Physics")

    timeout_seconds = _normalize_script_timeout_seconds(script_timeout_seconds)
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

    for step_label, script_path in zip(_QUANTUM_STEP_LABELS, script_paths):
        print(f"\n{step_label}")
        try:
            subprocess.run(
                [sys.executable, str(script_path)],
                check=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Quantum script timed out: {script_path.name} "
                f"(timeout={timeout_seconds:.1f}s)"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Quantum script failed: {script_path.name} (exit={exc.returncode})"
            ) from exc

    print("\n--- QUANTUM INTEGRATION COMPLETE ---")
    return {
        "ok": True,
        "base_path": str(lab_path),
        "scripts": [p.name for p in script_paths],
    }

if __name__ == "__main__":
    run_quantum_suite()
