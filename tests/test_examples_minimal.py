# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Minimal Example Tests
# © 1998–2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Smoke tests for examples/minimal.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "minimal.py"


def test_minimal_example_file_exists() -> None:
    assert SCRIPT.exists()


def test_minimal_example_runs_equilibrium_and_controller() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--grid",
            "17",
            "--equilibrium-iters",
            "4",
            "--seed",
            "7",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=180.0,
    )
    assert result.returncode == 0, result.stderr
    assert "equilibrium_converged=" in result.stdout
    assert "controller_backend=" in result.stdout
