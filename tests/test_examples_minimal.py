# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Minimal Example Tests
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
