# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — 10kHz Verification Script Tests
"""Smoke tests for validation/verify_10khz_rust.py behavior."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "validation" / "verify_10khz_rust.py"


def test_verify_10khz_rust_mock_mode_smoke() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--mode",
            "mock",
            "--shot-seconds",
            "0.05",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout + result.stderr
    assert "SCPN Fusion Core: 10kHz Rust Migration Verification" in out
    assert "Avg latency per step: N/A (mock mode)" in out
    assert "Max jitter (step time): N/A (mock mode)" in out
    assert "[MOCK] Structural verification completed" in out
