# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — CLI tests for point-wise ψ RMSE validation
# ──────────────────────────────────────────────────────────────────────
"""
CLI-level regression tests for validation/psi_pointwise_rmse.py.

Ensures `main()` remains compatible with strict ASCII stdout encodings
that appear on some Windows terminals.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import validation.psi_pointwise_rmse as psi_rmse_mod
from validation.psi_pointwise_rmse import PsiRMSESummary


class _AsciiStrictStdout:
    """stdout stub that rejects non-ASCII writes."""

    encoding = "ascii"

    def __init__(self) -> None:
        self._parts: list[str] = []

    def write(self, text: str) -> int:
        text.encode("ascii", errors="strict")
        self._parts.append(text)
        return len(text)

    def flush(self) -> None:  # pragma: no cover - no side-effects to assert
        return None

    def getvalue(self) -> str:
        return "".join(self._parts)


def test_main_is_ascii_stdout_safe(
    tmp_path: Path, monkeypatch
) -> None:
    summary = PsiRMSESummary(
        count=1,
        mean_psi_rmse_norm=0.125,
        mean_psi_relative_l2=0.05,
        mean_gs_residual_l2=0.01,
        worst_psi_rmse_norm=0.125,
        worst_file="sample.geqdsk",
        rows=[
            {
                "file": "sample.geqdsk",
                "grid": "9x9",
                "psi_rmse_norm": 0.125,
                "psi_relative_l2": 0.05,
                "gs_residual_l2": 0.01,
                "sor_iterations": 12,
                "solve_time_ms": 3.4,
            }
        ],
    )

    monkeypatch.setattr(psi_rmse_mod, "validate_all_sparc", lambda: summary)
    monkeypatch.setattr(psi_rmse_mod, "ROOT", tmp_path)
    ascii_stdout = _AsciiStrictStdout()
    monkeypatch.setattr(sys, "stdout", ascii_stdout)

    rc = psi_rmse_mod.main()
    assert rc == 0
    assert "psi_N RMSE" in ascii_stdout.getvalue()

    report = tmp_path / "validation" / "reports" / "psi_pointwise_rmse.json"
    assert report.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["count"] == 1
    assert payload["worst_file"] == "sample.geqdsk"
