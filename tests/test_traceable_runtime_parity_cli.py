# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Traceable Runtime Parity CLI Tests
# ──────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import sys
from pathlib import Path

import validation.traceable_runtime_parity as parity_mod


def _summary(strict_ok: bool = True) -> dict[str, object]:
    return {
        "timestamp_utc": "2026-02-15T00:00:00+00:00",
        "steps": 8,
        "batch": 2,
        "seed": 1,
        "atol": 1e-8,
        "requested_backends": ["numpy"],
        "available_backends": ["numpy"],
        "spec": {
            "dt_s": 1e-3,
            "tau_s": 5e-3,
            "gain": 1.0,
            "command_limit": 1.0,
        },
        "reports": [
            {
                "backend": "numpy",
                "single_max_abs_err": 0.0,
                "batch_max_abs_err": 0.0,
                "single_within_tol": True,
                "batch_within_tol": True,
            }
        ],
        "strict_ok": strict_ok,
    }


def test_main_passes_backend_flags_and_writes_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run_parity_check(**kwargs):
        captured.update(kwargs)
        return _summary(strict_ok=True)

    json_out = tmp_path / "parity.json"
    md_out = tmp_path / "parity.md"
    monkeypatch.setattr(parity_mod, "run_parity_check", fake_run_parity_check)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "traceable_runtime_parity.py",
            "--steps",
            "8",
            "--batch",
            "2",
            "--backend",
            "numpy",
            "--backend",
            "torchscript",
            "--output-json",
            str(json_out),
            "--output-md",
            str(md_out),
        ],
    )

    rc = parity_mod.main()
    assert rc == 0
    assert captured["backends"] == ["numpy", "torchscript"]
    assert json_out.exists()
    assert md_out.exists()
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["strict_ok"] is True
    assert "Traceable Runtime Backend Parity" in md_out.read_text(encoding="utf-8")


def test_main_strict_returns_nonzero_when_not_ok(monkeypatch) -> None:
    monkeypatch.setattr(parity_mod, "run_parity_check", lambda **_kwargs: _summary(strict_ok=False))
    monkeypatch.setattr(sys, "argv", ["traceable_runtime_parity.py", "--strict"])
    rc = parity_mod.main()
    assert rc == 1
