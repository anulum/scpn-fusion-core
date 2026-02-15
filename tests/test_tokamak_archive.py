# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Tokamak Archive Tests
# ──────────────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import pytest

import scpn_fusion.io.tokamak_archive as archive


def test_load_diiid_reference_profiles_smoke() -> None:
    rows = archive.load_diiid_reference_profiles()
    assert rows
    assert all(r.machine == "DIII-D" for r in rows)
    assert any(r.disruption for r in rows)
    assert all(len(r.psi_contour) == 64 for r in rows)
    assert all(len(r.sensor_trace) == 96 for r in rows)
    assert np.isfinite([r.beta_n for r in rows]).all()


def test_load_cmod_reference_profiles_smoke() -> None:
    rows = archive.load_cmod_reference_profiles()
    assert rows
    assert all(r.machine == "C-Mod" for r in rows)
    assert all(len(r.psi_contour) == 64 for r in rows)
    assert all(len(r.sensor_trace) == 96 for r in rows)
    assert any(r.disruption for r in rows)
    assert np.isfinite([r.tau_e_ms for r in rows]).all()


def test_load_machine_profiles_live_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail_live(**_: object) -> list[archive.TokamakProfile]:
        raise RuntimeError("live unavailable")

    monkeypatch.setattr(archive, "fetch_mdsplus_profiles", _fail_live)
    rows, meta = archive.load_machine_profiles(
        machine="DIII-D",
        prefer_live=True,
        host="example.invalid",
        tree="EFIT01",
    )
    assert rows
    assert meta["live_attempted"] is True
    assert meta["source"] == "reference"
    assert "live unavailable" in str(meta["live_error"])


def test_fetch_mdsplus_profiles_rejects_empty_shots() -> None:
    with pytest.raises(ValueError, match="shots must be non-empty"):
        archive.fetch_mdsplus_profiles(
            machine="DIII-D",
            host="atlas.gat.com",
            tree="EFIT01",
            shots=[],
        )
