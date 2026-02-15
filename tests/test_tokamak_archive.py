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


def _sample_profile(shot: int, time_ms: float) -> archive.TokamakProfile:
    return archive.TokamakProfile(
        machine="DIII-D",
        shot=int(shot),
        time_ms=float(time_ms),
        beta_n=1.8,
        q95=4.6,
        tau_e_ms=120.0,
        psi_contour=tuple(float(v) for v in np.linspace(0.0, 1.0, 64)),
        sensor_trace=tuple(float(v) for v in np.linspace(0.1, 1.1, 96)),
        toroidal_n1_amp=0.12,
        toroidal_n2_amp=0.08,
        toroidal_n3_amp=0.04,
        disruption=True,
    )


def test_poll_mdsplus_feed_merges_live_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _live_fetch(**_: object) -> list[archive.TokamakProfile]:
        calls["n"] += 1
        if calls["n"] == 1:
            return [_sample_profile(shot=170001, time_ms=950.0)]
        return [
            _sample_profile(shot=170001, time_ms=950.0),
            _sample_profile(shot=170002, time_ms=980.0),
        ]

    monkeypatch.setattr(archive, "fetch_mdsplus_profiles", _live_fetch)
    rows, meta = archive.poll_mdsplus_feed(
        machine="DIII-D",
        host="atlas.gat.com",
        tree="EFIT01",
        shots=[170001, 170002],
        polls=2,
        poll_interval_ms=50,
    )
    assert len(rows) == 2
    assert meta["source"] == "live_stream"
    assert meta["live_total_profiles"] >= 3
    assert len(meta["poll_records"]) == 2


def test_poll_mdsplus_feed_falls_back_to_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    def _live_fail(**_: object) -> list[archive.TokamakProfile]:
        raise RuntimeError("mdsplus down")

    monkeypatch.setattr(archive, "fetch_mdsplus_profiles", _live_fail)
    rows, meta = archive.poll_mdsplus_feed(
        machine="DIII-D",
        host="atlas.gat.com",
        tree="EFIT01",
        shots=[170001],
        polls=2,
        fallback_to_reference=True,
    )
    assert rows
    assert meta["source"] == "reference_fallback"
    assert meta["fallback_meta"] is not None
    assert any("mdsplus down" in str(r["error"]) for r in meta["poll_records"])
