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


# ──────────────────────────────────────────────────────────────────────
# MDSplus mock tests (Phase 1 hardening)
# ──────────────────────────────────────────────────────────────────────


class _MockPayload:
    """Simulate MDSplus node data return."""

    def __init__(self, value: object) -> None:
        self._value = value

    def data(self) -> object:
        return self._value


class _MockConnection:
    """Simulate MDSplus.Connection with deterministic data."""

    def __init__(self, host: str) -> None:
        self.host = host
        self._open_tree: str | None = None
        self._open_shot: int | None = None

    def openTree(self, tree: str, shot: int) -> None:
        self._open_tree = tree
        self._open_shot = shot

    def get(self, node: str) -> _MockPayload:
        data_map: dict[str, object] = {
            "\\time_ms": np.array([950.0]),
            "\\betan": np.array([2.1]),
            "\\q95": np.array([4.5]),
            "\\taue_ms": np.array([120.0]),
            "\\psi_contour": np.linspace(0.0, 1.0, 64),
            "\\sensor_trace": np.linspace(0.1, 1.1, 96),
            "\\toroidal_n1_amp": np.array([0.12]),
            "\\toroidal_n2_amp": np.array([0.08]),
            "\\toroidal_n3_amp": np.array([0.04]),
            "\\disruption_flag": np.array([0.0]),
        }
        return _MockPayload(data_map.get(node, np.array([0.0])))


def test_fetch_mdsplus_profiles_with_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that fetch_mdsplus_profiles works with a mocked MDSplus.Connection."""
    import types

    mock_module = types.ModuleType("MDSplus")
    mock_module.Connection = _MockConnection  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "MDSplus", mock_module)

    rows = archive.fetch_mdsplus_profiles(
        machine="DIII-D",
        host="mock.local",
        tree="EFIT01",
        shots=[170001, 170002],
    )
    assert len(rows) == 2
    assert all(r.machine == "DIII-D" for r in rows)
    assert all(r.beta_n > 0.0 for r in rows)
    assert all(len(r.psi_contour) == 64 for r in rows)
    assert all(len(r.sensor_trace) == 96 for r in rows)


def test_fetch_mdsplus_profiles_partial_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that partial failures are handled with allow_partial=True."""
    import types

    call_count = {"n": 0}

    class _FailOnSecondConnection(_MockConnection):
        def openTree(self, tree: str, shot: int) -> None:
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("Shot not found")
            super().openTree(tree, shot)

    mock_module = types.ModuleType("MDSplus")
    mock_module.Connection = _FailOnSecondConnection  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "MDSplus", mock_module)

    rows = archive.fetch_mdsplus_profiles(
        machine="DIII-D",
        host="mock.local",
        tree="EFIT01",
        shots=[170001, 170002],
        allow_partial=True,
    )
    assert len(rows) == 1
    assert rows[0].shot == 170001


def test_poll_mdsplus_feed_with_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test poll_mdsplus_feed accumulation with mocked MDSplus."""
    import types

    mock_module = types.ModuleType("MDSplus")
    mock_module.Connection = _MockConnection  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "MDSplus", mock_module)

    rows, meta = archive.poll_mdsplus_feed(
        machine="DIII-D",
        host="mock.local",
        tree="EFIT01",
        shots=[170001],
        polls=2,
        poll_interval_ms=10,
    )
    assert len(rows) >= 1
    assert meta["source"] == "live_stream"
    assert meta["live_total_profiles"] >= 2
    assert len(meta["poll_records"]) == 2


def test_fetch_mdsplus_profiles_custom_node_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that custom node_map overrides work."""
    import types

    class _CustomNodeConnection(_MockConnection):
        def get(self, node: str) -> _MockPayload:
            if node == "\\custom_betan":
                return _MockPayload(np.array([3.5]))
            return super().get(node)

    mock_module = types.ModuleType("MDSplus")
    mock_module.Connection = _CustomNodeConnection  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "MDSplus", mock_module)

    rows = archive.fetch_mdsplus_profiles(
        machine="DIII-D",
        host="mock.local",
        tree="EFIT01",
        shots=[170001],
        node_map={"beta_n": "\\custom_betan"},
    )
    assert len(rows) == 1
    assert abs(rows[0].beta_n - 3.5) < 1e-6


def test_fetch_mdsplus_profiles_disruption_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test disruption flag parsing from MDSplus."""
    import types

    class _DisruptionConnection(_MockConnection):
        def get(self, node: str) -> _MockPayload:
            if node == "\\disruption_flag":
                return _MockPayload(np.array([1.0]))
            return super().get(node)

    mock_module = types.ModuleType("MDSplus")
    mock_module.Connection = _DisruptionConnection  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "MDSplus", mock_module)

    rows = archive.fetch_mdsplus_profiles(
        machine="DIII-D",
        host="mock.local",
        tree="EFIT01",
        shots=[170001],
    )
    assert len(rows) == 1
    assert rows[0].disruption is True


def test_fetch_mdsplus_profiles_cmod_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test C-Mod machine normalization with MDSplus mock."""
    import types

    mock_module = types.ModuleType("MDSplus")
    mock_module.Connection = _MockConnection  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "MDSplus", mock_module)

    rows = archive.fetch_mdsplus_profiles(
        machine="c-mod",
        host="mock.local",
        tree="EFIT01",
        shots=[1020101],
    )
    assert len(rows) == 1
    assert rows[0].machine == "C-Mod"
