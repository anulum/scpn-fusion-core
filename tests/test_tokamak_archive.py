# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tokamak Archive Tests

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

import scpn_fusion.io.tokamak_archive as archive
from scpn_fusion.fallback_telemetry import (
    reset_fallback_telemetry,
    snapshot_fallback_telemetry,
)


def test_load_diiid_reference_profiles_smoke() -> None:
    rows = archive.load_diiid_reference_profiles()
    assert rows
    assert all(r.machine == "DIII-D" for r in rows)
    assert any(r.disruption for r in rows)
    assert all(len(r.psi_contour) == 64 for r in rows)
    assert all(len(r.sensor_trace) == 96 for r in rows)
    assert np.isfinite([r.beta_n for r in rows]).all()


def test_load_diiid_reference_profiles_rejects_empty_directory(tmp_path: Path) -> None:
    """Reject DIII-D reference directories with no GEQDSK files."""
    with pytest.raises(ValueError, match="No DIII-D reference files"):
        archive.load_diiid_reference_profiles(reference_dir=tmp_path)


def test_load_cmod_reference_profiles_smoke() -> None:
    rows = archive.load_cmod_reference_profiles()
    assert rows
    assert all(r.machine == "C-Mod" for r in rows)
    assert all(len(r.psi_contour) == 64 for r in rows)
    assert all(len(r.sensor_trace) == 96 for r in rows)
    assert any(r.disruption for r in rows)
    assert np.isfinite([r.tau_e_ms for r in rows]).all()


def test_load_cmod_reference_profiles_rejects_missing_machine_rows(
    tmp_path: Path,
) -> None:
    """Reject C-Mod CSV inputs that contain no C-Mod machine rows."""
    csv_path = tmp_path / "itpa.csv"
    csv_path.write_text(
        "machine,shot,Ip_MA,BT_T,tau_E_s,H98y2,kappa,delta\n"
        "DIII-D,170001,1.2,2.1,0.05,0.95,1.8,0.3\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="No C-Mod rows"):
        archive.load_cmod_reference_profiles(itpa_csv_path=csv_path)


def test_default_data_root_can_be_overridden_by_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "reference_data"
    disruption_dir = data_root / "diiid" / "disruption_shots"
    synthetic_dir = data_root / "synthetic_shots"
    disruption_dir.mkdir(parents=True)
    synthetic_dir.mkdir(parents=True)
    (disruption_dir / "shot_env_override.npz").write_bytes(b"placeholder")
    (synthetic_dir / "shot_env_override.npz").write_bytes(b"placeholder")

    monkeypatch.setenv("SCPN_DATA_DIR", str(data_root))

    assert archive.default_reference_data_root() == data_root
    assert archive.list_disruption_shots() == ["shot_env_override"]
    assert archive.list_synthetic_shots() == ["shot_env_override"]


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


def test_load_machine_profiles_reports_missing_cmod_live_config() -> None:
    """Fall back to C-Mod reference data when live host or tree is omitted."""
    rows, meta = archive.load_machine_profiles(machine="c-mod", prefer_live=True)

    assert rows
    assert all(row.machine == "C-Mod" for row in rows)
    assert meta["live_attempted"] is True
    assert meta["live_error"] == "Missing host/tree for live MDSplus fetch."
    assert meta["source"] == "reference"


def test_load_machine_profiles_merges_live_and_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Merge live profiles over the reference set when live fetch succeeds."""
    live = _sample_profile(shot=999001, time_ms=1234.0)

    def _live_fetch(**_: object) -> list[archive.TokamakProfile]:
        return [live]

    monkeypatch.setattr(archive, "fetch_mdsplus_profiles", _live_fetch)

    rows, meta = archive.load_machine_profiles(
        machine="DIII-D",
        prefer_live=True,
        host="mock.local",
        tree="EFIT01",
        shots=[live.shot],
    )

    assert live in rows
    assert meta["source"] == "live+reference"
    assert meta["live_count"] == 1


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
    reset_fallback_telemetry()

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
    snap = snapshot_fallback_telemetry()
    assert snap["domain_counts"]["tokamak_archive"] >= 1


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


def _install_mock_mdsplus(
    monkeypatch: pytest.MonkeyPatch,
    connection_type: type[_MockConnection],
    *,
    mds_exception: type[BaseException] | None = None,
) -> types.ModuleType:
    """Install a typed mock MDSplus module into ``sys.modules``."""
    mock_module = types.ModuleType("MDSplus")
    mock_module.__dict__["Connection"] = connection_type
    if mds_exception is not None:
        exception_module = types.SimpleNamespace(MdsException=mds_exception)
        mock_module.__dict__["mdsExceptions"] = exception_module
    monkeypatch.setitem(sys.modules, "MDSplus", mock_module)
    return mock_module


def test_fetch_mdsplus_profiles_with_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that fetch_mdsplus_profiles works with a mocked MDSplus.Connection."""
    _install_mock_mdsplus(monkeypatch, _MockConnection)

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
    call_count = {"n": 0}

    class _FailOnSecondConnection(_MockConnection):
        def openTree(self, tree: str, shot: int) -> None:
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("Shot not found")
            super().openTree(tree, shot)

    _install_mock_mdsplus(monkeypatch, _FailOnSecondConnection)

    rows = archive.fetch_mdsplus_profiles(
        machine="DIII-D",
        host="mock.local",
        tree="EFIT01",
        shots=[170001, 170002],
        allow_partial=True,
    )
    assert len(rows) == 1
    assert rows[0].shot == 170001


def test_fetch_mdsplus_profiles_reraises_when_partial_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagate shot failures when partial live ingestion is disabled."""

    class _FailingConnection(_MockConnection):
        def openTree(self, tree: str, shot: int) -> None:
            raise RuntimeError(f"shot {shot} unavailable")

    _install_mock_mdsplus(monkeypatch, _FailingConnection)

    with pytest.raises(RuntimeError, match="shot 170001 unavailable"):
        archive.fetch_mdsplus_profiles(
            machine="DIII-D",
            host="mock.local",
            tree="EFIT01",
            shots=[170001],
            allow_partial=False,
        )


def test_fetch_mdsplus_profiles_partial_failure_uses_mds_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat MDSplus-specific exceptions as recoverable partial-shot failures."""

    class _MockMdsException(Exception):
        """Fake MDSplus base exception exposed by the optional dependency."""

    class _MdsExceptionConnection(_MockConnection):
        def openTree(self, tree: str, shot: int) -> None:
            if shot == 170002:
                raise _MockMdsException("shot unavailable")
            super().openTree(tree, shot)

    _install_mock_mdsplus(
        monkeypatch,
        _MdsExceptionConnection,
        mds_exception=_MockMdsException,
    )

    rows = archive.fetch_mdsplus_profiles(
        machine="DIII-D",
        host="mock.local",
        tree="EFIT01",
        shots=[170001, 170002],
        allow_partial=True,
    )

    assert [row.shot for row in rows] == [170001]


def test_poll_mdsplus_feed_with_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test poll_mdsplus_feed accumulation with mocked MDSplus."""
    _install_mock_mdsplus(monkeypatch, _MockConnection)

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

    class _CustomNodeConnection(_MockConnection):
        def get(self, node: str) -> _MockPayload:
            if node == "\\custom_betan":
                return _MockPayload(np.array([3.5]))
            return super().get(node)

    _install_mock_mdsplus(monkeypatch, _CustomNodeConnection)

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

    class _DisruptionConnection(_MockConnection):
        def get(self, node: str) -> _MockPayload:
            if node == "\\disruption_flag":
                return _MockPayload(np.array([1.0]))
            return super().get(node)

    _install_mock_mdsplus(monkeypatch, _DisruptionConnection)

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
    _install_mock_mdsplus(monkeypatch, _MockConnection)

    rows = archive.fetch_mdsplus_profiles(
        machine="c-mod",
        host="mock.local",
        tree="EFIT01",
        shots=[1020101],
    )
    assert len(rows) == 1
    assert rows[0].machine == "C-Mod"


def test_load_disruption_shot_rejects_object_array_payload(tmp_path: Path) -> None:
    """Disruption shot loader should reject object arrays under secure defaults."""
    path = tmp_path / "bad_disruption.npz"
    zeros = np.zeros(1000, dtype=np.float64)
    np.savez(
        path,
        time_s=zeros,
        Ip_MA=zeros,
        BT_T=zeros,
        beta_N=zeros,
        q95=zeros,
        ne_1e19=zeros,
        n1_amp=zeros,
        n2_amp=zeros,
        locked_mode_amp=zeros,
        dBdt_gauss_per_s=zeros,
        vertical_position_m=zeros,
        is_disruption=np.array([True], dtype=object),
        disruption_time_idx=np.array([500], dtype=object),
        disruption_type=np.array(["locked_mode"], dtype=object),
    )
    with pytest.raises(ValueError):
        archive.load_disruption_shot(path)


def test_shot_loaders_reject_non_npz_extension(tmp_path: Path) -> None:
    bad = tmp_path / "not_npz.txt"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="\\.npz"):
        archive.load_synthetic_shot(bad)
    with pytest.raises(ValueError, match="\\.npz"):
        archive.load_disruption_shot(bad)
