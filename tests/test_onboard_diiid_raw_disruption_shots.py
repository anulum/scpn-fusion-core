# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for tools/onboard_diiid_raw_disruption_shots.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "onboard_diiid_raw_disruption_shots.py"
SPEC = importlib.util.spec_from_file_location("onboard_diiid_raw_disruption_shots", MODULE_PATH)
assert SPEC and SPEC.loader
onboard_mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = onboard_mod
SPEC.loader.exec_module(onboard_mod)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _download_result(
    *,
    source: str = "cache",
    data: np.ndarray[Any, Any] | None = None,
    time: np.ndarray[Any, Any] | None = None,
    signal_name: str = "Ip",
) -> SimpleNamespace:
    if data is None:
        data = np.linspace(0.0, 1.0, 32, dtype=np.float64)
    if time is None:
        time = np.linspace(0.0, 0.031, data.size, dtype=np.float64)
    return SimpleNamespace(
        source=source,
        signals={signal_name: SimpleNamespace(data=data, time=time)},
    )


def _run_onboard(
    tmp_path: Path,
    *,
    spec: dict[str, Any],
    refresh_manifest: bool = False,
) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        onboard_mod.onboard_shots(
            spec=spec,
            shot_dir=tmp_path / "shots",
            metadata_path=tmp_path / "metadata.json",
            cache_dir=tmp_path / "cache",
            force_download=False,
            refresh_manifest=refresh_manifest,
            manifest_path=tmp_path / "manifest.json",
        ),
    )


def test_onboard_shots_creates_npz_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Package a downloaded disruptive shot and record its raw provenance."""
    shot_dir = tmp_path / "shots"
    metadata_path = tmp_path / "disruption_shot_metadata.json"
    manifest_path = tmp_path / "disruption_shots_manifest.json"

    def _fake_download(**kwargs: Any) -> SimpleNamespace:
        del kwargs
        t = np.linspace(0.0, 0.159, 160, dtype=np.float64)
        signal = 0.8 + 0.2 * np.sin(2.0 * np.pi * 3.0 * t)
        return SimpleNamespace(
            source="mdsplus",
            signals={"Ip": SimpleNamespace(data=signal, time=t)},
        )

    monkeypatch.setattr(onboard_mod, "download_shot_data", _fake_download)
    monkeypatch.setattr(onboard_mod, "_refresh_manifest", lambda **_: None)

    summary = onboard_mod.onboard_shots(
        spec={
            "shots": [
                {
                    "shot": 163303,
                    "scenario": "raw_hmode",
                    "label": "disruptive",
                    "signals": ["Ip"],
                    "is_disruption": True,
                    "disruption_time_s": 0.12,
                }
            ]
        },
        shot_dir=shot_dir,
        metadata_path=metadata_path,
        cache_dir=tmp_path / "cache",
        force_download=False,
        refresh_manifest=False,
        manifest_path=manifest_path,
    )

    assert summary["created_count"] == 1
    npz_path = shot_dir / "shot_163303_raw_hmode.npz"
    assert npz_path.exists()
    with np.load(npz_path, allow_pickle=False) as payload:
        assert "dBdt_gauss_per_s" in payload
        assert "n1_amp" in payload
        assert "n2_amp" in payload
        assert bool(payload["is_disruption"])
        assert int(payload["disruption_time_idx"]) > 0

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    override = metadata["shot_overrides"]["shot_163303_raw_hmode.npz"]
    assert override["source_type"] == "raw_diiid_mdsplus_proxy"
    assert override["label"] == "disruptive"
    assert metadata["manifest_overrides"]["data_license"] == "mixed-v1"


def test_onboard_shots_reference_source_stays_non_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep reference fallback provenance distinct from raw facility data."""
    shot_dir = tmp_path / "shots"
    metadata_path = tmp_path / "disruption_shot_metadata.json"
    manifest_path = tmp_path / "disruption_shots_manifest.json"

    def _fake_download(**kwargs: Any) -> SimpleNamespace:
        del kwargs
        t = np.linspace(0.0, 0.159, 160, dtype=np.float64)
        signal = np.linspace(0.0, 1.0, 160, dtype=np.float64)
        return SimpleNamespace(
            source="reference",
            signals={"Ip": SimpleNamespace(data=signal, time=t)},
        )

    monkeypatch.setattr(onboard_mod, "download_shot_data", _fake_download)
    monkeypatch.setattr(onboard_mod, "_refresh_manifest", lambda **_: None)

    summary = onboard_mod.onboard_shots(
        spec={
            "shots": [
                {
                    "shot": 170000,
                    "scenario": "raw_reference",
                    "label": "safe",
                    "signals": ["Ip"],
                    "is_disruption": False,
                }
            ]
        },
        shot_dir=shot_dir,
        metadata_path=metadata_path,
        cache_dir=tmp_path / "cache",
        force_download=False,
        refresh_manifest=False,
        manifest_path=manifest_path,
    )

    assert summary["created_count"] == 1
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    override = metadata["shot_overrides"]["shot_170000_raw_reference.npz"]
    assert override["source_type"] == "reference_diiid_proxy"


def test_load_download_shot_data_uses_direct_script_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the direct-script import only when the package module is absent."""
    expected = object()
    calls: list[str] = []

    def _import(name: str) -> SimpleNamespace:
        calls.append(name)
        if name == "tools.download_diiid_data":
            error = ModuleNotFoundError("missing package module")
            error.name = name
            raise error
        return SimpleNamespace(download_shot_data=expected)

    monkeypatch.setattr(onboard_mod.importlib, "import_module", _import)
    assert onboard_mod._load_download_shot_data() is expected
    assert calls == ["tools.download_diiid_data", "download_diiid_data"]


def test_load_download_shot_data_preserves_nested_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagate missing nested dependencies instead of masking them."""

    def _import(_name: str) -> SimpleNamespace:
        error = ModuleNotFoundError("missing dependency")
        error.name = "nested_dependency"
        raise error

    monkeypatch.setattr(onboard_mod.importlib, "import_module", _import)
    with pytest.raises(ModuleNotFoundError, match="missing dependency"):
        onboard_mod._load_download_shot_data()


def test_download_reference_fallback_never_starts_live_profile_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloader = __import__("tools.download_diiid_data", fromlist=["_try_reference_data"])
    from scpn_fusion.io import tokamak_archive

    calls: list[dict[str, Any]] = []

    def fake_fetch_profiles(**kwargs: Any) -> list[Any]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr(
        tokamak_archive,
        "fetch_mdsplus_profiles",
        fake_fetch_profiles,
    )

    assert downloader._try_reference_data("DIII-D", 163303, ["Ip"], tmp_path) is None
    assert calls == []


def test_path_json_and_scenario_helpers(tmp_path: Path) -> None:
    """Resolve paths and reject malformed JSON or empty scenario names."""
    assert onboard_mod._resolve("relative.json") == onboard_mod.REPO_ROOT / "relative.json"
    assert onboard_mod._resolve(str(tmp_path)) == tmp_path

    object_path = tmp_path / "object.json"
    _write_json(object_path, {"value": 3})
    assert onboard_mod._load_json(object_path) == {"value": 3}

    list_path = tmp_path / "list.json"
    list_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        onboard_mod._load_json(list_path)

    assert onboard_mod._sanitize_scenario(" Raw H-mode / 2 ") == "raw_h_mode_2"
    with pytest.raises(ValueError, match="alphanumeric"):
        onboard_mod._sanitize_scenario(" -- ")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (np.array([], dtype=np.float64), np.array([0.0, 1.0])),
        (np.array([4.0], dtype=np.float64), np.array([0.0, 1.0])),
        (np.array([0.0, np.nan, 2.0]), np.arange(3, dtype=np.float64)),
        (np.array([0.0, 0.0, 2.0]), np.arange(3, dtype=np.float64)),
        (np.array([0.0, 0.5, 2.0]), np.array([0.0, 0.5, 2.0])),
    ],
)
def test_ensure_monotonic_timebase(
    raw: np.ndarray[Any, Any], expected: np.ndarray[Any, Any]
) -> None:
    """Repair short, non-finite, and non-monotonic sample axes."""
    np.testing.assert_array_equal(onboard_mod._ensure_monotonic_timebase(raw), expected)


def test_disruption_signal_validation_and_fallbacks() -> None:
    """Reject invalid signals and keep degenerate gradients finite."""
    with pytest.raises(ValueError, match="at least 2"):
        onboard_mod._derive_disruption_signal(
            np.array([1.0], dtype=np.float64), np.array([0.0], dtype=np.float64)
        )
    with pytest.raises(ValueError, match="non-finite"):
        onboard_mod._derive_disruption_signal(
            np.array([1.0, np.nan], dtype=np.float64), np.array([0.0, 1.0], dtype=np.float64)
        )

    duplicate_time = np.zeros(4, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        derived = onboard_mod._derive_disruption_signal(
            np.arange(4, dtype=np.float64), duplicate_time
        )
    assert np.all(np.isfinite(derived))

    constant = onboard_mod._derive_disruption_signal(
        np.ones(4, dtype=np.float64), np.arange(4, dtype=np.float64)
    )
    np.testing.assert_array_equal(constant, np.zeros(4, dtype=np.float64))


def test_disruption_index_selection_and_source_types() -> None:
    """Choose bounded disruption indices and explicit provenance labels."""
    time_s = np.linspace(0.0, 0.9, 10, dtype=np.float64)
    assert (
        onboard_mod._choose_disruption_index(
            n_samples=10,
            is_disruption=False,
            disruption_time_idx=None,
            disruption_time_s=None,
            time_s=time_s,
        )
        == -1
    )
    assert (
        onboard_mod._choose_disruption_index(
            n_samples=10,
            is_disruption=True,
            disruption_time_idx=-5,
            disruption_time_s=None,
            time_s=time_s,
        )
        == 1
    )
    assert (
        onboard_mod._choose_disruption_index(
            n_samples=10,
            is_disruption=True,
            disruption_time_idx=None,
            disruption_time_s=0.45,
            time_s=time_s,
        )
        == 5
    )
    assert (
        onboard_mod._choose_disruption_index(
            n_samples=10,
            is_disruption=True,
            disruption_time_idx=None,
            disruption_time_s=None,
            time_s=time_s,
        )
        == 8
    )
    assert onboard_mod._derive_source_type("cache") == "raw_diiid_cache_proxy"
    assert onboard_mod._derive_source_type("unknown") == "unknown_diiid_proxy"


def test_metadata_helpers_validate_and_preserve_overrides(tmp_path: Path) -> None:
    """Initialize metadata and validate both override mappings."""
    metadata_path = tmp_path / "metadata.json"
    assert onboard_mod._load_metadata(metadata_path) == {
        "manifest_overrides": {},
        "shot_overrides": {},
    }

    _write_json(
        metadata_path,
        {"manifest_overrides": {"data_license": "owner-v1"}, "shot_overrides": {}},
    )
    assert onboard_mod._load_metadata(metadata_path)["manifest_overrides"] == {
        "data_license": "owner-v1"
    }

    for key in ("manifest_overrides", "shot_overrides"):
        _write_json(metadata_path, {"manifest_overrides": {}, "shot_overrides": {}, key: []})
        with pytest.raises(ValueError, match=key):
            onboard_mod._load_metadata(metadata_path)


def test_refresh_manifest_uses_exact_generator_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invoke the manifest generator with exact paths and fail-fast semantics."""
    captured: dict[str, Any] = {}

    def _run(command: list[str], *, cwd: Path, check: bool) -> None:
        captured.update(command=command, cwd=cwd, check=check)

    monkeypatch.setattr(onboard_mod.subprocess, "run", _run)
    onboard_mod._refresh_manifest(
        shot_dir=tmp_path / "shots",
        metadata_path=tmp_path / "metadata.json",
        manifest_path=tmp_path / "manifest.json",
    )
    assert captured["command"][1] == "tools/generate_disruption_shot_manifest.py"
    assert captured["cwd"] == onboard_mod.REPO_ROOT
    assert captured["check"] is True


@pytest.mark.parametrize(
    ("shots", "message"),
    [
        ([], "non-empty"),
        (["bad"], "must be an object"),
        ([{"shot": True}], "positive integer"),
        ([{"shot": 0}], "positive integer"),
        ([{"shot": 1, "scenario": "---"}], "alphanumeric"),
        ([{"shot": 1, "label": "maybe"}], "label"),
        ([{"shot": 1, "signals": 3}], "list or CSV"),
        ([{"shot": 1, "signals": " , "}], "cannot be empty"),
    ],
)
def test_onboard_shots_rejects_malformed_spec(
    tmp_path: Path, shots: list[Any], message: str
) -> None:
    """Reject invalid shot definitions before downloader access."""
    with pytest.raises(ValueError, match=message):
        _run_onboard(tmp_path, spec={"shots": shots})


@pytest.mark.parametrize(
    ("result", "error"),
    [
        (SimpleNamespace(source="missing", signals={}), "no signals returned"),
        (_download_result(data=np.arange(8, dtype=np.float64)), "insufficient samples"),
        (
            _download_result(data=np.array([np.nan] * 16, dtype=np.float64)),
            "non-finite",
        ),
    ],
)
def test_onboard_shots_aggregates_nonfatal_data_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result: SimpleNamespace,
    error: str,
) -> None:
    """Aggregate missing, short, and non-finite downloaded signal failures."""
    monkeypatch.setattr(onboard_mod, "download_shot_data", lambda **_: result)
    summary = _run_onboard(tmp_path, spec={"shots": [{"shot": 123}]})
    assert summary["created_count"] == 0
    assert summary["failed_count"] == 1
    assert error in summary["failures"][0]["error"]


def test_onboard_shots_aggregates_download_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Record downloader exceptions without aborting the onboarding batch."""

    def _fail(**_kwargs: Any) -> None:
        raise RuntimeError("offline")

    monkeypatch.setattr(onboard_mod, "download_shot_data", _fail)
    summary = _run_onboard(tmp_path, spec={"shots": [{"shot": 123}]})
    assert summary["failures"] == [{"shot": 123, "error": "download failed: offline"}]


def test_onboard_shots_uses_fallback_signal_and_preserves_manifest_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use a returned fallback signal without replacing owner metadata."""
    monkeypatch.setattr(
        onboard_mod,
        "download_shot_data",
        lambda **_: _download_result(signal_name="beta_N"),
    )
    refresh_calls: list[dict[str, Path]] = []
    monkeypatch.setattr(
        onboard_mod, "_refresh_manifest", lambda **kwargs: refresh_calls.append(kwargs)
    )
    metadata_path = tmp_path / "metadata.json"
    _write_json(
        metadata_path,
        {
            "manifest_overrides": {
                "data_license": "owner-v1",
                "real_data_notice": "owner notice",
            },
            "shot_overrides": {},
        },
    )
    summary = _run_onboard(
        tmp_path,
        spec={"shots": [{"shot": 123, "signals": "Ip, beta_N", "primary_signal": "Ip"}]},
        refresh_manifest=True,
    )
    assert summary["created_files"] == ["shot_123_raw_123.npz"]
    metadata = onboard_mod._load_json(metadata_path)
    assert metadata["manifest_overrides"]["data_license"] == "owner-v1"
    assert metadata["manifest_overrides"]["real_data_notice"] == "owner notice"
    assert refresh_calls == [
        {
            "shot_dir": tmp_path / "shots",
            "metadata_path": tmp_path / "metadata.json",
            "manifest_path": tmp_path / "manifest.json",
        }
    ]


def test_onboard_shots_rejects_defensively_malformed_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep a defensive guard for malformed metadata returned by integrations."""
    monkeypatch.setattr(
        onboard_mod,
        "_load_metadata",
        lambda _path: {"manifest_overrides": [], "shot_overrides": {}},
    )
    with pytest.raises(ValueError, match="metadata payload malformed"):
        _run_onboard(tmp_path, spec={"shots": [{"shot": 123}]})


def test_onboard_shots_real_download_cache_integration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Onboard through the production downloader using only a local fresh cache."""
    downloader = __import__("tools.download_diiid_data", fromlist=["download_shot_data"])
    cache_dir = tmp_path / "cache"
    cache_path = downloader._cache_path(cache_dir, "DIII-D", 163303, ["Ip"])
    signal = downloader.SignalResult(
        name="Ip",
        data=np.linspace(0.0, 1.0, 32, dtype=np.float64),
        time=np.linspace(0.0, 0.031, 32, dtype=np.float64),
    )
    downloader._save_to_cache(cache_path, {"Ip": signal})
    monkeypatch.setattr(onboard_mod, "download_shot_data", downloader.download_shot_data)

    summary = cast(
        dict[str, Any],
        onboard_mod.onboard_shots(
            spec={"shots": [{"shot": 163303, "signals": ["Ip"]}]},
            shot_dir=tmp_path / "shots",
            metadata_path=tmp_path / "metadata.json",
            cache_dir=cache_dir,
            force_download=False,
            refresh_manifest=False,
            manifest_path=tmp_path / "manifest.json",
        ),
    )
    assert summary["created_count"] == 1
    assert (tmp_path / "shots" / "shot_163303_raw_163303.npz").is_file()


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        ({"created_count": 1, "failed_count": 0}, 0),
        ({"created_count": 0, "failed_count": 1}, 1),
        ({"created_count": 1, "failed_count": 1}, 2),
    ],
)
def test_main_writes_summary_and_returns_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    summary: dict[str, int],
    expected: int,
) -> None:
    """Write the CLI summary and distinguish clean, failed, and partial runs."""
    spec_path = tmp_path / "spec.json"
    summary_path = tmp_path / "summary.json"
    _write_json(spec_path, {"shots": [{"shot": 1}]})
    monkeypatch.setattr(onboard_mod, "onboard_shots", lambda **_: summary)
    status = onboard_mod.main(
        [
            "--spec",
            str(spec_path),
            "--shot-dir",
            str(tmp_path / "shots"),
            "--metadata",
            str(tmp_path / "metadata.json"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--summary-json",
            str(summary_path),
            "--force-download",
            "--skip-refresh-manifest",
        ]
    )
    assert status == expected
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary


def test_main_requires_existing_spec(tmp_path: Path) -> None:
    """Fail before mutation when the requested onboarding spec is absent."""
    try:
        onboard_mod.main(
            [
                "--spec",
                str(tmp_path / "missing_spec.json"),
                "--shot-dir",
                str(tmp_path / "shots"),
                "--metadata",
                str(tmp_path / "disruption_shot_metadata.json"),
            ]
        )
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError for missing spec file")
