"""Tests for tools/onboard_diiid_raw_disruption_shots.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "onboard_diiid_raw_disruption_shots.py"
SPEC = importlib.util.spec_from_file_location("onboard_diiid_raw_disruption_shots", MODULE_PATH)
assert SPEC and SPEC.loader
onboard_mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = onboard_mod
SPEC.loader.exec_module(onboard_mod)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_onboard_shots_creates_npz_and_metadata(tmp_path: Path, monkeypatch) -> None:
    shot_dir = tmp_path / "shots"
    metadata_path = tmp_path / "disruption_shot_metadata.json"
    manifest_path = tmp_path / "disruption_shots_manifest.json"

    def _fake_download(**kwargs):  # type: ignore[no-untyped-def]
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


def test_onboard_shots_reference_source_stays_non_raw(tmp_path: Path, monkeypatch) -> None:
    shot_dir = tmp_path / "shots"
    metadata_path = tmp_path / "disruption_shot_metadata.json"
    manifest_path = tmp_path / "disruption_shots_manifest.json"

    def _fake_download(**kwargs):  # type: ignore[no-untyped-def]
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


def test_main_requires_existing_spec(tmp_path: Path) -> None:
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
