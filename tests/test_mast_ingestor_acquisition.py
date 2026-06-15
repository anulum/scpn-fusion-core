# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — MAST ingestion acquisition regression tests
"""Regression tests for bounded FAIR-MAST acquisition edge cases."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

from scpn_fusion.io import mast_ingestor
from scpn_fusion.io.mast_ingestor import MastIngestor

ROOT = Path(__file__).resolve().parents[1]
ACQUIRE_PATH = ROOT / "tools" / "acquire_mast_level2_panel.py"
SPEC = importlib.util.spec_from_file_location("acquire_mast_level2_panel", ACQUIRE_PATH)
assert SPEC is not None and SPEC.loader is not None
acquire_mast_level2_panel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acquire_mast_level2_panel)


class _FakeSummaryDataset:
    time = types.SimpleNamespace(values=np.array([0.0, 1.0, 2.0], dtype=np.float64))
    ip = types.SimpleNamespace(values=np.array([1.0, 2.0, 3.0], dtype=np.float64))

    def __contains__(self, key: str) -> bool:
        return False


class _FakeFileSystem:
    def get_mapper(self, url: str) -> str:
        return url


def test_load_shot_summary_uses_nan_density_when_missing(monkeypatch, tmp_path: Path) -> None:
    """MAST shots without line-average density remain usable."""

    monkeypatch.setattr(
        mast_ingestor.fsspec, "filesystem", lambda *args, **kwargs: _FakeFileSystem()
    )
    monkeypatch.setattr(
        mast_ingestor.xr, "open_zarr", lambda *args, **kwargs: _FakeSummaryDataset()
    )

    summary = MastIngestor(cache_dir=tmp_path).load_shot_summary(30447)

    np.testing.assert_allclose(summary["time"], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(summary["ip"], [1.0, 2.0, 3.0])
    assert np.isnan(summary["density"]).all()


def test_evict_shot_cache_removes_hashed_simplecache_entries(monkeypatch, tmp_path: Path) -> None:
    """Shot retries can clear corrupt simplecache objects without full-cache deletion."""

    remote_paths = [
        "mast/level2/shots/30454.zarr/summary/ip/c/0",
        "mast/level2/shots/30454.zarr/magnetics/b_field_pol_probe_cc_field/c/0/0",
    ]
    for remote_path in remote_paths:
        (tmp_path / hashlib.sha256(remote_path.encode()).hexdigest()).write_text(
            "stale", encoding="utf-8"
        )

    class _FakeS3FileSystem:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def find(self, remote_root: str) -> list[str]:
            assert remote_root == "mast/level2/shots/30454.zarr"
            return remote_paths

    fake_s3fs = types.SimpleNamespace(S3FileSystem=_FakeS3FileSystem)
    monkeypatch.setitem(sys.modules, "s3fs", fake_s3fs)

    report = acquire_mast_level2_panel.evict_shot_cache(
        30454, tmp_path, "https://s3.echo.stfc.ac.uk"
    )

    assert report == {
        "shot_id": 30454,
        "remote_object_count": 2,
        "removed_cache_entry_count": 2,
    }
    assert not any(tmp_path.iterdir())
