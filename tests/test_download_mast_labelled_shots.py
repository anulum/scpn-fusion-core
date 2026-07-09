# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — MAST Labelled Shot Download Tests

from __future__ import annotations

import importlib.util
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "download_mast_labelled_shots.py"
SPEC = importlib.util.spec_from_file_location("tools.download_mast_labelled_shots", MODULE_PATH)
assert SPEC and SPEC.loader
download_mast = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = download_mast
SPEC.loader.exec_module(download_mast)


def _write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as JSON."""
    path.write_text(json.dumps(payload), encoding="utf-8")


def _valid_manifest() -> dict[str, Any]:
    """Return a minimal independent MAST label manifest."""
    return {
        "manifest_version": "mast-independent-disruption-labels-v1",
        "dataset": "FAIR-MAST Level-2 bounded disruption panel",
        "label_authority": "facility log export",
        "shots": [
            {
                "shot_id": 30456,
                "label": "disruptive",
                "disruption_time_s": 0.2218,
                "source_type": "facility_log",
                "source_reference": "facility-log://mast/30456",
                "labeled_by": "UKAEA facility log",
                "labeled_at_utc": "2026-06-16T00:00:00Z",
                "review_status": "accepted",
            },
            {
                "shot_id": 30420,
                "label": "non_disruptive",
                "source_type": "published_table",
                "source_reference": "doi:10.example/mast-controls",
                "labeled_by": "published table",
                "labeled_at_utc": "2026-06-16T00:00:00Z",
                "review_status": "accepted",
            },
        ],
    }


class FakeMastIngestor:
    """Local FAIR-MAST ingestor fake used to avoid network downloads."""

    instances: list[FakeMastIngestor] = []

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.closed = False
        FakeMastIngestor.instances.append(self)

    def load_shot_summary(self, shot_id: int) -> dict[str, NDArray[np.float64]]:
        """Return deterministic summary arrays for ``shot_id``."""
        return {
            "time": np.array([0.0, 0.1, 0.2], dtype=np.float64),
            "ip": np.array([float(shot_id), float(shot_id) + 1.0, np.nan], dtype=np.float64),
            "density": np.array([1.0, np.inf, 3.0], dtype=np.float64),
        }

    def load_magnetic_probes(self, shot_id: int) -> dict[str, NDArray[np.float64]]:
        """Return deterministic magnetic probes for ``shot_id``."""
        return {
            "time": np.array([0.0, 0.2], dtype=np.float64),
            "bp": np.array([1.0, np.nan], dtype=np.float64),
            "bt": np.array([5.0], dtype=np.float64),
        }

    def close(self) -> None:
        """Record explicit downloader cleanup."""
        self.closed = True


def test_missing_manifest_writes_blocked_download_report(tmp_path: Path) -> None:
    """Missing manifests write fail-closed download reports."""
    report_path = tmp_path / "download_report.json"

    rc = download_mast.main(
        [
            "--manifest",
            str(tmp_path / "missing.json"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--report",
            str(report_path),
        ]
    )

    assert rc == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "blocked_invalid_or_missing_independent_labels"
    assert report["target_count"] == 0
    assert report["downloaded_count"] == 0


def test_malformed_manifest_writes_blocked_download_report(tmp_path: Path) -> None:
    """Malformed manifests are reported instead of raising."""
    manifest_path = tmp_path / "independent_labels.json"
    report_path = tmp_path / "download_report.json"
    manifest_path.write_text("{", encoding="utf-8")

    rc = download_mast.main(
        [
            "--manifest",
            str(manifest_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--report",
            str(report_path),
        ]
    )

    assert rc == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "blocked_invalid_or_missing_independent_labels"
    assert report["errors"]


def test_non_object_manifest_writes_blocked_download_report(tmp_path: Path) -> None:
    """Non-object JSON manifests are blocked."""
    manifest_path = tmp_path / "independent_labels.json"
    report_path = tmp_path / "download_report.json"
    _write_json(manifest_path, [])

    rc = download_mast.main(
        [
            "--manifest",
            str(manifest_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--report",
            str(report_path),
        ]
    )

    assert rc == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "top-level object" in report["errors"][0]


def test_valid_manifest_dry_run_targets_only_labelled_shots(tmp_path: Path) -> None:
    """Dry-run mode writes target shot metadata without downloading."""
    manifest_path = tmp_path / "independent_labels.json"
    report_path = tmp_path / "download_report.json"
    _write_json(manifest_path, _valid_manifest())

    rc = download_mast.main(
        [
            "--manifest",
            str(manifest_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--report",
            str(report_path),
            "--dry-run",
        ]
    )

    assert rc == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "dry_run"
    assert report["target_shots"] == [30456, 30420]
    assert report["target_count"] == 2


def test_manifest_shot_ids_keep_unique_positive_ints() -> None:
    """Manifest shot extraction ignores malformed rows and duplicates."""
    payload = {
        "shots": [
            {"shot_id": 1},
            {"shot_id": 1},
            {"shot_id": True},
            {"shot_id": "2"},
            "not-an-object",
            {"shot_id": 3},
        ]
    }

    assert download_mast._manifest_shot_ids(payload) == [1, 3]
    assert download_mast._manifest_shot_ids({"shots": "bad"}) == []


def test_resample_to_summary_time_shapes_and_sanitizes() -> None:
    """Resampling preserves summary shape and removes non-finite values."""
    equal = download_mast._resample_to_summary_time(
        np.array([1.0, np.nan, np.inf], dtype=np.float64), 3
    )
    empty = download_mast._resample_to_summary_time(np.array([], dtype=np.float64), 2)
    short = download_mast._resample_to_summary_time(np.array([2.0, 4.0], dtype=np.float64), 4)

    assert equal.tolist() == [1.0, 0.0, 0.0]
    assert empty.tolist() == [0.0, 0.0]
    assert short.tolist() == [2.0, 2.0, 2.0, 4.0]


def test_build_blocked_report_counts_unique_targets() -> None:
    """Blocked reports include recoverable shot targets from invalid manifests."""
    report = download_mast.build_blocked_report(
        manifest_path=Path("labels.json"),
        cache_dir=Path("cache"),
        errors=["bad"],
        payload={"shots": [{"shot_id": 7}, {"shot_id": 7}, {"shot_id": 8}]},
    )

    assert report["status"] == "blocked_invalid_or_missing_independent_labels"
    assert report["target_shots"] == [7, 8]
    assert report["target_count"] == 2


def test_materialise_shot_reports_existing_npz(tmp_path: Path) -> None:
    """Existing shot archives are not downloaded again."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    out = cache_dir / "mast_shot_30456.npz"
    out.write_bytes(b"cached")

    result = download_mast.materialise_shot(FakeMastIngestor(tmp_path), cache_dir, 30456)

    assert result["status"] == "already_present"
    assert result["bytes"] == len(b"cached")
    assert result["sha256"] == download_mast._sha256(out)


def test_materialise_shot_downloads_compact_npz(tmp_path: Path) -> None:
    """Missing shot archives are materialized into compact NPZ files."""
    cache_dir = tmp_path / "cache"

    result = download_mast.materialise_shot(FakeMastIngestor(tmp_path), cache_dir, 30456)

    out = Path(result["path"])
    assert result["status"] == "downloaded"
    assert result["summary_samples"] == 3
    assert result["magnetic_channel_count"] == 2
    with np.load(out) as arrays:
        assert arrays["time"].tolist() == [0.0, 0.1, 0.2]
        assert arrays["density"].tolist() == [1.0, float("inf"), 3.0]
        assert arrays["mag_bp_field"].tolist() == [1.0, 1.0, 0.0]


def test_main_downloads_labelled_shots_with_fake_ingestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI downloads all manifest targets and closes the ingestor."""
    FakeMastIngestor.instances.clear()
    manifest_path = tmp_path / "independent_labels.json"
    report_path = tmp_path / "download_report.json"
    _write_json(manifest_path, _valid_manifest())
    monkeypatch.setattr(download_mast, "MastIngestor", FakeMastIngestor)

    rc = download_mast.main(
        [
            "--manifest",
            str(manifest_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--report",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert report["status"] == "downloaded"
    assert report["downloaded_count"] == 2
    assert report["already_present_count"] == 0
    assert report["failed_count"] == 0
    assert FakeMastIngestor.instances[0].closed is True


def test_main_reports_partial_download_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Operational per-shot failures produce a partial-failure report."""
    manifest_path = tmp_path / "independent_labels.json"
    report_path = tmp_path / "download_report.json"
    _write_json(manifest_path, _valid_manifest())
    monkeypatch.setattr(download_mast, "MastIngestor", FakeMastIngestor)

    def fake_materialise(_ingestor: FakeMastIngestor, _cache_dir: Path, shot_id: int) -> dict[str, Any]:
        """Return one success and one operational failure."""
        if shot_id == 30420:
            raise RuntimeError("network unavailable")
        return {"shot_id": shot_id, "status": "already_present"}

    monkeypatch.setattr(download_mast, "materialise_shot", fake_materialise)

    rc = download_mast.main(
        [
            "--manifest",
            str(manifest_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--report",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert report["status"] == "partial_failure"
    assert report["already_present_count"] == 1
    assert report["failed_count"] == 1
    assert "RuntimeError: network unavailable" in report["results"][1]["error"]


def test_cli_resolves_repo_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relative manifest, cache, and report paths resolve from ``REPO_ROOT``."""
    manifest_path = tmp_path / "labels.json"
    report_path = tmp_path / "reports" / "download_report.json"
    _write_json(manifest_path, _valid_manifest())
    monkeypatch.setattr(download_mast, "REPO_ROOT", tmp_path)

    rc = download_mast.main(
        [
            "--manifest",
            "labels.json",
            "--cache-dir",
            "cache",
            "--report",
            "reports/download_report.json",
            "--dry-run",
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert report["manifest_path"] == str(manifest_path)
    assert report["cache_dir"] == str(tmp_path / "cache")


def test_script_entrypoint_delegates_to_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Running the script entrypoint delegates to ``main``."""
    manifest_path = tmp_path / "independent_labels.json"
    report_path = tmp_path / "download_report.json"
    _write_json(manifest_path, _valid_manifest())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_mast_labelled_shots.py",
            "--manifest",
            str(manifest_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--report",
            str(report_path),
            "--dry-run",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(MODULE_PATH), run_name="__main__")

    assert excinfo.value.code == 0
