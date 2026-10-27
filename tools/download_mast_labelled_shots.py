# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — MAST Labelled Shot Downloader
"""Download FAIR-MAST shots named by the independent label manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from scpn_fusion.io.mast_ingestor import MastIngestor
from tools.check_mast_label_manifest import (
    DEFAULT_MANIFEST,
    DEFAULT_REPORT,
    _load_json,
    validate_manifest,
)


DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "mast_snn_rust_cache"
DEFAULT_DOWNLOAD_REPORT = REPO_ROOT / "validation" / "reports" / "mast_labelled_shot_downloads.json"


def utc_now() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _load_manifest(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.exists():
        return None, [f"label manifest not found: {path}"]
    try:
        payload = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, [str(exc)]
    return payload, validate_manifest(payload)


def _manifest_shot_ids(payload: dict[str, Any]) -> list[int]:
    shots = payload.get("shots", [])
    shot_ids: list[int] = []
    if not isinstance(shots, list):
        return shot_ids
    for item in shots:
        if not isinstance(item, dict):
            continue
        shot_id = item.get("shot_id")
        if isinstance(shot_id, int) and not isinstance(shot_id, bool) and shot_id not in shot_ids:
            shot_ids.append(shot_id)
    return shot_ids


def _resample_to_summary_time(values: NDArray[np.float64], n_time: int) -> NDArray[np.float64]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(flat) == n_time:
        return np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    if len(flat) == 0:
        return np.zeros(n_time, dtype=np.float64)
    sample_idx = np.linspace(0, len(flat) - 1, n_time).astype(np.int64)
    return np.nan_to_num(flat[sample_idx], nan=0.0, posinf=0.0, neginf=0.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialise_shot(ingestor: MastIngestor, cache_dir: Path, shot_id: int) -> dict[str, Any]:
    """Download one FAIR-MAST shot and write the compact NPZ used by Rust validation."""
    out = cache_dir / f"mast_shot_{shot_id}.npz"
    if out.exists():
        return {
            "shot_id": shot_id,
            "status": "already_present",
            "path": str(out),
            "bytes": out.stat().st_size,
            "sha256": _sha256(out),
        }

    summary = ingestor.load_shot_summary(shot_id)
    magnetics = ingestor.load_magnetic_probes(shot_id)
    time_s = np.asarray(summary["time"], dtype=np.float64)
    arrays: dict[str, NDArray[np.float64]] = {
        "time": time_s,
        "ip": np.asarray(summary["ip"], dtype=np.float64),
        "density": np.asarray(summary["density"], dtype=np.float64),
    }
    for key, value in magnetics.items():
        if key == "time":
            continue
        arrays[f"mag_{key}_field"] = _resample_to_summary_time(np.asarray(value), len(time_s))

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".npz.tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(out)
    return {
        "shot_id": shot_id,
        "status": "downloaded",
        "path": str(out),
        "bytes": out.stat().st_size,
        "sha256": _sha256(out),
        "summary_samples": int(len(time_s)),
        "magnetic_channel_count": int(sum(1 for key in arrays if key.startswith("mag_"))),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a deterministic JSON payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_blocked_report(
    *,
    manifest_path: Path,
    cache_dir: Path,
    errors: list[str],
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the fail-closed report emitted when independent labels are invalid."""
    shot_ids = _manifest_shot_ids(payload) if payload is not None else []
    return {
        "schema": "scpn-fusion-core.mast-labelled-shot-downloads.v1",
        "created_utc": utc_now(),
        "status": "blocked_invalid_or_missing_independent_labels",
        "manifest_path": str(manifest_path),
        "cache_dir": str(cache_dir),
        "target_shots": shot_ids,
        "target_count": len(shot_ids),
        "downloaded_count": 0,
        "already_present_count": 0,
        "failed_count": 0,
        "errors": errors,
        "results": [],
        "label_readiness_report": str(DEFAULT_REPORT),
        "claim_boundary": (
            "This downloader only accepts MAST shots named by an independently "
            "validated disruptive/non_disruptive label manifest. It does not infer "
            "labels from current collapse, neighboring controls, or model output."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Download labelled FAIR-MAST shots after manifest validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_DOWNLOAD_REPORT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    cache_dir = args.cache_dir
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    report_path = args.report
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path

    payload, errors = _load_manifest(manifest_path)
    if errors:
        blocked_report = build_blocked_report(
            manifest_path=manifest_path,
            cache_dir=cache_dir,
            errors=errors,
            payload=payload,
        )
        write_json(report_path, blocked_report)
        print(f"MAST labelled shot download blocked: {len(errors)} label issue(s)")
        print(f"report={report_path}")
        return 1

    assert payload is not None
    targets = _manifest_shot_ids(payload)
    report: dict[str, Any] = {
        "schema": "scpn-fusion-core.mast-labelled-shot-downloads.v1",
        "created_utc": utc_now(),
        "status": "dry_run" if args.dry_run else "running",
        "manifest_path": str(manifest_path),
        "cache_dir": str(cache_dir),
        "target_shots": targets,
        "target_count": len(targets),
        "downloaded_count": 0,
        "already_present_count": 0,
        "failed_count": 0,
        "errors": [],
        "results": [],
        "claim_boundary": (
            "Downloaded shots inherit labels only from the accepted independent "
            "manifest; FAIR-MAST signal availability alone is not label evidence."
        ),
    }
    write_json(report_path, report)
    if args.dry_run:
        print(f"MAST labelled shot dry-run target_count={len(targets)}")
        print(f"report={report_path}")
        return 0

    ingestor = MastIngestor(cache_dir=cache_dir / "_zarr_simplecache")
    try:
        for shot_id in targets:
            try:
                result = materialise_shot(ingestor, cache_dir, shot_id)
            except Exception as exc:  # pragma: no cover - operational network boundary
                result = {
                    "shot_id": shot_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            report["results"].append(result)
            report["downloaded_count"] = sum(
                1 for item in report["results"] if item.get("status") == "downloaded"
            )
            report["already_present_count"] = sum(
                1 for item in report["results"] if item.get("status") == "already_present"
            )
            report["failed_count"] = sum(
                1 for item in report["results"] if item.get("status") == "failed"
            )
            report["updated_utc"] = utc_now()
            write_json(report_path, report)
            print(f"shot={shot_id} status={result['status']}", flush=True)
    finally:
        ingestor.close()

    report["finished_utc"] = utc_now()
    report["status"] = "downloaded" if report["failed_count"] == 0 else "partial_failure"
    write_json(report_path, report)
    print(
        "finished "
        f"downloaded={report['downloaded_count']} "
        f"already_present={report['already_present_count']} "
        f"failed={report['failed_count']}"
    )
    print(f"report={report_path}")
    return 0 if report["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
