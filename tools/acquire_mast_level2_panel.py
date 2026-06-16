# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — bounded FAIR-MAST Level-2 acquisition
"""Acquire a bounded FAIR-MAST Level-2 benchmark panel.

The script intentionally avoids downloading the whole FAIR-MAST archive. It
loads only the groups used by the current disruption benchmark lane: ``summary``
and ``magnetics``. fsspec's simplecache stores downloaded chunks under the
repository-local cache directory by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ENDPOINT_URL = "https://s3.echo.stfc.ac.uk"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


DEFAULT_CACHE_DIR = repo_root() / "data" / "mast_cache"
DEFAULT_RUN_DIR = repo_root() / "data" / "mast_runs"


def load_catalog(endpoint_url: str) -> list[int]:
    root = repo_root()
    sys.path.insert(0, str(root / "external"))

    import s3fs  # type: ignore

    fs = s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": endpoint_url})
    files = fs.ls("mast/level2/shots/")
    return sorted(
        int(Path(file_name).name.split(".")[0])
        for file_name in files
        if file_name.endswith(".zarr")
    )


def evict_shot_cache(shot_id: int, cache_dir: Path, endpoint_url: str) -> dict[str, Any]:
    """Remove hashed simplecache entries for one FAIR-MAST shot."""
    root = repo_root()
    sys.path.insert(0, str(root / "external"))

    import s3fs  # type: ignore

    remote_root = f"mast/level2/shots/{shot_id}.zarr"
    fs = s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": endpoint_url})
    removed: list[str] = []
    missing: list[str] = []
    for remote_path in fs.find(remote_root):
        cache_name = hashlib.sha256(remote_path.encode()).hexdigest()
        cache_path = cache_dir / cache_name
        if cache_path.exists():
            cache_path.unlink()
            removed.append(cache_name)
        else:
            missing.append(cache_name)
    return {
        "shot_id": shot_id,
        "remote_object_count": len(removed) + len(missing),
        "removed_cache_entry_count": len(removed),
    }


def select_targets(catalog: list[int], recent_count: int) -> list[int]:
    catalog_set = set(catalog)
    core = [30419, 30420, 30421, 30422, 30423, 30424, 29881, 29882, 30335, 30336]
    neighbors: list[int] = []
    for anchor in (29881, 30335, 30420):
        neighbors.extend(range(anchor - 3, anchor + 4))
    recent = catalog[-recent_count:] if recent_count > 0 else []

    targets: list[int] = []
    for shot_id in [*core, *neighbors, *recent]:
        if shot_id in catalog_set and shot_id not in targets:
            targets.append(shot_id)
    return targets


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def acquire_one(shot_id: int, cache_dir: Path, timeout_s: int) -> dict[str, Any]:
    root = repo_root()
    code = (
        "import json, sys; "
        f"sys.path.insert(0, {str(root / 'external')!r}); "
        f"sys.path.insert(0, {str(root / 'src')!r}); "
        "from scpn_fusion.io.mast_ingestor import MastIngestor; "
        f"ing=MastIngestor(cache_dir={str(cache_dir)!r}); "
        f"summary=ing.load_shot_summary({shot_id}); "
        f"mag=ing.load_magnetic_probes({shot_id}); "
        "print(json.dumps({"
        f"'shot_id': {shot_id}, "
        "'summary_samples': int(len(summary.get('time', []))), "
        "'summary_keys': sorted(summary.keys()), "
        "'magnetic_keys': sorted(mag.keys()), "
        "'magnetic_samples': int(len(mag.get('time', [])))"
        "}))"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'external'}:{root / 'src'}"
    env["SCPN_MAST_CACHE_DIR"] = str(cache_dir)

    started = time.time()
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        env=env,
        text=True,
        timeout=timeout_s,
    )
    entry: dict[str, Any] = {
        "shot_id": shot_id,
        "duration_s": round(time.time() - started, 3),
        "returncode": result.returncode,
    }
    if result.returncode == 0:
        entry.update(json.loads(result.stdout.strip().splitlines()[-1]))
        entry["status"] = "acquired"
    else:
        entry["status"] = "failed"
        entry["stderr_tail"] = result.stderr[-1600:]
        entry["stdout_tail"] = result.stdout[-1600:]
    return entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--recent-count", type=int, default=40)
    parser.add_argument(
        "--shots",
        nargs="+",
        type=int,
        help="Explicit shot IDs to acquire instead of the default benchmark panel.",
    )
    parser.add_argument("--timeout-s", type=int, default=240)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--evict-shot-cache",
        action="store_true",
        help="Delete existing simplecache entries for target shots before acquisition.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog(DEFAULT_ENDPOINT_URL)
    catalog_set = set(catalog)
    if args.shots:
        targets = []
        for shot_id in args.shots:
            if shot_id not in catalog_set:
                raise SystemExit(f"shot {shot_id} is not present in FAIR-MAST Level-2 catalog")
            if shot_id not in targets:
                targets.append(shot_id)
    else:
        targets = select_targets(catalog, args.recent_count)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = run_dir / f"level2_benchmark_panel_{run_id}.json"

    report: dict[str, Any] = {
        "schema": "scpn-control.mast-level2-benchmark-panel-run.v1",
        "created_utc": utc_now(),
        "repo": str(repo_root()),
        "cache_dir": str(cache_dir),
        "catalog_count": len(catalog),
        "catalog_min_shot": catalog[0] if catalog else None,
        "catalog_max_shot": catalog[-1] if catalog else None,
        "selection_policy": (
            "explicit shot retry panel"
            if args.shots
            else "known candidate disruption shots + neighboring controls + capped recent-shot panel"
        ),
        "target_count": len(targets),
        "targets": targets,
        "dry_run": args.dry_run,
        "evict_shot_cache": args.evict_shot_cache,
        "cache_evictions": [],
        "results": [],
    }
    write_report(report_path, report)
    print(f"report={report_path}")
    print(f"cache_dir={cache_dir}")
    print(f"target_count={len(targets)}")

    if args.dry_run:
        return 0

    for index, shot_id in enumerate(targets, start=1):
        if args.evict_shot_cache:
            eviction = evict_shot_cache(shot_id, cache_dir, DEFAULT_ENDPOINT_URL)
            report["cache_evictions"].append(eviction)
            report["updated_utc"] = utc_now()
            write_report(report_path, report)
            print(
                f"[{index}/{len(targets)}] shot={shot_id} "
                f"evicted_cache_entries={eviction['removed_cache_entry_count']}",
                flush=True,
            )
        print(f"[{index}/{len(targets)}] shot={shot_id} start", flush=True)
        entry = {"shot_id": shot_id, "status": "started", "started_utc": utc_now()}
        try:
            entry.update(acquire_one(shot_id, cache_dir, args.timeout_s))
        except subprocess.TimeoutExpired as exc:
            entry["status"] = "timeout"
            entry["stderr_tail"] = (exc.stderr or "")[-1600:] if isinstance(exc.stderr, str) else ""
            entry["stdout_tail"] = (exc.stdout or "")[-1600:] if isinstance(exc.stdout, str) else ""
        except Exception as exc:  # pragma: no cover - operational fault boundary
            entry["status"] = "error"
            entry["error"] = f"{type(exc).__name__}: {exc}"

        report["results"].append(entry)
        report["updated_utc"] = utc_now()
        report["acquired_count"] = sum(
            1 for result in report["results"] if result.get("status") == "acquired"
        )
        report["failed_count"] = len(report["results"]) - report["acquired_count"]
        write_report(report_path, report)
        print(
            f"[{index}/{len(targets)}] shot={shot_id} "
            f"status={entry.get('status')} duration_s={entry.get('duration_s')}",
            flush=True,
        )
        time.sleep(1)

    report["finished_utc"] = utc_now()
    write_report(report_path, report)
    print(
        f"finished acquired={report.get('acquired_count', 0)} failed={report.get('failed_count', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
