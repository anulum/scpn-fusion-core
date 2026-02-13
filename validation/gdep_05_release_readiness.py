# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GDEP-05 Release Readiness Validation
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""GDEP-05: deterministic release-readiness gate for v2.0-cutting-edge."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRACKER_PATH = ROOT / "docs" / "PHASE2_ADVANCED_RFC_TRACKER.md"
PHASE3_QUEUE_PATH = ROOT / "docs" / "PHASE3_EXECUTION_REGISTRY.md"
CHANGELOG_PATH = ROOT / "CHANGELOG.md"

REQUIRED_DONE_TASKS = (
    "GNEU-01",
    "GNEU-02",
    "GNEU-03",
    "GAI-01",
    "GAI-02",
    "GAI-03",
    "GMVR-01",
    "GMVR-02",
    "GMVR-03",
    "GDEP-01",
    "GDEP-02",
    "GDEP-03",
    "GDEP-05",
)

REQUIRED_CHANGELOG_PHRASE = (
    "Novum elevated - SNN + GyroSwin hybrids for resilient, 1000x fast control; "
    "MVR grounded in 2025-2026 liquid metal/HTS."
)


def parse_tracker_statuses(path: Path) -> dict[str, str]:
    statuses: dict[str, str] = {}
    pattern = re.compile(r"^\|\s*([A-Z]+-\d{2})\s*\|\s*([^|]+?)\s*\|")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        task_id = match.group(1).strip()
        status = match.group(2).strip()
        statuses[task_id] = status
    return statuses


def parse_phase3_active_statuses(path: Path) -> dict[str, str]:
    statuses: dict[str, str] = {}
    pattern = re.compile(r"^\-\s*(Completed|In progress):\s*`(S2-\d{3})`")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        status = match.group(1).strip()
        task_id = match.group(2).strip()
        statuses[task_id] = status
    return statuses


def run_campaign(
    *,
    tracker_path: Path = TRACKER_PATH,
    phase3_queue_path: Path = PHASE3_QUEUE_PATH,
    changelog_path: Path = CHANGELOG_PATH,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    statuses = parse_tracker_statuses(tracker_path)
    phase3_statuses = parse_phase3_active_statuses(phase3_queue_path)
    changelog_text = changelog_path.read_text(encoding="utf-8")

    missing_done = [
        task_id
        for task_id in REQUIRED_DONE_TASKS
        if statuses.get(task_id, "Missing") != "Done"
    ]

    changelog_phrase_present = REQUIRED_CHANGELOG_PHRASE in changelog_text
    passes = bool(not missing_done and changelog_phrase_present)
    s2_completed = sorted(
        task_id for task_id, status in phase3_statuses.items() if status == "Completed"
    )
    s2_in_progress = sorted(
        task_id for task_id, status in phase3_statuses.items() if status == "In progress"
    )

    return {
        "tracker_path": str(tracker_path),
        "phase3_queue_path": str(phase3_queue_path),
        "changelog_path": str(changelog_path),
        "required_done_tasks": list(REQUIRED_DONE_TASKS),
        "required_done_count": len(REQUIRED_DONE_TASKS),
        "done_count": len(REQUIRED_DONE_TASKS) - len(missing_done),
        "missing_done_tasks": missing_done,
        "s2_queue_health": {
            "completed_count": len(s2_completed),
            "in_progress_count": len(s2_in_progress),
            "in_progress_tasks": s2_in_progress,
            "completed_tasks": s2_completed,
            "parse_ok": bool(phase3_statuses),
        },
        "required_changelog_phrase": REQUIRED_CHANGELOG_PHRASE,
        "changelog_phrase_present": changelog_phrase_present,
        "passes_thresholds": passes,
        "runtime_seconds": float(time.perf_counter() - t0),
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gdep_05": run_campaign(**kwargs),
    }


def render_markdown(report: dict[str, Any]) -> str:
    g = report["gdep_05"]
    lines = [
        "# GDEP-05 Release Readiness",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{g['runtime_seconds']:.3f} s`",
        "",
        "## Tracker Coverage",
        "",
        f"- Required done tasks: `{g['required_done_count']}`",
        f"- Done count: `{g['done_count']}`",
        f"- Missing done tasks: `{', '.join(g['missing_done_tasks']) if g['missing_done_tasks'] else 'none'}`",
        "",
        "## Changelog Contract",
        "",
        f"- Phrase present: `{'YES' if g['changelog_phrase_present'] else 'NO'}`",
        f"- Required phrase: `{g['required_changelog_phrase']}`",
        "",
        "## Phase 3 Queue",
        "",
        f"- Parsed queue metadata: `{'YES' if g['s2_queue_health']['parse_ok'] else 'NO'}`",
        f"- Completed S2 tasks tracked: `{g['s2_queue_health']['completed_count']}`",
        f"- In-progress S2 tasks tracked: `{g['s2_queue_health']['in_progress_count']}`",
        f"- Active S2 tasks: `{', '.join(g['s2_queue_health']['in_progress_tasks']) if g['s2_queue_health']['in_progress_tasks'] else 'none'}`",
        "",
        f"- Overall pass: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker-path", default=str(TRACKER_PATH))
    parser.add_argument("--phase3-queue-path", default=str(PHASE3_QUEUE_PATH))
    parser.add_argument("--changelog-path", default=str(CHANGELOG_PATH))
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "gdep_05_release_readiness.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "gdep_05_release_readiness.md"),
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = generate_report(
        tracker_path=Path(args.tracker_path),
        phase3_queue_path=Path(args.phase3_queue_path),
        changelog_path=Path(args.changelog_path),
    )

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = report["gdep_05"]
    print("GDEP-05 release readiness validation complete.")
    print(f"passes_thresholds={g['passes_thresholds']}")
    print(
        "Summary -> "
        f"done_count={g['done_count']}/{g['required_done_count']}, "
        f"changelog_phrase_present={g['changelog_phrase_present']}"
    )

    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
