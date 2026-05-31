#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Download public upstream sources for full-fidelity parity work.

The downloaded repositories and web pages are cached under ``data/external``,
which is gitignored. The tracked output is a provenance report with revisions,
checksums, and cache locations. These raw snapshots are not accepted parity
artifacts until converted into schema-valid JSON/NPZ reference artifacts with
license, provenance, thresholds, and observables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "data" / "external" / "full_fidelity_public_sources"
REPORT_DIR = ROOT / "validation" / "reports"
JSON_REPORT = REPORT_DIR / "full_fidelity_public_source_downloads.json"
MD_REPORT = REPORT_DIR / "full_fidelity_public_source_downloads.md"


@dataclass(frozen=True)
class GitSource:
    """Public git source required for downstream parity artifact extraction."""

    name: str
    url: str
    cache_subdir: str
    surface: str
    purpose: str


@dataclass(frozen=True)
class WebSource:
    """Public documentation or landing page source with checksum capture."""

    name: str
    url: str
    cache_subdir: str
    surface: str
    purpose: str


GIT_SOURCES: tuple[GitSource, ...] = (
    GitSource(
        name="dream",
        url="https://github.com/chalmersplasmatheory/DREAM.git",
        cache_subdir="repos/dream",
        surface="runaway_electrons",
        purpose="DREAM runaway-electron source, examples, and benchmark extraction target.",
    ),
    GitSource(
        name="aurora",
        url="https://github.com/fsciortino/Aurora.git",
        cache_subdir="repos/aurora",
        surface="impurity_transport",
        purpose="Aurora impurity-transport source, examples, and ADAS-facing contract target.",
    ),
    GitSource(
        name="freegs",
        url="https://github.com/freegs-plasma/freegs.git",
        cache_subdir="repos/freegs",
        surface="free_boundary_equilibrium",
        purpose="FreeGS free-boundary equilibrium source and public example target.",
    ),
    GitSource(
        name="freegsnke",
        url="https://github.com/FusionComputingLab/freegsnke.git",
        cache_subdir="repos/freegsnke",
        surface="free_boundary_equilibrium",
        purpose="FreeGSNKE evolutive free-boundary source and coil-metadata target.",
    ),
    GitSource(
        name="gs2",
        url="https://gitlab.com/gyrokinetics/gs2.git",
        cache_subdir="repos/gs2",
        surface="native_nonlinear_gyrokinetics",
        purpose="GS2 nonlinear gyrokinetic source and benchmark-deck extraction target.",
    ),
    GitSource(
        name="gacode",
        url="https://github.com/gafusion/gacode.git",
        cache_subdir="repos/gacode",
        surface="native_nonlinear_gyrokinetics",
        purpose="GACODE/CGYRO source and public nonlinear deck extraction target.",
    ),
)

WEB_SOURCES: tuple[WebSource, ...] = (
    WebSource(
        name="genecode_landing",
        url="https://genecode.org/",
        cache_subdir="web/genecode.org.html",
        surface="native_nonlinear_gyrokinetics",
        purpose="GENE public landing page and acquisition provenance.",
    ),
    WebSource(
        name="genecode_main",
        url="https://genecode.org/main.html",
        cache_subdir="web/genecode_main.html",
        surface="native_nonlinear_gyrokinetics",
        purpose="GENE public documentation page and acquisition provenance.",
    ),
    WebSource(
        name="cgyro_docs",
        url="https://gafusion.github.io/doc/cgyro.html",
        cache_subdir="web/cgyro.html",
        surface="native_nonlinear_gyrokinetics",
        purpose="CGYRO public documentation and benchmark-contract provenance.",
    ),
    WebSource(
        name="gacode_cgyro_docs",
        url="https://gacode.io/cgyro.html",
        cache_subdir="web/gacode_cgyro.html",
        surface="native_nonlinear_gyrokinetics",
        purpose="Current GACODE CGYRO documentation mirror and provenance.",
    ),
    WebSource(
        name="gs2_docs",
        url="https://gyrokinetics.gitlab.io/gs2/",
        cache_subdir="web/gs2.html",
        surface="native_nonlinear_gyrokinetics",
        purpose="GS2 public documentation landing page.",
    ),
    WebSource(
        name="gs2_user_manual",
        url="https://gyrokinetics.gitlab.io/gs2/page/user_manual/index.html",
        cache_subdir="web/gs2_user_manual.html",
        surface="native_nonlinear_gyrokinetics",
        purpose="GS2 public user manual for benchmark-deck interpretation.",
    ),
    WebSource(
        name="dream_docs",
        url="https://ft.nephy.chalmers.se/dream/",
        cache_subdir="web/dream.html",
        surface="runaway_electrons",
        purpose="DREAM public documentation landing page.",
    ),
    WebSource(
        name="aurora_docs",
        url="https://aurora-fusion.readthedocs.io/",
        cache_subdir="web/aurora.html",
        surface="impurity_transport",
        purpose="Aurora public documentation landing page.",
    ),
    WebSource(
        name="freegs_docs",
        url="https://freegs.readthedocs.io/",
        cache_subdir="web/freegs.html",
        surface="free_boundary_equilibrium",
        purpose="FreeGS public documentation landing page.",
    ),
)


def _rel(path: Path) -> str:
    """Return a repository-relative path string."""
    return str(path.relative_to(ROOT))


def _run(args: list[str], *, cwd: Path | None = None, timeout: int = 600) -> None:
    """Run a trusted local command with shell disabled."""
    subprocess.run(  # nosec B603: args are allowlisted git invocations, shell is disabled.
        args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _capture(args: list[str], *, cwd: Path | None = None, timeout: int = 120) -> str:
    """Capture a trusted local command with shell disabled."""
    result = subprocess.run(  # nosec B603: args are allowlisted git invocations, shell is disabled.
        args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout.strip()


def _file_count(path: Path) -> int:
    """Count non-.git files below a cached repository path."""
    count = 0
    for child in path.rglob("*"):
        if child.is_file() and ".git" not in child.parts:
            count += 1
    return count


def _download_git(source: GitSource) -> dict[str, Any]:
    """Clone or update one public git source and return report metadata."""
    git = shutil.which("git")
    if git is None:
        return {
            "kind": "git",
            "name": source.name,
            "status": "failed",
            "reason": "git executable not found",
            "surface": source.surface,
            "url": source.url,
            "tracked_in_repo": False,
        }

    cache_path = CACHE_ROOT / source.cache_subdir
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if (cache_path / ".git").is_dir():
            _run([git, "-C", str(cache_path), "remote", "set-url", "origin", source.url])
            _run([git, "-C", str(cache_path), "pull", "--ff-only", "--depth", "1"])
        else:
            _run([git, "clone", "--depth", "1", source.url, str(cache_path)])

        commit = _capture([git, "-C", str(cache_path), "rev-parse", "HEAD"])
        try:
            branch = _capture([git, "-C", str(cache_path), "symbolic-ref", "--short", "HEAD"])
        except subprocess.CalledProcessError:
            branch = "detached"
        return {
            "branch": branch,
            "cache_path": _rel(cache_path),
            "commit": commit,
            "file_count": _file_count(cache_path),
            "kind": "git",
            "name": source.name,
            "purpose": source.purpose,
            "status": "downloaded",
            "surface": source.surface,
            "tracked_in_repo": False,
            "url": source.url,
        }
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {
            "cache_path": _rel(cache_path),
            "kind": "git",
            "name": source.name,
            "purpose": source.purpose,
            "reason": str(exc),
            "status": "failed",
            "surface": source.surface,
            "tracked_in_repo": False,
            "url": source.url,
        }


def _download_web(source: WebSource, *, timeout: int) -> dict[str, Any]:
    """Download one public web page and return report metadata."""
    cache_path = CACHE_ROOT / source.cache_subdir
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        source.url,
        headers={"User-Agent": "scpn-fusion-core-public-source-downloader/1.0"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # nosec B310: fixed public URLs.
            payload = response.read()
        cache_path.write_bytes(payload)
        return {
            "bytes": len(payload),
            "cache_path": _rel(cache_path),
            "kind": "web",
            "name": source.name,
            "purpose": source.purpose,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "status": "downloaded",
            "surface": source.surface,
            "tracked_in_repo": False,
            "url": source.url,
        }
    except (HTTPError, URLError, TimeoutError) as exc:
        return {
            "cache_path": _rel(cache_path),
            "kind": "web",
            "name": source.name,
            "purpose": source.purpose,
            "reason": str(exc),
            "status": "failed",
            "surface": source.surface,
            "tracked_in_repo": False,
            "url": source.url,
        }


def build_report(*, timeout: int) -> dict[str, Any]:
    """Download all public sources and return the provenance report."""
    items = [_download_git(source) for source in GIT_SOURCES]
    items.extend(_download_web(source, timeout=timeout) for source in WEB_SOURCES)
    return {
        "all_reachable_downloads_completed": all(item["status"] == "downloaded" for item in items),
        "cache_root": _rel(CACHE_ROOT),
        "gitignored_cache": True,
        "items": items,
        "note": (
            "Raw upstream source snapshots are cached under data/external and are not parity "
            "artifacts until converted into validated JSON/NPZ reference artifacts with license, "
            "provenance, checksum, and thresholds."
        ),
        "schema": "full-fidelity-public-source-downloads.v1",
    }


def write_reports(report: dict[str, Any]) -> None:
    """Write machine-readable and Markdown provenance reports."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Full-Fidelity Public Source Downloads",
        "",
        report["note"],
        "",
        f"- Cache root: `{report['cache_root']}`",
        f"- Gitignored cache: `{report['gitignored_cache']}`",
        f"- All reachable downloads completed: `{report['all_reachable_downloads_completed']}`",
        "",
        "| Source | Surface | Kind | Status | Revision / SHA256 | Cache |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in report["items"]:
        revision = item.get("commit") or item.get("sha256") or item.get("reason", "")
        lines.append(
            "| {name} | {surface} | {kind} | {status} | `{revision}` | `{cache}` |".format(
                name=item["name"],
                surface=item["surface"],
                kind=item["kind"],
                status=item["status"],
                revision=revision,
                cache=item.get("cache_path", ""),
            )
        )
    lines.append("")
    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=30, help="Per-page web timeout in seconds.")
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Write reports even if one public source cannot be reached.",
    )
    args = parser.parse_args(argv)

    report = build_report(timeout=args.timeout)
    write_reports(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["all_reachable_downloads_completed"] or args.allow_failures:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
