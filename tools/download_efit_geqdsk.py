#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — DIII-D EFIT GEQDSK Downloader (1A.2)
# (c) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Download DIII-D EFIT GEQDSK equilibrium files from the MDSplus archive.

If the ``MDSplus`` Python module is available, connects to the DIII-D
MDSplus server (atlas.gat.com) and fetches EFIT GEQDSKs for a set of
canonical validation shots.  When MDSplus is unavailable (e.g. in CI),
the script falls back to checking the local cache directory and reports
which files are present vs. missing.

Usage::

    python tools/download_efit_geqdsk.py
    python tools/download_efit_geqdsk.py --cache-dir /tmp/diiid_cache

Importable entry point::

    from tools.download_efit_geqdsk import download_geqdsks
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ── Target shots ─────────────────────────────────────────────────────

DIIID_TARGET_SHOTS: Dict[int, str] = {
    163303: "H-mode",
    154406: "hybrid",
    175970: "neg-delta",
    166549: "snowflake",
    176673: "high-beta",
}
"""Canonical DIII-D validation shots with scenario labels."""

DEFAULT_MDSPLUS_HOST = "atlas.gat.com"
DEFAULT_MDSPLUS_TREE = "efit01"
DEFAULT_EFIT_NODE = "\\efit01::gEQDSK"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPO_ROOT / "validation" / "reference_data" / "diiid"


# ── Data structures ──────────────────────────────────────────────────

@dataclass(frozen=True)
class ShotStatus:
    """Status of a single GEQDSK file for one shot."""

    shot: int
    scenario: str
    available: bool
    source: str  # "cache", "mdsplus", or "missing"
    path: Optional[Path]
    error: Optional[str] = None


# ── Core logic ───────────────────────────────────────────────────────

def _geqdsk_filename(shot: int) -> str:
    """Return the canonical cache filename for a given shot number."""
    scenario = DIIID_TARGET_SHOTS.get(shot, "unknown")
    safe_scenario = scenario.replace("-", "").replace(" ", "_").lower()
    return f"diiid_{safe_scenario}_{shot}.geqdsk"


def _check_cache(shot: int, cache_dir: Path) -> Optional[Path]:
    """Check if a GEQDSK for *shot* already exists in *cache_dir*.

    Looks for files matching the canonical name as well as any file
    containing the shot number in its stem.
    """
    canonical = cache_dir / _geqdsk_filename(shot)
    if canonical.is_file():
        return canonical

    # Fallback: any .geqdsk file whose name contains the shot number
    for p in cache_dir.glob("*.geqdsk"):
        if str(shot) in p.stem:
            return p

    # Also check existing reference files that may match by scenario tag
    scenario = DIIID_TARGET_SHOTS.get(shot, "")
    scenario_lower = scenario.replace("-", "").replace(" ", "_").lower()
    if scenario_lower:
        for p in cache_dir.glob("*.geqdsk"):
            if scenario_lower in p.stem.lower().replace("-", "").replace(" ", "_"):
                return p

    return None


def _try_mdsplus_download(
    shot: int,
    cache_dir: Path,
    *,
    host: str = DEFAULT_MDSPLUS_HOST,
    tree: str = DEFAULT_MDSPLUS_TREE,
    node: str = DEFAULT_EFIT_NODE,
) -> Tuple[Optional[Path], Optional[str]]:
    """Attempt to download a GEQDSK from MDSplus.

    Returns ``(path, None)`` on success or ``(None, error_message)`` on failure.
    """
    try:
        import MDSplus  # type: ignore[import-untyped]
    except ImportError:
        return None, "MDSplus Python module not installed"

    out_path = cache_dir / _geqdsk_filename(shot)

    try:
        conn = MDSplus.Connection(host)
        conn.openTree(tree, shot)
        data = conn.get(node)
        raw: Any = data.data() if hasattr(data, "data") else data

        # MDSplus may return the GEQDSK as a byte string or as structured data.
        # If it's a string (raw file content), write directly.  Otherwise, we
        # would need to reconstruct the file, which is beyond the scope of this
        # simple downloader -- log and treat as failure.
        if isinstance(raw, (str, bytes)):
            text = raw.decode("ascii") if isinstance(raw, bytes) else raw
            cache_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding="ascii")
            return out_path, None
        else:
            return None, (
                f"MDSplus returned non-string data for shot {shot} "
                f"(type={type(raw).__name__}); manual EFIT fetch may be needed"
            )
    except Exception as exc:  # noqa: BLE001
        return None, f"MDSplus error for shot {shot}: {exc}"


def download_geqdsks(
    *,
    cache_dir: Optional[Path] = None,
    shots: Optional[Sequence[int]] = None,
    host: str = DEFAULT_MDSPLUS_HOST,
    tree: str = DEFAULT_MDSPLUS_TREE,
    try_mdsplus: bool = True,
) -> List[ShotStatus]:
    """Download or locate DIII-D EFIT GEQDSK files.

    Parameters
    ----------
    cache_dir : Path, optional
        Directory to store/look for cached files.  Defaults to
        ``validation/reference_data/diiid/`` relative to the repo root.
    shots : sequence of int, optional
        Shot numbers to fetch.  Defaults to :data:`DIIID_TARGET_SHOTS`.
    host : str
        MDSplus server hostname.
    tree : str
        MDSplus tree name.
    try_mdsplus : bool
        Whether to attempt MDSplus download for missing files.

    Returns
    -------
    list of ShotStatus
        Per-shot status indicating availability and source.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if shots is None:
        shots = list(DIIID_TARGET_SHOTS.keys())

    results: List[ShotStatus] = []

    for shot in shots:
        scenario = DIIID_TARGET_SHOTS.get(shot, "unknown")

        # 1. Check local cache first
        cached = _check_cache(shot, cache_dir)
        if cached is not None:
            results.append(
                ShotStatus(
                    shot=shot,
                    scenario=scenario,
                    available=True,
                    source="cache",
                    path=cached,
                )
            )
            continue

        # 2. Try MDSplus download if requested
        if try_mdsplus:
            path, err = _try_mdsplus_download(
                shot, cache_dir, host=host, tree=tree,
            )
            if path is not None:
                results.append(
                    ShotStatus(
                        shot=shot,
                        scenario=scenario,
                        available=True,
                        source="mdsplus",
                        path=path,
                    )
                )
                continue

            # Log the error but continue to mark as missing
            logger.warning("Shot %d (%s): %s", shot, scenario, err)
            results.append(
                ShotStatus(
                    shot=shot,
                    scenario=scenario,
                    available=False,
                    source="missing",
                    path=None,
                    error=err,
                )
            )
        else:
            results.append(
                ShotStatus(
                    shot=shot,
                    scenario=scenario,
                    available=False,
                    source="missing",
                    path=None,
                    error="MDSplus download disabled",
                )
            )

    return results


# ── CLI ──────────────────────────────────────────────────────────────

def _print_status(results: List[ShotStatus]) -> None:
    """Pretty-print the download/cache status table."""
    available_count = sum(1 for r in results if r.available)
    total = len(results)

    print(f"\nDIII-D EFIT GEQDSK Status ({available_count}/{total} available)")
    print("=" * 72)
    print(f"{'Shot':>8}  {'Scenario':<14}  {'Status':<10}  {'Source':<10}  Path / Error")
    print("-" * 72)

    for r in results:
        status_str = "OK" if r.available else "MISSING"
        detail = str(r.path) if r.path else (r.error or "")
        print(f"{r.shot:>8}  {r.scenario:<14}  {status_str:<10}  {r.source:<10}  {detail}")

    print("-" * 72)

    if available_count < total:
        missing = [r for r in results if not r.available]
        print(
            f"\n{len(missing)} file(s) missing.  "
            "Install MDSplus and ensure network access to atlas.gat.com, "
            "or place .geqdsk files manually in the cache directory."
        )
    else:
        print("\nAll target shots are available.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download or check DIII-D EFIT GEQDSK files for validation.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory to store/look for cached GEQDSK files. "
            f"Default: {DEFAULT_CACHE_DIR}"
        ),
    )
    parser.add_argument(
        "--no-mdsplus",
        action="store_true",
        help="Skip MDSplus download attempts; only check local cache.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        nargs="*",
        default=None,
        help="Specific shot numbers to check (default: all 5 canonical shots).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    results = download_geqdsks(
        cache_dir=args.cache_dir,
        shots=args.shots,
        try_mdsplus=not args.no_mdsplus,
    )
    _print_status(results)

    # Exit code: 0 if all available, 1 if any missing
    if all(r.available for r in results):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
