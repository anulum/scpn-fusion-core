#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — DIII-D EFIT GEQDSK Downloader (1A.2)
"""Download DIII-D EFIT GEQDSK files from MDSplus or inspect a local cache.

If the ``MDSplus`` Python module is available, connects to the DIII-D
server and fetches canonical validation shots. Without MDSplus, the script
reports exact shot-bound local cache entries as present or missing.

Usage::

    python tools/download_efit_geqdsk.py
    python tools/download_efit_geqdsk.py --cache-dir /tmp/diiid_cache

"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence, cast, runtime_checkable

logger = logging.getLogger(__name__)

DIIID_TARGET_SHOTS: dict[int, str] = {
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
MAX_GEQDSK_DOWNLOAD_BYTES = 10 * 1024 * 1024

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPO_ROOT / "validation" / "reference_data" / "diiid"


@dataclass(frozen=True)
class ShotStatus:
    """Status of a single GEQDSK file for one shot."""

    shot: int
    scenario: str
    available: bool
    source: Literal["cache", "mdsplus", "missing"]
    path: Path | None
    error: str | None = None


class _MDSplusConnection(Protocol):
    """Typed subset of the legacy MDSplus connection used by this tool."""

    def openTree(self, tree: str, shot: int) -> object:
        """Open one tree for one shot."""

    def get(self, node: str) -> object:
        """Return the requested node payload."""


class _MDSplusModule(Protocol):
    """Typed subset of the dynamically imported MDSplus module."""

    def Connection(self, host: str) -> _MDSplusConnection:
        """Create a connection to ``host``."""


@runtime_checkable
class _MDSplusData(Protocol):
    """MDSplus value wrapper exposing its underlying payload."""

    def data(self) -> object:
        """Return the wrapped value."""


def _geqdsk_filename(shot: int) -> str:
    """Return the canonical cache filename for a given shot number."""
    scenario = DIIID_TARGET_SHOTS.get(shot, "unknown")
    safe_scenario = scenario.replace("-", "").replace(" ", "_").lower()
    return f"diiid_{safe_scenario}_{shot}.geqdsk"


def _is_usable_cache_file(path: Path) -> bool:
    """Return whether a cache candidate is a bounded non-symlink regular file."""
    try:
        size = path.stat().st_size
        return path.is_file() and not path.is_symlink() and 0 < size <= MAX_GEQDSK_DOWNLOAD_BYTES
    except OSError:
        return False


def _check_cache(shot: int, cache_dir: Path) -> Path | None:
    """Return a bounded cache file named canonically or by exact shot number."""
    canonical = cache_dir / _geqdsk_filename(shot)
    if _is_usable_cache_file(canonical):
        return canonical

    # Fallback: any .geqdsk file whose name contains the shot number
    for p in cache_dir.glob("*.geqdsk"):
        if re.search(rf"(?<!\d){shot}(?!\d)", p.stem) and _is_usable_cache_file(p):
            return p

    return None


def _load_mdsplus() -> _MDSplusModule | None:
    """Load the optional legacy MDSplus client behind a typed protocol."""
    try:
        module = importlib.import_module("MDSplus")
    except ImportError:
        return None
    return cast(_MDSplusModule, module)


def _payload_bytes(raw: object, *, shot: int) -> tuple[bytes | None, str | None]:
    """Validate and encode a bounded ASCII GEQDSK provider payload."""
    if isinstance(raw, _MDSplusData):
        raw = raw.data()
    try:
        if isinstance(raw, str):
            payload = raw.encode("ascii")
        elif isinstance(raw, bytes):
            payload = raw
        else:
            return None, (
                f"MDSplus returned non-string data for shot {shot} "
                f"(type={type(raw).__name__}); manual EFIT fetch may be needed"
            )
    except UnicodeEncodeError:
        return None, f"MDSplus returned non-ASCII GEQDSK text for shot {shot}"
    if not payload.strip():
        return None, f"MDSplus returned an empty GEQDSK payload for shot {shot}"
    if len(payload) > MAX_GEQDSK_DOWNLOAD_BYTES:
        return None, (
            f"MDSplus GEQDSK payload for shot {shot} is too large: "
            f"{len(payload)} bytes exceeds {MAX_GEQDSK_DOWNLOAD_BYTES}"
        )
    return payload, None


def _write_payload_atomically(path: Path, payload: bytes) -> None:
    """Write one validated payload atomically inside its cache directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".part",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _try_mdsplus_download(
    shot: int,
    cache_dir: Path,
    *,
    host: str = DEFAULT_MDSPLUS_HOST,
    tree: str = DEFAULT_MDSPLUS_TREE,
    node: str = DEFAULT_EFIT_NODE,
) -> tuple[Path | None, str | None]:
    """Return ``(path, None)`` on download success or ``(None, error)``."""
    mdsplus = _load_mdsplus()
    if mdsplus is None:
        return None, "MDSplus Python module not installed"

    out_path = cache_dir / _geqdsk_filename(shot)

    try:
        conn = mdsplus.Connection(host)
        conn.openTree(tree, shot)
        payload, error = _payload_bytes(conn.get(node), shot=shot)
        if payload is None:
            return None, error
        _write_payload_atomically(out_path, payload)
        return out_path, None
    except Exception as exc:  # noqa: BLE001
        return None, f"MDSplus error for shot {shot}: {exc}"


def download_geqdsks(
    *,
    cache_dir: Path | None = None,
    shots: Sequence[int] | None = None,
    host: str = DEFAULT_MDSPLUS_HOST,
    tree: str = DEFAULT_MDSPLUS_TREE,
    try_mdsplus: bool = True,
) -> list[ShotStatus]:
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
    requested_shots = list(DIIID_TARGET_SHOTS) if shots is None else list(shots)
    if not requested_shots:
        raise ValueError("At least one DIII-D shot must be requested")
    if any(isinstance(shot, bool) or shot <= 0 for shot in requested_shots):
        raise ValueError("DIII-D shot numbers must be positive integers")
    if len(set(requested_shots)) != len(requested_shots):
        raise ValueError("DIII-D shot numbers must be unique")

    resolved_cache = DEFAULT_CACHE_DIR if cache_dir is None else Path(cache_dir)
    resolved_cache.mkdir(parents=True, exist_ok=True)
    results: list[ShotStatus] = []

    for shot in requested_shots:
        scenario = DIIID_TARGET_SHOTS.get(shot, "unknown")

        # 1. Check local cache first
        cached = _check_cache(shot, resolved_cache)
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
                shot,
                resolved_cache,
                host=host,
                tree=tree,
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


def _print_status(results: Sequence[ShotStatus]) -> None:
    """Pretty-print the download/cache status table."""
    if not results:
        raise ValueError("Cannot print an empty GEQDSK status collection")
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


class _Arguments(argparse.Namespace):
    """Typed command-line arguments for the GEQDSK downloader."""

    cache_dir: Path | None
    no_mdsplus: bool
    shots: list[int] | None
    verbose: bool


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download or check DIII-D EFIT GEQDSK files for validation.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(f"Directory to store/look for cached GEQDSK files. Default: {DEFAULT_CACHE_DIR}"),
    )
    parser.add_argument(
        "--no-mdsplus",
        action="store_true",
        help="Skip MDSplus download attempts; only check local cache.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        nargs="+",
        default=None,
        help="Specific shot numbers to check (default: all 5 canonical shots).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args(argv, namespace=_Arguments())

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
    raise SystemExit(main())
