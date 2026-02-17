#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — DIII-D Shot Data Downloader (1D.2)
# (c) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Download shot-level time-series data from tokamak MDSplus archives with
retry logic and NPZ caching.

When the ``MDSplus`` Python module is installed and the target server
is reachable, the script fetches the requested signals, applies
exponential-backoff retry logic (3 attempts by default), and writes
the results to a compressed ``.npz`` cache file.  On subsequent runs
the cached file is reused if it exists and is sufficiently recent.

When ``MDSplus`` is **not** installed the script falls back to
reference data in ``validation/reference_data/`` (if available) and
reports the missing dependency otherwise.

Usage::

    python tools/download_diiid_data.py --machine DIII-D --shot 163303 --signals Ip,q95,beta_N
    python tools/download_diiid_data.py --machine DIII-D --shot 163303 --signals Ip --cache-dir /tmp/npz

Importable entry point::

    from tools.download_diiid_data import download_shot_data
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPO_ROOT / "validation" / "reference_data" / "diiid"

# Map of canonical signal names to MDSplus node paths, per machine.
# Users can override or extend this via the ``node_overrides`` parameter.
DEFAULT_NODE_MAPS: Dict[str, Dict[str, str]] = {
    "DIII-D": {
        "Ip": "\\ip",
        "q95": "\\q95",
        "beta_N": "\\betan",
        "ne_bar": "\\denv03",
        "Wmhd": "\\wmhd",
        "Pinj": "\\pinj",
        "Prad": "\\prad_tot",
        "li": "\\li",
        "H98": "\\h98",
        "kappa": "\\kappa",
    },
    "C-Mod": {
        "Ip": "\\ip",
        "q95": "\\q95",
        "beta_N": "\\betan",
        "ne_bar": "\\ne_bar",
        "Wmhd": "\\wmhd",
    },
}

DEFAULT_HOSTS: Dict[str, str] = {
    "DIII-D": "atlas.gat.com",
    "C-Mod": "alcdata.psfc.mit.edu",
}

DEFAULT_TREES: Dict[str, str] = {
    "DIII-D": "d3d",
    "C-Mod": "cmod",
}

MAX_RETRIES = 3
BACKOFF_BASE_S = 1.0  # base delay for exponential backoff
CACHE_MAX_AGE_S = 7 * 24 * 3600  # 1 week


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class SignalResult:
    """Container for one fetched signal."""

    name: str
    data: NDArray[np.float64]
    time: NDArray[np.float64]
    unit: str = ""
    node_path: str = ""


@dataclass
class ShotDownloadResult:
    """Aggregate result of downloading signals for one shot."""

    machine: str
    shot: int
    signals: Dict[str, SignalResult] = field(default_factory=dict)
    source: str = "missing"  # "cache", "mdsplus", "reference", "missing"
    cache_path: Optional[Path] = None
    errors: Dict[str, str] = field(default_factory=dict)


# ── NPZ cache helpers ────────────────────────────────────────────────

def _cache_key(machine: str, shot: int, signals: Sequence[str]) -> str:
    """Compute a deterministic cache filename."""
    sig_hash = hashlib.md5(
        ",".join(sorted(signals)).encode(), usedforsecurity=False,
    ).hexdigest()[:8]
    safe_machine = machine.replace("-", "").replace(" ", "_").lower()
    return f"{safe_machine}_{shot}_{sig_hash}.npz"


def _cache_path(cache_dir: Path, machine: str, shot: int, signals: Sequence[str]) -> Path:
    return cache_dir / _cache_key(machine, shot, signals)


def _cache_is_fresh(path: Path, max_age_s: float = CACHE_MAX_AGE_S) -> bool:
    """Return True if *path* exists and is younger than *max_age_s*."""
    if not path.is_file():
        return False
    age = time.time() - path.stat().st_mtime
    return age < max_age_s


def _load_from_cache(path: Path) -> Dict[str, SignalResult]:
    """Load signals from a cached NPZ file."""
    data = np.load(str(path), allow_pickle=False)
    signals: Dict[str, SignalResult] = {}

    # Metadata key stores signal names as comma-separated string
    meta_key = "__signal_names__"
    if meta_key not in data:
        # Legacy or hand-placed cache: treat every array as a signal
        for key in data.files:
            arr = data[key]
            signals[key] = SignalResult(
                name=key,
                data=arr,
                time=np.arange(arr.size, dtype=np.float64),
            )
        return signals

    signal_names = str(data[meta_key]).split(",")
    for name in signal_names:
        data_key = f"{name}_data"
        time_key = f"{name}_time"
        if data_key in data and time_key in data:
            signals[name] = SignalResult(
                name=name,
                data=np.asarray(data[data_key], dtype=np.float64),
                time=np.asarray(data[time_key], dtype=np.float64),
            )
    return signals


def _save_to_cache(path: Path, signals: Dict[str, SignalResult]) -> None:
    """Save signals to an NPZ cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, Any] = {
        "__signal_names__": np.array(",".join(sorted(signals.keys()))),
    }
    for name, sig in signals.items():
        arrays[f"{name}_data"] = sig.data
        arrays[f"{name}_time"] = sig.time
    np.savez_compressed(str(path), **arrays)
    logger.info("Cached %d signals to %s", len(signals), path)


# ── Reference data fallback ─────────────────────────────────────────

def _try_reference_data(
    machine: str,
    shot: int,
    signals: Sequence[str],
    cache_dir: Path,
) -> Optional[Dict[str, SignalResult]]:
    """Try to load signals from pre-existing reference data NPZ files.

    Searches the cache_dir for any NPZ file containing the shot number.
    Falls back to tokamak_archive.fetch_mdsplus_profiles() if available.
    """
    # 1. Search for any pre-existing NPZ containing this shot number
    for npz_file in cache_dir.glob("*.npz"):
        if str(shot) in npz_file.stem:
            try:
                loaded = _load_from_cache(npz_file)
                if loaded:
                    logger.info(
                        "Found reference data for shot %d in %s",
                        shot, npz_file,
                    )
                    return loaded
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to load %s: %s", npz_file, exc)

    # 2. Try tokamak_archive.fetch_mdsplus_profiles() as an alternative
    try:
        from scpn_fusion.io.tokamak_archive import fetch_mdsplus_profiles

        profiles = fetch_mdsplus_profiles(machine=machine, shot=shot)
        if profiles:
            result: Dict[str, SignalResult] = {}
            for sig_name in signals:
                if sig_name in profiles:
                    prof = profiles[sig_name]
                    result[sig_name] = SignalResult(
                        name=sig_name,
                        data=np.asarray(prof.get("data", []), dtype=np.float64),
                        time=np.asarray(prof.get("time", []), dtype=np.float64),
                    )
            if result:
                return result
    except (ImportError, Exception) as exc:  # noqa: BLE001
        logger.debug("tokamak_archive fallback failed: %s", exc)

    return None


# ── MDSplus fetch with retry ─────────────────────────────────────────

def _fetch_signal_mdsplus(
    conn: Any,
    signal_name: str,
    node_path: str,
    *,
    max_retries: int = MAX_RETRIES,
    backoff_base: float = BACKOFF_BASE_S,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fetch a single signal from an open MDSplus connection with retry.

    Returns ``(data, time)`` arrays.

    Raises
    ------
    RuntimeError
        If all retry attempts fail.
    """
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            raw_data = conn.get(node_path)
            data = np.asarray(
                raw_data.data() if hasattr(raw_data, "data") else raw_data,
                dtype=np.float64,
            ).ravel()

            # Try to get the time base from the same node via dim_of()
            try:
                raw_time = conn.get(f"dim_of({node_path})")
                t = np.asarray(
                    raw_time.data() if hasattr(raw_time, "data") else raw_time,
                    dtype=np.float64,
                ).ravel()
            except Exception:  # noqa: BLE001
                # No time dimension available; fabricate indices
                t = np.arange(data.size, dtype=np.float64)

            if t.size != data.size:
                min_len = min(t.size, data.size)
                t = t[:min_len]
                data = data[:min_len]

            return data, t

        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_retries - 1:
                delay = backoff_base * (2 ** attempt)
                logger.warning(
                    "Attempt %d/%d for %s (%s) failed: %s  — retrying in %.1fs",
                    attempt + 1, max_retries, signal_name, node_path, exc, delay,
                )
                time.sleep(delay)

    assert last_exc is not None  # mypy satisfaction
    raise RuntimeError(
        f"All {max_retries} attempts failed for signal '{signal_name}' "
        f"(node={node_path}): {last_exc}"
    ) from last_exc


# ── Public API ───────────────────────────────────────────────────────

def download_shot_data(
    machine: str,
    shot: int,
    signals: Sequence[str],
    cache_dir: Optional[Path] = None,
    *,
    host: Optional[str] = None,
    tree: Optional[str] = None,
    node_overrides: Optional[Dict[str, str]] = None,
    max_retries: int = MAX_RETRIES,
    cache_max_age_s: float = CACHE_MAX_AGE_S,
    force_download: bool = False,
) -> ShotDownloadResult:
    """Download (or load from cache) time-series signals for a single shot.

    Parameters
    ----------
    machine : str
        Machine identifier, e.g. ``"DIII-D"`` or ``"C-Mod"``.
    shot : int
        Shot number.
    signals : sequence of str
        Signal names to fetch (e.g. ``["Ip", "q95", "beta_N"]``).
    cache_dir : Path, optional
        NPZ cache directory.  Defaults to ``validation/reference_data/diiid/``.
    host : str, optional
        MDSplus server hostname.  Defaults based on machine.
    tree : str, optional
        MDSplus tree name.  Defaults based on machine.
    node_overrides : dict, optional
        Override or extend the default signal-to-node mapping.
    max_retries : int
        Maximum number of retry attempts per signal.
    cache_max_age_s : float
        Maximum age (seconds) for a cache file to be considered fresh.
    force_download : bool
        If True, ignore existing cache and re-download.

    Returns
    -------
    ShotDownloadResult
        Downloaded signals, source tag, and any per-signal errors.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)

    result = ShotDownloadResult(machine=machine, shot=shot)
    signals = list(signals)

    # 1. Check cache
    npz_path = _cache_path(cache_dir, machine, shot, signals)
    result.cache_path = npz_path

    if not force_download and _cache_is_fresh(npz_path, cache_max_age_s):
        logger.info("Loading shot %d from cache: %s", shot, npz_path)
        try:
            result.signals = _load_from_cache(npz_path)
            result.source = "cache"
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cache load failed for %s: %s — will re-download", npz_path, exc)

    # 2. Try MDSplus
    mdsplus_available = False
    try:
        import MDSplus  # type: ignore[import-untyped]
        mdsplus_available = True
    except ImportError:
        pass

    if mdsplus_available:
        # Resolve host / tree
        resolved_host = host or DEFAULT_HOSTS.get(machine)
        resolved_tree = tree or DEFAULT_TREES.get(machine)
        if not resolved_host or not resolved_tree:
            msg = f"No default host/tree for machine '{machine}'. Provide --host and --tree."
            logger.error(msg)
            for sig in signals:
                result.errors[sig] = msg
        else:
            # Resolve node map
            node_map = dict(DEFAULT_NODE_MAPS.get(machine, {}))
            if node_overrides:
                node_map.update(node_overrides)

            # Fetch signals
            try:
                conn = MDSplus.Connection(resolved_host)
                conn.openTree(resolved_tree, shot)

                for sig_name in signals:
                    node_path = node_map.get(sig_name)
                    if node_path is None:
                        result.errors[sig_name] = (
                            f"No MDSplus node mapping for signal '{sig_name}' on {machine}. "
                            f"Available: {list(node_map.keys())}"
                        )
                        continue

                    try:
                        data, t = _fetch_signal_mdsplus(
                            conn, sig_name, node_path, max_retries=max_retries,
                        )
                        result.signals[sig_name] = SignalResult(
                            name=sig_name,
                            data=data,
                            time=t,
                            node_path=node_path,
                        )
                    except Exception as exc:  # noqa: BLE001
                        result.errors[sig_name] = str(exc)

            except Exception as exc:  # noqa: BLE001
                msg = f"MDSplus connection failed ({resolved_host}, tree={resolved_tree}, shot={shot}): {exc}"
                logger.error(msg)
                for sig in signals:
                    if sig not in result.signals:
                        result.errors[sig] = msg

    # 3. Fall back to reference data if MDSplus was unavailable or failed
    if not result.signals:
        logger.info(
            "MDSplus unavailable or returned no data; trying reference data for shot %d",
            shot,
        )
        ref_signals = _try_reference_data(machine, shot, signals, cache_dir)
        if ref_signals:
            result.signals = ref_signals
            result.source = "reference"
            # Clear errors for signals we got from reference
            for sig_name in ref_signals:
                result.errors.pop(sig_name, None)
        else:
            if not mdsplus_available:
                msg = (
                    "MDSplus Python module is not installed and no reference data found.  "
                    "Install it with: pip install mdsplus  "
                    "or place pre-cached .npz files in the cache directory."
                )
                for sig in signals:
                    if sig not in result.errors:
                        result.errors[sig] = msg
            result.source = "missing"

    # 4. Cache the results if we got at least one signal
    if result.signals and result.source != "missing":
        try:
            _save_to_cache(npz_path, result.signals)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write cache %s: %s", npz_path, exc)
        if result.source != "reference":
            result.source = "mdsplus"

    return result


# ── CLI ──────────────────────────────────────────────────────────────

def _print_result(result: ShotDownloadResult) -> None:
    """Pretty-print download results."""
    print(f"\nShot Data Download: {result.machine} #{result.shot}")
    print(f"Source: {result.source}")
    if result.cache_path:
        print(f"Cache:  {result.cache_path}")
    print("=" * 60)

    if result.signals:
        print(f"\n{'Signal':<12}  {'Points':>8}  {'Time Range':>20}  {'Data Range':>24}")
        print("-" * 68)
        for name, sig in sorted(result.signals.items()):
            n = sig.data.size
            if n > 0:
                t_range = f"[{sig.time[0]:.2f}, {sig.time[-1]:.2f}]"
                d_range = f"[{sig.data.min():.4g}, {sig.data.max():.4g}]"
            else:
                t_range = "empty"
                d_range = "empty"
            print(f"{name:<12}  {n:>8}  {t_range:>20}  {d_range:>24}")

    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for name, err in sorted(result.errors.items()):
            print(f"  {name}: {err}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download tokamak shot data via MDSplus with NPZ caching.",
    )
    parser.add_argument(
        "--machine",
        type=str,
        default="DIII-D",
        help="Machine name (default: DIII-D).",
    )
    parser.add_argument(
        "--shot",
        type=int,
        required=True,
        help="Shot number to download.",
    )
    parser.add_argument(
        "--signals",
        type=str,
        required=True,
        help="Comma-separated list of signal names (e.g. Ip,q95,beta_N).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=f"NPZ cache directory (default: {DEFAULT_CACHE_DIR}).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="MDSplus server hostname (default: auto-detected from machine).",
    )
    parser.add_argument(
        "--tree",
        type=str,
        default=None,
        help="MDSplus tree name (default: auto-detected from machine).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached file exists.",
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

    signals = [s.strip() for s in args.signals.split(",") if s.strip()]
    if not signals:
        print("ERROR: No signals specified.", file=sys.stderr)
        return 1

    result = download_shot_data(
        machine=args.machine,
        shot=args.shot,
        signals=signals,
        cache_dir=args.cache_dir,
        host=args.host,
        tree=args.tree,
        force_download=args.force,
    )
    _print_result(result)

    if result.source == "missing":
        return 1
    if result.errors:
        return 2  # partial success
    return 0


if __name__ == "__main__":
    sys.exit(main())
