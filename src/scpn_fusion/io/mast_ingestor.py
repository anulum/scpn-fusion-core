# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — MAST Data Ingestor
"""
Formal ingestor for the UKAEA FAIR MAST public dataset.

Provides high-frequency streaming access to real tokamak data
via the S3/Zarr stack.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import weakref
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# 1. Resolve repository-hosted external libraries path
_EXTERNAL_ROOT = Path(__file__).resolve().parents[3] / "external"
if _EXTERNAL_ROOT.exists() and str(_EXTERNAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_ROOT))

# 2. Delayed imports to handle external lib requirement
try:
    fsspec: Any | None = importlib.import_module("fsspec")
    xr: Any | None = importlib.import_module("xarray")
    importlib.import_module("zarr")
    _HAS_FAIR_MAST_STACK = True
except ImportError:
    fsspec = None
    xr = None
    _HAS_FAIR_MAST_STACK = False
    logger.debug("FAIR MAST stack (zarr, s3fs, xarray) not available.")


def default_mast_cache_dir() -> Path:
    """Return the local cache directory for public MAST Zarr downloads."""
    override = os.environ.get("SCPN_MAST_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[3] / "data" / "mast_cache"


class MastIngestor:
    """Ingests real plasma data from the MAST (Mega Ampere Spherical Tokamak)."""

    ENDPOINT_URL = "https://s3.echo.stfc.ac.uk"
    BUCKET_NAME = "mast"

    def __init__(self, cache_dir: Optional[str | Path] = None) -> None:
        if not _HAS_FAIR_MAST_STACK:
            raise ImportError(
                "MAST ingestion requires zarr, s3fs, and xarray. "
                "Ensure the repository external/ directory or project environment is present."
            )

        if cache_dir is None:
            cache_dir = default_mast_cache_dir()

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._filesystems: list[Any] = []

    def _filesystem(self) -> Any:
        """Create a cache filesystem and retain it for explicit cleanup."""
        if fsspec is None:
            raise ImportError("MAST ingestion requires fsspec.")
        fs = fsspec.filesystem(
            "simplecache",
            cache_storage=str(self.cache_dir),
            target_protocol="s3",
            target_options={
                "anon": True,
                "endpoint_url": self.ENDPOINT_URL,
                "skip_instance_cache": True,
            },
        )
        self._filesystems.append(fs)
        return fs

    def close(self) -> None:
        """Close retained FAIR MAST S3 sessions before interpreter teardown."""
        loops: list[Any] = []
        while self._filesystems:
            fs = self._filesystems.pop()
            inner = getattr(fs, "fs", fs)
            loop = getattr(inner, "loop", None)
            if loop is not None:
                loops.append(loop)
            for client in (getattr(inner, "s3", None), getattr(inner, "_s3", None)):
                if client is None:
                    continue
                try:
                    client._endpoint.http_session._connector._close()
                except AttributeError:
                    pass

        registry = getattr(weakref.finalize, "_registry", {})
        for finalizer, info in list(registry.items()):
            args = cast(tuple[Any, ...], getattr(info, "args", ()))
            func = getattr(info, "func", None)
            if (
                len(args) >= 2
                and args[0] in loops
                and getattr(func, "__module__", "") == "s3fs.core"
                and getattr(func, "__name__", "") == "close_session"
            ):
                finalizer.detach()

    def load_shot_summary(self, shot_id: int) -> dict[str, NDArray[np.float64]]:
        """Load plasma current and time-series for a specific shot."""
        if xr is None:
            raise ImportError("MAST ingestion requires xarray.")
        url = f"s3://{self.BUCKET_NAME}/level2/shots/{shot_id}.zarr"

        fs = self._filesystem()
        store = fs.get_mapper(url)

        ds = xr.open_zarr(store, group="summary", consolidated=True)
        time = np.asarray(ds.time.values, dtype=np.float64)
        if "line_average_n_e" in ds:
            density = np.asarray(ds.line_average_n_e.values, dtype=np.float64)
        else:
            density = np.full(time.shape, np.nan, dtype=np.float64)
        return {
            "time": time,
            "ip": np.asarray(ds.ip.values, dtype=np.float64),
            "density": density,
        }

    def load_magnetic_probes(
        self, shot_id: int, probe_ids: Optional[list[str]] = None
    ) -> dict[str, NDArray[np.float64]]:
        """Load raw magnetic probe signals."""
        if xr is None:
            raise ImportError("MAST ingestion requires xarray.")
        url = f"s3://{self.BUCKET_NAME}/level2/shots/{shot_id}.zarr"

        fs = self._filesystem()
        store = fs.get_mapper(url)

        ds = xr.open_zarr(store, group="magnetics", consolidated=True)

        if probe_ids is None:
            probe_ids = list(ds.data_vars)[:10]

        out: dict[str, NDArray[np.float64]] = {"time": np.asarray(ds.time.values, dtype=np.float64)}
        for pid in probe_ids:
            if pid in ds.data_vars:
                out[pid] = np.asarray(ds[pid].values, dtype=np.float64)

        return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        ingestor = MastIngestor()
        data = ingestor.load_shot_summary(30421)
        print(f"Loaded Shot 30421: {len(data['time'])} samples.")
    except Exception as e:
        print(f"MAST Ingestor Check Failed: {e}")
