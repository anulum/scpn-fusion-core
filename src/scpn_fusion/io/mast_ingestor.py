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

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# 1. Resolve NTFS-hosted external libraries path
_EXTERNAL_ROOT = Path(__file__).resolve().parents[3] / "external"
if _EXTERNAL_ROOT.exists() and str(_EXTERNAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_ROOT))

# 2. Delayed imports to handle external lib requirement
try:
    import fsspec
    import xarray as xr
    import zarr as _zarr  # noqa: F401

    _HAS_FAIR_MAST_STACK = True
except ImportError:
    _HAS_FAIR_MAST_STACK = False
    logger.debug("FAIR MAST stack (zarr, s3fs, xarray) not available.")


def default_mast_cache_dir() -> Path:
    """Return the local cache directory for public MAST Zarr downloads."""
    override = os.environ.get("SCPN_MAST_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[3] / "data" / "mast_cache"


class MastIngestor:
    """
    Ingests real plasma data from the MAST (Mega Ampere Spherical Tokamak).
    """

    ENDPOINT_URL = "https://s3.echo.stfc.ac.uk"
    BUCKET_NAME = "mast"

    def __init__(self, cache_dir: Optional[str | Path] = None) -> None:
        if not _HAS_FAIR_MAST_STACK:
            raise ImportError(
                "MAST ingestion requires zarr, s3fs, and xarray. "
                "Ensure NTFS external/ directory is present."
            )

        if cache_dir is None:
            cache_dir = default_mast_cache_dir()

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_shot_summary(self, shot_id: int) -> Dict[str, NDArray[np.float64]]:
        """Load plasma current and time-series for a specific shot."""
        url = f"s3://{self.BUCKET_NAME}/level2/shots/{shot_id}.zarr"

        fs = fsspec.filesystem(
            "simplecache",
            cache_storage=str(self.cache_dir),
            target_protocol="s3",
            target_options={"anon": True, "endpoint_url": self.ENDPOINT_URL},
        )
        store = fs.get_mapper(url)

        ds = xr.open_zarr(store, group="summary", consolidated=True)
        return {
            "time": ds.time.values,
            "ip": ds.ip.values,
            "density": ds.line_average_n_e.values,
        }

    def load_magnetic_probes(
        self, shot_id: int, probe_ids: Optional[List[str]] = None
    ) -> Dict[str, NDArray[np.float64]]:
        """Load raw magnetic probe signals."""
        url = f"s3://{self.BUCKET_NAME}/level2/shots/{shot_id}.zarr"

        fs = fsspec.filesystem(
            "simplecache",
            cache_storage=str(self.cache_dir),
            target_protocol="s3",
            target_options={"anon": True, "endpoint_url": self.ENDPOINT_URL},
        )
        store = fs.get_mapper(url)

        ds = xr.open_zarr(store, group="magnetics", consolidated=True)

        if probe_ids is None:
            probe_ids = list(ds.data_vars)[:10]

        out = {"time": ds.time.values}
        for pid in probe_ids:
            if pid in ds.data_vars:
                out[pid] = ds[pid].values

        return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        ingestor = MastIngestor()
        data = ingestor.load_shot_summary(30421)
        print(f"Loaded Shot 30421: {len(data['time'])} samples.")
    except Exception as e:
        print(f"MAST Ingestor Check Failed: {e}")
