# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tokamak Disruption Archive Utilities
"""Disruption shot NPZ loaders extracted from tokamak_archive monolith."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def list_disruption_shots(
    *,
    disruption_dir: Path,
) -> list[str]:
    """Return sorted disruption-shot NPZ names (without extension)."""
    d = Path(disruption_dir)
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.npz"))


def load_disruption_shot(
    shot_name_or_path: str | Path,
    *,
    disruption_dir: Path,
) -> dict[str, Any]:
    """Load a single disruption shot NPZ file."""
    p = Path(shot_name_or_path)
    if not p.suffix:
        p = Path(disruption_dir) / f"{p.name}.npz"
    if p.suffix.lower() != ".npz":
        raise ValueError(f"Disruption shot file must be .npz: {p}")
    if not p.exists():
        raise FileNotFoundError(f"Disruption shot file not found: {p}")

    with np.load(str(p), allow_pickle=False) as raw:
        required_array_keys = {
            "time_s",
            "Ip_MA",
            "BT_T",
            "beta_N",
            "q95",
            "ne_1e19",
            "n1_amp",
            "n2_amp",
            "locked_mode_amp",
            "dBdt_gauss_per_s",
            "vertical_position_m",
        }
        required_scalar_keys = {"is_disruption", "disruption_time_idx", "disruption_type"}
        all_required = required_array_keys | required_scalar_keys
        present = set(raw.files)
        missing = all_required - present
        if missing:
            raise ValueError(f"NPZ file {p.name} missing keys: {sorted(missing)}")

        result: dict[str, Any] = {}
        for k in required_array_keys:
            result[k] = np.asarray(raw[k], dtype=np.float64)
        result["is_disruption"] = bool(np.asarray(raw["is_disruption"]).reshape(()).item())
        result["disruption_time_idx"] = int(
            np.asarray(raw["disruption_time_idx"]).reshape(()).item()
        )
        result["disruption_type"] = str(np.asarray(raw["disruption_type"]).reshape(()).item())
        return result
