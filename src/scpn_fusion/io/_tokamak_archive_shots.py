# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tokamak Archive Shot Wrappers
"""Disruption and synthetic shot wrapper APIs for the tokamak archive facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scpn_fusion.io.tokamak_archive_profiles import (
    default_disruption_dir,
    default_synthetic_dir,
)
from scpn_fusion.io.tokamak_disruption_archive import (
    list_disruption_shots as _list_disruption_shots_impl,
    load_disruption_shot as _load_disruption_shot_impl,
)
from scpn_fusion.io.tokamak_synthetic_archive import (
    generate_synthetic_shot_database as _generate_synthetic_shot_database_impl,
    list_synthetic_shots as _list_synthetic_shots_impl,
    load_synthetic_shot as _load_synthetic_shot_impl,
)


def list_disruption_shots(
    *,
    disruption_dir: Path | None = None,
) -> list[str]:
    """
    Return the names of all available disruption shot NPZ files.

    Each name corresponds to a file in ``disruption_dir`` without the
    ``.npz`` extension and sorted alphabetically.
    """
    d = disruption_dir if disruption_dir is not None else default_disruption_dir()
    return _list_disruption_shots_impl(disruption_dir=Path(d))


def load_disruption_shot(
    shot_name_or_path: str | Path,
    *,
    disruption_dir: Path | None = None,
) -> dict[str, Any]:
    """Load one disruption shot NPZ payload through the hardened archive loader."""
    d = disruption_dir if disruption_dir is not None else default_disruption_dir()
    return _load_disruption_shot_impl(shot_name_or_path, disruption_dir=Path(d))


def generate_synthetic_shot_database(
    *,
    n_shots: int = 60,
    output_dir: Path | None = None,
    seed: int = 20260218,
) -> list[dict[str, Any]]:
    """Generate synthetic tokamak shot NPZ files through the synthetic archive loader."""
    out_dir = Path(output_dir) if output_dir is not None else default_synthetic_dir()
    return _generate_synthetic_shot_database_impl(
        n_shots=n_shots,
        output_dir=out_dir,
        seed=seed,
    )


def list_synthetic_shots(
    *,
    synthetic_dir: Path | None = None,
) -> list[str]:
    """Return names of all available synthetic shot NPZ files without extension."""
    d = Path(synthetic_dir) if synthetic_dir is not None else default_synthetic_dir()
    return _list_synthetic_shots_impl(synthetic_dir=d)


def load_synthetic_shot(
    shot_name_or_path: str | Path,
    *,
    synthetic_dir: Path | None = None,
) -> dict[str, Any]:
    """Load one synthetic shot NPZ payload through the hardened archive loader."""
    d = Path(synthetic_dir) if synthetic_dir is not None else default_synthetic_dir()
    return _load_synthetic_shot_impl(shot_name_or_path, synthetic_dir=d)
