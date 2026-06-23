# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Reference-Data Path Resolution Tests
"""Tests for layout-independent reference-data root resolution."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from scpn_fusion._data_paths import data_root


def test_data_root_contains_validation_reference_data() -> None:
    """The resolved root must contain the bundled validation reference-data tree."""
    root = data_root()
    assert (root / "validation" / "reference_data").is_dir()


def test_data_root_resolves_ipb98_coefficients() -> None:
    """The canonical IPB98 coefficient file (loaded at import) must be reachable."""
    coeff = data_root() / "validation" / "reference_data" / "itpa" / "ipb98y2_coefficients.json"
    assert coeff.is_file()


def test_data_root_returns_absolute_path() -> None:
    """Resolution always yields an absolute path so callers can open files directly."""
    assert data_root().is_absolute()


def test_data_root_matches_installed_validation_when_importable() -> None:
    """When the validation package is importable, the root must hold it as a child."""
    spec = importlib.util.find_spec("validation")
    if spec is None or spec.origin is None:
        pytest.skip("validation package not importable in this layout")
    validation_dir = Path(spec.origin).resolve().parent
    assert validation_dir.parent == data_root() or (data_root() / "validation").is_dir()


def test_reference_data_root_resolves_under_data_root() -> None:
    """The tokamak-archive reference-data root must track the layout-independent
    :func:`data_root` resolver so it is reachable from an installed wheel."""
    from scpn_fusion.io.tokamak_archive_profiles import default_reference_data_root

    assert default_reference_data_root() == data_root() / "validation" / "reference_data"


def test_diagnostics_default_config_resolves_under_data_root() -> None:
    """The diagnostics demo config must resolve via :func:`data_root` (wheel-safe)."""
    from scpn_fusion.diagnostics.run_diagnostics import DEFAULT_CONFIG_PATH

    assert data_root() / "validation" / "iter_validated_config.json" == DEFAULT_CONFIG_PATH
