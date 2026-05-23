# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Šotek. All rights reserved.
# © Code 2020-2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — IMAS Connector Common Tests
"""Module-specific tests for shared IMAS connector coercion contracts."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "scpn_fusion" / "io" / "imas_connector_common.py"
SPEC = importlib.util.spec_from_file_location("imas_connector_common", MODULE_PATH)
assert SPEC and SPEC.loader
common = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = common
SPEC.loader.exec_module(common)

from imas_connector_common import (
    REQUIRED_DIGITAL_TWIN_STATE_KEYS,
    REQUIRED_PROFILE_1D_KEYS,
    _coerce_finite_real,
    _coerce_finite_real_sequence,
    _coerce_int,
    _coerce_profiles_1d,
    _missing_required_keys,
)


def test_missing_required_keys_preserves_required_order() -> None:
    """Missing-key reporting follows the published required-key order."""
    payload = {REQUIRED_PROFILE_1D_KEYS[1]: [1.0]}

    missing = _missing_required_keys(payload, REQUIRED_PROFILE_1D_KEYS)

    assert missing == ["rho_norm", "electron_density_1e20_m3"]


def test_coerce_int_rejects_bool_and_values_below_minimum() -> None:
    """Integer coercion rejects bools and enforces lower physical bounds."""
    assert _coerce_int("shot", 42, minimum=0) == 42

    with pytest.raises(ValueError, match="shot must be an integer"):
        _coerce_int("shot", True, minimum=0)
    with pytest.raises(ValueError, match="shot must be >= 0"):
        _coerce_int("shot", -1, minimum=0)


def test_coerce_finite_real_rejects_bool_nan_and_low_values() -> None:
    """Real coercion rejects non-physical sentinel values and lower-bound breaks."""
    assert _coerce_finite_real("density", 1, minimum=0.0) == 1.0

    for invalid in (False, math.nan, math.inf):
        with pytest.raises(ValueError, match="density must be a finite number"):
            _coerce_finite_real("density", invalid, minimum=0.0)
    with pytest.raises(ValueError, match="density must be >= 0.0"):
        _coerce_finite_real("density", -0.1, minimum=0.0)


def test_coerce_finite_real_sequence_enforces_bounds_length_and_monotonicity() -> None:
    """Profile-axis coercion enforces sequence shape and radial ordering."""
    assert _coerce_finite_real_sequence(
        "rho_norm",
        [0.0, 0.5, 1.0],
        minimum_len=2,
        minimum=0.0,
        maximum=1.0,
        strictly_increasing=True,
    ) == [0.0, 0.5, 1.0]

    with pytest.raises(ValueError, match="rho_norm must be a sequence"):
        _coerce_finite_real_sequence("rho_norm", "0,1")
    with pytest.raises(ValueError, match="rho_norm must contain at least 2 values"):
        _coerce_finite_real_sequence("rho_norm", [0.0], minimum_len=2)
    with pytest.raises(ValueError, match=r"rho_norm\[1\] must be <= 1.0"):
        _coerce_finite_real_sequence("rho_norm", [0.0, 1.1], maximum=1.0)
    with pytest.raises(ValueError, match="rho_norm must be strictly increasing"):
        _coerce_finite_real_sequence("rho_norm", [0.0, 0.0], strictly_increasing=True)


def test_coerce_profiles_1d_returns_consistent_float_profiles() -> None:
    """A valid IMAS profile payload is normalised into aligned float arrays."""
    payload = {
        "rho_norm": [0, 0.5, 1],
        "electron_temp_keV": [4, 2, 0.5],
        "electron_density_1e20_m3": [0.8, 0.9, 0.7],
    }

    profiles = _coerce_profiles_1d(payload, name="state.profiles_1d")

    assert profiles == {
        "rho_norm": [0.0, 0.5, 1.0],
        "electron_temp_keV": [4.0, 2.0, 0.5],
        "electron_density_1e20_m3": [0.8, 0.9, 0.7],
    }


def test_coerce_profiles_1d_reports_missing_and_misaligned_profile_payloads() -> None:
    """Profile coercion rejects incomplete and length-mismatched IDS payloads."""
    assert set(REQUIRED_PROFILE_1D_KEYS).issubset(REQUIRED_DIGITAL_TWIN_STATE_KEYS)

    with pytest.raises(ValueError, match="state.profiles_1d missing keys"):
        _coerce_profiles_1d({"rho_norm": [0.0, 1.0]}, name="state.profiles_1d")

    payload = {
        "rho_norm": [0.0, 0.5, 1.0],
        "electron_temp_keV": [4.0, 2.0],
        "electron_density_1e20_m3": [0.8, 0.9, 0.7],
    }
    with pytest.raises(ValueError, match="electron_temp_keV length must match"):
        _coerce_profiles_1d(payload, name="state.profiles_1d")
