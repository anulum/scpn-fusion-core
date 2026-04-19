# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.io.tokamak_archive_profiles import (
    _coerce_finite,
    _coerce_int,
    _normalize_machine,
    _profile_key,
    _stable_shot_from_text,
    _synthetic_cmod_psi_contour,
    TokamakProfile,
    default_reference_data_root,
)


def test_default_reference_data_root_honors_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCPN_DATA_DIR", "C:/tmp/custom_data")
    assert str(default_reference_data_root()).replace("\\", "/").endswith("/tmp/custom_data")


def test_profile_key_and_machine_normalization() -> None:
    profile = TokamakProfile(
        machine="DIII-D",
        shot=123456,
        time_ms=100.4,
        beta_n=2.0,
        q95=4.2,
        tau_e_ms=120.0,
        psi_contour=(0.0, 0.5, 1.0),
        sensor_trace=(0.1, 0.2, 0.3),
        toroidal_n1_amp=0.1,
        toroidal_n2_amp=0.05,
        toroidal_n3_amp=0.02,
        disruption=False,
    )
    assert _profile_key(profile) == ("DIII-D", 123456, 100)
    assert _normalize_machine("diiid") == "DIII-D"
    assert _normalize_machine("c-mod") == "C-Mod"


@pytest.mark.parametrize("value", [True, "x", -1])
def test_integer_coercion_rejects_invalid(value: object) -> None:
    with pytest.raises(ValueError):
        _coerce_int("shot", value, minimum=0)


@pytest.mark.parametrize("value", [True, "x", np.nan, np.inf, -np.inf, -1.0])
def test_finite_coercion_rejects_invalid(value: object) -> None:
    with pytest.raises(ValueError):
        _coerce_finite("beta_n", value, minimum=0.0)


def test_stable_shot_hash_and_synthetic_contour_range() -> None:
    a = _stable_shot_from_text("sparc_0001")
    b = _stable_shot_from_text("sparc_0001")
    c = _stable_shot_from_text("sparc_0002")
    assert a == b
    assert a != c

    psi = _synthetic_cmod_psi_contour(kappa=1.8, delta=0.3, points=64)
    assert psi.shape == (64,)
    assert np.all(np.isfinite(psi))
    assert float(np.min(psi)) >= 0.0
    assert float(np.max(psi)) <= 1.0
