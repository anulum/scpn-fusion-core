# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Impurity Transport Diagnostics Tests
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_fusion.core.impurity_transport_diagnostics import (
    neoclassical_impurity_pinch,
    tungsten_accumulation_diagnostic,
)


def test_impurity_pinch() -> None:
    rho = np.linspace(0, 1, 50)
    # Peaked density
    ne = 1e20 * (1.0 - 0.8 * rho**2)
    # Flat Ti
    Ti = 5000.0 * np.ones(50)
    q = np.ones(50)
    eps = 0.2 + 0.2 * rho

    V_W = neoclassical_impurity_pinch(74, ne, 5000.0 * np.ones(50), Ti, q, rho, 6.2, 2.0, eps)

    # With radial coordinate increasing outward, a peaked density has 1/L_n > 0.
    # The trace neoclassical pinch should therefore point inward: V_r < 0.
    assert V_W[25] < 0.0


def test_impurity_pinch_temperature_screening_contract() -> None:
    rho = np.linspace(0, 1, 50)
    ne = 1e20 * np.ones(50)
    Ti_hot_core = 5000.0 * (1.0 - 0.6 * rho**2)
    q = 1.0 + rho
    eps = 0.2 + 0.2 * rho

    V_W = neoclassical_impurity_pinch(
        74, ne, 5000.0 * np.ones(50), Ti_hot_core, q, rho, 6.2, 2.0, eps
    )

    assert V_W[25] < 0.0
    assert abs(V_W[-2]) > abs(V_W[2])


def test_impurity_pinch_rejects_invalid_domain() -> None:
    rho = np.linspace(0, 1, 50)
    ne = np.ones(50) * 1e20
    Ti = np.ones(50) * 5000.0
    q = np.ones(50)
    eps = 0.2 + 0.2 * rho

    with pytest.raises(ValueError, match="Z"):
        neoclassical_impurity_pinch(0, ne, Ti, Ti, q, rho, 6.2, 2.0, eps)
    with pytest.raises(ValueError, match="ne"):
        neoclassical_impurity_pinch(74, np.zeros(50), Ti, Ti, q, rho, 6.2, 2.0, eps)
    with pytest.raises(ValueError, match="matching shapes"):
        neoclassical_impurity_pinch(74, ne[:-1], Ti, Ti, q, rho, 6.2, 2.0, eps)


def test_accumulation_diagnostic() -> None:
    ne = np.ones(50) * 1e20
    nW = np.ones(50) * 1e16  # c_W = 1e-4 -> critical
    nW[0] = 1e17  # Core peaked -> peaking factor 10

    diag = tungsten_accumulation_diagnostic(nW, ne)

    assert diag["danger_level"] == "critical"
    assert diag["peaking_factor"] == 10.0


def _pinch_kwargs() -> dict[str, Any]:
    return {
        "Z": 18,
        "ne": np.full(5, 1e19),
        "Te_eV": np.full(5, 1000.0),
        "Ti_eV": np.full(5, 1000.0),
        "q": np.full(5, 2.0),
        "rho": np.linspace(0.1, 1.0, 5),
        "R0": 6.2,
        "a": 2.0,
        "epsilon": np.linspace(0.05, 0.3, 5),
    }


class TestNeoclassicalPinchValidation:
    def test_rejects_short_rho(self) -> None:
        kw = _pinch_kwargs()
        for key in ("ne", "Te_eV", "Ti_eV", "q", "rho", "epsilon"):
            kw[key] = kw[key][:2]
        with pytest.raises(ValueError, match="at least three points"):
            neoclassical_impurity_pinch(**kw)

    def test_rejects_non_increasing_rho(self) -> None:
        kw = _pinch_kwargs()
        kw["rho"] = np.array([1.0, 0.7, 0.5, 0.3, 0.1])
        with pytest.raises(ValueError, match="rho must be finite and strictly increasing"):
            neoclassical_impurity_pinch(**kw)

    def test_rejects_nonpositive_ti(self) -> None:
        kw = _pinch_kwargs()
        kw["Ti_eV"] = np.array([1000.0, 0.0, 1000.0, 1000.0, 1000.0])
        with pytest.raises(ValueError, match="Ti_eV must be finite and positive"):
            neoclassical_impurity_pinch(**kw)

    def test_rejects_nonpositive_q(self) -> None:
        kw = _pinch_kwargs()
        kw["q"] = np.array([2.0, 2.0, 0.0, 2.0, 2.0])
        with pytest.raises(ValueError, match="q must be finite and positive"):
            neoclassical_impurity_pinch(**kw)

    def test_rejects_negative_epsilon(self) -> None:
        kw = _pinch_kwargs()
        kw["epsilon"] = np.array([0.05, -0.1, 0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match="epsilon must be finite and non-negative"):
            neoclassical_impurity_pinch(**kw)

    def test_rejects_bad_major_radius(self) -> None:
        kw = _pinch_kwargs()
        kw["R0"] = 0.0
        with pytest.raises(ValueError, match="R0 must be finite and positive"):
            neoclassical_impurity_pinch(**kw)

    def test_rejects_bad_minor_radius(self) -> None:
        kw = _pinch_kwargs()
        kw["a"] = 0.0
        with pytest.raises(ValueError, match="a must be finite and positive"):
            neoclassical_impurity_pinch(**kw)


def test_tungsten_diagnostic_danger_levels() -> None:
    ne = np.full(3, 1e20)
    safe = tungsten_accumulation_diagnostic(np.full(3, 1e14), ne)
    assert safe["danger_level"] == "safe"
    warning = tungsten_accumulation_diagnostic(np.full(3, 3e15), ne)
    assert warning["danger_level"] == "warning"
