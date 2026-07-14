# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Global Design Evaluator Dispatch Tests
"""Rust <-> NumPy dispatch tests for the reactor-design evaluator kernel.

These exercise the pure-NumPy ``GlobalDesignExplorer`` physics-scaling surrogate
and the Rust ``py_evaluate_design`` binding through the class-kernel dispatcher.
They require only NumPy and the compiled extension, so they run in the core-only
CI matrix.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.global_design_scanner import (
    GlobalDesignExplorer,
    _DesignEvaluatorRustKernel,
    create_design_evaluator,
)

_NUMERIC_KEYS = (
    "P_fus",
    "Q",
    "Wall_Load",
    "Div_Load_Baseline",
    "Shadow_Fraction",
    "Div_Load_Optimized",
    "Div_Load",
    "B_peak_HTS_T",
    "Zeff_Est",
    "beta_N_eff",
    "Cost",
)


def test_design_dispatch_registers_both_tiers() -> None:
    """The class-kernel registry carries RUST and NUMPY design-evaluator tiers."""
    from scpn_fusion.core import _multi_compat as multi

    kernels = multi.registered_kernel_classes()
    assert "global_design_scan" in kernels
    tiers = [tier.rstrip("*") for tier in kernels["global_design_scan"]]
    assert "rust" in tiers
    assert "numpy" in tiers


def test_design_numpy_floor_without_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    """The factory resolves to the NumPy explorer when Rust is unavailable."""
    from scpn_fusion.core import _multi_compat as multi

    multi._ensure_probed()
    monkeypatch.setitem(multi._availability, multi.BackendTier.RUST, False)
    monkeypatch.delitem(multi._class_dispatch_cache, "global_design_scan", raising=False)
    try:
        evaluator = create_design_evaluator("dummy")
        assert isinstance(evaluator, GlobalDesignExplorer)
        design = evaluator.evaluate_design(6.0, 5.3, 15.0)
        assert design["Model_Regime"] == "physics_scaling_surrogate"
        assert bool(np.isfinite(float(design["P_fus"])))
    finally:
        multi._class_dispatch_cache.pop("global_design_scan", None)


def test_rust_kernel_matches_explorer() -> None:
    """The Rust tier reproduces the Python explorer to floating-point round-off.

    Both tiers run the identical physics-scaling surrogate (Troyon/H-mode
    ``beta_N`` shaping, Eich divertor scaling, and the HEAT-ML magnetic-shadow
    ridge attenuation with the same frozen weights), so every numeric metric
    agrees to ~1e-12 relative and ``Constraint_OK`` is identical across the
    design envelope.
    """
    pytest.importorskip("scpn_fusion_rs")
    from scpn_fusion.core import _multi_compat_providers as providers

    numpy_kernel = providers._load_numpy_design_evaluator()("dummy")
    rust_kernel = providers._load_rust_design_evaluator()("dummy")
    assert isinstance(rust_kernel, _DesignEvaluatorRustKernel)

    rng = np.random.default_rng(2026)
    for _ in range(256):
        r = float(rng.uniform(1.1, 9.0))
        b = float(rng.uniform(4.0, 12.2))
        i_p = float(rng.uniform(2.0, 25.0))
        py = numpy_kernel.evaluate_design(r, b, i_p)
        rs = rust_kernel.evaluate_design(r, b, i_p)
        for key in _NUMERIC_KEYS:
            assert float(rs[key]) == pytest.approx(float(py[key]), rel=1e-11, abs=1e-11)
        assert bool(rs["Constraint_OK"]) == bool(py["Constraint_OK"])
        assert rs["Model_Regime"] == py["Model_Regime"]


def test_rust_kernel_caps_thread_through() -> None:
    """Custom engineering caps flip ``Constraint_OK`` identically in both tiers."""
    pytest.importorskip("scpn_fusion_rs")

    # A tight divertor cap must fail the constraint on both tiers.
    tight = dict(divertor_flux_cap_mw_m2=0.5, zeff_cap=0.4, hts_peak_cap_t=21.0)
    py = GlobalDesignExplorer("dummy", **tight)
    rs = _DesignEvaluatorRustKernel("dummy", **tight)
    py_design = py.evaluate_design(6.0, 5.3, 15.0)
    rs_design = rs.evaluate_design(6.0, 5.3, 15.0)
    assert bool(py_design["Constraint_OK"]) is False
    assert bool(rs_design["Constraint_OK"]) == bool(py_design["Constraint_OK"])


def test_rust_kernel_rejects_nonpositive_inputs() -> None:
    """The Rust tier reproduces the explorer's finite-positive input guards."""
    pytest.importorskip("scpn_fusion_rs")

    kernel = _DesignEvaluatorRustKernel("dummy")
    with pytest.raises(ValueError, match="R_maj"):
        kernel.evaluate_design(0.0, 5.3, 15.0)
    with pytest.raises(ValueError, match="B_field"):
        kernel.evaluate_design(6.0, -1.0, 15.0)
    with pytest.raises(ValueError, match="I_plasma"):
        kernel.evaluate_design(6.0, 5.3, float("nan"))


def test_rust_kernel_validates_caps() -> None:
    """The Rust tier reproduces the explorer's cap validation."""
    pytest.importorskip("scpn_fusion_rs")

    with pytest.raises(ValueError, match="zeff_cap must be <= 1.0"):
        _DesignEvaluatorRustKernel("dummy", zeff_cap=1.2)
    with pytest.raises(ValueError, match="divertor_flux_cap_mw_m2"):
        _DesignEvaluatorRustKernel("dummy", divertor_flux_cap_mw_m2=-1.0)
    with pytest.raises(ValueError, match="hts_peak_cap_t"):
        _DesignEvaluatorRustKernel("dummy", hts_peak_cap_t=float("inf"))


def test_create_design_evaluator_returns_dict_surface() -> None:
    """The dispatched evaluator exposes the evaluate_design dict contract."""
    evaluator = create_design_evaluator("dummy")
    assert callable(evaluator.evaluate_design)
    design = evaluator.evaluate_design(3.2, 8.1, 9.5)
    for key in (*_NUMERIC_KEYS, "R", "B", "Ip", "Model_Regime", "Constraint_OK"):
        assert key in design
    assert float(design["Cost"]) == pytest.approx(3.2**3 * 8.1)
