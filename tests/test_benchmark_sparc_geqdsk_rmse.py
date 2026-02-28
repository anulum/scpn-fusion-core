# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — SPARC RMSE Benchmark Hardening Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────

from __future__ import annotations

import sys
import types

import numpy as np

from validation import benchmark_sparc_geqdsk_rmse


def test_reduced_order_proxy_preserves_shape_and_is_finite() -> None:
    eq = benchmark_sparc_geqdsk_rmse._build_sparc_like_equilibrium(NR=33, NZ=35)
    proxy = benchmark_sparc_geqdsk_rmse._reduced_order_proxy(eq["psi"], coarse_points=12)

    assert proxy.shape == eq["psi"].shape
    assert np.all(np.isfinite(proxy))
    assert np.min(proxy) >= 0.0
    assert np.max(proxy) <= 1.0


def test_run_neural_surrogate_fallback_reports_backend(monkeypatch) -> None:
    fake_module = types.ModuleType("scpn_fusion.core.neural_equilibrium")

    class _FailingAccelerator:
        def __init__(self) -> None:
            raise RuntimeError("synthetic-missing-weights")

    fake_module.NeuralEquilibriumAccelerator = _FailingAccelerator
    monkeypatch.setitem(sys.modules, "scpn_fusion.core.neural_equilibrium", fake_module)

    eq = benchmark_sparc_geqdsk_rmse._build_sparc_like_equilibrium(NR=33, NZ=33)
    pred, backend, reason = benchmark_sparc_geqdsk_rmse._run_neural_surrogate(eq)

    assert pred.shape == eq["psi"].shape
    assert backend == "reduced_order_proxy"
    assert "RuntimeError" in str(reason)


def test_run_benchmark_strict_backend_enforces_neural_requirement(monkeypatch) -> None:
    def _proxy_only(eq):  # type: ignore[no-untyped-def]
        return np.asarray(eq["psi"], dtype=np.float64), "reduced_order_proxy", "unit-test"

    monkeypatch.setattr(benchmark_sparc_geqdsk_rmse, "_run_neural_surrogate", _proxy_only)
    strict_result = benchmark_sparc_geqdsk_rmse.run_benchmark(
        grid_sizes=[33],
        require_neural_backend=True,
    )
    relaxed_result = benchmark_sparc_geqdsk_rmse.run_benchmark(
        grid_sizes=[33],
        require_neural_backend=False,
    )

    assert strict_result["passes"] is False
    assert strict_result["all_cases_neural_backend"] is False
    assert relaxed_result["passes"] is True
    assert relaxed_result["all_cases_neural_backend"] is False
    for row in strict_result["cases"]:
        assert row["surrogate_backend"] == "reduced_order_proxy"
        assert row["backend_requirement_satisfied"] is False
