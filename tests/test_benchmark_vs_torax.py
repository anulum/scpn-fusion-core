# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — TORAX Benchmark Hardening Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────

from __future__ import annotations

import sys
import types

import numpy as np

from validation import benchmark_vs_torax


def test_stable_seed_is_deterministic_and_bounded() -> None:
    seed_a = benchmark_vs_torax._stable_seed_from_name("ITER-baseline")
    seed_b = benchmark_vs_torax._stable_seed_from_name("ITER-baseline")
    seed_c = benchmark_vs_torax._stable_seed_from_name("SPARC-V2C")

    assert seed_a == seed_b
    assert seed_a != seed_c
    assert 0 <= seed_a < 2**31
    assert 0 <= seed_c < 2**31


def test_run_our_transport_fallback_reports_reason_and_seed(monkeypatch) -> None:
    fake_module = types.ModuleType("scpn_fusion.core.neural_transport")

    class _FailingModel:
        def predict_profile(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("forced-fallback")

    fake_module.NeuralTransportModel = _FailingModel
    fake_module.TransportInputs = object
    monkeypatch.setitem(sys.modules, "scpn_fusion.core.neural_transport", fake_module)

    case = benchmark_vs_torax.CASES[0]
    out_a = benchmark_vs_torax._run_our_transport(case)
    out_b = benchmark_vs_torax._run_our_transport(case)

    assert out_a["__backend__"] == "analytic_fallback"
    assert out_a["__seed__"] == benchmark_vs_torax._stable_seed_from_name(case.name)
    assert "inference_error:" in str(out_a["__fallback_reason__"])
    np.testing.assert_allclose(out_a["te_keV"], out_b["te_keV"], rtol=0.0, atol=0.0)


def test_run_benchmark_includes_transport_backend_fields(monkeypatch) -> None:
    def _stub_transport(case):  # type: ignore[no-untyped-def]
        rho = np.linspace(0.0, 1.0, case.n_rho)
        te = np.ones(case.n_rho, dtype=np.float64)
        return {
            "rho": rho,
            "te_keV": te,
            "ti_keV": 0.85 * te,
            "ne_1e19": te,
            "__backend__": "stub",
            "__fallback_reason__": "unit-test",
            "__seed__": 123,
        }

    monkeypatch.setattr(benchmark_vs_torax, "_run_our_transport", _stub_transport)
    result = benchmark_vs_torax.run_benchmark()

    assert "cases" in result and result["cases"]
    for row in result["cases"]:
        assert row["transport_backend"] == "stub"
        assert row["fallback_reason"] == "unit-test"
        assert row["fallback_seed"] == 123
