# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — SPARC RMSE Benchmark Hardening Tests

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
import pytest

from validation import benchmark_sparc_geqdsk_rmse


def test_reduced_order_proxy_preserves_shape_and_is_finite() -> None:
    eq = benchmark_sparc_geqdsk_rmse._build_sparc_like_equilibrium(NR=33, NZ=35)
    proxy = benchmark_sparc_geqdsk_rmse._reduced_order_proxy(eq["psi"], coarse_points=12)

    assert proxy.shape == eq["psi"].shape
    assert np.all(np.isfinite(proxy))
    assert np.min(proxy) >= 0.0
    assert np.max(proxy) <= 1.0


def test_run_neural_surrogate_fallback_reports_backend(monkeypatch) -> None:
    fake_module = cast(Any, types.ModuleType("scpn_fusion.core.neural_equilibrium"))
    fake_module.DEFAULT_WEIGHTS_PATH = Path("synthetic_weights_missing.npz")

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
    def _proxy_only(eq: dict[str, Any]) -> tuple[np.ndarray, str, str]:
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


def test_run_benchmark_uses_reference_geqdsk_mode_when_cases_available(monkeypatch) -> None:
    psi = np.eye(9, dtype=np.float64)

    monkeypatch.setattr(
        benchmark_sparc_geqdsk_rmse,
        "_load_sparc_geqdsk_cases",
        lambda: [
            {
                "name": "sparc_ref_case",
                "machine": "sparc",
                "source_file": "sparc_ref_case.geqdsk",
                "psi": psi,
                "feature_vector_full": np.array([-8.7, -12.2, 1.85, 0.0, 1.0, 1.0, 0.0, 2.37]),
                "Ip": -8.7,
                "B0": -12.2,
                "R0": 1.85,
                "kappa": 1.8,
                "geqdsk_contract": {"geqdsk_contract_pass": True, "psi_span": 1.0},
            }
        ],
    )

    def _proxy_only(eq: dict[str, Any]) -> tuple[np.ndarray, str, str]:
        return np.asarray(eq["psi"], dtype=np.float64), "reduced_order_proxy", "unit-test"

    monkeypatch.setattr(benchmark_sparc_geqdsk_rmse, "_run_neural_surrogate", _proxy_only)
    result = benchmark_sparc_geqdsk_rmse.run_benchmark(grid_sizes=[9], require_neural_backend=False)

    assert result["mode"] == "reference_geqdsk"
    assert result["reference_case_count"] == 1
    assert result["machine_counts"] == {"sparc": 1}
    assert len(result["cases"]) == 1
    assert result["cases"][0]["machine"] == "sparc"
    assert result["cases"][0]["source_file"] == "sparc_ref_case.geqdsk"
    assert result["cases"][0]["geqdsk_contract_pass"] is True


def test_sparc_loader_includes_geqdsk_and_eqdsk_extensions(monkeypatch, tmp_path) -> None:
    ref_dir = tmp_path / "sparc"
    ref_dir.mkdir()
    geqdsk_path = ref_dir / "case_a.geqdsk"
    eqdsk_path = ref_dir / "case_b.eqdsk"
    geqdsk_path.write_text("stub", encoding="utf-8")
    eqdsk_path.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(benchmark_sparc_geqdsk_rmse, "SPARC_REFERENCE_DIR", ref_dir)

    paths = benchmark_sparc_geqdsk_rmse._sparc_geqdsk_paths()

    assert [path.name for path in paths] == ["case_a.geqdsk", "case_b.eqdsk"]


def test_reference_loader_preserves_machine_provenance(monkeypatch, tmp_path) -> None:
    sparc_dir = tmp_path / "sparc"
    diiid_dir = tmp_path / "diiid"
    jet_dir = tmp_path / "jet"
    for directory in (sparc_dir, diiid_dir, jet_dir):
        directory.mkdir()
    (sparc_dir / "sparc_case.eqdsk").write_text("stub", encoding="utf-8")
    (diiid_dir / "diiid_case.geqdsk").write_text("stub", encoding="utf-8")
    (jet_dir / "jet_case.geqdsk").write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        benchmark_sparc_geqdsk_rmse,
        "REFERENCE_MACHINE_DIRS",
        {"sparc": sparc_dir, "diiid": diiid_dir, "jet": jet_dir},
    )

    paths = benchmark_sparc_geqdsk_rmse._reference_geqdsk_paths()

    assert [(machine, path.name) for machine, path in paths] == [
        ("diiid", "diiid_case.geqdsk"),
        ("jet", "jet_case.geqdsk"),
        ("sparc", "sparc_case.eqdsk"),
    ]


def test_geqdsk_contract_metrics_validate_axis_and_boundary() -> None:
    from scpn_fusion.core.eqdsk import GEqdsk

    r = np.linspace(1.0, 2.0, 9)
    z = np.linspace(-0.5, 0.5, 9)
    _rr, zz = np.meshgrid(r, z)
    psi = 0.25 * zz**2
    eq = GEqdsk(
        nw=9,
        nh=9,
        rdim=1.0,
        zdim=1.0,
        rleft=1.0,
        zmid=0.0,
        rmaxis=1.5,
        zmaxis=0.0,
        simag=0.0,
        sibry=float(np.max(psi)),
        psirz=psi,
        fpol=np.ones(9),
        pres=np.linspace(1.0, 0.0, 9),
        ffprime=np.full(9, -0.5),
        pprime=np.zeros(9),
        qpsi=np.linspace(1.0, 3.0, 9),
        rbdry=np.array([1.25, 1.75, 1.75, 1.25]),
        zbdry=np.array([-0.25, -0.25, 0.25, 0.25]),
    )

    metrics = benchmark_sparc_geqdsk_rmse._geqdsk_contract_metrics(eq)

    assert metrics["geqdsk_contract_pass"] is True
    assert metrics["geqdsk_source_contract_pass"] is True
    assert metrics["axis_index_interior"] is True
    assert metrics["boundary_inside_grid"] is True
    assert metrics["axis_error_m"] <= metrics["axis_tolerance_m"]
    assert metrics["axis_psi_error_fraction"] <= 1e-2
    assert metrics["q_finite_nonzero"] is True
    assert metrics["gs_profile_source_ok"] is True
    assert metrics["gs_profile_source_rel_l2"] <= metrics["gs_profile_source_threshold"]
    assert metrics["gs_profile_source_best_fit_convention"] == "canonical"
    assert metrics["gs_profile_source_best_fit_scale"] == pytest.approx(1.0)
    assert metrics["gs_profile_source_best_fit_rel_l2"] < 1e-12


def test_geqdsk_contract_metrics_reject_inconsistent_profile_source() -> None:
    from scpn_fusion.core.eqdsk import GEqdsk

    r = np.linspace(1.0, 2.0, 9)
    z = np.linspace(-0.5, 0.5, 9)
    _rr, zz = np.meshgrid(r, z)
    psi = 0.25 * zz**2
    eq = GEqdsk(
        nw=9,
        nh=9,
        rdim=1.0,
        zdim=1.0,
        rleft=1.0,
        zmid=0.0,
        rmaxis=1.5,
        zmaxis=0.0,
        simag=0.0,
        sibry=float(np.max(psi)),
        psirz=psi,
        fpol=np.ones(9),
        pres=np.linspace(1.0, 0.0, 9),
        ffprime=np.zeros(9),
        pprime=np.zeros(9),
        qpsi=np.linspace(1.0, 3.0, 9),
        rbdry=np.array([1.25, 1.75, 1.75, 1.25]),
        zbdry=np.array([-0.25, -0.25, 0.25, 0.25]),
    )

    metrics = benchmark_sparc_geqdsk_rmse._geqdsk_contract_metrics(eq)

    assert metrics["geqdsk_contract_pass"] is True
    assert metrics["geqdsk_source_contract_pass"] is False
    assert metrics["gs_profile_source_ok"] is False
    assert metrics["gs_profile_source_rel_l2"] > metrics["gs_profile_source_threshold"]


def test_geqdsk_contract_metrics_attributes_two_pi_source_scale() -> None:
    from scpn_fusion.core.eqdsk import GEqdsk

    r = np.linspace(1.0, 2.0, 9)
    z = np.linspace(-0.5, 0.5, 9)
    _rr, zz = np.meshgrid(r, z)
    psi = 0.25 * zz**2
    eq = GEqdsk(
        nw=9,
        nh=9,
        rdim=1.0,
        zdim=1.0,
        rleft=1.0,
        zmid=0.0,
        rmaxis=1.5,
        zmaxis=0.0,
        simag=0.0,
        sibry=float(np.max(psi)),
        psirz=psi,
        fpol=np.ones(9),
        pres=np.linspace(1.0, 0.0, 9),
        ffprime=np.full(9, -0.5 / (2.0 * np.pi)),
        pprime=np.zeros(9),
        qpsi=np.linspace(1.0, 3.0, 9),
        rbdry=np.array([1.25, 1.75, 1.75, 1.25]),
        zbdry=np.array([-0.25, -0.25, 0.25, 0.25]),
    )

    metrics = benchmark_sparc_geqdsk_rmse._geqdsk_contract_metrics(eq)

    assert metrics["geqdsk_contract_pass"] is True
    assert metrics["geqdsk_source_contract_pass"] is False
    assert metrics["gs_profile_source_best_fit_convention"] == "scaled_by_2pi"
    assert metrics["gs_profile_source_best_fit_scale"] == pytest.approx(2.0 * np.pi)
    assert metrics["gs_profile_source_best_fit_rel_l2"] < 1e-12


def test_run_benchmark_strict_source_contract_enforces_geqdsk_pde(monkeypatch) -> None:
    psi = np.eye(9, dtype=np.float64)

    monkeypatch.setattr(
        benchmark_sparc_geqdsk_rmse,
        "_load_sparc_geqdsk_cases",
        lambda: [
            {
                "name": "sparc_ref_case",
                "machine": "sparc",
                "source_file": "sparc_ref_case.geqdsk",
                "psi": psi,
                "feature_vector_full": np.array([-8.7, -12.2, 1.85, 0.0, 1.0, 1.0, 0.0, 2.37]),
                "Ip": -8.7,
                "B0": -12.2,
                "R0": 1.85,
                "kappa": 1.8,
                "geqdsk_contract": {
                    "geqdsk_contract_pass": True,
                    "geqdsk_source_contract_pass": False,
                    "psi_span": 1.0,
                },
            }
        ],
    )

    def _proxy_only(eq: dict[str, Any]) -> tuple[np.ndarray, str, str]:
        return np.asarray(eq["psi"], dtype=np.float64), "reduced_order_proxy", "unit-test"

    monkeypatch.setattr(benchmark_sparc_geqdsk_rmse, "_run_neural_surrogate", _proxy_only)
    relaxed = benchmark_sparc_geqdsk_rmse.run_benchmark(
        grid_sizes=[9],
        strict_source_contract=False,
    )
    strict = benchmark_sparc_geqdsk_rmse.run_benchmark(
        grid_sizes=[9],
        strict_source_contract=True,
    )

    assert relaxed["passes"] is True
    assert strict["passes"] is False
    assert strict["cases"][0]["geqdsk_source_contract_pass"] is False
