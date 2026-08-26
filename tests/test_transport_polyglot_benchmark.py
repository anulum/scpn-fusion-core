# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Transport Polyglot Benchmark Tests
"""Tests for the public Rust/PyO3 and NumPy transport comparison."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import numpy as np
import pytest

from benchmarks import bench_transport_polyglot as benchmark


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "validation" / "polyglot" / "performance_comparison.schema.json"
REPORT_PATH = ROOT / "validation" / "reports" / "transport_polyglot_comparison.json"


class _FakeTransportSolver:
    """NumPy-backed stand-in for deterministic report-shape tests."""

    def __init__(self) -> None:
        self.rho = np.zeros(0, dtype=np.float64)
        self.profile = np.zeros(0, dtype=np.float64)
        self.chi = np.zeros(0, dtype=np.float64)
        self.source = np.zeros(0, dtype=np.float64)
        self.dt = 0.0

    def build_profile(self) -> str:
        return "release"

    def set_transport_state(
        self,
        rho: np.ndarray,
        t_e_kev: np.ndarray,
        _t_i_kev: np.ndarray,
        _n_e_19: np.ndarray,
        _n_impurity: np.ndarray,
        chi: np.ndarray,
        dt: float,
    ) -> None:
        self.rho = np.asarray(rho, dtype=np.float64).copy()
        self.profile = np.asarray(t_e_kev, dtype=np.float64).copy()
        self.chi = np.asarray(chi, dtype=np.float64).copy()
        self.source = np.zeros_like(self.rho)
        self.dt = dt

    def evolve_profiles(self, p_aux_mw: float) -> None:
        assert p_aux_mw == 0.0
        self.profile = benchmark.crank_nicolson_step(
            self.profile,
            self.chi,
            self.source,
            self.rho,
            float(self.rho[1] - self.rho[0]),
            self.dt,
            T_edge=benchmark.EDGE_KEV,
            use_jax=False,
        )

    def electron_temperature_profile(self) -> np.ndarray:
        return self.profile.copy()


def _fake_module() -> SimpleNamespace:
    return SimpleNamespace(PyTransportSolver=_FakeTransportSolver)


def test_report_passes_public_schema_and_correctness_gate() -> None:
    """The shared schema binds timing disclosure to numerical parity."""
    report = benchmark.build_report(
        benchmark.BenchmarkConfig(
            nodes=129,
            steps=10,
            dt_s=1e-3,
            discarded_warmups=1,
            warm_samples=3,
        ),
        rust_module=_fake_module(),
    )
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    jsonschema.validate(report, schema)
    assert report["gate"] == {
        "passes": True,
        "correctness_passes": True,
        "release_build_verified": True,
    }
    assert report["correctness"]["maximum_profile_difference_kev"] == pytest.approx(0.0)
    assert report["disclosure"]["portable_performance_claim_admitted"] is False
    assert {row["backend"] for row in report["timing"]["rows"]} == {
        "numpy",
        "rust_pyo3",
    }
    assert all(len(row["warm_samples_s"]) == 3 for row in report["timing"]["rows"])


def test_markdown_uses_factual_local_machine_wording() -> None:
    """Public prose reports the observation without a portable guarantee."""
    report = benchmark.build_report(
        benchmark.BenchmarkConfig(
            nodes=17,
            steps=1,
            dt_s=1e-3,
            discarded_warmups=0,
            warm_samples=3,
        ),
        rust_module=_fake_module(),
    )

    rendered = benchmark.render_markdown(report)
    assert "On the disclosed local machine" in rendered
    assert "not a portable performance guarantee" in rendered
    assert "Maximum Rust/NumPy profile difference" in rendered
    assert "warm_samples_s" not in rendered


@pytest.mark.parametrize(
    "config, message",
    [
        (benchmark.BenchmarkConfig(nodes=2), "nodes"),
        (benchmark.BenchmarkConfig(steps=0), "steps"),
        (benchmark.BenchmarkConfig(dt_s=0.0), "dt_s"),
        (benchmark.BenchmarkConfig(discarded_warmups=-1), "discarded_warmups"),
        (benchmark.BenchmarkConfig(warm_samples=2), "warm_samples"),
    ],
)
def test_case_contract_rejects_invalid_controls(
    config: benchmark.BenchmarkConfig,
    message: str,
) -> None:
    """Invalid benchmark controls fail before timings are collected."""
    with pytest.raises(ValueError, match=message):
        benchmark.build_report(config, rust_module=_fake_module())


def test_cli_writes_json_and_markdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The CLI writes both public report formats from one measured payload."""
    monkeypatch.setattr(benchmark, "_load_rust_module", _fake_module)
    output_json = tmp_path / "transport.json"
    output_markdown = tmp_path / "transport.md"

    result = benchmark.main(
        [
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
            "--nodes",
            "129",
            "--steps",
            "10",
            "--warmups",
            "0",
            "--samples",
            "3",
        ]
    )

    assert result == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema"] == benchmark.SCHEMA
    assert payload["gate"]["passes"] is True
    assert output_markdown.read_text(encoding="utf-8").startswith(
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->"
    )


def test_tracked_report_and_public_contract_are_cross_wired() -> None:
    """Keep the checked-in result discoverable and schema-valid."""
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(report, schema)
    assert report["gate"]["passes"] is True
    assert report["disclosure"]["portable_performance_claim_admitted"] is False

    readme = (ROOT / "benchmarks" / "README.md").read_text(encoding="utf-8")
    contract = (ROOT / "benchmarks" / "POLYGLOT_PERFORMANCE_CONTRACT.md").read_text(
        encoding="utf-8"
    )
    assert benchmark.COMMAND in readme
    assert "Every promoted polyglot backend" in contract
    assert "transport_polyglot_comparison.json" in contract


def test_installed_binding_discloses_build_profile() -> None:
    """The real extension exposes its compile profile when installed."""
    try:
        rust_module = benchmark._load_rust_module()
    except RuntimeError:
        pytest.skip("scpn_fusion_rs is not installed")
    assert rust_module.PyTransportSolver().build_profile() in {"debug", "release"}
