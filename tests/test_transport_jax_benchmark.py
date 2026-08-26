# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — JAX Transport Benchmark Tests

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped]
import pytest

from benchmarks import bench_transport_jax as benchmark
from benchmarks import bench_transport_polyglot as common

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "validation" / "polyglot" / "performance_comparison.schema.json"
REPORT_PATH = ROOT / "validation" / "reports" / "transport_jax_comparison.json"


@pytest.fixture(scope="module")
def measured_report() -> dict[str, Any]:
    """Run a short-sample real JAX/NumPy cohort on the frozen science case."""
    return benchmark.build_report(
        common.BenchmarkConfig(
            nodes=129,
            steps=10,
            dt_s=1.0e-3,
            discarded_warmups=0,
            warm_samples=3,
        )
    )


def test_real_report_passes_schema_accuracy_and_gradient_gates(
    measured_report: dict[str, Any],
) -> None:
    """The actual JAX runtime passes the shared public evidence contract."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(measured_report, schema)

    assert measured_report["gate"] == {
        "passes": True,
        "correctness_passes": True,
        "candidate_build_verified": True,
    }
    assert measured_report["correctness"]["maximum_profile_difference_kev"] <= 2.0e-14
    assert measured_report["correctness"]["source_gradient"]["relative_error"] <= 0.01
    assert measured_report["disclosure"]["portable_performance_claim_admitted"] is False
    assert {row["backend"] for row in measured_report["timing"]["rows"]} == {
        "numpy",
        "jax",
    }


def test_markdown_reports_local_observation_without_speed_promotion(
    measured_report: dict[str, Any],
) -> None:
    """Public prose states the measured ordering without a superiority label."""
    rendered = benchmark.render_markdown(measured_report)
    assert "On the disclosed local machine" in rendered
    assert "not a portable performance guarantee" in rendered
    assert "Maximum JAX/NumPy profile difference" in rendered
    assert "Source-gradient relative error" in rendered
    assert "so automatic dispatch retains NumPy" in rendered
    assert "faster" not in rendered.lower()
    assert "superior" not in rendered.lower()


def test_cli_writes_real_schema_valid_reports(tmp_path: Path) -> None:
    """The command-line surface executes both backends and writes both formats."""
    output_json = tmp_path / "jax_transport.json"
    output_markdown = tmp_path / "jax_transport.md"
    result = benchmark.main(
        [
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
            "--warmups",
            "0",
            "--samples",
            "3",
        ]
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(payload, schema)
    assert result == 0
    assert payload["gate"]["passes"] is True
    assert output_markdown.read_text(encoding="utf-8").startswith(
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->"
    )


def test_tracked_report_is_reproducible_and_discoverable() -> None:
    """The checked-in report binds its sources and public entry points."""
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(report, schema)
    assert report["gate"]["passes"] is True

    source_paths = {
        "benchmark": ROOT / "benchmarks" / "bench_transport_jax.py",
        "benchmark_common": ROOT / "benchmarks" / "bench_transport_polyglot.py",
        "jax_transport": ROOT / "src" / "scpn_fusion" / "core" / "jax_transport_solver.py",
        "canonical_operator": ROOT / "src" / "scpn_fusion" / "core" / "jax_solvers.py",
    }
    expected_hashes = report["environment"]["source_sha256"]
    for name, path in source_paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hashes[name]

    readme = (ROOT / "benchmarks" / "README.md").read_text(encoding="utf-8")
    contract = (ROOT / "benchmarks" / "POLYGLOT_PERFORMANCE_CONTRACT.md").read_text(
        encoding="utf-8"
    )
    assert benchmark.COMMAND in readme
    assert "transport_jax_comparison.json" in contract
