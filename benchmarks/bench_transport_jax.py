#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — JAX Transport Performance Comparison
"""Compare JAX and NumPy cylindrical Crank–Nicolson transport."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from scipy.special import j0

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPORT_JSON = ROOT / "validation" / "reports" / "transport_jax_comparison.json"
REPORT_MD = ROOT / "validation" / "reports" / "transport_jax_comparison.md"
SCHEMA = "scpn-fusion-core.polyglot-performance-comparison.v1"
COMMAND = ".venv/bin/python benchmarks/bench_transport_jax.py"
MAX_PARITY_ERROR_KEV = 2.0e-14
MAX_ANALYTIC_RMSE_KEV = 2.2829306504541543e-6
MAX_GRADIENT_RELATIVE_ERROR = 0.01

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from benchmarks import bench_transport_polyglot as common  # noqa: E402
from scpn_fusion.core.jax_solvers import crank_nicolson_step  # noqa: E402
from scpn_fusion.core.jax_transport_solver import simulate_scenario_jax  # noqa: E402

FloatArray: TypeAlias = NDArray[np.float64]
JsonObject: TypeAlias = dict[str, Any]


def _run_jax(case: common.TransportCase, config: common.BenchmarkConfig) -> FloatArray:
    initial = jnp.asarray(case.initial, dtype=jnp.float64)
    diffusivity = jnp.asarray(case.diffusivity, dtype=jnp.float64)
    source_history = jnp.broadcast_to(
        jnp.asarray(case.source, dtype=jnp.float64),
        (config.steps, config.nodes),
    )
    rho = jnp.asarray(case.rho, dtype=jnp.float64)
    electron_history, _ = simulate_scenario_jax(
        initial,
        initial,
        diffusivity,
        diffusivity,
        source_history,
        source_history,
        rho,
        config.dt_s,
        t_edge_e=common.EDGE_KEV,
        t_edge_i=common.EDGE_KEV,
    )
    electron_history.block_until_ready()
    return np.asarray(electron_history[-1], dtype=np.float64)


def _correctness(
    numpy_profile: FloatArray,
    jax_profile: FloatArray,
    case: common.TransportCase,
    config: common.BenchmarkConfig,
) -> JsonObject:
    final_time = config.dt_s * config.steps
    exact = np.asarray(
        common.EDGE_KEV
        + common.AMPLITUDE_KEV
        * j0(common.J01 * case.rho)
        * math.exp(-common.DIFFUSIVITY_M2_S * common.J01 * common.J01 * final_time),
        dtype=np.float64,
    )
    parity_error = float(np.max(np.abs(jax_profile - numpy_profile)))
    analytic_rmse = float(np.sqrt(np.mean((jax_profile - exact) ** 2)))
    edge_exact = bool(jax_profile[-1] == common.EDGE_KEV)
    finite_positive = bool(np.all(np.isfinite(jax_profile)) and np.all(jax_profile > 0.0))
    return {
        "maximum_profile_difference_kev": parity_error,
        "maximum_profile_difference_kev_limit": MAX_PARITY_ERROR_KEV,
        "analytic_rmse_kev": analytic_rmse,
        "analytic_rmse_kev_limit": MAX_ANALYTIC_RMSE_KEV,
        "edge_exact": edge_exact,
        "finite_positive_profiles": finite_positive,
        "numpy_profile_sha256": common._sha256_bytes(np.ascontiguousarray(numpy_profile).tobytes()),
        "jax_profile_sha256": common._sha256_bytes(np.ascontiguousarray(jax_profile).tobytes()),
        "passes": bool(
            parity_error <= MAX_PARITY_ERROR_KEV
            and analytic_rmse <= MAX_ANALYTIC_RMSE_KEV
            and edge_exact
            and finite_positive
        ),
    }


def _gradient_check(
    case: common.TransportCase,
    config: common.BenchmarkConfig,
) -> JsonObject:
    source_shape = np.asarray(1.0 - case.rho**2, dtype=np.float64)
    rho_jax = jnp.asarray(case.rho, dtype=jnp.float64)
    initial_jax = jnp.asarray(case.initial, dtype=jnp.float64)
    diffusivity_jax = jnp.asarray(case.diffusivity, dtype=jnp.float64)
    source_shape_jax = jnp.asarray(source_shape, dtype=jnp.float64)

    def jax_cost(amplitude: jax.Array) -> jax.Array:
        source_history = jnp.broadcast_to(
            amplitude * source_shape_jax,
            (config.steps, config.nodes),
        )
        electron_history, _ = simulate_scenario_jax(
            initial_jax,
            initial_jax,
            diffusivity_jax,
            diffusivity_jax,
            source_history,
            source_history,
            rho_jax,
            config.dt_s,
            t_edge_e=common.EDGE_KEV,
            t_edge_i=common.EDGE_KEV,
        )
        return jnp.mean(electron_history[-1, :-1])

    def numpy_cost(amplitude: float) -> float:
        profile = case.initial.copy()
        source = amplitude * source_shape
        for _ in range(config.steps):
            profile = crank_nicolson_step(
                profile,
                case.diffusivity,
                source,
                case.rho,
                float(case.rho[1] - case.rho[0]),
                config.dt_s,
                T_edge=common.EDGE_KEV,
                use_jax=False,
            )
        return float(np.mean(profile[:-1]))

    amplitude = 0.2
    autodiff = float(jax.grad(jax_cost)(jnp.asarray(amplitude)).block_until_ready())
    epsilon = 1.0e-5
    finite_difference = (numpy_cost(amplitude + epsilon) - numpy_cost(amplitude - epsilon)) / (
        2.0 * epsilon
    )
    relative_error = abs(autodiff - finite_difference) / max(abs(finite_difference), 1.0e-15)
    return {
        "source_amplitude_kev_s": amplitude,
        "finite_difference_epsilon": epsilon,
        "autodiff": autodiff,
        "central_finite_difference": finite_difference,
        "relative_error": relative_error,
        "relative_error_limit": MAX_GRADIENT_RELATIVE_ERROR,
        "passes": bool(
            math.isfinite(autodiff)
            and math.isfinite(finite_difference)
            and relative_error <= MAX_GRADIENT_RELATIVE_ERROR
        ),
    }


def _jax_devices() -> list[JsonObject]:
    return [
        {
            "platform": str(device.platform),
            "device_kind": str(device.device_kind),
            "id": int(device.id),
        }
        for device in jax.devices()
    ]


def build_report(config: common.BenchmarkConfig | None = None) -> JsonObject:
    """Run the frozen JAX/NumPy comparison and return its complete report."""
    config = config or common.BenchmarkConfig()
    case = common._make_case(config)
    load_before = common._load_average()

    def run_numpy() -> FloatArray:
        return common._run_numpy(case, config)

    def run_jax() -> FloatArray:
        return _run_jax(case, config)

    numpy_cold_s, numpy_cold_profile = common._measure(run_numpy)
    jax_cold_s, jax_cold_profile = common._measure(run_jax)
    correctness = _correctness(numpy_cold_profile, jax_cold_profile, case, config)

    for index in range(config.discarded_warmups):
        if index % 2 == 0:
            run_numpy()
            run_jax()
        else:
            run_jax()
            run_numpy()

    numpy_samples: list[float] = []
    jax_samples: list[float] = []
    final_numpy = numpy_cold_profile
    final_jax = jax_cold_profile
    for index in range(config.warm_samples):
        if index % 2 == 0:
            numpy_s, final_numpy = common._measure(run_numpy)
            jax_s, final_jax = common._measure(run_jax)
        else:
            jax_s, final_jax = common._measure(run_jax)
            numpy_s, final_numpy = common._measure(run_numpy)
        numpy_samples.append(numpy_s)
        jax_samples.append(jax_s)

    if _correctness(final_numpy, final_jax, case, config) != correctness:
        raise RuntimeError("cold and warm correctness projections differ")

    numpy_row = common._timing_row(
        backend="numpy",
        language="Python",
        implementation="scpn_fusion.core.jax_solvers.crank_nicolson_step(use_jax=False)",
        build_profile="CPython/NumPy",
        cold_sample_s=numpy_cold_s,
        warm_samples_s=numpy_samples,
    )
    jax_row = common._timing_row(
        backend="jax",
        language="JAX/XLA via Python",
        implementation="scpn_fusion.core.jax_transport_solver.simulate_scenario_jax",
        build_profile=f"JAX float64/{jax.default_backend()}",
        cold_sample_s=jax_cold_s,
        warm_samples_s=jax_samples,
    )
    ratio = float(numpy_row["warm_median_s"] / jax_row["warm_median_s"])
    gradient = _gradient_check(case, config)
    candidate_verified = bool(cast(Any, jax.config).read("jax_enable_x64"))
    correctness_passes = bool(correctness["passes"] and gradient["passes"])
    report: JsonObject = {
        "schema": SCHEMA,
        "benchmark": "cylindrical_transport_jax_vs_numpy",
        "generated_at": datetime.now().astimezone().isoformat(),
        "command": COMMAND,
        "case": common._case_payload(config),
        "correctness": {**correctness, "source_gradient": gradient},
        "timing": {
            "paired_order": "alternating NumPy-JAX and JAX-NumPy pairs",
            "discarded_warmups": config.discarded_warmups,
            "rows": [numpy_row, jax_row],
            "numpy_over_candidate_ratio": ratio,
        },
        "environment": {
            "cpu_model": common._cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "affinity_cpus": common._affinity(),
            "governors": common._governors(),
            "memory_total_bytes": common._memory_total_bytes(),
            "process_count": common._process_count(),
            "load_average_before": load_before,
            "load_average_after": common._load_average(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": common._package_version("scipy"),
            "jax": str(jax.__version__),
            "jaxlib": common._package_version("jaxlib"),
            "jax_enable_x64": bool(cast(Any, jax.config).read("jax_enable_x64")),
            "jax_default_backend": str(jax.default_backend()),
            "jax_devices": _jax_devices(),
            "thread_environment": common._thread_environment(),
            "source_sha256": {
                "benchmark": common._sha256_file(Path(__file__)),
                "benchmark_common": common._sha256_file(
                    ROOT / "benchmarks" / "bench_transport_polyglot.py"
                ),
                "jax_transport": common._sha256_file(
                    ROOT / "src" / "scpn_fusion" / "core" / "jax_transport_solver.py"
                ),
                "canonical_operator": common._sha256_file(
                    ROOT / "src" / "scpn_fusion" / "core" / "jax_solvers.py"
                ),
            },
        },
        "disclosure": {
            "classification": "local_machine_observation",
            "portable_performance_claim_admitted": False,
            "interpretation": (
                "The ratio is the observed result on the disclosed local machine. "
                "It is not a portable performance guarantee."
            ),
            "timing_includes": [
                "backend array construction",
                "host-to-device transfer for JAX",
                f"one compiled public rollout containing {config.steps} solver steps",
                "device synchronization",
                "final device-to-host transfer/readback",
            ],
            "timing_excludes": ["module import"],
        },
        "gate": {
            "passes": bool(correctness_passes and candidate_verified),
            "correctness_passes": correctness_passes,
            "candidate_build_verified": candidate_verified,
        },
    }
    validate_report(report)
    return report


def validate_report(report: JsonObject) -> None:
    """Fail closed on incomplete or misleading JAX comparison reports."""
    if report.get("schema") != SCHEMA:
        raise ValueError("unexpected polyglot report schema")
    timing = cast(JsonObject, report.get("timing"))
    rows = cast(list[JsonObject], timing.get("rows"))
    if len(rows) != 2 or {str(row.get("backend")) for row in rows} != {"numpy", "jax"}:
        raise ValueError("report must contain exactly the NumPy and JAX rows")
    for row in rows:
        samples = cast(list[float], row.get("warm_samples_s"))
        if len(samples) < 3 or any(not math.isfinite(value) or value < 0.0 for value in samples):
            raise ValueError("timing rows require at least three finite non-negative samples")
    ratio = float(timing.get("numpy_over_candidate_ratio", 0.0))
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("timing ratio must be finite and positive")
    disclosure = cast(JsonObject, report.get("disclosure"))
    if disclosure.get("portable_performance_claim_admitted") is not False:
        raise ValueError("local polyglot reports cannot admit a portable performance claim")


def render_markdown(report: JsonObject) -> str:
    """Render the factual public JAX/NumPy comparison."""
    case = cast(JsonObject, report["case"])
    correctness = cast(JsonObject, report["correctness"])
    gradient = cast(JsonObject, correctness["source_gradient"])
    timing = cast(JsonObject, report["timing"])
    rows = cast(list[JsonObject], timing["rows"])
    environment = cast(JsonObject, report["environment"])
    disclosure = cast(JsonObject, report["disclosure"])
    row_by_backend = {str(row["backend"]): row for row in rows}
    numpy_row = row_by_backend["numpy"]
    jax_row = row_by_backend["jax"]
    affinity = cast(list[int], environment["affinity_cpus"])
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "<!-- Commercial license available -->",
        "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
        "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
        "<!-- ORCID: 0009-0009-3560-0851 -->",
        "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
        "<!-- SCPN Fusion Core — JAX Transport Comparison -->",
        "",
        "# JAX and NumPy cylindrical transport comparison",
        "",
        "## Outcome",
        "",
        (
            "On the disclosed local machine, the NumPy/JAX median wall-time ratio was "
            f"{float(timing['numpy_over_candidate_ratio']):.4f} for this case. The final "
            "profiles differ by "
            f"{float(correctness['maximum_profile_difference_kev']):.6e} keV."
        ),
        (
            "For this workload, the observed JAX median was "
            f"{1.0 / float(timing['numpy_over_candidate_ratio']):.4f} times the NumPy median, "
            "so automatic dispatch retains NumPy."
        ),
        "",
        str(disclosure["interpretation"]),
        "",
        "## Side-by-side timing",
        "",
        "| Backend | Language | Build profile | Cold (s) | Warm P05 (s) | Warm median (s) | Warm P95 (s) | Samples |",
        "|---|---|---|---:|---:|---:|---:|---:|",
        (
            f"| NumPy | Python | {numpy_row['build_profile']} | "
            f"{float(numpy_row['cold_sample_s']):.9f} | {float(numpy_row['warm_p05_s']):.9f} | "
            f"{float(numpy_row['warm_median_s']):.9f} | {float(numpy_row['warm_p95_s']):.9f} | "
            f"{len(cast(list[float], numpy_row['warm_samples_s']))} |"
        ),
        (
            f"| JAX | JAX/XLA via Python | {jax_row['build_profile']} | "
            f"{float(jax_row['cold_sample_s']):.9f} | {float(jax_row['warm_p05_s']):.9f} | "
            f"{float(jax_row['warm_median_s']):.9f} | {float(jax_row['warm_p95_s']):.9f} | "
            f"{len(cast(list[float], jax_row['warm_samples_s']))} |"
        ),
        "",
        "## Numerical checks",
        "",
        "| Check | Result | Limit |",
        "|---|---:|---:|",
        (
            "| Maximum JAX/NumPy profile difference | "
            f"{float(correctness['maximum_profile_difference_kev']):.6e} keV | "
            f"{float(correctness['maximum_profile_difference_kev_limit']):.6e} keV |"
        ),
        (
            f"| Analytic Bessel RMSE | {float(correctness['analytic_rmse_kev']):.6e} keV | "
            f"{float(correctness['analytic_rmse_kev_limit']):.6e} keV |"
        ),
        (
            f"| Source-gradient relative error | {float(gradient['relative_error']):.6e} | "
            f"{float(gradient['relative_error_limit']):.6e} |"
        ),
        f"| Exact outer edge | {correctness['edge_exact']} | `true` |",
        f"| Finite positive profile | {correctness['finite_positive_profiles']} | `true` |",
        "",
        "## Timed scope",
        "",
        f"- Grid: {case['nodes']} radial nodes, float64.",
        f"- Evolution: {case['steps']} steps at dt={float(case['dt_s']):.6g} s.",
        "- Included: array construction, transfers, solver calls, JAX synchronization, and readback.",
        "- Excluded: module import.",
        f"- Pair order: {timing['paired_order']}.",
        f"- Discarded paired warmups: {timing['discarded_warmups']}.",
        "",
        "## Environment",
        "",
        f"- CPU: {environment['cpu_model']}",
        f"- Logical CPUs: {environment['logical_cpu_count']}",
        f"- Affinity: {len(affinity)} CPUs (`{affinity}`)",
        f"- Governors: `{environment['governors']}`",
        f"- Process count: {environment['process_count']}",
        f"- Load average before: `{environment['load_average_before']}`",
        f"- Load average after: `{environment['load_average_after']}`",
        f"- Platform: {environment['platform']}",
        f"- Python / NumPy / SciPy: {environment['python']} / {environment['numpy']} / {environment['scipy']}",
        f"- JAX / jaxlib: {environment['jax']} / {environment['jaxlib']}",
        f"- JAX backend and devices: {environment['jax_default_backend']} / `{environment['jax_devices']}`",
        f"- Thread environment: `{environment['thread_environment']}`",
        "",
        "## Reproduce",
        "",
        "```bash",
        COMMAND,
        "```",
        "",
        "The JSON companion retains every raw warm sample, gradient values, and source hashes.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public comparison and write JSON plus Markdown reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=REPORT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=REPORT_MD)
    parser.add_argument("--nodes", type=int, default=129)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=31)
    args = parser.parse_args(argv)
    report = build_report(
        common.BenchmarkConfig(
            nodes=int(args.nodes),
            steps=int(args.steps),
            dt_s=float(args.dt),
            discarded_warmups=int(args.warmups),
            warm_samples=int(args.samples),
        )
    )
    output_json = Path(args.output_json)
    output_markdown = Path(args.output_markdown)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(output_json)
    print(output_markdown)
    return 0 if bool(cast(JsonObject, report["gate"])["passes"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
