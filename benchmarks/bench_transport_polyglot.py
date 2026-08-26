#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Transport Polyglot Performance Comparison
"""Compare release Rust/PyO3 and canonical NumPy cylindrical transport."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from scipy.special import j0

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPORT_JSON = ROOT / "validation" / "reports" / "transport_polyglot_comparison.json"
REPORT_MD = ROOT / "validation" / "reports" / "transport_polyglot_comparison.md"
SCHEMA = "scpn-fusion-core.polyglot-performance-comparison.v1"
COMMAND = ".venv/bin/python benchmarks/bench_transport_polyglot.py"
J01 = 2.4048255576957728
EDGE_KEV = 0.1
AMPLITUDE_KEV = 0.9
DIFFUSIVITY_M2_S = 1.0
MAX_PARITY_ERROR_KEV = 2e-14
MAX_ANALYTIC_RMSE_KEV = 2.2829306504541543e-6

sys.path.insert(0, str(SRC))

from scpn_fusion.core.jax_solvers import crank_nicolson_step  # noqa: E402

FloatArray: TypeAlias = NDArray[np.float64]
JsonObject: TypeAlias = dict[str, Any]


class RustTransportSolver(Protocol):
    """Typed subset of the PyO3 transport surface used by this benchmark."""

    def build_profile(self) -> str: ...

    def set_transport_state(
        self,
        rho: FloatArray,
        t_e_kev: FloatArray,
        t_i_kev: FloatArray,
        n_e_19: FloatArray,
        n_impurity: FloatArray,
        chi: FloatArray,
        dt: float,
    ) -> None: ...

    def evolve_profiles(self, p_aux_mw: float) -> None: ...

    def electron_temperature_profile(self) -> FloatArray: ...


class RustTransportModule(Protocol):
    """Typed constructor exposed by the native extension."""

    PyTransportSolver: type[RustTransportSolver]


@dataclass(frozen=True)
class BenchmarkConfig:
    """Frozen public comparison controls."""

    nodes: int = 129
    steps: int = 10
    dt_s: float = 1e-3
    discarded_warmups: int = 10
    warm_samples: int = 31


@dataclass(frozen=True)
class TransportCase:
    """Shared arrays for both public backends."""

    rho: FloatArray
    initial: FloatArray
    density: FloatArray
    impurity: FloatArray
    diffusivity: FloatArray
    source: FloatArray


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: JsonObject) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256_bytes(encoded.encode("utf-8"))


def _load_rust_module() -> RustTransportModule:
    try:
        module = importlib.import_module("scpn_fusion_rs")
    except ImportError as exc:
        raise RuntimeError(
            "scpn_fusion_rs is unavailable; build it with maturin develop --release"
        ) from exc
    if not hasattr(module, "PyTransportSolver"):
        raise RuntimeError("scpn_fusion_rs does not expose PyTransportSolver")
    return cast(RustTransportModule, module)


def _make_case(config: BenchmarkConfig) -> TransportCase:
    if config.nodes < 3:
        raise ValueError("nodes must be >= 3")
    if config.steps < 1:
        raise ValueError("steps must be >= 1")
    if not math.isfinite(config.dt_s) or config.dt_s <= 0.0:
        raise ValueError("dt_s must be finite and > 0")
    if config.discarded_warmups < 0:
        raise ValueError("discarded_warmups must be >= 0")
    if config.warm_samples < 3:
        raise ValueError("warm_samples must be >= 3")

    rho = np.linspace(0.0, 1.0, config.nodes, dtype=np.float64)
    initial = np.asarray(EDGE_KEV + AMPLITUDE_KEV * j0(J01 * rho), dtype=np.float64)
    return TransportCase(
        rho=rho,
        initial=initial,
        density=np.full(config.nodes, 10.0, dtype=np.float64),
        impurity=np.zeros(config.nodes, dtype=np.float64),
        diffusivity=np.full(config.nodes, DIFFUSIVITY_M2_S, dtype=np.float64),
        source=np.zeros(config.nodes, dtype=np.float64),
    )


def _run_numpy(case: TransportCase, config: BenchmarkConfig) -> FloatArray:
    profile = case.initial.copy()
    drho = float(case.rho[1] - case.rho[0])
    for _ in range(config.steps):
        profile = crank_nicolson_step(
            profile,
            case.diffusivity,
            case.source,
            case.rho,
            drho,
            config.dt_s,
            T_edge=EDGE_KEV,
            use_jax=False,
        )
    return np.asarray(profile, dtype=np.float64)


def _run_rust(
    rust_module: RustTransportModule,
    case: TransportCase,
    config: BenchmarkConfig,
) -> tuple[FloatArray, str]:
    solver = rust_module.PyTransportSolver()
    build_profile = solver.build_profile()
    solver.set_transport_state(
        case.rho,
        case.initial,
        case.initial,
        case.density,
        case.impurity,
        case.diffusivity,
        config.dt_s,
    )
    for _ in range(config.steps):
        solver.evolve_profiles(0.0)
    result = np.asarray(solver.electron_temperature_profile(), dtype=np.float64).copy()
    return result, build_profile


def _measure(run: Callable[[], FloatArray]) -> tuple[float, FloatArray]:
    start_ns = time.perf_counter_ns()
    result = run()
    elapsed_s = (time.perf_counter_ns() - start_ns) * 1e-9
    if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
        raise RuntimeError("benchmark clock produced an invalid duration")
    return elapsed_s, result


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must lie in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timing_row(
    *,
    backend: str,
    language: str,
    implementation: str,
    build_profile: str,
    cold_sample_s: float,
    warm_samples_s: Sequence[float],
) -> JsonObject:
    samples = [float(value) for value in warm_samples_s]
    return {
        "backend": backend,
        "language": language,
        "implementation": implementation,
        "build_profile": build_profile,
        "cold_sample_s": float(cold_sample_s),
        "warm_samples_s": samples,
        "warm_median_s": float(statistics.median(samples)),
        "warm_p05_s": _percentile(samples, 0.05),
        "warm_p95_s": _percentile(samples, 0.95),
    }


def _load_average() -> list[float]:
    try:
        return [float(value) for value in os.getloadavg()]
    except OSError:
        return []


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _governors() -> JsonObject:
    counts: dict[str, int] = {}
    paths = Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor")
    for path in paths:
        try:
            governor = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        counts[governor] = counts.get(governor, 0) + 1
    return dict(sorted(counts.items()))


def _process_count() -> int | None:
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    try:
        return sum(1 for entry in proc.iterdir() if entry.name.isdigit())
    except OSError:
        return None


def _memory_total_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) * 1024
    return None


def _tool_version(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(command),
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"
    return (completed.stdout or completed.stderr).strip()


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _affinity() -> list[int]:
    try:
        return sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    except AttributeError:
        return []


def _thread_environment() -> dict[str, str]:
    names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
    )
    return {name: os.environ[name] for name in names if name in os.environ}


def _case_payload(config: BenchmarkConfig) -> JsonObject:
    payload: JsonObject = {
        "nodes": config.nodes,
        "steps": config.steps,
        "dt_s": config.dt_s,
        "precision": "float64",
        "diffusivity_m2_s": DIFFUSIVITY_M2_S,
        "sources": "zero",
        "edge_temperature_kev": EDGE_KEV,
        "initial_condition": f"0.1 + 0.9*J0({J01}*rho)",
        "timed_scope": (
            f"construct backend state, transfer inputs, execute {config.steps} public steps, "
            "and transfer/read the final profile"
        ),
    }
    payload["case_sha256"] = _canonical_sha256(payload)
    return payload


def _correctness(
    numpy_profile: FloatArray,
    rust_profile: FloatArray,
    case: TransportCase,
    config: BenchmarkConfig,
) -> JsonObject:
    final_time = config.dt_s * config.steps
    exact = np.asarray(
        EDGE_KEV
        + AMPLITUDE_KEV * j0(J01 * case.rho) * math.exp(-DIFFUSIVITY_M2_S * J01 * J01 * final_time),
        dtype=np.float64,
    )
    parity_error = float(np.max(np.abs(rust_profile - numpy_profile)))
    analytic_rmse = float(np.sqrt(np.mean((rust_profile - exact) ** 2)))
    edge_exact = bool(rust_profile[-1] == EDGE_KEV and numpy_profile[-1] == EDGE_KEV)
    finite_positive = bool(
        np.all(np.isfinite(rust_profile))
        and np.all(np.isfinite(numpy_profile))
        and np.all(rust_profile > 0.0)
        and np.all(numpy_profile > 0.0)
    )
    passes = (
        parity_error <= MAX_PARITY_ERROR_KEV
        and analytic_rmse <= MAX_ANALYTIC_RMSE_KEV
        and edge_exact
        and finite_positive
    )
    return {
        "maximum_profile_difference_kev": parity_error,
        "maximum_profile_difference_kev_limit": MAX_PARITY_ERROR_KEV,
        "analytic_rmse_kev": analytic_rmse,
        "analytic_rmse_kev_limit": MAX_ANALYTIC_RMSE_KEV,
        "edge_exact": edge_exact,
        "finite_positive_profiles": finite_positive,
        "numpy_profile_sha256": _sha256_bytes(np.ascontiguousarray(numpy_profile).tobytes()),
        "rust_profile_sha256": _sha256_bytes(np.ascontiguousarray(rust_profile).tobytes()),
        "passes": passes,
    }


def build_report(
    config: BenchmarkConfig | None = None,
    *,
    rust_module: RustTransportModule | None = None,
) -> JsonObject:
    """Run the paired benchmark and return its complete report."""
    config = config or BenchmarkConfig()
    case = _make_case(config)
    module = rust_module or _load_rust_module()
    probe_solver = module.PyTransportSolver()
    rust_build_profile = probe_solver.build_profile()
    load_before = _load_average()

    def run_numpy() -> FloatArray:
        return _run_numpy(case, config)

    def run_rust() -> FloatArray:
        profile, _ = _run_rust(module, case, config)
        return profile

    numpy_cold_s, numpy_cold_profile = _measure(run_numpy)
    rust_cold_s, rust_cold_profile = _measure(run_rust)
    correctness = _correctness(numpy_cold_profile, rust_cold_profile, case, config)

    for index in range(config.discarded_warmups):
        if index % 2 == 0:
            run_numpy()
            run_rust()
        else:
            run_rust()
            run_numpy()

    numpy_samples: list[float] = []
    rust_samples: list[float] = []
    last_numpy = numpy_cold_profile
    last_rust = rust_cold_profile
    for index in range(config.warm_samples):
        if index % 2 == 0:
            numpy_s, last_numpy = _measure(run_numpy)
            rust_s, last_rust = _measure(run_rust)
        else:
            rust_s, last_rust = _measure(run_rust)
            numpy_s, last_numpy = _measure(run_numpy)
        numpy_samples.append(numpy_s)
        rust_samples.append(rust_s)

    final_correctness = _correctness(last_numpy, last_rust, case, config)
    if final_correctness != correctness:
        raise RuntimeError("cold and warm correctness projections differ")

    numpy_row = _timing_row(
        backend="numpy",
        language="Python",
        implementation="scpn_fusion.core.jax_solvers.crank_nicolson_step(use_jax=False)",
        build_profile="CPython/NumPy",
        cold_sample_s=numpy_cold_s,
        warm_samples_s=numpy_samples,
    )
    rust_row = _timing_row(
        backend="rust_pyo3",
        language="Rust via PyO3",
        implementation="scpn_fusion_rs.PyTransportSolver.evolve_profiles",
        build_profile=rust_build_profile,
        cold_sample_s=rust_cold_s,
        warm_samples_s=rust_samples,
    )
    ratio = float(numpy_row["warm_median_s"] / rust_row["warm_median_s"])
    release_verified = rust_build_profile == "release"
    correctness_passes = bool(correctness["passes"])

    report: JsonObject = {
        "schema": SCHEMA,
        "benchmark": "cylindrical_transport_rust_pyo3_vs_numpy",
        "generated_at": datetime.now().astimezone().isoformat(),
        "command": COMMAND,
        "case": _case_payload(config),
        "correctness": correctness,
        "timing": {
            "paired_order": "alternating NumPy-Rust and Rust-NumPy pairs",
            "discarded_warmups": config.discarded_warmups,
            "rows": [numpy_row, rust_row],
            "numpy_over_rust_ratio": ratio,
        },
        "environment": {
            "cpu_model": _cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "affinity_cpus": _affinity(),
            "governors": _governors(),
            "memory_total_bytes": _memory_total_bytes(),
            "process_count": _process_count(),
            "load_average_before": load_before,
            "load_average_after": _load_average(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": _package_version("scipy"),
            "scpn_fusion_rs": _package_version("scpn-fusion-rs"),
            "rustc": _tool_version(["rustc", "--version"]),
            "cargo": _tool_version(["cargo", "--version"]),
            "thread_environment": _thread_environment(),
            "source_sha256": {
                "rust_transport": _sha256_file(
                    ROOT / "scpn-fusion-rs" / "crates" / "fusion-core" / "src" / "transport.rs"
                ),
                "pyo3_binding": _sha256_file(
                    ROOT
                    / "scpn-fusion-rs"
                    / "crates"
                    / "fusion-python"
                    / "src"
                    / "bindings"
                    / "transport.rs"
                ),
                "numpy_reference": _sha256_file(
                    ROOT / "src" / "scpn_fusion" / "core" / "jax_solvers.py"
                ),
                "benchmark": _sha256_file(Path(__file__)),
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
                "backend construction",
                "input state transfer",
                "ten public solver calls",
                "final result transfer/readback",
            ],
            "timing_excludes": ["module import", "extension compilation"],
        },
        "gate": {
            "passes": correctness_passes and release_verified,
            "correctness_passes": correctness_passes,
            "release_build_verified": release_verified,
        },
    }
    validate_report(report)
    return report


def validate_report(report: JsonObject) -> None:
    """Fail closed on incomplete, non-finite, or misleading reports."""
    if report.get("schema") != SCHEMA:
        raise ValueError("unexpected polyglot report schema")
    timing = cast(JsonObject, report.get("timing"))
    rows = cast(list[JsonObject], timing.get("rows"))
    if len(rows) != 2 or {str(row.get("backend")) for row in rows} != {"numpy", "rust_pyo3"}:
        raise ValueError("report must contain exactly the NumPy and Rust/PyO3 rows")
    for row in rows:
        samples = cast(list[float], row.get("warm_samples_s"))
        if len(samples) < 3 or any(not math.isfinite(value) or value < 0.0 for value in samples):
            raise ValueError("timing rows require at least three finite non-negative samples")
    ratio = float(timing.get("numpy_over_rust_ratio", 0.0))
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("timing ratio must be finite and positive")
    disclosure = cast(JsonObject, report.get("disclosure"))
    if disclosure.get("portable_performance_claim_admitted") is not False:
        raise ValueError("local polyglot reports cannot admit a portable performance claim")


def render_markdown(report: JsonObject) -> str:
    """Render the public factual side-by-side report."""
    case = cast(JsonObject, report["case"])
    correctness = cast(JsonObject, report["correctness"])
    timing = cast(JsonObject, report["timing"])
    rows = cast(list[JsonObject], timing["rows"])
    environment = cast(JsonObject, report["environment"])
    disclosure = cast(JsonObject, report["disclosure"])
    row_by_backend = {str(row["backend"]): row for row in rows}
    numpy_row = row_by_backend["numpy"]
    rust_row = row_by_backend["rust_pyo3"]
    affinity = cast(list[int], environment["affinity_cpus"])
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "<!-- Commercial license available -->",
        "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
        "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
        "<!-- ORCID: 0009-0009-3560-0851 -->",
        "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
        "<!-- SCPN Fusion Core — Transport Polyglot Comparison -->",
        "",
        "# Rust/PyO3 and NumPy cylindrical transport comparison",
        "",
        "## Outcome",
        "",
        (
            "On the disclosed local machine, the NumPy/Rust median wall-time ratio was "
            f"{float(timing['numpy_over_rust_ratio']):.4f} for this case. The final "
            "profiles differ by "
            f"{float(correctness['maximum_profile_difference_kev']):.6e} keV."
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
            f"| Rust/PyO3 | Rust via Python | {rust_row['build_profile']} | "
            f"{float(rust_row['cold_sample_s']):.9f} | {float(rust_row['warm_p05_s']):.9f} | "
            f"{float(rust_row['warm_median_s']):.9f} | {float(rust_row['warm_p95_s']):.9f} | "
            f"{len(cast(list[float], rust_row['warm_samples_s']))} |"
        ),
        "",
        "## Numerical checks",
        "",
        "| Check | Result | Limit |",
        "|---|---:|---:|",
        (
            "| Maximum Rust/NumPy profile difference | "
            f"{float(correctness['maximum_profile_difference_kev']):.6e} keV | "
            f"{float(correctness['maximum_profile_difference_kev_limit']):.6e} keV |"
        ),
        (
            f"| Analytic Bessel RMSE | {float(correctness['analytic_rmse_kev']):.6e} keV | "
            f"{float(correctness['analytic_rmse_kev_limit']):.6e} keV |"
        ),
        f"| Exact outer edge | {correctness['edge_exact']} | `true` |",
        f"| Finite positive profiles | {correctness['finite_positive_profiles']} | `true` |",
        "",
        "## Timed scope",
        "",
        f"- Grid: {case['nodes']} radial nodes, float64.",
        f"- Evolution: {case['steps']} steps at dt={float(case['dt_s']):.6g} s.",
        f"- Included: {case['timed_scope']}.",
        "- Excluded: extension compilation and module import.",
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
        f"- Rust: {environment['rustc']}",
        f"- Thread environment: `{environment['thread_environment']}`",
        "",
        "## Reproduce",
        "",
        "```bash",
        'VIRTUAL_ENV="$PWD/.venv" maturin develop --release \\',
        "  --manifest-path scpn-fusion-rs/crates/fusion-python/Cargo.toml",
        COMMAND,
        "```",
        "",
        "The JSON companion retains every raw warm sample and the source hashes.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public comparison and write JSON plus Markdown reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=REPORT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=REPORT_MD)
    parser.add_argument("--nodes", type=int, default=129)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=31)
    args = parser.parse_args(argv)

    report = build_report(
        BenchmarkConfig(
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
