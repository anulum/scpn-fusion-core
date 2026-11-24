# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Coupled TORAX Parity Tests
"""Production-boundary tests for coupled TORAX/native evidence."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from validation import benchmark_torax_coupled_transport_parity as benchmark
from validation.reference_data.torax.coupled_transport_model_intersection_deck import (
    MODEL_INTERSECTION,
)


def _payload() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(benchmark.REFERENCE_PATH.read_text(encoding="utf-8")))


def _pin_native_wall_times(
    monkeypatch: pytest.MonkeyPatch, *, cold_s: float, warm_s: float
) -> None:
    """Run every native case for real, but report fixed wall times for it.

    The warm-cost gate divides the TORAX warm time recorded on the reference host by
    the native warm time measured in this process. On a runner much faster or slower
    than that host, the ratio crosses its limit with no change in the code. A test
    that asserts the gate's decision therefore fixes the native times. The first
    call is the cold run; every later call reports ``warm_s``.
    """
    run_native = benchmark._run_native
    calls = 0

    def pinned(reference_case: Any, *, dt_s: float) -> tuple[dict[str, Any], float]:
        """Return the real native case with the pinned wall time for this call."""
        nonlocal calls
        case, _measured = run_native(reference_case, dt_s=dt_s)
        calls += 1
        return case, cold_s if calls == 1 else warm_s

    monkeypatch.setattr(benchmark, "_run_native", pinned)


def _tracked_native_runtime() -> dict[str, float]:
    """Native wall times recorded with the tracked report on the reference host."""
    report = json.loads(benchmark.REPORT_JSON.read_text(encoding="utf-8"))
    runtime = report["runtime_seconds"]
    return {"cold": float(runtime["native_cold"]), "warm": float(runtime["native_warm"])}


def test_reference_authenticates_pinned_deck_and_deterministic_replay() -> None:
    reference = benchmark.load_reference()
    assert reference["provenance"]["torax_version"] == "1.4.3"
    assert reference["determinism"]["byte_identical_scientific_projection"] is True
    assert reference["primary"]["sim_error"] == "SimError.NO_ERROR"
    assert reference["refined"]["sim_error"] == "SimError.NO_ERROR"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("schema", "wrong"), "unexpected schema"),
        (("provenance.torax_version", "0.0.0"), "pinned TORAX 1.4.3"),
        (("provenance.deck_sha256", "0" * 64), "deck source digest is stale"),
        (
            ("determinism.byte_identical_scientific_projection", False),
            "lacks deterministic replay",
        ),
        (("primary.sim_error", "SimError.NUMERICAL_ERROR"), "did not complete cleanly"),
        (("runtime_samples_seconds", "x"), "lacks warm runtime samples"),
        (("runtime_samples_seconds.warm_primary", "abc"), "must be a sequence of wall times"),
        (("runtime_samples_seconds.warm_primary", [0.1, 0.2]), "needs at least 3 positive"),
        (("runtime_samples_seconds.warm_primary", [0.1, 0.2, 0.0]), "needs at least 3 positive"),
        (("runtime_seconds.warm_primary", 1.0), "not the minimum of its samples"),
        (("determinism.warm_projection_sha256s", "abc"), "do not reproduce the primary"),
        (("determinism.warm_projection_sha256s", ["0" * 64] * 5), "do not reproduce the primary"),
    ],
)
def test_reference_rejects_mutated_custody(
    tmp_path: Path,
    mutation: tuple[str, object],
    message: str,
) -> None:
    payload = _payload()
    key, value = mutation
    cursor: dict[str, Any] = payload
    parts = key.split(".")
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = value
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        benchmark.load_reference(path)


def test_reference_needs_one_identical_replay_per_warm_sample(tmp_path: Path) -> None:
    payload = _payload()
    determinism = payload["determinism"]
    determinism["warm_projection_sha256s"] = determinism["warm_projection_sha256s"][:-1]
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="do not reproduce the primary"):
        benchmark.load_reference(path)


def test_build_report_exercises_four_state_pareto_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    tracked = _tracked_native_runtime()
    _pin_native_wall_times(monkeypatch, cold_s=tracked["cold"], warm_s=tracked["warm"])
    report_json = tmp_path / "parity.json"
    report_md = tmp_path / "parity.md"
    argv = ["--report-json", str(report_json), "--report-md", str(report_md), "--strict"]

    assert benchmark.main(argv) == 0

    assert json.loads(capsys.readouterr().out)["cost_completeness"] is True
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert benchmark.check_report(report_json=report_json, report_md=report_md) == []
    assert report["passes_thresholds"] is True
    assert all(report["gates"].values())
    assert set(report["state_metrics"]) == {
        "ion_temperature_kev",
        "electron_temperature_kev",
        "electron_density_m3",
        "poloidal_flux_wb_per_rad",
    }
    assert report["source_budget_metrics"]["torax_max_relative_error"] < 1e-12
    assert report["source_budget_metrics"]["native_max_relative_error"] < 1e-12
    assert report["conservation_metrics"]["native_max_linear_residual_linf"] < 1e-10
    assert report["determinism"]["native_byte_identical"] is True
    assert report["determinism"]["torax_byte_identical"] is True
    replay_shas = report["determinism"]["native_replay_projection_sha256s"]
    assert replay_shas == [report["determinism"]["native_primary_projection_sha256"]] * (
        benchmark.NATIVE_WARM_REPEATS
    )
    samples = report["runtime_samples_seconds"]
    assert len(samples["native_warm"]) == benchmark.NATIVE_WARM_REPEATS
    assert report["runtime_seconds"]["native_warm"] == min(samples["native_warm"])
    assert report["runtime_seconds"]["torax_warm"] == min(samples["torax_warm"])
    assert report["performance_superiority_claimed"] is False
    assert report["physics_equivalence_claimed"] is False


@pytest.mark.parametrize("direction", ["native_faster", "native_slower"])
def test_warm_cost_gate_fails_strict_beyond_the_ratio_limit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    direction: str,
) -> None:
    """A warm ratio past the limit, in either direction, fails only the cost gate."""
    torax_warm = float(benchmark.load_reference()["runtime_seconds"]["warm_primary"])
    limit = float(MODEL_INTERSECTION["thresholds"]["maximum_warm_cost_ratio"])
    warm_s = (
        torax_warm / (2.0 * limit) if direction == "native_faster" else torax_warm * 2.0 * limit
    )
    _pin_native_wall_times(monkeypatch, cold_s=_tracked_native_runtime()["cold"], warm_s=warm_s)
    report_json = tmp_path / "parity.json"
    report_md = tmp_path / "parity.md"
    argv = ["--report-json", str(report_json), "--report-md", str(report_md), "--strict"]

    assert benchmark.main(argv) == 1

    gates = json.loads(capsys.readouterr().out)
    assert gates.pop("cost_completeness") is False
    assert all(gates.values())
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["passes_thresholds"] is False
    assert report["runtime_seconds"]["warm_max_over_min_ratio"] == pytest.approx(2.0 * limit)


def test_tracked_report_and_markdown_are_current() -> None:
    assert benchmark.check_report() == []
    report = json.loads(benchmark.REPORT_JSON.read_text(encoding="utf-8"))
    markdown = benchmark.render_markdown(report)
    assert "Overall pass: `True`" in markdown
    assert "Performance superiority claimed: `False`" in markdown
    assert "Physics equivalence claimed: `False`" in markdown
    assert "`source_budgets`: `True`" in markdown
    assert "## Warm cost" in markdown
    assert "(limit `100`)" in markdown


def test_check_report_rejects_mutated_projection_and_runtime(tmp_path: Path) -> None:
    report = json.loads(benchmark.REPORT_JSON.read_text(encoding="utf-8"))
    report["state_metrics"]["ion_temperature_kev"] = 9.0
    report["runtime_seconds"]["native_warm"] = 0.0
    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"
    report_json.write_text(json.dumps(report), encoding="utf-8")
    report_md.write_text(benchmark.render_markdown(report), encoding="utf-8")

    errors = benchmark.check_report(report_json=report_json, report_md=report_md)

    assert "coupled parity report scientific projection digest is stale" in errors
    assert "coupled parity report runtime evidence is incomplete" in errors


def _drop_samples(report: dict[str, Any]) -> None:
    del report["runtime_samples_seconds"]


def _shorten_native_samples(report: dict[str, Any]) -> None:
    report["runtime_samples_seconds"]["native_warm"] = [1.0]


def _double_torax_warm(report: dict[str, Any]) -> None:
    report["runtime_seconds"]["torax_warm"] *= 2.0


def _flatten_ratio(report: dict[str, Any]) -> None:
    report["runtime_seconds"]["warm_max_over_min_ratio"] = 1.0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_drop_samples, "coupled parity report lacks warm runtime samples"),
        (
            _shorten_native_samples,
            "coupled parity report native_warm samples needs at least 3 positive wall times",
        ),
        (_double_torax_warm, "coupled parity report torax_warm is not the minimum of its samples"),
        (_flatten_ratio, "coupled parity report warm cost ratio does not follow its warm times"),
    ],
)
def test_check_report_rejects_warm_times_that_do_not_follow_their_samples(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None], message: str
) -> None:
    report = json.loads(benchmark.REPORT_JSON.read_text(encoding="utf-8"))
    mutate(report)
    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"
    report_json.write_text(json.dumps(report), encoding="utf-8")
    report_md.write_text(benchmark.render_markdown(report), encoding="utf-8")

    assert message in benchmark.check_report(report_json=report_json, report_md=report_md)


def test_cli_check_reports_success(capsys: pytest.CaptureFixture[str]) -> None:
    assert benchmark.main(["--check"]) == 0
    assert capsys.readouterr().err == ""
