# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — tests of the SCPN inverse fit on FreeGSNKE's public inverse problems
"""Contract and live tests for the SCPN inverse fit on FreeGSNKE's public inverse problems.

The contract tests use the real fit types on a small grid and hand-built reports; the live
test runs the MAST-U-like case end to end with one Levenberg-Marquardt iteration.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("freegsnke")

import validation.benchmark_freegsnke_inverse as base
from freegsnke import GSstaticsolver
import validation.benchmark_freegsnke_inverse_fit as benchmark
import validation.benchmark_freegsnke_multimachine as machines
from scpn_fusion.core.free_boundary_coil_fit import (
    CoilFitProblem,
    CoilFitResult,
    IsofluxTargets,
    StopReason,
)
from scpn_fusion.core.jax_free_boundary_predictive import build_response_matrix

CASES = machines.CASES


def _small_problem() -> CoilFitProblem:
    r, z = np.linspace(1.0, 2.0, 9), np.linspace(-1.0, 1.0, 9)
    response, wall, source = build_response_matrix(jnp.asarray(r), jnp.asarray(z))
    return CoilFitProblem(
        r_grid=r,
        z_grid=z,
        filament_r=np.array([1.1, 1.9]),
        filament_z=np.array([0.8, -0.8]),
        incidence=np.eye(2),
        psin_knots=np.linspace(0.0, 1.0, 4),
        pprime=np.zeros(4),
        ffprime=np.zeros(4),
        ip_target=1.0e5,
        response_matrix=response,
        wall_idx=wall,
        source_idx=source,
        targets=IsofluxTargets(
            np.array([1.3, 1.5, 1.7]), np.array([0.0, 0.2, -0.2]), np.array([1.5]), np.array([0.5])
        ),
        control=np.array([True, True]),
        fixed_currents=np.array([1.0e3, -1.0e3]),
        tikhonov=np.zeros(2),
    )


def _result(controls: np.ndarray, *, psi: Any, steps: int, stop: StopReason) -> CoilFitResult:
    return CoilFitResult(
        controls=controls,
        objective=1.0,
        max_abs_residual=1.0,
        objective_history=(2.0, 1.0),
        stop_reason=stop,
        outer_iterations=1,
        jacobian_singular_values=(3.0, 1.0),
        bounds_active=0,
        continuation_steps=steps,
        psi=psi,
    )


def test_tikhonov_weights_follow_the_upstream_cases() -> None:
    assert benchmark.tikhonov_weights(CASES["mastu_like"], 11).tolist() == list(base.MASTU_L2_REG)
    assert benchmark.tikhonov_weights(CASES["sparc_like"], 11).tolist() == [1.0e-16] * 10 + [1e-5]
    assert benchmark.tikhonov_weights(CASES["iter_like"], 3).tolist() == [1.0e-14] * 3
    with pytest.raises(ValueError, match="10 controls"):
        benchmark.tikhonov_weights(CASES["sparc_like"], 10)


def test_current_limits_exist_for_mastu_only_and_are_open_where_upstream_is() -> None:
    lower, upper = benchmark.current_limits(CASES["mastu_like"], 11)
    assert lower is not None and upper is not None
    assert upper[0] == 5.0e3 and lower[-2] == -10.0e3
    assert np.isinf(upper[-1]) and np.isinf(lower[-1]) and lower[-1] < 0.0
    assert benchmark.current_limits(CASES["sparc_like"], 11) == (None, None)
    with pytest.raises(ValueError, match="for 10 controls"):
        benchmark.current_limits(CASES["mastu_like"], 10)


def test_scpn_loss_splits_the_residual_into_freegsnke_blocks() -> None:
    targets = IsofluxTargets(np.ones(3), np.arange(3.0), np.array([1.0]), np.array([0.0]))
    loss = benchmark.scpn_constraint_loss(targets, np.array([3.0, 0.0, 4.0, 1.0, 2.0]))
    assert loss == {"isoflux_l2": 5.0, "null_b_r_l2": 1.0, "null_b_z_l2": 2.0, "total_l2": 30**0.5}
    with pytest.raises(ValueError, match="does not match"):
        benchmark.scpn_constraint_loss(targets, np.zeros(4))


def test_starts_are_freegsnke_currents_or_a_seeded_perturbation() -> None:
    reference = np.array([1.0e3, -2.0e3, 5.0e2])
    same = benchmark.start_controls(reference, "freegsnke")
    assert same is not reference and same.tolist() == reference.tolist()
    perturbed = benchmark.start_controls(reference, "perturbed")
    assert perturbed.tolist() == benchmark.start_controls(reference, "perturbed").tolist()
    ratio = perturbed / reference
    assert np.all(np.abs(ratio - 1.0) <= benchmark.PERTURBATION) and np.any(ratio != 1.0)
    with pytest.raises(ValueError, match="unknown start"):
        benchmark.start_controls(reference, "cold")


def test_fit_record_reports_final_numbers_only_for_a_converged_solution() -> None:
    problem = _small_problem()
    reference = np.array([1.0e3, -1.0e3])
    start = np.array([2.0e3, -3.0e3])
    psi = jnp.zeros((9, 9))

    direct = benchmark.fit_record(
        problem, reference, start, _result(start, psi=psi, steps=0, stop="stalled")
    )
    assert direct["continuation_fraction_reached"] is None
    assert direct["converged_solution"] is True
    assert set(direct["loss"]) == {"isoflux_l2", "null_b_r_l2", "null_b_z_l2", "total_l2"}
    assert direct["max_abs_change_from_freegsnke_a"] == 2.0e3
    assert direct["relative_l2_change_from_freegsnke"] == pytest.approx(5.0**0.5 / 2.0**0.5)

    walked = benchmark.fit_record(
        problem, reference, start, _result(start, psi=psi, steps=3, stop="stalled")
    )
    assert walked["continuation_fraction_reached"] == 1.0

    quarter = reference + 0.25 * (start - reference)
    stopped = benchmark.fit_record(
        problem, reference, start, _result(quarter, psi=None, steps=7, stop="start_not_converged")
    )
    assert stopped["continuation_fraction_reached"] == pytest.approx(0.25)
    assert stopped["converged_solution"] is False
    assert "loss" not in stopped and "controls_a" not in stopped


def test_freegsnke_numbers_at_unconverged_currents_are_withheld(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Unconverged:
        relative_change = 1.0

        def __init__(self, eq: object) -> None:
            self.eq = eq

        def solve(self, **_kwargs: object) -> None:
            return None

    seen: list[np.ndarray] = []
    eq = SimpleNamespace(tokamak=SimpleNamespace(set_all_coil_currents=seen.append))
    monkeypatch.setattr(GSstaticsolver, "NKGSsolver", Unconverged)

    record = benchmark.freegsnke_at_currents(eq, object(), CASES["sparc_like"], np.ones(3))

    assert record == {"converged": False, "loss": None, "relative_change": 1.0}
    assert [values.tolist() for values in seen] == [[1.0, 1.0, 1.0]]


def test_an_unreached_start_is_not_scored_in_freegsnke() -> None:
    """A fit without a converged state never reaches FreeGSNKE: nothing is cross-evaluated."""
    problem = dataclasses.replace(_small_problem(), inner_iterations=1)
    reference = np.array([1.0e3, -1.0e3])

    entry = benchmark.score_start(
        object(), object(), CASES["sparc_like"], problem, reference, jnp.zeros((9, 9)), "freegsnke"
    )

    assert entry["stop_reason"] == "start_not_converged"
    assert entry["converged_solution"] is False
    assert "freegsnke_at_scpn_currents" not in entry
    assert entry["seconds_orientation_only"] >= 0.0


def test_a_freegsnke_controlled_passive_structure_is_refused() -> None:
    eq = SimpleNamespace(tokamak=SimpleNamespace(n_active_coils=2))
    with pytest.raises(ValueError, match="passive structure"):
        benchmark.build_problem(
            Path("."), CASES["sparc_like"], eq, object(), np.array([True, True, True])
        )


def _case(
    *, refinement: float = 1e-10, scpn: bool = True, cross: bool = True, seconds: float = 6.2
) -> dict[str, Any]:
    return {
        "freegsnke_loss": {"total_l2": 1.0e-5},
        "freegsnke_run": {
            "forward_refinement_relative_change": refinement,
            "seconds_orientation_only": seconds,
        },
        "scpn_at_freegsnke_currents": {"converged": scpn, "loss": {"total_l2": 2.0e-3}},
        "scpn_evaluator_on_freegsnke_flux": {"total_l2": 1.1e-5},
        "starts": {
            "freegsnke": {
                "freegsnke_at_scpn_currents": {"converged": cross, "loss": {"total_l2": 4.0e-3}},
                "loss": {"total_l2": 1.0e-8},
                "relative_l2_change_from_freegsnke": 0.014,
                "stop_reason": "stalled",
            }
        },
    }


@pytest.mark.parametrize(
    ("changes", "failed"),
    [
        ({}, set()),
        ({"refinement": 1e-8}, {"freegsnke_refinement_converged"}),
        ({"scpn": False}, {"scpn_forward_at_freegsnke_currents_converged"}),
        ({"cross": False}, {"freegsnke_refinements_at_scpn_currents_converged"}),
    ],
)
def test_case_gates_concern_completeness(changes: dict[str, Any], failed: set[str]) -> None:
    checks = benchmark.case_checks(_case(**changes))
    assert {name for name, ok in checks.items() if not ok} == failed


def _partial(start: str, **changes: Any) -> dict[str, Any]:
    case = _case(**changes)
    case["starts"] = {start: case["starts"]["freegsnke"] | {"stop_reason": f"{start}-stop"}}
    case["checks"] = benchmark.case_checks(case)
    case["passed"] = all(case["checks"].values())
    report: dict[str, Any] = {
        "benchmark_id": benchmark.BENCHMARK_ID,
        "cases": {"mastu_like": case},
        "claim_boundary": benchmark.CLAIM_BOUNDARY,
        "generator": {"path": "g", "sha256": "0" * 64},
        "method_differences": list(benchmark.METHOD_DIFFERENCES),
        "provenance": {"host": {"cpu_model": "x"}, "scpn_source_revision": "r"},
        "schema_version": benchmark.SCHEMA_VERSION,
        "settings": {"max_outer": benchmark.MAX_OUTER},
    }
    return benchmark.sign_report(report)


def test_merge_joins_the_starts_of_partial_reports_and_signs_again() -> None:
    merged = benchmark.merge_reports(
        [_partial("freegsnke"), _partial("perturbed", cross=False, seconds=6.6)]
    )
    case = merged["cases"]["mastu_like"]
    assert list(case["starts"]) == ["freegsnke", "perturbed"]
    # Each partial run solved FreeGSNKE again: only the first run time is kept.
    assert case["freegsnke_run"]["seconds_orientation_only"] == 6.2
    assert merged["merged_by"] == {
        "path": "validation/benchmark_freegsnke_inverse_fit.py",
        "sha256": hashlib.sha256(Path(benchmark.__file__).read_bytes()).hexdigest(),
    }
    assert case["checks"]["freegsnke_refinements_at_scpn_currents_converged"] is False
    assert merged["status"] == "fail"
    assert benchmark.merge_reports([copy.deepcopy(merged)]) == merged


def test_merge_refuses_inconsistent_or_altered_partials() -> None:
    with pytest.raises(ValueError, match="nothing to merge"):
        benchmark.merge_reports([])
    altered = _partial("freegsnke")
    altered["cases"]["mastu_like"]["freegsnke_loss"]["total_l2"] = 0.0
    with pytest.raises(ValueError, match="payload digest"):
        benchmark.merge_reports([altered])
    other_host = _partial("perturbed")
    other_host["provenance"]["host"] = {"cpu_model": "y"}
    with pytest.raises(ValueError, match="generator, settings or provenance"):
        benchmark.merge_reports([_partial("freegsnke"), benchmark.sign_report(other_host)])
    with pytest.raises(ValueError, match="disagree on the FreeGSNKE case"):
        benchmark.merge_reports([_partial("freegsnke"), _partial("perturbed", refinement=2e-10)])
    with pytest.raises(ValueError, match="appears twice"):
        benchmark.merge_reports([_partial("freegsnke"), _partial("freegsnke")])


def test_markdown_shows_both_evaluators_and_dashes_for_withheld_numbers() -> None:
    report = _partial("perturbed")
    entry = report["cases"]["mastu_like"]["starts"]["perturbed"]
    for key in ("freegsnke_at_scpn_currents", "loss", "relative_l2_change_from_freegsnke"):
        del entry[key]
    rendered = benchmark.render_markdown(benchmark.sign_report(report))
    assert rendered.splitlines()[:7] == list(benchmark.MARKDOWN_HEADER)
    assert "Status: **PASS**" in rendered
    assert "| mastu_like | perturbed | perturbed-stop | 1e-05 | 1.1e-05 | 0.002 | — | — | — |" in (
        rendered
    )
    assert all(item in rendered for item in benchmark.METHOD_DIFFERENCES)


def test_command_line_merges_partials_and_honours_strict(tmp_path: Path) -> None:
    paths = []
    for start, cross in (("freegsnke", True), ("perturbed", False)):
        path = tmp_path / f"{start}.json"
        path.write_text(json.dumps(_partial(start, cross=cross)), encoding="utf-8")
        paths.append(str(path))
    output, markdown = tmp_path / "fit.json", tmp_path / "fit.md"
    outputs = ["--output", str(output), "--markdown", str(markdown)]

    assert benchmark.main(["--merge", *paths, *outputs]) == 0
    assert benchmark.main(["--merge", *paths, *outputs, "--strict"]) == 1
    assert set(json.loads(output.read_text(encoding="utf-8"))["cases"]["mastu_like"]["starts"]) == {
        "freegsnke",
        "perturbed",
    }
    assert markdown.read_text(encoding="utf-8").startswith(benchmark.MARKDOWN_HEADER[0])


def test_command_line_runs_the_selected_case_and_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_case(_source: Path, case: machines.MachineCase, starts: tuple[str, ...]) -> Any:
        calls.append((case.name, starts))
        record = _case()
        record["checks"] = benchmark.case_checks(record)
        record["passed"] = True
        return record

    monkeypatch.setattr(benchmark, "run_case", fake_case)
    output = tmp_path / "fit.json"
    arguments = ["--case", "sparc_like", "--start", "perturbed", "--strict", "--output"]

    assert benchmark.main([*arguments, str(output), "--markdown", str(tmp_path / "f.md")]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert calls == [("sparc_like", ("perturbed",))]
    assert report["provenance"]["freegsnke_source_commit"] == base.UPSTREAM_COMMIT
    assert len(report["provenance"]["scpn_source_revision"]) == 40
    assert set(report["provenance"]["scpn_source_sha256"]) == set(benchmark.SOURCE_PATHS)
    assert report["settings"]["max_outer"] == benchmark.MAX_OUTER


def test_selection_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown or empty case selection"):
        benchmark.build_report(case_names=("not_a_machine",))
    with pytest.raises(ValueError, match="unknown or empty start selection"):
        benchmark.build_report(case_names=("mastu_like",), starts=())
    with pytest.raises(ValueError, match="unknown or empty start selection"):
        benchmark.build_report(case_names=("mastu_like",), starts=("cold",))


def test_script_entry_point_runs_from_the_repository_root() -> None:
    root = Path(benchmark.__file__).resolve().parents[1]
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    completed = subprocess.run(
        [sys.executable, "validation/benchmark_freegsnke_inverse_fit.py", "--help"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--start {freegsnke,perturbed}" in completed.stdout


@pytest.mark.slow
def test_live_mastu_like_case_with_one_fit_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the MAST-U-like case end to end with one Levenberg-Marquardt iteration."""
    monkeypatch.setattr(benchmark, "MAX_OUTER", 1)
    source = Path(benchmark.DEFAULT_SOURCE).resolve()

    record = benchmark.run_case(source, CASES["mastu_like"], ("freegsnke",))

    assert record["passed"], record["checks"]
    assert "Solenoid" not in record["controlled_circuits"]
    assert len(record["controlled_circuits"]) == 11
    assert record["tikhonov"] == list(base.MASTU_L2_REG)
    own = record["freegsnke_loss"]["total_l2"]
    on_flux = record["scpn_evaluator_on_freegsnke_flux"]["total_l2"]
    assert np.isfinite(own) and np.isfinite(on_flux)
    entry = record["starts"]["freegsnke"]
    assert entry["converged_solution"] and entry["outer_iterations"] == 1
    assert entry["loss"]["total_l2"] <= record["scpn_at_freegsnke_currents"]["loss"]["total_l2"]
    assert entry["freegsnke_at_scpn_currents"]["converged"]
    assert np.isfinite(entry["freegsnke_at_scpn_currents"]["loss"]["total_l2"])
    assert len(entry["jacobian_singular_values"]) == 11


def test_import_puts_the_repository_and_source_tree_on_the_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Imported from outside the repository, the module puts both roots on the path itself."""
    root, source = str(benchmark.ROOT), str(benchmark.ROOT / "src")
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry not in (root, source)])
    importlib.reload(benchmark)
    assert sys.path[:2] == [source, root]
