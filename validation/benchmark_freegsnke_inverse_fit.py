#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — SCPN inverse coil-current fit on FreeGSNKE's public inverse problems
"""Fit SCPN's coil currents to FreeGSNKE's public inverse targets and score both solutions.

FreeGSNKE 3.0.1 ships three static inverse problems, solved here exactly as in
:mod:`validation.benchmark_freegsnke_multimachine`: MAST-U-like, SPARC-like and ITER-like.
For every selected case and start this benchmark

1. runs the FreeGSNKE inverse solve and its forward refinement, and evaluates FreeGSNKE's own
   constraint loss (``Inverse_optimizer.build_lsq``) at that solution;
2. poses the same isoflux and null-point targets, controlled circuits, Tikhonov weights and
   current limits to :func:`scpn_fusion.core.free_boundary_coil_fit.fit_coil_currents`, on
   SCPN's sampling of the converged FreeGSNKE profile;
3. fits SCPN's currents from FreeGSNKE's currents (start ``freegsnke``: how far they must move
   for SCPN to meet the same targets), or from a seeded +-20 % perturbation of them (start
   ``perturbed``, walked to by continuation from FreeGSNKE's currents when it does not converge
   directly);
4. refines FreeGSNKE at SCPN's fitted currents and evaluates FreeGSNKE's loss there, so the
   currents of each code are scored by both evaluators.

The two methods differ, and the report says how (``method_differences``). The gates concern
completeness, not agreement: every FreeGSNKE refinement and every SCPN solve that a reported
number comes from must have converged. A perturbed start that SCPN cannot reach is a robustness
result of SCPN's forward solver; it is recorded, not gated. The cases are synthetic public
machine descriptions: there is no facility, control, safety or speed claim and no comparison
with measured discharges.

A run may select cases and starts. ``--merge`` joins such partial reports into one, provided
they come from the same generator, settings, SCPN revision, FreeGSNKE source and host. Each
partial run solves FreeGSNKE's case again, so its FreeGSNKE run time differs; every other
field of a case must agree. The merged case keeps the run time of the first partial report
that names it, and the merged report records the digest of the code that merged it.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# The base benchmark enables JAX float64 at import time, before any array is created here.
import jax
import jax.numpy as jnp

from freegsnke import GSstaticsolver
from freegsnke.inverse import Inverse_optimizer
from scpn_fusion.core.free_boundary_coil_fit import (
    DEFAULT_INNER_ITERATIONS,
    DEFAULT_INNER_TOLERANCE,
    CoilFitProblem,
    CoilFitResult,
    IsofluxTargets,
    fit_coil_currents,
    forward,
    require_float64,
)
from scpn_fusion.core.jax_free_boundary_predictive import build_response_matrix
from validation import benchmark_freegsnke_inverse as base
from validation import benchmark_freegsnke_multimachine as machines

DEFAULT_SOURCE = base.DEFAULT_SOURCE
REPORT_PATH = ROOT / "validation/reports/freegsnke_inverse_fit.json"
MARKDOWN_PATH = ROOT / "validation/reports/freegsnke_inverse_fit.md"
SCHEMA_VERSION = "scpn-fusion.freegsnke-inverse-fit.v1"
BENCHMARK_ID = "freegsnke-public-machines-inverse-fit"
STARTS = ("freegsnke", "perturbed")
PERTURBATION = 0.2
PERTURBATION_SEED = 0
MAX_OUTER = 30
SOURCE_PATHS = (
    "src/scpn_fusion/core/free_boundary_coil_fit.py",
    "src/scpn_fusion/core/jax_free_boundary_predictive.py",
    "validation/benchmark_freegsnke_inverse.py",
    "validation/benchmark_freegsnke_multimachine.py",
)
CLAIM_BOUNDARY = {
    "control_admission": False,
    "experimental_validation": False,
    "facility_validation": False,
    "same_case_inverse_comparison": True,
    "speed_comparison": False,
}
METHOD_DIFFERENCES = (
    "FreeGSNKE linearises the targets with the plasma frozen; SCPN takes central differences "
    "through the full free-boundary solve.",
    "FreeGSNKE penalises current-limit violations through slack variables; SCPN projects the "
    "currents onto the limits.",
    "Both apply the Tikhonov weight to the current step.",
    "FreeGSNKE interpolates the plasma flux with a bicubic spline; SCPN with Keys cubic "
    "convolution.",
)
FloatArray = base.FloatArray
MARKDOWN_HEADER = (
    "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
    "<!-- Commercial license available -->",
    "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
    "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
    "<!-- ORCID: 0009-0009-3560-0851 -->",
    "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
    "<!-- SCPN Fusion Core — SCPN inverse fit on FreeGSNKE's public inverse problems -->",
)


def tikhonov_weights(case: machines.MachineCase, control_count: int) -> FloatArray:
    """Return the case's upstream Tikhonov weights, one per controlled circuit."""
    weights = base.MASTU_L2_REG if case.name == "mastu_like" else case.inverse_kwargs["l2_reg"]
    vector = np.asarray(weights, dtype=np.float64)
    if vector.ndim == 0:
        return np.full(control_count, float(vector))
    if vector.shape != (control_count,):
        raise ValueError(
            f"{case.name}: {vector.size} Tikhonov weights for {control_count} controls"
        )
    return vector


def current_limits(
    case: machines.MachineCase, control_count: int
) -> tuple[FloatArray | None, FloatArray | None]:
    """Return the upstream (lower, upper) limits per controlled circuit; ``None`` is unbounded."""
    if case.name != "mastu_like":
        return None, None
    upper_row, lower_row = base.MASTU_COIL_LIMITS
    upper = np.array([np.inf if value is None else value for value in upper_row])
    lower = np.array([-np.inf if value is None else value for value in lower_row])
    if upper.size != control_count:
        raise ValueError(f"{case.name}: {upper.size} current limits for {control_count} controls")
    return lower, upper


def _blocks(isoflux: float, b_r: float, b_z: float) -> dict[str, float]:
    return {
        "isoflux_l2": isoflux,
        "null_b_r_l2": b_r,
        "null_b_z_l2": b_z,
        "total_l2": float(np.sqrt(isoflux**2 + b_r**2 + b_z**2)),
    }


def scpn_constraint_loss(targets: IsofluxTargets, residual: FloatArray) -> dict[str, float]:
    """Split SCPN's residual into FreeGSNKE's loss blocks: isoflux pairs, B_R and B_Z."""
    pairs, nulls = targets.pairs.shape[0], targets.null_r.size
    if residual.shape != (targets.residual_count,):
        raise ValueError("residual does not match the targets")
    return _blocks(
        float(np.linalg.norm(residual[:pairs])),
        float(np.linalg.norm(residual[pairs : pairs + nulls])),
        float(np.linalg.norm(residual[pairs + nulls :])),
    )


def freegsnke_constraint_loss(eq: Any, case: machines.MachineCase) -> tuple[dict[str, float], Any]:
    """Evaluate FreeGSNKE's own constraint loss of ``eq`` for the case's targets.

    Returns the loss blocks and FreeGSNKE's control mask over all of ``eq``'s coils.
    """
    constrain = Inverse_optimizer(
        null_points=[list(case.null_points[0]), list(case.null_points[1])],
        isoflux_set=np.array([[list(case.isoflux[0]), list(case.isoflux[1])]], dtype=np.float64),
    )
    constrain.prepare_for_solve(eq)
    constrain.build_plasma_vals(trial_plasma_psi=eq.plasma_psi)
    constrain.build_lsq(np.asarray(eq.tokamak.getCurrentsVec(), dtype=np.float64))
    # One isoflux set and the null points give exactly three blocks: isoflux, B_R, B_Z.
    isoflux, b_r, b_z = (float(value) for value in constrain.loss)
    return _blocks(isoflux, b_r, b_z), constrain.control_mask


def build_problem(
    source: Path, case: machines.MachineCase, eq: Any, profiles: Any, control_mask: Any
) -> tuple[CoilFitProblem, FloatArray, jax.Array, list[dict[str, Any]]]:
    """Pose FreeGSNKE's converged case as an SCPN fit with FreeGSNKE's control set.

    Returns the problem, FreeGSNKE's controlled currents, its flux map as a warm start
    ``(nz, nr)`` and the circuit contract.
    """
    circuit_count = int(eq.tokamak.n_active_coils)
    mask = np.asarray(control_mask, dtype=bool)
    if mask[circuit_count:].any():
        raise ValueError(f"{case.name}: FreeGSNKE controls a passive structure")
    control = mask[:circuit_count]
    # A copy: FreeGSNKE keeps the returned vector and overwrites it when currents are set.
    currents = np.array(eq.tokamak.getCurrentsVec(), dtype=np.float64)[:circuit_count].copy()
    coil_r, coil_z, _filament_i, incidence, circuits = base._active_filaments(
        source, currents, machines.machine_paths(source, case)["active_coils"]
    )
    r_grid = np.asarray(eq.R[:, 0], dtype=np.float64)
    z_grid = np.asarray(eq.Z[0, :], dtype=np.float64)
    knots = np.linspace(0.0, 1.0, base.PROFILE_SAMPLE_COUNT, dtype=np.float64)
    response, wall_idx, source_idx = build_response_matrix(jnp.asarray(r_grid), jnp.asarray(z_grid))
    lower, upper = current_limits(case, int(control.sum()))
    problem = CoilFitProblem(
        r_grid=r_grid,
        z_grid=z_grid,
        filament_r=coil_r,
        filament_z=coil_z,
        incidence=incidence,
        psin_knots=knots,
        pprime=np.asarray(profiles.pprime(knots), dtype=np.float64),
        ffprime=np.asarray(profiles.ffprime(knots), dtype=np.float64),
        ip_target=case.ip_a,
        response_matrix=response,
        wall_idx=wall_idx,
        source_idx=source_idx,
        targets=IsofluxTargets(
            np.asarray(case.isoflux[0]),
            np.asarray(case.isoflux[1]),
            np.asarray(case.null_points[0]),
            np.asarray(case.null_points[1]),
        ),
        control=control,
        fixed_currents=currents,
        tikhonov=tikhonov_weights(case, int(control.sum())),
        lower=lower,
        upper=upper,
    )
    psi_reference = jnp.asarray(np.asarray(eq.psi(), dtype=np.float64).T)
    return problem, currents[control], psi_reference, circuits


def start_controls(reference: FloatArray, start: str) -> FloatArray:
    """Return the start: FreeGSNKE's currents, or their seeded +-20 % perturbation."""
    if start == "freegsnke":
        return reference.copy()
    if start != "perturbed":
        raise ValueError(f"unknown start {start!r}")
    rng = np.random.default_rng(PERTURBATION_SEED)
    return reference * (1.0 + PERTURBATION * rng.uniform(-1.0, 1.0, reference.size))


def fit_record(
    problem: CoilFitProblem, reference: FloatArray, start: FloatArray, result: CoilFitResult
) -> dict[str, Any]:
    """Describe one fit. Numbers of the final state appear only when its solve converged.

    ``continuation_fraction_reached`` is the share of the walk from FreeGSNKE's currents to
    the start that converged: 1.0 once the walk arrived, ``None`` when no walk was needed.
    """
    target = problem.project(start)
    walk = target - reference
    reached: float | None = None
    if result.continuation_steps:
        stopped = result.stop_reason == "start_not_converged"
        reached = float((result.controls - reference) @ walk / (walk @ walk)) if stopped else 1.0
    record: dict[str, Any] = {
        "continuation_fraction_reached": reached,
        "continuation_steps": result.continuation_steps,
        "converged_solution": result.psi is not None,
        "outer_iterations": result.outer_iterations,
        "start_controls_a": target.tolist(),
        "stop_reason": result.stop_reason,
    }
    if result.psi is None:
        return record
    residual = problem.residual(result.psi, problem.circuit_currents(result.controls))
    change = result.controls - reference
    record.update(
        {
            "bounds_active": result.bounds_active,
            "controls_a": result.controls.tolist(),
            "jacobian_singular_values": list(result.jacobian_singular_values),
            "loss": scpn_constraint_loss(problem.targets, residual),
            "max_abs_change_from_freegsnke_a": float(np.max(np.abs(change))),
            "objective_history": list(result.objective_history),
            "relative_l2_change_from_freegsnke": float(
                np.linalg.norm(change) / np.linalg.norm(reference)
            ),
        }
    )
    return record


def freegsnke_at_currents(
    eq: Any, profiles: Any, case: machines.MachineCase, circuit_currents: FloatArray
) -> dict[str, Any]:
    """Refine FreeGSNKE at the given active-circuit currents and evaluate its loss there."""
    eq.tokamak.set_all_coil_currents(np.asarray(circuit_currents, dtype=np.float64))
    solver = GSstaticsolver.NKGSsolver(eq)
    solver.solve(
        eq=eq,
        profiles=profiles,
        constrain=None,
        target_relative_tolerance=machines.REFINEMENT_TOLERANCE,
        verbose=False,
    )
    change = float(solver.relative_change)
    converged = change <= machines.REFINEMENT_TOLERANCE
    return {
        "converged": converged,
        "loss": freegsnke_constraint_loss(eq, case)[0] if converged else None,
        "relative_change": change,
    }


def score_start(
    eq: Any,
    profiles: Any,
    case: machines.MachineCase,
    problem: CoilFitProblem,
    reference: FloatArray,
    psi_reference: jax.Array,
    start: str,
) -> dict[str, Any]:
    """Fit SCPN from one start; when the fit converged, score its currents in FreeGSNKE too."""
    started = time.perf_counter()
    initial = start_controls(reference, start)
    result = fit_coil_currents(
        problem,
        initial,
        psi_reference,
        continuation_reference=reference if start == "perturbed" else None,
        max_outer=MAX_OUTER,
    )
    entry = fit_record(problem, reference, initial, result)
    if result.psi is not None:
        entry["freegsnke_at_scpn_currents"] = freegsnke_at_currents(
            eq, profiles, case, problem.circuit_currents(result.controls)
        )
    entry["seconds_orientation_only"] = time.perf_counter() - started
    return entry


def case_checks(case: dict[str, Any]) -> dict[str, bool]:
    """Return the completeness gates of one case record."""
    refinement = case["freegsnke_run"]["forward_refinement_relative_change"]
    cross = [
        start["freegsnke_at_scpn_currents"]["converged"]
        for start in case["starts"].values()
        if "freegsnke_at_scpn_currents" in start
    ]
    return {
        "freegsnke_refinement_converged": refinement <= machines.REFINEMENT_TOLERANCE,
        "freegsnke_refinements_at_scpn_currents_converged": all(cross),
        "scpn_forward_at_freegsnke_currents_converged": bool(
            case["scpn_at_freegsnke_currents"]["converged"]
        ),
    }


def run_case(source: Path, case: machines.MachineCase, starts: tuple[str, ...]) -> dict[str, Any]:
    """Solve one case in FreeGSNKE, fit SCPN from each start and score both currents."""
    eq, profiles, freegsnke_run = machines.solve_freegsnke_case(source, case)
    freegsnke_loss, control_mask = freegsnke_constraint_loss(eq, case)
    problem, reference, psi_reference, circuits = build_problem(
        source, case, eq, profiles, control_mask
    )
    at_reference = forward(problem, reference, psi_reference)
    on_freegsnke_flux = problem.residual(psi_reference, problem.circuit_currents(reference))
    record: dict[str, Any] = {
        "circuits": circuits,
        "controlled_circuits": [
            circuit["circuit"] for circuit, on in zip(circuits, problem.control) if on
        ],
        "freegsnke_controls_a": reference.tolist(),
        "freegsnke_loss": freegsnke_loss,
        "freegsnke_run": freegsnke_run,
        "grid_shape_r_z": list(case.grid),
        # SCPN's evaluator on FreeGSNKE's own flux map: separates the evaluators' difference
        # from the difference between the two codes' equilibria.
        "scpn_evaluator_on_freegsnke_flux": scpn_constraint_loss(
            problem.targets, on_freegsnke_flux
        ),
        "scpn_at_freegsnke_currents": {
            "converged": at_reference.converged,
            "inner_iterations": at_reference.iterations,
            "loss": (
                scpn_constraint_loss(problem.targets, at_reference.residual)
                if at_reference.converged
                else None
            ),
        },
        "starts": {},
        "targets": {
            "isoflux_points": problem.targets.isoflux_r.size,
            "null_points": len(case.null_points[0]),
        },
        "tikhonov": problem.tikhonov.tolist(),
        "upstream_reference": case.upstream_reference,
    }
    for start in starts:
        record["starts"][start] = score_start(
            eq, profiles, case, problem, reference, psi_reference, start
        )
    record["checks"] = case_checks(record)
    record["passed"] = all(record["checks"].values())
    return record


def sign_report(report: dict[str, Any]) -> dict[str, Any]:
    """Set the report's status from its case gates and its payload digest."""
    report["status"] = (
        "pass" if all(case["passed"] for case in report["cases"].values()) else "fail"
    )
    report["payload_sha256"] = base._payload_sha256(report)
    return report


def build_report(
    source: Path = DEFAULT_SOURCE,
    case_names: tuple[str, ...] = tuple(machines.CASES),
    starts: tuple[str, ...] = STARTS,
) -> dict[str, Any]:
    """Run every selected case and start and return the signed report."""
    require_float64()
    unknown = [name for name in case_names if name not in machines.CASES]
    if unknown or not case_names:
        raise ValueError(f"unknown or empty case selection: {unknown or case_names}")
    if not starts or any(start not in STARTS for start in starts):
        raise ValueError(f"unknown or empty start selection: {starts}")
    source = source.resolve()
    versions = base.validate_source(source)
    # Taken before the run, so a checkout without a revision fails at once.
    provenance = {
        "freegsnke_source_commit": versions["commit"],
        "host": machines.host_identity(),
        "scpn_source_revision": base._git_head(ROOT),
        "scpn_source_sha256": {path: base._file_sha256(ROOT / path) for path in SOURCE_PATHS},
        "versions": versions,
    }
    report: dict[str, Any] = {
        "benchmark_id": BENCHMARK_ID,
        "cases": {name: run_case(source, machines.CASES[name], starts) for name in case_names},
        "claim_boundary": CLAIM_BOUNDARY,
        "generator": {
            "path": "validation/benchmark_freegsnke_inverse_fit.py",
            "sha256": base._file_sha256(Path(__file__).resolve()),
        },
        "method_differences": list(METHOD_DIFFERENCES),
        "provenance": provenance,
        "schema_version": SCHEMA_VERSION,
        "settings": {
            "inner_iterations": DEFAULT_INNER_ITERATIONS,
            "inner_tolerance": DEFAULT_INNER_TOLERANCE,
            "max_outer": MAX_OUTER,
            "perturbation": PERTURBATION,
            "perturbation_seed": PERTURBATION_SEED,
            "refinement_tolerance": machines.REFINEMENT_TOLERANCE,
        },
    }
    return sign_report(report)


def _freegsnke_case(case: dict[str, Any]) -> dict[str, Any]:
    """Return the FreeGSNKE part of a case: no starts, no derived gates, no run time."""
    shared = {
        key: copy.deepcopy(value)
        for key, value in case.items()
        if key not in ("starts", "checks", "passed")
    }
    shared.get("freegsnke_run", {}).pop("seconds_orientation_only", None)
    return shared


def merge_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Join partial reports of one generator, revision and host into one signed report."""
    if not reports:
        raise ValueError("nothing to merge")
    shared = (
        "benchmark_id",
        "claim_boundary",
        "generator",
        "method_differences",
        "provenance",
        "schema_version",
        "settings",
    )
    merged = {key: copy.deepcopy(reports[0][key]) for key in shared}
    merged["cases"] = {}
    for report in reports:
        if report.get("payload_sha256") != base._payload_sha256(report):
            raise ValueError("a partial report does not match its payload digest")
        if any(report[key] != merged[key] for key in shared):
            raise ValueError("partial reports differ in generator, settings or provenance")
        for name, case in report["cases"].items():
            known = merged["cases"].setdefault(name, {**copy.deepcopy(case), "starts": {}})
            if _freegsnke_case(case) != _freegsnke_case(known):
                raise ValueError(f"{name}: partial reports disagree on the FreeGSNKE case")
            for start, entry in case["starts"].items():
                if start in known["starts"]:
                    raise ValueError(f"{name}: start {start!r} appears twice")
                known["starts"][start] = copy.deepcopy(entry)
    for case in merged["cases"].values():
        case["checks"] = case_checks(case)
        case["passed"] = all(case["checks"].values())
    merged["merged_by"] = {
        "path": "validation/benchmark_freegsnke_inverse_fit.py",
        "sha256": base._file_sha256(Path(__file__).resolve()),
    }
    return sign_report(merged)


def _total(loss: dict[str, float] | None) -> str:
    return "—" if loss is None else f"{loss['total_l2']:.3g}"


def render_markdown(report: dict[str, Any]) -> str:
    """Render one row per case and start, with both evaluators' losses."""
    lines = [
        *MARKDOWN_HEADER,
        "# SCPN inverse coil-current fit on FreeGSNKE's public inverse problems",
        "",
        f"Status: **{report['status'].upper()}** (completeness gates; agreement is reported, "
        "not gated)",
        "",
        "Synthetic public machine descriptions only. No facility, control, safety or speed "
        "claim; no comparison with measured discharges.",
        "",
        "Loss columns are L2 norms over all targets. At FreeGSNKE's currents: FreeGSNKE's "
        "loss, SCPN's evaluator on FreeGSNKE's flux map, and SCPN's loss on its own "
        "equilibrium. At SCPN's fitted currents: SCPN's loss and FreeGSNKE's.",
        "",
        "| Case | Start | Stop | FreeGSNKE own | SCPN evaluator, FreeGSNKE flux "
        "| SCPN at FreeGSNKE currents | SCPN fit | FreeGSNKE at SCPN currents "
        "| Current change (rel. L2) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, case in report["cases"].items():
        for start, entry in case["starts"].items():
            cross = entry.get("freegsnke_at_scpn_currents", {}).get("loss")
            change = entry.get("relative_l2_change_from_freegsnke")
            lines.append(
                f"| {name} | {start} | {entry['stop_reason']} "
                f"| {_total(case['freegsnke_loss'])} "
                f"| {_total(case['scpn_evaluator_on_freegsnke_flux'])} "
                f"| {_total(case['scpn_at_freegsnke_currents']['loss'])} "
                f"| {_total(entry.get('loss'))} | {_total(cross)} "
                f"| {'—' if change is None else f'{change:.3g}'} |"
            )
    lines += ["", "Method differences:", ""]
    lines += [f"- {item}" for item in report["method_differences"]]
    lines += ["", f"Payload SHA-256: `{report['payload_sha256']}`", ""]
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the source checkout, outputs, case and start selection, merge inputs and strictness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--markdown", type=Path, default=MARKDOWN_PATH)
    parser.add_argument("--case", action="append", choices=sorted(machines.CASES))
    parser.add_argument("--start", action="append", choices=STARTS)
    parser.add_argument("--merge", nargs="+", type=Path, metavar="PARTIAL")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write the JSON and Markdown reports; exit 1 under ``--strict`` when a gate fails."""
    args = _parse_args(argv)
    if args.merge:
        partials = [json.loads(path.read_text(encoding="utf-8")) for path in args.merge]
        report = merge_reports(partials)
    else:
        report = build_report(
            args.source,
            tuple(args.case) if args.case else tuple(machines.CASES),
            tuple(args.start) if args.start else STARTS,
        )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    return 1 if args.strict and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
