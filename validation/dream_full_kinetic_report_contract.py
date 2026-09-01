# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — DREAM Full Kinetic Report Contract
"""Fail-closed acceptance contract for full-kinetic DREAM parity evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import numpy as np

from validation.dream_full_kinetic_reference import (
    DREAM_COMMIT,
    ION_RATE_AUXILIARY_QUANTITIES,
    REQUIRED_AUXILIARY_QUANTITIES,
    REQUIRED_RUNAWAY_QUANTITIES,
    sha256_file,
)
from validation.reference_data.dream.full_kinetic_radial_parity_deck import (
    REQUESTED_OTHER_QUANTITIES,
)


SCHEMA: Final[str] = "scpn-fusion.dream-full-kinetic-parity.v1"
DREAMI_SHA256: Final[str] = "03d4251092864062ef4913a5944fb18e3931a7fe3a8e34491dd64ebe0fc69f39"
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DECK_RELATIVE_PATH: Final[Path] = Path(
    "validation/reference_data/dream/full_kinetic_radial_parity_deck.py"
)
LOCK_RELATIVE_PATH: Final[Path] = Path(
    "validation/reference_data/dream/full_kinetic_radial_parity_lock.json"
)
NATIVE_SOURCE_PATHS: Final[tuple[Path, ...]] = (
    Path("src/scpn_fusion/core/runaway_kinetic_coefficients.py"),
    Path("src/scpn_fusion/core/runaway_kinetic_diagnostics.py"),
    Path("src/scpn_fusion/core/runaway_kinetic_grid.py"),
    Path("src/scpn_fusion/core/runaway_kinetic_operator.py"),
    Path("src/scpn_fusion/core/runaway_kinetic_solver.py"),
    Path("validation/dream_full_kinetic_metrics.py"),
    Path("validation/dream_full_kinetic_reference.py"),
    Path("validation/dream_full_kinetic_report_contract.py"),
    Path("validation/benchmark_dream_full_kinetic_parity.py"),
)
THRESHOLDS: Final[dict[str, float]] = {
    "native_distribution_residual_max": 5.0e-7,
    "native_density_residual_max": 2.0e-4,
    "pitch_advection_reconstruction_max": 1.0e-7,
    "avalanche_source_reconstruction_max": 1.0e-12,
    "radial_loss_reconstruction_max": 1.0e-12,
    "current_moment_reconstruction_max": 1.0e-7,
    "distribution_convergence_error_max": 0.25,
    "density_convergence_error_max": 0.05,
    "current_convergence_error_max": 0.10,
    "growth_convergence_error_max": 0.02,
    "operator_convergence_error_max": 0.25,
    "convergence_improvement_ratio_max": 1.0,
}
REQUIRED_GATES: Final[frozenset[str]] = frozenset(
    {
        "pinned_dreami_binary",
        "frozen_custody_lock",
        "frozen_case_reconstructed",
        "complete_execution_receipts",
        "all_axes_evolved",
        "all_requested_auxiliary_outputs_present",
        "all_configured_active_auxiliary_outputs_nonzero",
        "all_required_operators_nonzero",
        "native_distribution_residual",
        "native_density_residual",
        "pitch_advection_reconstruction",
        "avalanche_source_reconstruction",
        "radial_loss_reconstruction",
        "current_moment_reconstruction",
        "distribution_converged",
        "density_converged",
        "current_converged",
        "growth_converged",
        "operator_coefficients_converged",
        "state_convergence_improves",
        "operator_convergence_improves",
    }
)
CONVERGENCE_RESOLUTIONS: Final[frozenset[str]] = frozenset({"coarse", "medium", "fine"})
ACCEPTED_PROCESS_EXIT_STATUSES: Final[frozenset[int]] = frozenset({0, 14, 124})
REQUIRED_ACTIVE_AUXILIARY: Final[frozenset[str]] = frozenset(
    {
        "fluid/GammaAva",
        "fluid/runawayRate",
        "fluid/W_re",
        "scalar/energyloss_f_re",
        "scalar/radialloss_f_re",
    }
)
REQUIRED_NONZERO_OPERATORS: Final[frozenset[str]] = frozenset(
    {
        "Drr",
        "Dpp",
        "Dxx",
        "S_ava",
        "synchrotron_f1",
        "synchrotron_f2",
        "bremsstrahlung_f1",
        "nu_D_f1",
        "nu_D_f2",
    }
)
RADIAL_FACE_OPERATORS: Final[frozenset[str]] = frozenset({"Ar", "Drr"})
MOMENTUM_FACE_OPERATORS: Final[frozenset[str]] = frozenset(
    {
        "Ap1",
        "Dpp",
        "Dpx",
        "lnLambda_ee_f1",
        "lnLambda_ei_f1",
        "nu_D_f1",
        "nu_s_f1",
        "nu_par_f1",
        "synchrotron_f1",
        "bremsstrahlung_f1",
    }
)
PITCH_FACE_OPERATORS: Final[frozenset[str]] = frozenset(
    {
        "Ap2",
        "Dxp",
        "Dxx",
        "lnLambda_ee_f2",
        "lnLambda_ei_f2",
        "nu_D_f2",
        "nu_s_f2",
        "nu_par_f2",
        "synchrotron_f2",
    }
)
NATIVE_SCALAR_LIMITS: Final[dict[str, str]] = {
    "distribution_residual_max": "native_distribution_residual_max",
    "density_residual_max": "native_density_residual_max",
    "pitch_advection_reconstruction_max": "pitch_advection_reconstruction_max",
    "avalanche_source_reconstruction_max": "avalanche_source_reconstruction_max",
    "radial_loss_reconstruction_max": "radial_loss_reconstruction_max",
    "current_moment_reconstruction_max": "current_moment_reconstruction_max",
}
NATIVE_SEQUENCE_METRICS: Final[frozenset[str]] = frozenset(
    {
        "distribution_residual_relative_l2",
        "density_residual_relative_l2",
        "density_source_budgets",
        "pitch_advection_reconstruction_relative_l2",
        "avalanche_source_reconstruction_relative_l2",
        "pitch_scattering_particle_conservation",
        "radial_loss_reconstruction_relative_l2",
        "current_moment_reconstruction_relative_l2",
        "radiation_budgets",
    }
)
STATE_METRICS: Final[frozenset[str]] = frozenset(
    {
        "distribution_relative_l2",
        "density_relative_l2",
        "current_relative_l2",
        "growth_ratio_absolute_error",
    }
)
OPERATOR_METRICS: Final[frozenset[str]] = frozenset(
    {
        "radial_transport_Drr",
        "pitch_scattering_Dxx",
        "synchrotron_momentum",
        "bremsstrahlung_momentum",
        "partial_screening_nu_D",
    }
)
NATIVE_SEQUENCE_MAXIMA: Final[dict[str, str]] = {
    "distribution_residual_relative_l2": "distribution_residual_max",
    "density_residual_relative_l2": "density_residual_max",
    "pitch_advection_reconstruction_relative_l2": "pitch_advection_reconstruction_max",
    "avalanche_source_reconstruction_relative_l2": "avalanche_source_reconstruction_max",
    "pitch_scattering_particle_conservation": "pitch_scattering_particle_conservation_max",
    "radial_loss_reconstruction_relative_l2": "radial_loss_reconstruction_max",
    "current_moment_reconstruction_relative_l2": "current_moment_reconstruction_max",
}
DENSITY_BUDGET_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "radial_transport_m3_s",
        "avalanche_generation_m3_s",
        "external_source_m3_s",
        "total_tendency_m3_s",
        "finite_difference_m3_s",
    }
)
RADIATION_BUDGET_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "synchrotron_phase_space_rate",
        "bremsstrahlung_phase_space_rate",
    }
)
INITIAL_CURRENT_DEFECT_INTERPRETATION: Final[str] = (
    "the pinned DREAM commit saves zero initial j_re although the initialized "
    "f_re has nonzero parallel-current moment; every completed step "
    "reconstructs j_re within the declared threshold"
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_nonnegative_metrics(values: object, keys: frozenset[str]) -> bool:
    return bool(
        isinstance(values, dict)
        and frozenset(values) == keys
        and all(
            isinstance(values[key], (int, float))
            and np.isfinite(values[key])
            and values[key] >= 0.0
            for key in keys
        )
    )


def _is_finite_number(value: object, *, nonnegative: bool = False) -> bool:
    """Return whether one JSON scalar is finite and satisfies its sign contract."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    return bool(np.isfinite(value) and (not nonnegative or value >= 0.0))


def _is_finite_vector(value: object, *, length: int, nonnegative: bool = False) -> bool:
    """Validate an exact-length JSON numeric vector without accepting booleans."""

    return bool(
        isinstance(value, list)
        and len(value) == length
        and all(_is_finite_number(item, nonnegative=nonnegative) for item in value)
    )


def _density_budgets_are_complete(
    value: object,
    *,
    nt: int,
    nr: int,
    reported_residuals: list[float],
) -> bool:
    """Validate the saved radial, avalanche and total-density budget sequence."""

    if not isinstance(value, list) or len(value) != nt:
        return False
    radial_active = False
    avalanche_active = False
    for step, raw_budget in enumerate(value):
        if not isinstance(raw_budget, dict) or frozenset(raw_budget) != DENSITY_BUDGET_FIELDS:
            return False
        budget: dict[str, Any] = raw_budget
        if not all(_is_finite_vector(budget[field], length=nr) for field in DENSITY_BUDGET_FIELDS):
            return False
        radial = np.asarray(budget["radial_transport_m3_s"], dtype=np.float64)
        avalanche = np.asarray(budget["avalanche_generation_m3_s"], dtype=np.float64)
        external = np.asarray(budget["external_source_m3_s"], dtype=np.float64)
        total = np.asarray(budget["total_tendency_m3_s"], dtype=np.float64)
        finite_difference = np.asarray(budget["finite_difference_m3_s"], dtype=np.float64)
        if not (
            np.all(external == 0.0)
            and np.allclose(total, radial + avalanche + external, rtol=2.0e-15, atol=0.0)
        ):
            return False
        reconstructed_residual = float(
            np.linalg.norm(total - finite_difference)
            / max(float(np.linalg.norm(finite_difference)), 1.0)
        )
        if not np.isclose(
            reconstructed_residual,
            reported_residuals[step],
            rtol=2.0e-13,
            atol=1.0e-15,
        ):
            return False
        radial_active = radial_active or bool(np.any(radial != 0.0))
        avalanche_active = avalanche_active or bool(np.any(avalanche != 0.0))
    return radial_active and avalanche_active


def _radiation_budgets_are_complete(value: object, *, nt: int) -> bool:
    """Validate finite active synchrotron and bremsstrahlung budget sequences."""

    if not isinstance(value, list) or len(value) != nt:
        return False
    active = {field: False for field in RADIATION_BUDGET_FIELDS}
    for raw_budget in value:
        if not isinstance(raw_budget, dict) or frozenset(raw_budget) != RADIATION_BUDGET_FIELDS:
            return False
        for field in RADIATION_BUDGET_FIELDS:
            scalar = raw_budget[field]
            if not _is_finite_number(scalar):
                return False
            active[field] = active[field] or scalar != 0.0
    return all(active.values())


def _native_metrics_are_complete(value: dict[str, Any], *, nt: int, nr: int) -> bool:
    """Fail closed on malformed, non-finite or internally inconsistent native evidence."""

    metrics = value
    expected_keys = (
        NATIVE_SEQUENCE_METRICS
        | frozenset(NATIVE_SCALAR_LIMITS)
        | frozenset(
            {
                "pitch_scattering_particle_conservation_max",
                "initial_current_initialization_defect",
            }
        )
    )
    if frozenset(metrics) != expected_keys:
        return False

    sequences: dict[str, list[float]] = {}
    for sequence_name, maximum_name in NATIVE_SEQUENCE_MAXIMA.items():
        raw_sequence = metrics[sequence_name]
        if not _is_finite_vector(raw_sequence, length=nt, nonnegative=True):
            return False
        sequence: list[float] = raw_sequence
        sequences[sequence_name] = sequence
        maximum = metrics[maximum_name]
        if not (_is_finite_number(maximum, nonnegative=True) and maximum == max(sequence)):
            return False

    if not all(
        metrics[metric] <= THRESHOLDS[threshold]
        for metric, threshold in NATIVE_SCALAR_LIMITS.items()
    ):
        return False
    if not _density_budgets_are_complete(
        metrics["density_source_budgets"],
        nt=nt,
        nr=nr,
        reported_residuals=sequences["density_residual_relative_l2"],
    ):
        return False
    if not _radiation_budgets_are_complete(metrics["radiation_budgets"], nt=nt):
        return False

    raw_defect = metrics["initial_current_initialization_defect"]
    if not isinstance(raw_defect, dict):
        return False
    defect: dict[str, Any] = raw_defect
    return bool(
        frozenset(defect) == {"saved_j_re_norm", "distribution_moment_norm", "interpretation"}
        and defect["saved_j_re_norm"] == 0.0
        and _is_finite_number(defect["saved_j_re_norm"], nonnegative=True)
        and _is_finite_number(defect["distribution_moment_norm"], nonnegative=True)
        and defect["distribution_moment_norm"] > 0.0
        and defect["interpretation"] == INITIAL_CURRENT_DEFECT_INTERPRETATION
    )


def _repo_relative(path: Path) -> str:
    """Serialize tracked paths consistently for reproducible custody reports."""

    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _post_save_process_status(exit_status: int) -> str:
    """Describe the observed wrapper outcome after a complete HDF5 save."""

    return {
        0: "clean_exit",
        14: "MPI_Comm_get_attr_after_MPI_FINALIZE_abort",
        124: "outer_timeout_after_complete_save_during_MPI_FINALIZE_cleanup",
    }.get(exit_status, "unaccepted_process_exit")


def is_accepted_full_kinetic_parity(report: dict[str, Any]) -> bool:
    """Admit only the complete frozen DREAM parity-report contract."""

    gates = report.get("gates")
    custody = report.get("custody")
    reference = report.get("reference")
    execution = report.get("execution")
    native_parity = report.get("native_parity")
    convergence = report.get("convergence")
    if not (
        report.get("schema") == SCHEMA
        and report.get("all_pass") is True
        and report.get("thresholds") == THRESHOLDS
        and isinstance(gates, dict)
        and frozenset(gates) == REQUIRED_GATES
        and all(value is True for value in gates.values())
        and isinstance(custody, dict)
        and custody.get("dream_commit") == DREAM_COMMIT
        and custody.get("lock_schema") == "scpn-fusion.dream-full-kinetic-radial-lock.v2"
        and custody.get("convergence_family_revision") == 3
        and _is_sha256(custody.get("lock_sha256"))
        and _is_sha256(custody.get("deck_sha256"))
        and isinstance(reference, dict)
        and frozenset(reference) == CONVERGENCE_RESOLUTIONS
        and isinstance(execution, dict)
        and frozenset(execution) == CONVERGENCE_RESOLUTIONS
        and isinstance(native_parity, dict)
        and frozenset(native_parity) == CONVERGENCE_RESOLUTIONS
        and all(
            isinstance(native_parity[name], dict) and bool(native_parity[name])
            for name in CONVERGENCE_RESOLUTIONS
        )
        and isinstance(convergence, dict)
        and {
            "coarse_to_medium",
            "medium_to_fine",
            "state_improvement_ratio",
            "operator_improvement_ratio",
        }.issubset(convergence)
        and all(
            isinstance(convergence[name], dict) and bool(convergence[name])
            for name in (
                "coarse_to_medium",
                "medium_to_fine",
                "state_improvement_ratio",
                "operator_improvement_ratio",
            )
        )
    ):
        return False

    try:
        lock = json.loads((REPO_ROOT / LOCK_RELATIVE_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    dreami = custody.get("dreami")
    settings = custody.get("settings")
    native_source_provenance = custody.get("native_source_provenance")
    if not (
        isinstance(dreami, dict)
        and dreami.get("sha256") == DREAMI_SHA256
        and dreami.get("expected_sha256") == DREAMI_SHA256
        and isinstance(settings, dict)
        and frozenset(settings) == CONVERGENCE_RESOLUTIONS
        and custody.get("deck_path") == str(DECK_RELATIVE_PATH)
        and custody.get("lock_path") == str(LOCK_RELATIVE_PATH)
        and custody.get("deck_sha256") == sha256_file(REPO_ROOT / DECK_RELATIVE_PATH)
        and custody.get("lock_sha256") == sha256_file(REPO_ROOT / LOCK_RELATIVE_PATH)
        and isinstance(lock, dict)
        and lock.get("schema") == custody.get("lock_schema")
        and lock.get("convergence_family_revision") == custody.get("convergence_family_revision")
        and lock.get("dream_commit") == custody.get("dream_commit")
        and lock.get("thresholds") == THRESHOLDS
        and lock.get("deck_sha256") == custody.get("deck_sha256")
        and isinstance(native_source_provenance, dict)
        and frozenset(native_source_provenance)
        == frozenset(str(path) for path in NATIVE_SOURCE_PATHS)
        and all(
            native_source_provenance[str(path)] == sha256_file(REPO_ROOT / path)
            for path in NATIVE_SOURCE_PATHS
        )
    ):
        return False

    required_auxiliary = set(REQUIRED_AUXILIARY_QUANTITIES)
    for name in CONVERGENCE_RESOLUTIONS:
        setting = settings.get(name)
        output = reference.get(name)
        receipt = execution.get(name)
        locked_resolution = lock.get("resolutions", {}).get(name)
        if not (
            isinstance(setting, dict)
            and _is_sha256(setting.get("sha256"))
            and isinstance(locked_resolution, dict)
            and setting.get("sha256") == locked_resolution.get("settings_sha256")
            and isinstance(output, dict)
            and output.get("dream_commit") == DREAM_COMMIT
            and _is_sha256(output.get("sha256"))
            and tuple(output.get("requested_quantities", ())) == REQUESTED_OTHER_QUANTITIES
            and isinstance(output.get("auxiliary_diagnostics"), dict)
            and required_auxiliary.issubset(output["auxiliary_diagnostics"])
            and isinstance(output.get("operator_nonzero"), dict)
            and all(
                output["operator_nonzero"].get(operator) is True
                for operator in REQUIRED_NONZERO_OPERATORS
            )
            and isinstance(output.get("operator_shapes"), dict)
            and frozenset(output["operator_shapes"]) == frozenset(REQUIRED_RUNAWAY_QUANTITIES)
            and isinstance(receipt, dict)
            and receipt.get("saved_complete_output") is True
            and type(receipt.get("exit_status")) is int
            and receipt.get("exit_status") in ACCEPTED_PROCESS_EXIT_STATUSES
            and receipt.get("post_save_process_status")
            == _post_save_process_status(receipt["exit_status"])
            and isinstance(receipt.get("elapsed_wall_seconds"), (int, float))
            and np.isfinite(receipt["elapsed_wall_seconds"])
            and receipt["elapsed_wall_seconds"] > 0.0
        ):
            return False
        auxiliary = output["auxiliary_diagnostics"]
        for quantity in required_auxiliary:
            if quantity in ION_RATE_AUXILIARY_QUANTITIES:
                expected_shape = [
                    locked_resolution["nt"],
                    21,
                    locked_resolution["nr"],
                ]
            elif quantity.startswith("fluid/"):
                expected_shape = [
                    locked_resolution["nt"],
                    locked_resolution["nr"],
                ]
            else:
                expected_shape = [locked_resolution["nt"], 1]
            if not (
                isinstance(auxiliary[quantity], dict)
                and auxiliary[quantity].get("shape") == expected_shape
                and isinstance(auxiliary[quantity].get("nonzero"), bool)
            ):
                return False
        if not all(
            auxiliary[quantity].get("nonzero") is True for quantity in REQUIRED_ACTIVE_AUXILIARY
        ):
            return False
        for operator, shape in output["operator_shapes"].items():
            if operator in RADIAL_FACE_OPERATORS:
                expected_shape = [
                    locked_resolution["nt"],
                    locked_resolution["nr"] + 1,
                    locked_resolution["nxi"],
                    locked_resolution["np"],
                ]
            elif operator in MOMENTUM_FACE_OPERATORS:
                expected_shape = [
                    locked_resolution["nt"],
                    locked_resolution["nr"],
                    locked_resolution["nxi"],
                    locked_resolution["np"] + 1,
                ]
            elif operator in PITCH_FACE_OPERATORS:
                expected_shape = [
                    locked_resolution["nt"],
                    locked_resolution["nr"],
                    locked_resolution["nxi"] + 1,
                    locked_resolution["np"],
                ]
            else:
                expected_shape = [
                    locked_resolution["nt"],
                    locked_resolution["nr"],
                    locked_resolution["nxi"],
                    locked_resolution["np"],
                ]
            if shape != expected_shape:
                return False
        grid = output.get("grid")
        if not (
            isinstance(grid, dict)
            and all(isinstance(grid.get(axis), int) for axis in ("nr", "nxi", "np", "nt"))
            and all(grid[axis] > 1 for axis in ("nr", "nxi", "np"))
            and grid["nt"] > 0
            and grid == {axis: locked_resolution[axis] for axis in ("nr", "nxi", "np", "nt")}
            and isinstance(output.get("final_time_s"), (int, float))
            and output.get("final_time_s") == lock.get("case", {}).get("simulation_time_s")
        ):
            return False

        if not _native_metrics_are_complete(
            native_parity[name],
            nt=locked_resolution["nt"],
            nr=locked_resolution["nr"],
        ):
            return False

    coarse_medium = convergence["coarse_to_medium"]
    medium_fine = convergence["medium_to_fine"]
    state_improvement = convergence["state_improvement_ratio"]
    operator_improvement = convergence["operator_improvement_ratio"]
    return bool(
        frozenset(coarse_medium) == {"state", "operator"}
        and frozenset(medium_fine) == {"state", "operator"}
        and _finite_nonnegative_metrics(coarse_medium["state"], STATE_METRICS)
        and _finite_nonnegative_metrics(medium_fine["state"], STATE_METRICS)
        and _finite_nonnegative_metrics(coarse_medium["operator"], OPERATOR_METRICS)
        and _finite_nonnegative_metrics(medium_fine["operator"], OPERATOR_METRICS)
        and _finite_nonnegative_metrics(state_improvement, STATE_METRICS)
        and _finite_nonnegative_metrics(operator_improvement, OPERATOR_METRICS)
        and medium_fine["state"]["distribution_relative_l2"]
        <= THRESHOLDS["distribution_convergence_error_max"]
        and medium_fine["state"]["density_relative_l2"]
        <= THRESHOLDS["density_convergence_error_max"]
        and medium_fine["state"]["current_relative_l2"]
        <= THRESHOLDS["current_convergence_error_max"]
        and medium_fine["state"]["growth_ratio_absolute_error"]
        <= THRESHOLDS["growth_convergence_error_max"]
        and max(medium_fine["operator"].values()) <= THRESHOLDS["operator_convergence_error_max"]
        and max(state_improvement.values()) <= THRESHOLDS["convergence_improvement_ratio_max"]
        and max(operator_improvement.values()) <= THRESHOLDS["convergence_improvement_ratio_max"]
    )
