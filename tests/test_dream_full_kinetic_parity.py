# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — DREAM Full Kinetic Parity Tests
"""Real-surface tests for the frozen DREAM deck and native operator adapter."""

from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from validation import dream_full_kinetic_report_contract as report_contract
from validation.benchmark_dream_full_kinetic_parity import (
    CONVERGENCE_RESOLUTIONS,
    DECK_RELATIVE_PATH,
    DREAMI_SHA256,
    LOCK_RELATIVE_PATH,
    MOMENTUM_FACE_OPERATORS,
    NATIVE_SOURCE_PATHS,
    NATIVE_SCALAR_LIMITS,
    OPERATOR_METRICS,
    PITCH_FACE_OPERATORS,
    RADIAL_FACE_OPERATORS,
    REQUIRED_GATES,
    REQUIRED_NONZERO_OPERATORS,
    STATE_METRICS,
    THRESHOLDS,
    _interpolate,
    _post_save_process_status,
    _repo_relative,
    _same_encoded_scalar,
    is_accepted_full_kinetic_parity,
)
from validation.dream_full_kinetic_reference import (
    DreamFullKineticOutput,
    ELECTRON_CHARGE_C,
    ELECTRON_MASS_KG,
    SPEED_OF_LIGHT_M_PER_S,
    VACUUM_PERMITTIVITY_F_M,
    ION_RATE_AUXILIARY_QUANTITIES,
    REQUIRED_AUXILIARY_QUANTITIES,
    REQUIRED_RUNAWAY_QUANTITIES,
    sha256_file,
)
from validation.dream_full_kinetic_report_contract import (
    INITIAL_CURRENT_DEFECT_INTERPRETATION,
)
from validation.reference_data.dream.full_kinetic_radial_parity_deck import (
    DREAM_COMMIT,
    REQUESTED_OTHER_QUANTITIES,
    build_settings,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DREAM_ROOT = REPO_ROOT / "data/external/full_fidelity_public_sources/repos/dream"
DECK_PATH = REPO_ROOT / "validation/reference_data/dream/full_kinetic_radial_parity_deck.py"
LOCK_PATH = REPO_ROOT / "validation/reference_data/dream/full_kinetic_radial_parity_lock.json"


def _accepted_native_metrics(*, nt: int, nr: int) -> dict[str, Any]:
    """Build structurally complete native evidence for acceptance mutation tests."""

    zero_sequence = [0.0] * nt
    radial = [1.0, *([0.0] * (nr - 1))]
    avalanche = [2.0, *([0.0] * (nr - 1))]
    external = [0.0] * nr
    total = [3.0, *([0.0] * (nr - 1))]
    return {
        **{metric: 0.0 for metric in NATIVE_SCALAR_LIMITS},
        "distribution_residual_relative_l2": zero_sequence.copy(),
        "density_residual_relative_l2": zero_sequence.copy(),
        "density_source_budgets": [
            {
                "radial_transport_m3_s": radial.copy(),
                "avalanche_generation_m3_s": avalanche.copy(),
                "external_source_m3_s": external.copy(),
                "total_tendency_m3_s": total.copy(),
                "finite_difference_m3_s": total.copy(),
            }
            for _ in range(nt)
        ],
        "pitch_advection_reconstruction_relative_l2": zero_sequence.copy(),
        "avalanche_source_reconstruction_relative_l2": zero_sequence.copy(),
        "pitch_scattering_particle_conservation": zero_sequence.copy(),
        "pitch_scattering_particle_conservation_max": 0.0,
        "radial_loss_reconstruction_relative_l2": zero_sequence.copy(),
        "current_moment_reconstruction_relative_l2": zero_sequence.copy(),
        "initial_current_initialization_defect": {
            "saved_j_re_norm": 0.0,
            "distribution_moment_norm": 1.0,
            "interpretation": INITIAL_CURRENT_DEFECT_INTERPRETATION,
        },
        "radiation_budgets": [
            {
                "synchrotron_phase_space_rate": -1.0,
                "bremsstrahlung_phase_space_rate": -1.0,
            }
            for _ in range(nt)
        ],
    }


def _accepted_report_payload() -> dict[str, Any]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    references = {
        name: {
            "sha256": "d" * 64,
            "dream_commit": DREAM_COMMIT,
            "grid": {axis: lock["resolutions"][name][axis] for axis in ("nr", "nxi", "np", "nt")},
            "final_time_s": lock["case"]["simulation_time_s"],
            "requested_quantities": list(REQUESTED_OTHER_QUANTITIES),
            "auxiliary_diagnostics": {
                quantity: {
                    "shape": (
                        [
                            lock["resolutions"][name]["nt"],
                            21,
                            lock["resolutions"][name]["nr"],
                        ]
                        if quantity in ION_RATE_AUXILIARY_QUANTITIES
                        else [
                            lock["resolutions"][name]["nt"],
                            lock["resolutions"][name]["nr"],
                        ]
                        if quantity.startswith("fluid/")
                        else [lock["resolutions"][name]["nt"], 1]
                    ),
                    "nonzero": True,
                }
                for quantity in REQUIRED_AUXILIARY_QUANTITIES
            },
            "operator_nonzero": {operator: True for operator in REQUIRED_NONZERO_OPERATORS},
            "operator_shapes": {
                operator: (
                    [
                        lock["resolutions"][name]["nt"],
                        lock["resolutions"][name]["nr"] + 1,
                        lock["resolutions"][name]["nxi"],
                        lock["resolutions"][name]["np"],
                    ]
                    if operator in RADIAL_FACE_OPERATORS
                    else [
                        lock["resolutions"][name]["nt"],
                        lock["resolutions"][name]["nr"],
                        lock["resolutions"][name]["nxi"],
                        lock["resolutions"][name]["np"] + 1,
                    ]
                    if operator in MOMENTUM_FACE_OPERATORS
                    else [
                        lock["resolutions"][name]["nt"],
                        lock["resolutions"][name]["nr"],
                        lock["resolutions"][name]["nxi"] + 1,
                        lock["resolutions"][name]["np"],
                    ]
                    if operator in PITCH_FACE_OPERATORS
                    else [
                        lock["resolutions"][name]["nt"],
                        lock["resolutions"][name]["nr"],
                        lock["resolutions"][name]["nxi"],
                        lock["resolutions"][name]["np"],
                    ]
                )
                for operator in REQUIRED_RUNAWAY_QUANTITIES
            },
        }
        for name in CONVERGENCE_RESOLUTIONS
    }
    return {
        "schema": "scpn-fusion.dream-full-kinetic-parity.v1",
        "all_pass": True,
        "thresholds": THRESHOLDS,
        "gates": {name: True for name in REQUIRED_GATES},
        "custody": {
            "dream_commit": DREAM_COMMIT,
            "deck_path": str(DECK_RELATIVE_PATH),
            "lock_schema": "scpn-fusion.dream-full-kinetic-radial-lock.v2",
            "convergence_family_revision": 3,
            "lock_path": str(LOCK_RELATIVE_PATH),
            "lock_sha256": sha256_file(LOCK_PATH),
            "deck_sha256": sha256_file(DECK_PATH),
            "dreami": {
                "sha256": DREAMI_SHA256,
                "expected_sha256": DREAMI_SHA256,
            },
            "native_source_provenance": {
                str(path): sha256_file(REPO_ROOT / path) for path in NATIVE_SOURCE_PATHS
            },
            "settings": {
                name: {"sha256": lock["resolutions"][name]["settings_sha256"]}
                for name in CONVERGENCE_RESOLUTIONS
            },
        },
        "reference": references,
        "execution": {
            name: {
                "saved_complete_output": True,
                "exit_status": 124,
                "elapsed_wall_seconds": 1.0,
                "post_save_process_status": _post_save_process_status(124),
            }
            for name in CONVERGENCE_RESOLUTIONS
        },
        "native_parity": {
            name: _accepted_native_metrics(
                nt=lock["resolutions"][name]["nt"],
                nr=lock["resolutions"][name]["nr"],
            )
            for name in CONVERGENCE_RESOLUTIONS
        },
        "convergence": {
            "coarse_to_medium": {
                "state": {metric: 0.5 for metric in STATE_METRICS},
                "operator": {metric: 0.5 for metric in OPERATOR_METRICS},
            },
            "medium_to_fine": {
                "state": {metric: 0.0 for metric in STATE_METRICS},
                "operator": {metric: 0.0 for metric in OPERATOR_METRICS},
            },
            "state_improvement_ratio": {metric: 0.5 for metric in STATE_METRICS},
            "operator_improvement_ratio": {metric: 0.5 for metric in OPERATOR_METRICS},
        },
    }


@pytest.mark.skipif(not (DREAM_ROOT / "py/DREAM").is_dir(), reason="pinned DREAM absent")
def test_frozen_deck_enables_every_required_full_kinetic_term(tmp_path: Path) -> None:
    h5py = importlib.import_module("h5py")
    output = tmp_path / "output.h5"
    settings_path = tmp_path / "settings.h5"
    settings = build_settings(
        dream_root=DREAM_ROOT,
        resolution="coarse",
        output=output,
    )
    settings.save(str(settings_path))
    dreami = DREAM_ROOT / "build/iface/dreami"
    assert dreami.is_file()
    assert sha256_file(dreami) == DREAMI_SHA256

    with h5py.File(settings_path, "r") as handle:
        assert int(handle["radialgrid/nr"][0]) == 4
        assert int(handle["runawaygrid/nxi"][0]) == 16
        assert int(handle["runawaygrid/np"][0]) == 40
        assert int(handle["collisions/collfreq_type"][0]) == 3
        assert int(handle["collisions/bremsstrahlung_mode"][0]) == 2
        assert int(handle["eqsys/n_re/avalanche"][0]) == 4
        assert int(handle["eqsys/f_re/synchrotronmode"][0]) == 2
        assert int(handle["eqsys/f_re/transport/type"][0]) == 3
        assert float(handle["solver/tolerance/reltol"][0]) == 2.0e-8
        assert np.all(handle["solver/tolerance/reltols"][()] == 2.0e-8)
        requested = b"".join(handle["other/include"][()].tolist()).decode()
        assert tuple(requested.split(";")) == REQUESTED_OTHER_QUANTITIES


def test_real_dream_output_closes_native_full_distribution_residual() -> None:
    raw_path = os.environ.get("SCPN_DREAM_FULL_KINETIC_OUTPUT")
    if raw_path is None:
        pytest.skip("SCPN_DREAM_FULL_KINETIC_OUTPUT is not set")
    output = DreamFullKineticOutput.load(Path(raw_path))
    summary = output.summary()
    assert set(summary["operator_shapes"]) == set(REQUIRED_RUNAWAY_QUANTITIES)
    assert summary["operator_shapes"]["Drr"] == [
        output.times_s.size - 1,
        output.grid.nr + 1,
        output.grid.nxi,
        output.grid.np,
    ]
    assert summary["operator_shapes"]["S_ava"] == [
        output.times_s.size - 1,
        output.grid.nr,
        output.grid.nxi,
        output.grid.np,
    ]

    h5py = importlib.import_module("h5py")
    saved_auxiliary: set[str] = set()
    with h5py.File(output.path, "r") as handle:

        def collect_saved_auxiliary(name: str, item: object) -> None:
            if not name.startswith("runaway/") and hasattr(item, "shape"):
                saved_auxiliary.add(name)

        handle["other"].visititems(collect_saved_auxiliary)
    assert set(output.auxiliary_diagnostics) == saved_auxiliary

    assert output.commit == DREAM_COMMIT
    assert output.grid.nr > 1
    assert output.grid.nxi > 1
    assert output.grid.np > 1
    distribution_residuals = []
    density_residuals = []
    for step, dt_s in enumerate(np.diff(output.times_s)):
        tendencies = output.native_operator(step).evaluate(
            output.distribution[step + 1], output.density_m3[step + 1]
        )
        finite_difference = (output.distribution[step + 1] - output.distribution[step]) / dt_s
        numerator = np.sqrt(
            np.sum((tendencies.total - finite_difference) ** 2 * output.geometry.cell_measure)
        )
        denominator = np.sqrt(np.sum(finite_difference**2 * output.geometry.cell_measure))
        distribution_residuals.append(float(numerator / denominator))

        density_difference = (output.density_m3[step + 1] - output.density_m3[step]) / dt_s
        density_denominator = max(float(np.linalg.norm(density_difference)), 1.0)
        density_residuals.append(
            float(
                np.linalg.norm(tendencies.runaway_density_tendency_m3_s - density_difference)
                / density_denominator
            )
        )
        assert np.any(tendencies.runaway_density_radial_transport_m3_s != 0.0)
        assert np.any(tendencies.runaway_density_avalanche_generation_m3_s != 0.0)
        assert np.all(tendencies.runaway_density_external_source_m3_s == 0.0)
        np.testing.assert_allclose(
            tendencies.runaway_density_tendency_m3_s,
            tendencies.runaway_density_radial_transport_m3_s
            + tendencies.runaway_density_avalanche_generation_m3_s
            + tendencies.runaway_density_external_source_m3_s,
            rtol=0.0,
            atol=0.0,
        )
    assert max(distribution_residuals) <= THRESHOLDS["native_distribution_residual_max"]
    assert max(density_residuals) <= THRESHOLDS["native_density_residual_max"]
    assert np.any(output.auxiliary_diagnostics["scalar/radialloss_f_re"] != 0.0)
    assert np.all(output.auxiliary_diagnostics["scalar/radialloss_n_re"] == 0.0)

    momentum_cutoff_mc = float(output.grid.momentum_faces_mc[0])
    epsilon_mass_c = (
        4.0 * np.pi * VACUUM_PERMITTIVITY_F_M * ELECTRON_MASS_KG * SPEED_OF_LIGHT_M_PER_S
    )
    prefactor_m3_s = ELECTRON_CHARGE_C**4 / (epsilon_mass_c**2 * SPEED_OF_LIGHT_M_PER_S)
    expected_rate = (
        2.0
        * np.pi
        * prefactor_m3_s
        * (np.sqrt(1.0 + momentum_cutoff_mc**2) + 1.0)
        / momentum_cutoff_mc**2
        * output.total_electron_density_m3[1]
    )
    np.testing.assert_allclose(
        output.native_operator(0).coefficients.total_density_avalanche_rate_s_inv,
        expected_rate,
        rtol=2.0e-15,
        atol=0.0,
    )


def test_thresholds_require_all_full_fidelity_gates() -> None:
    assert THRESHOLDS == {
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


def test_acceptance_requires_the_complete_frozen_report_contract() -> None:
    valid = _accepted_report_payload()
    assert is_accepted_full_kinetic_parity(valid)

    rejected: list[dict[str, Any]] = []

    missing_gate = copy.deepcopy(valid)
    missing_gate["gates"].pop("growth_converged")
    rejected.append(missing_gate)

    extra_gate = copy.deepcopy(valid)
    extra_gate["gates"]["presence_only_shortcut"] = True
    rejected.append(extra_gate)

    relaxed_threshold = copy.deepcopy(valid)
    relaxed_threshold["thresholds"]["growth_convergence_error_max"] = 1.0
    rejected.append(relaxed_threshold)

    stale_family = copy.deepcopy(valid)
    stale_family["custody"]["convergence_family_revision"] = 2
    rejected.append(stale_family)

    wrong_binary = copy.deepcopy(valid)
    wrong_binary["custody"]["dreami"]["sha256"] = "0" * 64
    rejected.append(wrong_binary)

    detached_lock = copy.deepcopy(valid)
    detached_lock["custody"]["lock_sha256"] = "0" * 64
    rejected.append(detached_lock)

    stale_native_source = copy.deepcopy(valid)
    first_native_source = next(iter(stale_native_source["custody"]["native_source_provenance"]))
    stale_native_source["custody"]["native_source_provenance"][first_native_source] = "0" * 64
    rejected.append(stale_native_source)

    missing_resolution = copy.deepcopy(valid)
    missing_resolution["reference"].pop("fine")
    rejected.append(missing_resolution)

    empty_native_metrics = copy.deepcopy(valid)
    empty_native_metrics["native_parity"]["medium"] = {}
    rejected.append(empty_native_metrics)

    extra_native_metric = copy.deepcopy(valid)
    extra_native_metric["native_parity"]["medium"]["presence_only"] = [0.0]
    rejected.append(extra_native_metric)

    nonfinite_native_sequence = copy.deepcopy(valid)
    nonfinite_native_sequence["native_parity"]["medium"]["distribution_residual_relative_l2"][0] = (
        float("nan")
    )
    rejected.append(nonfinite_native_sequence)

    nonnumeric_native_sequence = copy.deepcopy(valid)
    nonnumeric_native_sequence["native_parity"]["medium"]["distribution_residual_relative_l2"][
        0
    ] = True
    rejected.append(nonnumeric_native_sequence)

    shortened_native_sequence = copy.deepcopy(valid)
    shortened_native_sequence["native_parity"]["fine"][
        "current_moment_reconstruction_relative_l2"
    ].pop()
    rejected.append(shortened_native_sequence)

    inconsistent_native_maximum = copy.deepcopy(valid)
    inconsistent_native_maximum["native_parity"]["coarse"]["distribution_residual_max"] = (
        THRESHOLDS["native_distribution_residual_max"] / 2.0
    )
    rejected.append(inconsistent_native_maximum)

    over_threshold_native_sequence = copy.deepcopy(valid)
    excessive_residual = THRESHOLDS["native_distribution_residual_max"] * 2.0
    over_threshold_native_sequence["native_parity"]["coarse"][
        "distribution_residual_relative_l2"
    ] = [excessive_residual] * valid["reference"]["coarse"]["grid"]["nt"]
    over_threshold_native_sequence["native_parity"]["coarse"]["distribution_residual_max"] = (
        excessive_residual
    )
    rejected.append(over_threshold_native_sequence)

    missing_density_budget_sequence = copy.deepcopy(valid)
    missing_density_budget_sequence["native_parity"]["medium"]["density_source_budgets"] = []
    rejected.append(missing_density_budget_sequence)

    malformed_density_budget = copy.deepcopy(valid)
    malformed_density_budget["native_parity"]["medium"]["density_source_budgets"][0] = []
    rejected.append(malformed_density_budget)

    malformed_density_vector = copy.deepcopy(valid)
    malformed_density_vector["native_parity"]["medium"]["density_source_budgets"][0][
        "radial_transport_m3_s"
    ].pop()
    rejected.append(malformed_density_vector)

    inconsistent_density_residual = copy.deepcopy(valid)
    inconsistent_density_residual["native_parity"]["medium"]["density_source_budgets"][0][
        "finite_difference_m3_s"
    ][0] = 4.0
    rejected.append(inconsistent_density_residual)

    inactive_radial_budget = copy.deepcopy(valid)
    for budget in inactive_radial_budget["native_parity"]["medium"]["density_source_budgets"]:
        budget["radial_transport_m3_s"] = [0.0] * valid["reference"]["medium"]["grid"]["nr"]
        budget["total_tendency_m3_s"] = budget["avalanche_generation_m3_s"].copy()
        budget["finite_difference_m3_s"] = budget["avalanche_generation_m3_s"].copy()
    rejected.append(inactive_radial_budget)

    inconsistent_density_budget = copy.deepcopy(valid)
    inconsistent_density_budget["native_parity"]["fine"]["density_source_budgets"][0][
        "external_source_m3_s"
    ][0] = 1.0
    rejected.append(inconsistent_density_budget)

    inactive_radiation_budget = copy.deepcopy(valid)
    for budget in inactive_radiation_budget["native_parity"]["coarse"]["radiation_budgets"]:
        budget["synchrotron_phase_space_rate"] = 0.0
    rejected.append(inactive_radiation_budget)

    missing_radiation_budget_sequence = copy.deepcopy(valid)
    missing_radiation_budget_sequence["native_parity"]["coarse"]["radiation_budgets"] = []
    rejected.append(missing_radiation_budget_sequence)

    malformed_radiation_budget = copy.deepcopy(valid)
    malformed_radiation_budget["native_parity"]["coarse"]["radiation_budgets"][0]["unexpected"] = (
        0.0
    )
    rejected.append(malformed_radiation_budget)

    nonnumeric_radiation_budget = copy.deepcopy(valid)
    nonnumeric_radiation_budget["native_parity"]["coarse"]["radiation_budgets"][0][
        "synchrotron_phase_space_rate"
    ] = True
    rejected.append(nonnumeric_radiation_budget)

    invalid_initial_current_defect = copy.deepcopy(valid)
    invalid_initial_current_defect["native_parity"]["medium"][
        "initial_current_initialization_defect"
    ]["distribution_moment_norm"] = 0.0
    rejected.append(invalid_initial_current_defect)

    malformed_initial_current_defect = copy.deepcopy(valid)
    malformed_initial_current_defect["native_parity"]["medium"][
        "initial_current_initialization_defect"
    ] = []
    rejected.append(malformed_initial_current_defect)

    incomplete_requested_outputs = copy.deepcopy(valid)
    incomplete_requested_outputs["reference"]["coarse"]["requested_quantities"].pop()
    rejected.append(incomplete_requested_outputs)

    incomplete_auxiliary_outputs = copy.deepcopy(valid)
    incomplete_auxiliary_outputs["reference"]["medium"]["auxiliary_diagnostics"].pop(
        "scalar/radialloss_f_re"
    )
    rejected.append(incomplete_auxiliary_outputs)

    inactive_required_auxiliary = copy.deepcopy(valid)
    inactive_required_auxiliary["reference"]["medium"]["auxiliary_diagnostics"][
        "scalar/radialloss_f_re"
    ]["nonzero"] = False
    rejected.append(inactive_required_auxiliary)

    collapsed_radial_auxiliary = copy.deepcopy(valid)
    collapsed_radial_auxiliary["reference"]["medium"]["auxiliary_diagnostics"]["fluid/GammaAva"][
        "shape"
    ] = [24, 1]
    rejected.append(collapsed_radial_auxiliary)

    missing_operator = copy.deepcopy(valid)
    missing_operator["reference"]["fine"]["operator_nonzero"]["S_ava"] = False
    rejected.append(missing_operator)

    collapsed_radial_operator = copy.deepcopy(valid)
    collapsed_radial_operator["reference"]["medium"]["operator_shapes"]["Drr"] = [
        24,
        1,
        48,
        120,
    ]
    rejected.append(collapsed_radial_operator)

    incomplete_grid = copy.deepcopy(valid)
    incomplete_grid["reference"]["fine"]["grid"]["nxi"] = 1
    rejected.append(incomplete_grid)

    incomplete_execution = copy.deepcopy(valid)
    incomplete_execution["execution"]["fine"]["saved_complete_output"] = False
    rejected.append(incomplete_execution)

    unrecognized_exit = copy.deepcopy(valid)
    unrecognized_exit["execution"]["fine"]["exit_status"] = 137
    unrecognized_exit["execution"]["fine"]["post_save_process_status"] = _post_save_process_status(
        137
    )
    rejected.append(unrecognized_exit)

    invalid_elapsed_time = copy.deepcopy(valid)
    invalid_elapsed_time["execution"]["coarse"]["elapsed_wall_seconds"] = float("nan")
    rejected.append(invalid_elapsed_time)

    excessive_native_residual = copy.deepcopy(valid)
    excessive_native_residual["native_parity"]["coarse"]["distribution_residual_max"] = (
        THRESHOLDS["native_distribution_residual_max"] * 2.0
    )
    rejected.append(excessive_native_residual)

    partial_convergence = copy.deepcopy(valid)
    partial_convergence["convergence"]["medium_to_fine"]["operator"].pop("radial_transport_Drr")
    rejected.append(partial_convergence)

    failed_growth_metric = copy.deepcopy(valid)
    failed_growth_metric["convergence"]["medium_to_fine"]["state"][
        "growth_ratio_absolute_error"
    ] = THRESHOLDS["growth_convergence_error_max"] * 2.0
    rejected.append(failed_growth_metric)

    for report in rejected:
        assert not is_accepted_full_kinetic_parity(report)


def test_acceptance_fails_closed_when_custody_lock_is_unreadable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    valid = _accepted_report_payload()
    missing_lock = tmp_path / "missing-lock.json"
    monkeypatch.setattr(report_contract, "LOCK_RELATIVE_PATH", missing_lock)

    assert not is_accepted_full_kinetic_parity(valid)


def test_interpolation_snaps_only_roundoff_sized_endpoint_drift() -> None:
    source_axes = (
        np.array([0.0, 0.11, np.nextafter(0.22, 0.0)]),
        np.array([-1.0, 0.0, 1.0]),
        np.array([0.02, 4.01, 8.0]),
    )
    source = np.add.outer(
        np.add.outer(source_axes[0], source_axes[1]),
        source_axes[2],
    )
    target_axes = (
        np.array([0.0, 0.22]),
        np.array([-1.0, 1.0]),
        np.array([0.02, 8.0]),
    )

    interpolated = _interpolate(source_axes, source, target_axes)

    assert interpolated.shape == (2, 2, 2)
    assert np.all(np.isfinite(interpolated))
    assert interpolated[-1, -1, -1] == pytest.approx(
        source_axes[0][-1] + source_axes[1][-1] + source_axes[2][-1]
    )

    outside_axes = (np.array([0.0, 0.220001]), target_axes[1], target_axes[2])
    with pytest.raises(ValueError, match="outside source interpolation domain"):
        _interpolate(source_axes, source, outside_axes)


def test_serialized_scalar_comparison_accepts_ulps_not_material_drift() -> None:
    assert _same_encoded_scalar(np.nextafter(0.22, 0.0), 0.22)
    assert _same_encoded_scalar(np.nextafter(8.0, np.inf), 8.0)
    assert not _same_encoded_scalar(0.220001, 0.22)


def test_custody_paths_are_canonical_inside_the_repository(tmp_path: Path) -> None:
    assert _repo_relative(DECK_PATH) == str(DECK_RELATIVE_PATH)
    external = tmp_path / "external.h5"
    assert _repo_relative(external) == str(external)
    assert _post_save_process_status(0) == "clean_exit"
    assert "MPI_FINALIZE" in _post_save_process_status(14)
    assert "complete_save" in _post_save_process_status(124)
    assert _post_save_process_status(137) == "unaccepted_process_exit"


def test_lock_freezes_deck_thresholds_and_three_resolutions_before_fine_output() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    assert lock["schema"] == "scpn-fusion.dream-full-kinetic-radial-lock.v2"
    assert lock["dream_commit"] == DREAM_COMMIT
    assert lock["deck_sha256"] == sha256_file(DECK_PATH)
    assert lock["thresholds"] == THRESHOLDS
    assert lock["frozen_before_fine_output"] is True
    assert lock["relocked_before_ultrafine_output"] is True
    assert lock["prior_family_receipt"]["all_pass"] is False
    assert set(lock["resolutions"]) == {"coarse", "medium", "fine"}
