# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — DREAM Full Kinetic Parity Benchmark
"""Benchmark full DREAM and native radius-momentum-pitch operator fidelity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from validation.dream_full_kinetic_metrics import (
    _interpolate,
    _native_metrics,
    _operator_convergence,
    _ratio,
    _same_encoded_scalar,
    _state_convergence,
)
from validation.dream_full_kinetic_reference import (
    DreamFullKineticOutput,
    REQUIRED_AUXILIARY_QUANTITIES,
    sha256_file,
)
from validation.dream_full_kinetic_report_contract import (
    ACCEPTED_PROCESS_EXIT_STATUSES,
    CONVERGENCE_RESOLUTIONS,
    DECK_RELATIVE_PATH,
    DREAMI_SHA256,
    LOCK_RELATIVE_PATH,
    MOMENTUM_FACE_OPERATORS,
    NATIVE_SCALAR_LIMITS,
    NATIVE_SEQUENCE_METRICS,
    NATIVE_SOURCE_PATHS,
    OPERATOR_METRICS,
    PITCH_FACE_OPERATORS,
    RADIAL_FACE_OPERATORS,
    REPO_ROOT,
    REQUIRED_GATES,
    REQUIRED_NONZERO_OPERATORS,
    SCHEMA,
    STATE_METRICS,
    THRESHOLDS,
    _post_save_process_status,
    _repo_relative,
    is_accepted_full_kinetic_parity,
)
from validation.reference_data.dream.full_kinetic_radial_parity_deck import (
    REQUESTED_OTHER_QUANTITIES,
)


__all__ = [
    "CONVERGENCE_RESOLUTIONS",
    "DECK_RELATIVE_PATH",
    "DREAMI_SHA256",
    "LOCK_RELATIVE_PATH",
    "MOMENTUM_FACE_OPERATORS",
    "NATIVE_SCALAR_LIMITS",
    "NATIVE_SEQUENCE_METRICS",
    "NATIVE_SOURCE_PATHS",
    "OPERATOR_METRICS",
    "PITCH_FACE_OPERATORS",
    "RADIAL_FACE_OPERATORS",
    "REQUIRED_GATES",
    "REQUIRED_NONZERO_OPERATORS",
    "STATE_METRICS",
    "THRESHOLDS",
    "_interpolate",
    "_post_save_process_status",
    "_repo_relative",
    "_same_encoded_scalar",
    "benchmark",
    "is_accepted_full_kinetic_parity",
    "main",
]


def benchmark(
    *,
    coarse_path: Path,
    medium_path: Path,
    fine_path: Path,
    lock_path: Path,
    deck_path: Path,
    coarse_settings_path: Path,
    medium_settings_path: Path,
    fine_settings_path: Path,
    coarse_wall_seconds: float,
    medium_wall_seconds: float,
    fine_wall_seconds: float,
    coarse_exit_status: int,
    medium_exit_status: int,
    fine_exit_status: int,
    upstream_settings_path: Path,
    upstream_output_path: Path,
    dreami_path: Path,
) -> dict[str, Any]:
    """Run every custody, physics, native-parity and convergence gate."""

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    settings_paths = {
        "coarse": coarse_settings_path,
        "medium": medium_settings_path,
        "fine": fine_settings_path,
    }
    custody = {
        "lock_path": _repo_relative(lock_path),
        "lock_sha256": sha256_file(lock_path),
        "lock_schema": lock.get("schema"),
        "convergence_family_revision": lock.get("convergence_family_revision"),
        "prior_family_receipt": lock.get("prior_family_receipt"),
        "aborted_preoutput_receipt": lock.get("aborted_preoutput_receipt"),
        "dream_commit": lock.get("dream_commit"),
        "deck_path": _repo_relative(deck_path),
        "deck_sha256": sha256_file(deck_path),
        "settings": {
            name: {
                "artifact_filename": path.name,
                "sha256": sha256_file(path),
            }
            for name, path in settings_paths.items()
        },
        "dreami": {
            "path": str(dreami_path),
            "sha256": sha256_file(dreami_path),
            "expected_sha256": DREAMI_SHA256,
            "build_configuration": "Release/O3",
            "petsc_version": "3.25.3",
            "hdf5_version": "2.1.0",
            "gsl_version": "2.8",
            "mpi_enabled": True,
        },
        "native_source_provenance": {
            str(path): sha256_file(REPO_ROOT / path) for path in NATIVE_SOURCE_PATHS
        },
    }
    coarse = DreamFullKineticOutput.load(coarse_path)
    medium = DreamFullKineticOutput.load(medium_path)
    fine = DreamFullKineticOutput.load(fine_path)
    outputs = {"coarse": coarse, "medium": medium, "fine": fine}
    lock_matches = bool(
        lock.get("schema") == "scpn-fusion.dream-full-kinetic-radial-lock.v2"
        and all(lock.get("dream_commit") == output.commit for output in outputs.values())
        and lock.get("thresholds") == THRESHOLDS
        and lock.get("deck_sha256") == custody["deck_sha256"]
        and all(
            lock.get("resolutions", {}).get(name, {}).get("settings_sha256")
            == custody["settings"][name]["sha256"]
            for name in settings_paths
        )
        and all(
            {
                "nr": output.grid.nr,
                "nxi": output.grid.nxi,
                "np": output.grid.np,
                "nt": output.times_s.size - 1,
            }
            == {key: lock["resolutions"][name][key] for key in ("nr", "nxi", "np", "nt")}
            for name, output in outputs.items()
        )
        and all(
            float(output.times_s[-1]) == lock["case"]["simulation_time_s"]
            for output in outputs.values()
        )
    )
    case_matches = all(
        np.allclose(
            output.electric_field_v_m,
            lock["case"]["electric_field_v_per_m"],
            rtol=0.0,
            atol=0.0,
        )
        and _same_encoded_scalar(
            float(output.grid.radius_faces_m[-1]),
            float(lock["case"]["minor_radius_m"]),
        )
        and _same_encoded_scalar(
            float(output.grid.momentum_faces_mc[0]),
            float(lock["case"]["p_min_mc"]),
        )
        and _same_encoded_scalar(
            float(output.grid.momentum_faces_mc[-1]),
            float(lock["case"]["p_max_mc"]),
        )
        and _same_encoded_scalar(
            float(output.case_settings["magnetic_field_t"]),
            float(lock["case"]["magnetic_field_t"]),
        )
        and _same_encoded_scalar(
            float(output.case_settings["cold_temperature_ev"]),
            float(lock["case"]["cold_temperature_ev"]),
        )
        and _same_encoded_scalar(
            float(output.case_settings["magnetic_perturbation"]),
            float(lock["case"]["magnetic_perturbation"]),
        )
        and output.case_settings["ion_atomic_numbers"] == [1.0, 18.0]
        and _same_encoded_scalar(
            float(output.case_settings["prescribed_ion_charge_state_density_m3"][3]),
            float(lock["case"]["argon_density_m3"]),
        )
        and _same_encoded_scalar(
            float(
                sum(
                    output.case_settings["prescribed_ion_charge_state_density_m3"][index] * charge
                    for index, charge in ((1, 1), (3, 1))
                )
            ),
            float(lock["case"]["free_electron_density_m3"]),
        )
        for output in outputs.values()
    )

    native = {name: _native_metrics(output) for name, output in outputs.items()}

    coarse_medium_state = _state_convergence(coarse, medium)
    medium_fine_state = _state_convergence(medium, fine)
    coarse_medium_operator = _operator_convergence(coarse, medium)
    medium_fine_operator = _operator_convergence(medium, fine)
    improvement = {
        name: _ratio(medium_fine_state[name], coarse_medium_state[name])
        for name in coarse_medium_state
    }
    operator_improvement = {
        name: _ratio(medium_fine_operator[name], coarse_medium_operator[name])
        for name in coarse_medium_operator
    }

    gates = {
        "pinned_dreami_binary": custody["dreami"]["sha256"] == custody["dreami"]["expected_sha256"],
        "frozen_custody_lock": lock_matches,
        "frozen_case_reconstructed": case_matches,
        "complete_execution_receipts": all(
            wall_seconds > 0.0
            and np.isfinite(wall_seconds)
            and exit_status in ACCEPTED_PROCESS_EXIT_STATUSES
            for wall_seconds, exit_status in (
                (coarse_wall_seconds, coarse_exit_status),
                (medium_wall_seconds, medium_exit_status),
                (fine_wall_seconds, fine_exit_status),
            )
        ),
        "all_axes_evolved": all(
            output.grid.nr > 1 and output.grid.nxi > 1 and output.grid.np > 1
            for output in outputs.values()
        ),
        "all_requested_auxiliary_outputs_present": all(
            output.requested_quantities == REQUESTED_OTHER_QUANTITIES
            and set(REQUIRED_AUXILIARY_QUANTITIES).issubset(output.auxiliary_diagnostics)
            for output in outputs.values()
        ),
        "all_configured_active_auxiliary_outputs_nonzero": all(
            all(
                np.any(output.auxiliary_diagnostics[name] != 0.0)
                for name in (
                    "fluid/GammaAva",
                    "fluid/runawayRate",
                    "fluid/W_re",
                    "scalar/energyloss_f_re",
                    "scalar/radialloss_f_re",
                )
            )
            for output in outputs.values()
        ),
        "all_required_operators_nonzero": all(
            all(
                np.any(output.coefficients[name] != 0.0)
                for name in (
                    "Drr",
                    "Dpp",
                    "Dxx",
                    "S_ava",
                    "synchrotron_f1",
                    "synchrotron_f2",
                    "bremsstrahlung_f1",
                    "nu_D_f1",
                    "nu_D_f2",
                )
            )
            for output in outputs.values()
        ),
        "native_distribution_residual": max(
            metrics["distribution_residual_max"] for metrics in native.values()
        )
        <= THRESHOLDS["native_distribution_residual_max"],
        "native_density_residual": max(
            metrics["density_residual_max"] for metrics in native.values()
        )
        <= THRESHOLDS["native_density_residual_max"],
        "pitch_advection_reconstruction": max(
            metrics["pitch_advection_reconstruction_max"] for metrics in native.values()
        )
        <= THRESHOLDS["pitch_advection_reconstruction_max"],
        "avalanche_source_reconstruction": max(
            metrics["avalanche_source_reconstruction_max"] for metrics in native.values()
        )
        <= THRESHOLDS["avalanche_source_reconstruction_max"],
        "radial_loss_reconstruction": max(
            metrics["radial_loss_reconstruction_max"] for metrics in native.values()
        )
        <= THRESHOLDS["radial_loss_reconstruction_max"],
        "current_moment_reconstruction": max(
            metrics["current_moment_reconstruction_max"] for metrics in native.values()
        )
        <= THRESHOLDS["current_moment_reconstruction_max"],
        "distribution_converged": medium_fine_state["distribution_relative_l2"]
        <= THRESHOLDS["distribution_convergence_error_max"],
        "density_converged": medium_fine_state["density_relative_l2"]
        <= THRESHOLDS["density_convergence_error_max"],
        "current_converged": medium_fine_state["current_relative_l2"]
        <= THRESHOLDS["current_convergence_error_max"],
        "growth_converged": medium_fine_state["growth_ratio_absolute_error"]
        <= THRESHOLDS["growth_convergence_error_max"],
        "operator_coefficients_converged": max(medium_fine_operator.values())
        <= THRESHOLDS["operator_convergence_error_max"],
        "state_convergence_improves": max(improvement.values())
        <= THRESHOLDS["convergence_improvement_ratio_max"],
        "operator_convergence_improves": max(operator_improvement.values())
        <= THRESHOLDS["convergence_improvement_ratio_max"],
    }
    return {
        "schema": SCHEMA,
        "thresholds": THRESHOLDS,
        "custody": custody,
        "execution": {
            name: {
                "elapsed_wall_seconds": wall_seconds,
                "exit_status": exit_status,
                "saved_complete_output": True,
                "post_save_process_status": _post_save_process_status(exit_status),
            }
            for name, wall_seconds, exit_status in (
                ("coarse", coarse_wall_seconds, coarse_exit_status),
                ("medium", medium_wall_seconds, medium_exit_status),
                ("fine", fine_wall_seconds, fine_exit_status),
            )
        },
        "pinned_commit_adapter_notes": {
            "avalanche_source": (
                "OtherQuantityHandler retains the previous S_ava buffer before "
                "adding the next source; first differences recover the per-step "
                "source used by the equation system"
            ),
            "total_density_avalanche_source": (
                "src/Settings/RunawaySourceTerms.cpp assigns fluid-mode "
                "AvalancheSourceRP to n_re, and "
                "src/Equations/Kinetic/AvalancheSourceRP.cpp supplies its exact "
                "closed-form momentum integral above the positive runaway cutoff. It is "
                "evaluated independently of the finite f_re-grid source integral. This "
                "source-code-derived split corrects the native adapter only; it changes "
                "no frozen DREAM output, case, tolerance or threshold"
            ),
            "total_density_radial_transport": (
                "src/Settings/Equations/n_re.cpp gives n_re the momentum-pitch integral "
                "of the kinetic f_re radial transport term. scalar/radialloss_f_re is "
                "therefore active; "
                "scalar/radialloss_n_re is zero because direct fluid n_re transport "
                "is disabled in the frozen deck, not because radial transport is omitted"
            ),
            "runaway_rate_diagnostic": (
                "fluid/runawayRate is requested and retained as an independent DREAM "
                "diagnostic, but is not reused to construct the native parity operator"
            ),
            "initial_current": (
                "the saved initial j_re is zero although initialized f_re has a "
                "nonzero parallel-current moment; completed steps reconstruct normally"
            ),
            "process_cleanup": (
                "dreami saves complete output before its MPI attribute query after "
                "MPI_Finalize returns exit status 14"
            ),
        },
        "reference": {name: output.summary() for name, output in outputs.items()},
        "native_parity": native,
        "convergence": {
            "coarse_to_medium": {
                "state": coarse_medium_state,
                "operator": coarse_medium_operator,
            },
            "medium_to_fine": {
                "state": medium_fine_state,
                "operator": medium_fine_operator,
            },
            "state_improvement_ratio": improvement,
            "operator_improvement_ratio": operator_improvement,
        },
        "upstream_2kinetic_baseline": {
            "settings_sha256": sha256_file(upstream_settings_path),
            "output_sha256": sha256_file(upstream_output_path),
            "elapsed_wall_seconds": 2648.89,
            "maximum_resident_kib": 3792420,
            "exit_status": 1,
            "saved_times_s": [0.0],
            "failure": "negative n_cold followed by GSL infinite-or-NaN value",
            "interpretation": (
                "authenticated first-step numerical failure; not a physics divergence certificate"
            ),
        },
        "gates": gates,
        "all_pass": all(gates.values()),
    }


def _markdown(result: dict[str, Any]) -> str:
    status = "PASS" if result["all_pass"] else "FAIL"
    lines = [
        "# DREAM Full Kinetic Radius-Momentum-Pitch Parity",
        "",
        f"Overall: **{status}**",
        "",
        "The benchmark evolves radius, momentum and pitch and retains kinetic "
        "avalanche generation, partial screening, pitch scattering, synchrotron "
        "and bremsstrahlung losses, radial transport and explicit operator budgets.",
        "",
        "## Gates",
        "",
    ]
    for name, passed in result["gates"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.extend(
        [
            "",
            "## Authenticated executions",
            "",
            "| Resolution | Grid (r, xi, p, t) | Output SHA-256 | Wall [s] | Exit | "
            "n_re growth | Native f residual | Native n_re residual |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in ("coarse", "medium", "fine"):
        reference = result["reference"][name]
        grid = reference["grid"]
        execution = result["execution"][name]
        native = result["native_parity"][name]
        lines.append(
            f"| {name} | {grid['nr']}, {grid['nxi']}, {grid['np']}, {grid['nt']} | "
            f"`{reference['sha256']}` | {execution['elapsed_wall_seconds']:.3f} | "
            f"{execution['exit_status']} | {reference['runaway_density_growth_ratio']:.9g} | "
            f"{native['distribution_residual_max']:.6e} | "
            f"{native['density_residual_max']:.6e} |"
        )
    lines.extend(
        [
            "",
            f"Pinned `dreami` SHA-256: `{result['custody']['dreami']['sha256']}`.",
            "",
            "Observed exit 14, or wrapper exit 124 after the complete save, can result "
            "from the pinned binary querying MPI attributes after `MPI_Finalize`; output "
            "completeness is validated independently before either outcome is admitted.",
            "The pinned commit also saves zero initial `j_re` while its initialized `f_re` "
            "has a nonzero current moment. This initialization defect is retained in JSON; "
            "all completed-step current moments are compared to `j_re` at the frozen gate.",
            "The same commit accumulates saved `S_ava` buffers; the adapter takes exact "
            "first differences before comparing the kinetic avalanche source.",
            "The total-density avalanche source is reconstructed separately from DREAM's "
            "closed-form fluid-mode momentum integral above the frozen runaway cutoff; it "
            "is not approximated by integrating the finite kinetic source grid.",
            "The `n_re` radial budget is the momentum-pitch integral of kinetic `f_re` "
            "transport. Accordingly `scalar/radialloss_f_re` is nonzero while "
            "`scalar/radialloss_n_re` is zero because no separate direct fluid transport "
            "term is configured.",
            "",
            "## State convergence",
            "",
            "| Observable | Coarse to medium | Medium to fine | Improvement ratio |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    convergence = result["convergence"]
    for name in convergence["coarse_to_medium"]["state"]:
        lines.append(
            f"| `{name}` | {convergence['coarse_to_medium']['state'][name]:.6e} | "
            f"{convergence['medium_to_fine']['state'][name]:.6e} | "
            f"{convergence['state_improvement_ratio'][name]:.6e} |"
        )
    lines.extend(
        [
            "",
            "## Operator-coefficient convergence",
            "",
            "| Operator | Coarse to medium | Medium to fine | Improvement ratio |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name in convergence["coarse_to_medium"]["operator"]:
        lines.append(
            f"| `{name}` | {convergence['coarse_to_medium']['operator'][name]:.6e} | "
            f"{convergence['medium_to_fine']['operator'][name]:.6e} | "
            f"{convergence['operator_improvement_ratio'][name]:.6e} |"
        )
    upstream = result["upstream_2kinetic_baseline"]
    requested = result["reference"]["coarse"]["requested_quantities"]
    lines.extend(
        [
            "",
            "## Physics and diagnostics custody",
            "",
            "The evolved operator includes kinetic avalanche generation, full partial "
            "screening, momentum and pitch diffusion, synchrotron and bremsstrahlung "
            "radiation reaction, and kinetic Rechester-Rosenbluth radial transport.",
            "",
            f"Requested DREAM diagnostic groups: `{'; '.join(requested)}`.",
            "Every saved non-runaway auxiliary dataset is retained in the JSON "
            f"custody record ({len(result['reference']['coarse']['auxiliary_diagnostics'])} "
            "datasets for each resolution), alongside every explicitly requested "
            "runaway coefficient.",
            "Zero `Ar`, `Dpx`, and `Dxp` values are retained and reported; they are the "
            "configured cylindrical-geometry/operator result, not omitted channels.",
            "",
            "## Upstream 2kinetic baseline receipt",
            "",
            f"- Settings SHA-256: `{upstream['settings_sha256']}`",
            f"- Partial output SHA-256: `{upstream['output_sha256']}`",
            f"- Wall time: `{upstream['elapsed_wall_seconds']:.2f} s`",
            f"- Exit status: `{upstream['exit_status']}`",
            f"- Failure: `{upstream['failure']}`",
            f"- Interpretation: {upstream['interpretation']}.",
            "",
            "Raw DREAM outputs remain external artifacts; the JSON report binds "
            "their SHA-256 values, exact pinned commit, grids, metrics and failure receipt.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--medium", type=Path, required=True)
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--coarse-settings", type=Path, required=True)
    parser.add_argument("--medium-settings", type=Path, required=True)
    parser.add_argument("--fine-settings", type=Path, required=True)
    parser.add_argument("--coarse-wall-seconds", type=float, required=True)
    parser.add_argument("--medium-wall-seconds", type=float, required=True)
    parser.add_argument("--fine-wall-seconds", type=float, required=True)
    parser.add_argument("--coarse-exit-status", type=int, required=True)
    parser.add_argument("--medium-exit-status", type=int, required=True)
    parser.add_argument("--fine-exit-status", type=int, required=True)
    parser.add_argument("--upstream-settings", type=Path, required=True)
    parser.add_argument("--upstream-output", type=Path, required=True)
    parser.add_argument("--dreami", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(
        coarse_path=args.coarse,
        medium_path=args.medium,
        fine_path=args.fine,
        lock_path=args.lock,
        deck_path=args.deck,
        coarse_settings_path=args.coarse_settings,
        medium_settings_path=args.medium_settings,
        fine_settings_path=args.fine_settings,
        coarse_wall_seconds=args.coarse_wall_seconds,
        medium_wall_seconds=args.medium_wall_seconds,
        fine_wall_seconds=args.fine_wall_seconds,
        coarse_exit_status=args.coarse_exit_status,
        medium_exit_status=args.medium_exit_status,
        fine_exit_status=args.fine_exit_status,
        upstream_settings_path=args.upstream_settings,
        upstream_output_path=args.upstream_output,
        dreami_path=args.dreami,
    )
    args.json_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"all_pass": result["all_pass"], "gates": result["gates"]}))
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
