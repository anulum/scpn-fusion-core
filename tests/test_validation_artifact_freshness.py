# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — stale-artifact guard binding tracked evidence JSONs to their generators
"""Stale-artifact guard: tracked evidence JSONs must match the generator that made them.

Each tracked validation artefact records ``provenance.generator_sha256`` (the digest of its
generator script at generation time). These tests recompute the digest of the CURRENT
generator source and fail when they diverge — i.e. when someone edits a generator without
regenerating (and re-committing) its artefact, or edits a tracked artefact by hand. The
distinct-eye finding F2 on the real-data evidence lane.

Most checks are hermetic file reads and hashes. The output-policy cohort also
exercises one bounded real benchmark CLI.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from pathlib import Path

import pytest

from validation.evidence_output import resolve_evidence_outputs

REPO = Path(__file__).resolve().parents[1]

_PREFLIGHT_EVIDENCE_WRITERS = (
    "validation/benchmark_disruption_replay_pipeline.py",
    "validation/benchmark_disruption_transfer_generalization.py",
    "validation/benchmark_eped_domain_contract.py",
    "validation/benchmark_transport_uncertainty_envelope.py",
    "validation/benchmark_multi_ion_transport_conservation.py",
    "validation/vertical_control_replay_benchmark.py",
    "validation/scpn_end_to_end_latency.py",
)

# (tracked artefact, generator that must have produced it)
_BOUND_PAIRS = [
    (
        "artifacts/real_diiid_145419/real_145419_validation.json",
        "validation/validate_real_diiid_145419.py",
    ),
    (
        "artifacts/coilgrad_adjoint_fd_evidence.json",
        "validation/measure_coilgrad_adjoint_fd.py",
    ),
    (
        "artifacts/rung2_mg_preconditioner/iteration_counts.json",
        "validation/measure_mg_preconditioner_iterations.py",
    ),
    (
        "artifacts/rung2_mg_preconditioner/compiled_forward_speedup.json",
        "validation/measure_compiled_forward_speedup.py",
    ),
    (
        "artifacts/rung2_mg_preconditioner/warm_start_forward.json",
        "validation/measure_warm_start_forward.py",
    ),
    (
        "artifacts/rung2_mg_preconditioner/batched_forward_amortisation.json",
        "validation/measure_batched_forward.py",
    ),
    (
        "artifacts/disruption_transfer_generalization.json",
        "validation/benchmark_disruption_transfer_generalization.py",
    ),
    (
        "artifacts/disruption_threshold_sweep.json",
        "tools/sweep_disruption_threshold.py",
    ),
    (
        "artifacts/disruption_roc_curve.json",
        "tools/sweep_disruption_threshold.py",
    ),
]

# Fail-closed dependency contracts owned by the test suite, never by the artifact.
# The ordering is canonical because generators persist these exact lists as provenance.
_EXPECTED_LOGIC_SOURCES = {
    "artifacts/real_diiid_145419/real_145419_validation.json": (
        "src/scpn_fusion/core/jax_free_boundary_gs.py",
        "src/scpn_fusion/core/jax_free_boundary_gs_implicit.py",
        "src/scpn_fusion/core/jax_plasma_support.py",
        "src/scpn_fusion/core/jax_equilibrium_solver.py",
        "src/scpn_fusion/core/jax_o_point.py",
    ),
    "artifacts/coilgrad_adjoint_fd_evidence.json": (
        "src/scpn_fusion/core/jax_free_boundary_predictive.py",
        "src/scpn_fusion/core/jax_free_boundary_gs.py",
        "src/scpn_fusion/core/jax_plasma_support.py",
        "src/scpn_fusion/core/jax_continuation_history.py",
        "src/scpn_fusion/core/jax_multigrid_precond.py",
        "src/scpn_fusion/core/jax_equilibrium_solver.py",
        "src/scpn_fusion/core/jax_o_point.py",
        "src/scpn_fusion/core/jax_x_point.py",
    ),
    "artifacts/rung2_mg_preconditioner/iteration_counts.json": (
        "src/scpn_fusion/core/jax_free_boundary_predictive.py",
        "src/scpn_fusion/core/jax_free_boundary_gs.py",
        "src/scpn_fusion/core/jax_plasma_support.py",
        "src/scpn_fusion/core/jax_continuation_history.py",
        "src/scpn_fusion/core/jax_multigrid_precond.py",
        "src/scpn_fusion/core/jax_equilibrium_solver.py",
        "src/scpn_fusion/core/jax_o_point.py",
        "src/scpn_fusion/core/jax_x_point.py",
    ),
    "artifacts/rung2_mg_preconditioner/compiled_forward_speedup.json": (
        "src/scpn_fusion/core/jax_predictive_forward_compiled.py",
        "src/scpn_fusion/core/jax_free_boundary_predictive.py",
        "src/scpn_fusion/core/jax_free_boundary_gs.py",
        "src/scpn_fusion/core/jax_plasma_support.py",
        "src/scpn_fusion/core/jax_continuation_history.py",
        "src/scpn_fusion/core/jax_multigrid_precond.py",
        "src/scpn_fusion/core/jax_predictive_checkpoint_trace.py",
        "src/scpn_fusion/core/jax_equilibrium_solver.py",
        "src/scpn_fusion/core/jax_o_point.py",
        "src/scpn_fusion/core/jax_x_point.py",
    ),
    "artifacts/rung2_mg_preconditioner/warm_start_forward.json": (
        "src/scpn_fusion/core/jax_predictive_forward_compiled.py",
        "src/scpn_fusion/core/jax_free_boundary_predictive.py",
        "src/scpn_fusion/core/jax_free_boundary_gs.py",
        "src/scpn_fusion/core/jax_plasma_support.py",
        "src/scpn_fusion/core/jax_continuation_history.py",
        "src/scpn_fusion/core/jax_multigrid_precond.py",
        "src/scpn_fusion/core/jax_predictive_checkpoint_trace.py",
        "src/scpn_fusion/core/jax_equilibrium_solver.py",
        "src/scpn_fusion/core/jax_o_point.py",
        "src/scpn_fusion/core/jax_x_point.py",
    ),
    "artifacts/rung2_mg_preconditioner/batched_forward_amortisation.json": (
        "src/scpn_fusion/core/jax_predictive_forward_compiled.py",
        "src/scpn_fusion/core/jax_free_boundary_predictive.py",
        "src/scpn_fusion/core/jax_free_boundary_gs.py",
        "src/scpn_fusion/core/jax_plasma_support.py",
        "src/scpn_fusion/core/jax_continuation_history.py",
        "src/scpn_fusion/core/jax_multigrid_precond.py",
        "src/scpn_fusion/core/jax_predictive_checkpoint_trace.py",
        "src/scpn_fusion/core/jax_equilibrium_solver.py",
        "src/scpn_fusion/core/jax_o_point.py",
        "src/scpn_fusion/core/jax_x_point.py",
    ),
    "artifacts/disruption_transfer_generalization.json": (
        "validation/validate_real_shots.py",
        "src/scpn_fusion/control/disruption_predictor.py",
    ),
    "artifacts/disruption_threshold_sweep.json": (
        "src/scpn_fusion/control/disruption_predictor.py",
    ),
    "artifacts/disruption_roc_curve.json": ("src/scpn_fusion/control/disruption_predictor.py",),
}

_DISRUPTION_DEPENDENCIES = {
    "artifacts/disruption_transfer_generalization.json": (
        "validation/validate_real_shots.py",
        "src/scpn_fusion/control/disruption_predictor.py",
    ),
    "artifacts/disruption_threshold_sweep.json": (
        "src/scpn_fusion/control/disruption_predictor.py",
    ),
    "artifacts/disruption_roc_curve.json": ("src/scpn_fusion/control/disruption_predictor.py",),
}
_DISRUPTION_DATA = REPO / "validation" / "reference_data" / "diiid" / "disruption_shots"
_PUBLIC_JSON_DOCUMENTS = (
    "artifacts/real_shot_validation.json",
    "examples/neuro_symbolic_control_demo.ipynb",
    "examples/neuro_symbolic_control_demo_v2.ipynb",
)
_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")


def _json_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in _json_strings(item)]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _json_strings(item)]
    return []


@pytest.mark.parametrize("document_rel", _PUBLIC_JSON_DOCUMENTS)
def test_public_json_documents_do_not_expose_windows_absolute_paths(document_rel: str) -> None:
    document = json.loads((REPO / document_rel).read_text(encoding="utf-8"))
    leaks = [text for text in _json_strings(document) if _WINDOWS_ABSOLUTE_PATH.search(text)]
    assert leaks == [], f"{document_rel} contains host-specific Windows paths: {leaks}"


def test_routine_evidence_defaults_are_local_and_ignored(tmp_path: Path) -> None:
    outputs = resolve_evidence_outputs(
        root=tmp_path,
        canonical_json=Path("validation/reports/example.json"),
        canonical_markdown=Path("validation/reports/example.md"),
        requested_json=None,
        requested_markdown=None,
        commit_evidence=False,
    )
    assert outputs.json == tmp_path / "artifacts" / "_local_example.json"
    assert outputs.markdown == tmp_path / "artifacts" / "_local_example.md"

    gitignore = (REPO / ".gitignore").read_text(encoding="utf-8")
    assert "artifacts/*_local*.json" in gitignore
    assert "artifacts/*_local*.md" in gitignore


def test_explicit_non_evidence_output_paths_remain_supported(tmp_path: Path) -> None:
    requested_json = tmp_path / "exports" / "result.json"
    requested_markdown = tmp_path / "exports" / "result.md"
    outputs = resolve_evidence_outputs(
        root=tmp_path,
        canonical_json=Path("validation/reports/example.json"),
        canonical_markdown=Path("validation/reports/example.md"),
        requested_json=requested_json,
        requested_markdown=requested_markdown,
        commit_evidence=False,
    )
    assert outputs.json == requested_json
    assert outputs.markdown == requested_markdown


def test_explicit_output_paths_outside_repository_remain_supported(tmp_path: Path) -> None:
    export_root = tmp_path.parent / f"{tmp_path.name}_external_exports"
    requested_json = export_root / "result.json"
    requested_markdown = export_root / "result.md"
    outputs = resolve_evidence_outputs(
        root=tmp_path,
        canonical_json=Path("validation/reports/example.json"),
        canonical_markdown=Path("validation/reports/example.md"),
        requested_json=requested_json,
        requested_markdown=requested_markdown,
        commit_evidence=False,
    )
    assert outputs.json == requested_json
    assert outputs.markdown == requested_markdown


def test_explicit_local_artifact_paths_remain_supported(tmp_path: Path) -> None:
    requested_json = tmp_path / "artifacts" / "_local_preflight.json"
    requested_markdown = tmp_path / "artifacts" / "_local_preflight.md"
    outputs = resolve_evidence_outputs(
        root=tmp_path,
        canonical_json=Path("validation/reports/example.json"),
        canonical_markdown=Path("validation/reports/example.md"),
        requested_json=requested_json,
        requested_markdown=requested_markdown,
        commit_evidence=False,
    )
    assert outputs.json == requested_json
    assert outputs.markdown == requested_markdown


@pytest.mark.parametrize(
    "protected_path",
    [
        Path("validation/reports/example.json"),
        Path("artifacts/example.json"),
        Path("artifacts/nested/_local_example.json"),
    ],
)
def test_protected_evidence_paths_require_explicit_commit_flag(
    tmp_path: Path, protected_path: Path
) -> None:
    with pytest.raises(ValueError, match="without --commit-evidence"):
        resolve_evidence_outputs(
            root=tmp_path,
            canonical_json=Path("validation/reports/example.json"),
            canonical_markdown=Path("validation/reports/example.md"),
            requested_json=tmp_path / protected_path,
            requested_markdown=tmp_path / "exports" / "result.md",
            commit_evidence=False,
        )


def test_commit_evidence_selects_canonical_outputs(tmp_path: Path) -> None:
    outputs = resolve_evidence_outputs(
        root=tmp_path,
        canonical_json=Path("validation/reports/example.json"),
        canonical_markdown=Path("validation/reports/example.md"),
        requested_json=None,
        requested_markdown=None,
        commit_evidence=True,
    )
    assert outputs.json == tmp_path / "validation" / "reports" / "example.json"
    assert outputs.markdown == tmp_path / "validation" / "reports" / "example.md"


@pytest.mark.parametrize("writer", _PREFLIGHT_EVIDENCE_WRITERS)
def test_preflight_evidence_writers_use_explicit_output_policy(writer: str) -> None:
    source = (REPO / writer).read_text(encoding="utf-8")
    assert "add_evidence_output_arguments(parser)" in source
    assert "resolve_evidence_outputs(" in source


def test_benchmark_cli_default_writes_local_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("validation.benchmark_eped_domain_contract")
    monkeypatch.setattr(module, "ROOT", tmp_path)

    assert module.main(["--strict"]) == 0
    output_json = tmp_path / "artifacts" / "_local_eped_domain_contract_benchmark.json"
    output_markdown = tmp_path / "artifacts" / "_local_eped_domain_contract_benchmark.md"
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["eped_domain_contract_benchmark"]["passes_thresholds"] is True
    assert output_markdown.read_text(encoding="utf-8").startswith(
        "# EPED Domain Contract Benchmark"
    )
    assert not (tmp_path / "validation" / "reports").exists()


def test_benchmark_cli_rejects_canonical_output_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("validation.benchmark_eped_domain_contract")
    monkeypatch.setattr(module, "ROOT", tmp_path)

    with pytest.raises(SystemExit, match="2"):
        module.main(
            [
                "--output-json",
                str(tmp_path / "validation" / "reports" / "example.json"),
                "--output-md",
                str(tmp_path / "exports" / "example.md"),
            ]
        )


def _digest_paths(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        label = path.relative_to(REPO).as_posix()
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@pytest.mark.parametrize(("artifact_rel", "generator_rel"), _BOUND_PAIRS)
def test_tracked_artifact_matches_its_generator(artifact_rel: str, generator_rel: str) -> None:
    artifact = REPO / artifact_rel
    generator = REPO / generator_rel
    assert generator.exists(), f"generator missing: {generator_rel}"
    assert artifact.exists(), f"tracked artefact missing: {artifact_rel}"
    record = json.loads(artifact.read_text())
    recorded = record.get("provenance", {}).get("generator_sha256") or record.get(
        "generator_sha256"
    )
    assert recorded, (
        f"{artifact_rel} lacks a generator_sha256 provenance field — regenerate it with the "
        "current generator"
    )
    current = hashlib.sha256(generator.read_bytes()).hexdigest()
    assert recorded == current, (
        f"{artifact_rel} is STALE: it was generated by {generator_rel}@{recorded[:12]} but the "
        f"current generator is @{current[:12]} — rerun the generator and commit both together"
    )


def test_artifacts_have_environment_provenance() -> None:
    """The real-data artefact must state the exact package versions that produced it."""
    record = json.loads((REPO / _BOUND_PAIRS[0][0]).read_text())
    packages = record["provenance"]["packages"]
    for name in ("omas", "freeqdsk", "numpy", "scipy", "jax"):
        assert packages.get(name), f"missing package version in provenance: {name}"
    assert record["provenance"].get("pinned_requirements_sha256")


@pytest.mark.parametrize(("artifact_rel", "_generator_rel"), _BOUND_PAIRS)
def test_tracked_artifact_matches_pinned_requirements(
    artifact_rel: str, _generator_rel: str
) -> None:
    """Every bound artifact records the current hash-pinned environment contract.

    After a lock refresh, regenerate all bound artifacts from the repository root with::

        PINNED_PY=/media/anulum/GOTM/_scratch/fusion-pinned-venv2/bin/python
        PYTHONPATH=src:. $PINNED_PY validation/validate_real_diiid_145419.py
        PYTHONPATH=src:. $PINNED_PY validation/measure_coilgrad_adjoint_fd.py
        PYTHONPATH=src:. $PINNED_PY validation/measure_mg_preconditioner_iterations.py
        PYTHONPATH=src:. $PINNED_PY validation/measure_compiled_forward_speedup.py
        PYTHONPATH=src:. $PINNED_PY validation/measure_warm_start_forward.py
        PYTHONPATH=src:. $PINNED_PY validation/measure_batched_forward.py
        PYTHONPATH=src:. $PINNED_PY validation/benchmark_disruption_transfer_generalization.py --strict
        PYTHONPATH=src:. $PINNED_PY tools/sweep_disruption_threshold.py
    """
    artifact = json.loads((REPO / artifact_rel).read_text(encoding="utf-8"))
    recorded = artifact.get("provenance", {}).get("pinned_requirements_sha256")
    current = hashlib.sha256((REPO / "requirements" / "full.txt").read_bytes()).hexdigest()
    assert recorded == current, (
        f"{artifact_rel} is not bound to the current requirements/full.txt: "
        f"recorded={recorded!r}, current={current}. See this test's docstring for the exact "
        "pinned-venv regeneration commands."
    )


@pytest.mark.parametrize(("artifact_rel", "_generator_rel"), _BOUND_PAIRS)
def test_tracked_artifact_matches_logic_sources(artifact_rel: str, _generator_rel: str) -> None:
    """Scientific evidence must be rebound whenever executable solver logic changes."""
    artifact = json.loads((REPO / artifact_rel).read_text(encoding="utf-8"))
    provenance = artifact.get("provenance", {})
    logic_sources = provenance.get("logic_sources")
    expected = _EXPECTED_LOGIC_SOURCES[artifact_rel]
    assert logic_sources == list(expected), (
        f"{artifact_rel} has an incomplete or reordered provenance.logic_sources contract: "
        f"recorded={logic_sources!r}, expected={list(expected)!r}"
    )
    paths = tuple(REPO / rel for rel in expected)
    missing = [path.relative_to(REPO).as_posix() for path in paths if not path.is_file()]
    assert missing == [], f"{artifact_rel} names missing logic sources: {missing}"
    assert provenance.get("logic_sources_sha256") == _digest_paths(paths), (
        f"{artifact_rel} is STALE relative to its recorded solver logic; regenerate it "
        "with the current generator"
    )


def test_warm_start_equivalence_is_fail_closed_on_nonconvergence() -> None:
    """Never label two iteration-capped fields as a fixed-point equivalence result."""
    artifact = json.loads(
        (REPO / "artifacts" / "rung2_mg_preconditioner" / "warm_start_forward.json").read_text(
            encoding="utf-8"
        )
    )
    correctness = artifact["correctness_load_independent"]
    equivalences = correctness["warm_vs_cold_fixed_point_span_rel"]
    convergence = correctness["convergence"]
    assert equivalences.keys() == convergence.keys()
    for key, value in equivalences.items():
        all_converged = all(status["converged"] for status in convergence[key].values())
        assert (value is not None) is all_converged, (
            f"{key}: equivalence must be numeric iff every compared solve converged"
        )


@pytest.mark.parametrize(("artifact_rel", "dependency_rels"), _DISRUPTION_DEPENDENCIES.items())
def test_disruption_artifact_matches_logic_and_data_corpus(
    artifact_rel: str, dependency_rels: tuple[str, ...]
) -> None:
    """Disruption evidence becomes stale when logic or its NPZ corpus changes."""
    artifact = json.loads((REPO / artifact_rel).read_text(encoding="utf-8"))
    provenance = artifact["provenance"]
    dependencies = tuple(REPO / rel for rel in dependency_rels)
    data_files = tuple(_DISRUPTION_DATA.glob("*.npz"))

    assert provenance["logic_sources"] == list(dependency_rels)
    assert provenance["logic_sources_sha256"] == _digest_paths(dependencies)
    assert provenance["disruption_data_file_count"] == len(data_files)
    assert provenance["disruption_data_sha256"] == _digest_paths(data_files)
