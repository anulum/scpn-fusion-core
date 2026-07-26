# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — Coil-Vacuum Grid Diagnostic Tests
"""Entry-point and prerequisite tests for the coil-vacuum grid diagnostic."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

import jax
import numpy as np
import pytest

from validation import diagnose_ida_coil_vacuum_grid_convergence as diagnostic
from validation import ida_coil_vacuum_grid_contract as contract
from validation import ida_coil_vacuum_grid_fields as fields


def test_tracked_six_report_prerequisite_chain_is_exact() -> None:
    """All six tracked prerequisite payloads must validate and bind exactly."""
    reports = diagnostic._load_bound_reports()
    assert set(reports) == set(contract.EXPECTED_PAYLOADS)
    assert {
        name: report["payload_sha256"] for name, report in reports.items()
    } == contract.EXPECTED_PAYLOADS
    assert reports["response"]["closure"]["native_operator_response_max_abs_wb"] == pytest.approx(
        2.1288526497187377e-14
    )


def test_load_report_rejects_duplicate_keys_and_non_object_roots(
    tmp_path: Path,
) -> None:
    """JSON ambiguity and non-object report roots must fail before validation."""
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"first","schema_version":"second"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        diagnostic._load_report(duplicate)

    sequence = tmp_path / "sequence.json"
    sequence.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        diagnostic._load_report(sequence)


def test_runtime_source_artifact_binds_the_executed_module_bytes() -> None:
    """Runtime source binding must resolve the actual imported module file."""
    artifact = diagnostic._runtime_source_artifact(
        fields,
        logical_path=contract.SOURCE_PATHS["field_operations"],
        resource_name="ida_coil_vacuum_grid_fields.py",
    )
    assert artifact == {
        "path": contract.SOURCE_PATHS["field_operations"],
        "sha256": diagnostic._file_sha256(
            diagnostic.ROOT / contract.SOURCE_PATHS["field_operations"]
        ),
    }
    with pytest.raises(RuntimeError, match="does not resolve"):
        diagnostic._runtime_source_artifact(
            fields,
            logical_path=contract.SOURCE_PATHS["field_operations"],
            resource_name="wrong.py",
        )
    with pytest.raises(RuntimeError, match="no inspectable runtime source"):
        diagnostic._runtime_source_artifact(
            object(),
            logical_path="uninspectable",
            resource_name="missing.py",
        )


def test_repository_artifact_distinguishes_dirty_state_from_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repository provenance must never convert an unavailable probe into clean."""
    artifact = diagnostic._repository_artifact()
    assert artifact["git_commit"] == diagnostic._same_case._git_value("rev-parse", "HEAD")
    assert artifact["path"] == "."
    assert artifact["worktree_clean"] is False

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("git unavailable")),
    )
    with pytest.raises(RuntimeError, match="provenance is not inspectable"):
        diagnostic._repository_artifact()


def test_bound_report_loader_rejects_frozen_payload_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prerequisite whose validated payload differs from the freeze must fail."""
    reports = diagnostic._load_bound_reports()
    original_load = diagnostic._load_report

    def drifted_load(path: Path) -> dict[str, Any]:
        report = original_load(path)
        if path == diagnostic.ROOT / contract.SAME_CASE_PATH:
            report["payload_sha256"] = "0" * 64
        return report

    monkeypatch.setattr(diagnostic, "_load_report", drifted_load)
    monkeypatch.setattr(diagnostic._same_case, "validate_report", lambda report: None)
    with pytest.raises(ValueError, match="same_case report does not match"):
        diagnostic._load_bound_reports()
    assert reports["same_case"]["payload_sha256"] == contract.EXPECTED_PAYLOADS["same_case"]


@pytest.mark.dedicated_hardware
def test_anchor_rejects_forcing_and_response_digest_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 129 anchor must fail independently on forcing and response byte drift."""
    axis = np.linspace(0.1, 2.8, 5, dtype=np.float64)
    r_mesh, z_mesh = np.meshgrid(axis, axis)
    equilibrium = SimpleNamespace(
        plasma_psi=np.zeros((5, 5), dtype=np.float64),
        psi=lambda: np.asarray(r_mesh**2 + z_mesh**2, dtype=np.float64),
    )
    with pytest.raises(ValueError, match="forcing anchor drifted"):
        diagnostic._anchor(
            equilibrium=equilibrium,
            r_grid=axis,
            z_grid=axis,
            response_closure_max_abs_wb=0.0,
        )

    forcing = fields.zero_identity_wall(
        np.asarray(
            diagnostic._operator._native_lhs(
                equilibrium.psi(),
                r_grid=axis,
                z_grid=axis,
            ).T,
            dtype=np.float64,
        ),
        field="test anchor forcing",
    )
    monkeypatch.setattr(
        contract,
        "EXPECTED_ANCHOR_FORCING_SHA256",
        diagnostic._array_sha256(forcing),
    )
    with pytest.raises(ValueError, match="response anchor drifted"):
        diagnostic._anchor(
            equilibrium=equilibrium,
            r_grid=axis,
            z_grid=axis,
            response_closure_max_abs_wb=0.0,
        )


def test_plasma_support_mask_uses_frozen_profile_and_zeroes_walls() -> None:
    """Reference-current support must be evaluated on-grid with walls excluded."""
    equilibrium = SimpleNamespace(
        psi_bndry=0.0,
        psiRZ=lambda r, z: np.asarray(r + z, dtype=np.float64),
    )
    profiles = SimpleNamespace(
        Jtor=lambda r, z, psi, psi_bndry: np.where(
            (r > 0.5) & (r < 2.5) & (z > -1.0) & (z < 1.0),
            1.0,
            0.0,
        )
    )
    support = diagnostic._plasma_support_mask(
        equilibrium=equilibrium,
        profiles=profiles,
        resolution=33,
        r_bounds=contract.R_BOUNDS_M,
        z_bounds=contract.Z_BOUNDS_M,
    )
    assert support.shape == (33, 33)
    assert np.any(support)
    assert not np.any(support[[0, -1], :])
    assert not np.any(support[:, [0, -1]])


def test_plasma_support_mask_isolates_freegs_state_and_preserves_zr_orientation() -> None:
    """Nested-grid evaluation must not mutate FreeGS or swap runtime field axes."""
    equilibrium = SimpleNamespace(
        psi_bndry=0.0,
        psiRZ=lambda r, z: np.asarray(r + z, dtype=np.float64),
        _updateBoundaryPsi=lambda psi: pytest.fail("native equilibrium was mutated"),
    )

    class StatefulProfile:
        def __init__(self) -> None:
            self.eq = equilibrium
            self.evaluated = False

        def Jtor(
            self,
            r: np.ndarray[Any, Any],
            z: np.ndarray[Any, Any],
            psi: np.ndarray[Any, Any],
            psi_bndry: float,
        ) -> np.ndarray[Any, Any]:
            self.eq._updateBoundaryPsi(psi)
            assert np.all(np.diff(r[:, 0]) > 0.0)
            assert np.all(np.diff(z[0, :]) > 0.0)
            assert psi_bndry == self.eq.psi_bndry == 0.0
            self.evaluated = True
            return np.asarray((r > 1.0) & (z < 0.0), dtype=np.float64)

    profiles = StatefulProfile()
    support = diagnostic._plasma_support_mask(
        equilibrium=equilibrium,
        profiles=profiles,
        resolution=33,
        r_bounds=contract.R_BOUNDS_M,
        z_bounds=contract.Z_BOUNDS_M,
    )
    assert profiles.evaluated is False
    assert support[8, 20]
    assert not support[24, 20]
    assert not support[8, 5]


@pytest.mark.parametrize("mode", ["empty", "shape"])
def test_plasma_support_mask_rejects_empty_or_malformed_evaluation(mode: str) -> None:
    """Malformed or empty profile support must fail before grid execution."""
    equilibrium = SimpleNamespace(
        psi_bndry=0.0,
        psiRZ=lambda r, z: np.asarray(r + z, dtype=np.float64),
    )
    profiles = SimpleNamespace(
        Jtor=(
            (lambda r, z, psi, psi_bndry: np.zeros((3, 3), dtype=np.float64))
            if mode == "shape"
            else (lambda r, z, psi, psi_bndry: np.zeros_like(r))
        )
    )
    message = "evaluation is invalid" if mode == "shape" else "must not be empty"
    with pytest.raises(ValueError, match=message):
        diagnostic._plasma_support_mask(
            equilibrium=equilibrium,
            profiles=profiles,
            resolution=33,
            r_bounds=contract.R_BOUNDS_M,
            z_bounds=contract.Z_BOUNDS_M,
        )


def test_nested_plasma_support_masks_use_one_frozen_finest_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every coarser support must be an exact restriction of one finest mask."""
    axis = np.arange(257, dtype=np.int64)
    row, column = np.meshgrid(axis, axis, indexing="ij")
    finest = np.asarray((row + 2 * column) % 7 < 3, dtype=np.bool_)
    finest[[0, -1], :] = False
    finest[:, [0, -1]] = False
    calls: list[int] = []

    def frozen_support(**kwargs: Any) -> np.ndarray[Any, Any]:
        calls.append(int(kwargs["resolution"]))
        return finest

    monkeypatch.setattr(diagnostic, "_plasma_support_mask", frozen_support)
    masks = diagnostic._nested_plasma_support_masks(
        equilibrium=object(),
        profiles=object(),
        r_bounds=contract.R_BOUNDS_M,
        z_bounds=contract.Z_BOUNDS_M,
    )
    assert calls == [257]
    assert set(masks) == set(contract.GRID_RESOLUTIONS)
    for resolution, mask in masks.items():
        expected = fields.restrict_to_shape(finest, (resolution, resolution)) > 0.5
        assert np.array_equal(mask, expected)


def test_diagnostic_admission_rejects_non_fp64_and_missing_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execution must reject non-FP64 mode and a missing runtime backend."""
    update_config = cast(Callable[[str, bool], None], jax.config.update)
    update_config("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="requires JAX FP64"):
            diagnostic.run_diagnostic(generated_at="2026-07-25T20:00:00Z")
    finally:
        update_config("jax_enable_x64", True)

    monkeypatch.setattr(diagnostic, "_load_bound_reports", lambda: {})
    monkeypatch.setattr(
        diagnostic._source,
        "_solve_reference",
        lambda path: (None, None, object(), object(), object(), None, "missing"),
    )
    monkeypatch.setattr(
        diagnostic._source,
        "_import_freegs",
        lambda: (None, None, "not installed"),
    )
    with pytest.raises(RuntimeError, match="backend unavailable: not installed"):
        diagnostic.run_diagnostic(generated_at="2026-07-25T20:00:00Z")


@pytest.mark.parametrize(
    ("forcing_digest", "response_digest", "message"),
    [
        ("wrong-forcing", "expected-response", "forcing does not reproduce"),
        ("expected-forcing", "wrong-response", "response does not reproduce"),
    ],
)
def test_diagnostic_rejects_129_ladder_anchor_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    forcing_digest: str,
    response_digest: str,
    message: str,
) -> None:
    """The four-grid ladder must reproduce both independently bound 129 digests."""
    reports = diagnostic._load_bound_reports()
    expected_sources = reports["operator"]["source_artifacts"]
    spec = SimpleNamespace(
        r_min=0.1,
        r_max=2.8,
        z_min=-1.8,
        z_max=1.8,
        example_path=diagnostic.ROOT / "README.md",
    )
    equilibrium = SimpleNamespace(
        R_1D=np.linspace(0.1, 2.8, 129, dtype=np.float64),
        Z_1D=np.linspace(-1.8, 1.8, 129, dtype=np.float64),
    )
    monkeypatch.setattr(diagnostic, "_load_bound_reports", lambda: reports)
    monkeypatch.setattr(
        diagnostic._source,
        "_solve_reference",
        lambda path: (None, None, spec, object(), equilibrium, None, "test"),
    )
    monkeypatch.setattr(
        diagnostic._source,
        "_import_freegs",
        lambda: (object(), None, None),
    )
    monkeypatch.setattr(diagnostic._source, "_evaluation_case", lambda report: object())
    monkeypatch.setattr(
        diagnostic._operator,
        "_source_artifacts",
        lambda **kwargs: {
            name: expected_sources[name]
            for name in ("freegs_boundary", "freegs_operator", "freegs_public_example")
        },
    )
    monkeypatch.setattr(diagnostic, "extract_coil_manifest", lambda tokamak: ())
    monkeypatch.setattr(diagnostic, "validate_frozen_manifest", lambda *args, **kwargs: ())
    monkeypatch.setattr(
        diagnostic,
        "_anchor",
        lambda **kwargs: (
            {
                "forcing_sha256": "expected-forcing",
                "response_closure_max_abs_wb": 0.0,
                "response_sha256": "expected-response",
            },
            np.zeros((129, 129), dtype=np.float64),
            np.zeros((129, 129), dtype=np.float64),
        ),
    )
    monkeypatch.setattr(
        diagnostic,
        "_plasma_support_mask",
        lambda **kwargs: np.ones((int(kwargs["resolution"]),) * 2, dtype=np.bool_),
    )

    def grid_result(**kwargs: Any) -> SimpleNamespace:
        resolution = int(kwargs["resolution"])
        return SimpleNamespace(
            resolution=resolution,
            report={
                "forcing_partition": {
                    "total": {
                        "field_sha256": (forcing_digest if resolution == 129 else "coarse-forcing")
                    }
                },
                "response_partition": {
                    "total": {
                        "field_sha256": (
                            response_digest if resolution == 129 else "coarse-response"
                        )
                    }
                },
            },
        )

    monkeypatch.setattr(diagnostic, "run_grid", grid_result)
    with pytest.raises(ValueError, match=message):
        diagnostic.run_diagnostic(generated_at="2026-07-25T20:00:00Z")


def test_cli_requires_timestamp_and_rejects_invalid_existing_report(
    tmp_path: Path,
) -> None:
    """Execution needs an explicit timestamp and validation rejects forged input."""
    with pytest.raises(SystemExit) as missing:
        diagnostic.main([])
    assert missing.value.code == 2

    forged = tmp_path / "forged.json"
    forged.write_text(
        json.dumps({"schema_version": "forged"}, allow_nan=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="top-level fields"):
        diagnostic.main(["--validate-report", str(forged)])


def test_write_report_validates_before_creating_files(tmp_path: Path) -> None:
    """Invalid evidence must not create either output file."""
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    with pytest.raises(ValueError, match="top-level fields"):
        diagnostic.write_report(
            {"schema_version": "forged"},
            json_path=json_path,
            markdown_path=markdown_path,
        )
    assert not json_path.exists()
    assert not markdown_path.exists()


@pytest.mark.experimental
@pytest.mark.external_reference
@pytest.mark.dedicated_hardware
def test_real_four_grid_cli_writes_and_validates_evidence(tmp_path: Path) -> None:
    """The real CLI must execute all grids and write a self-validating evidence pair."""
    json_path = tmp_path / "ida_coil_vacuum_grid_convergence.json"
    markdown_path = tmp_path / "ida_coil_vacuum_grid_convergence.md"
    assert (
        diagnostic.main(
            [
                "--generated-at",
                "2026-07-25T20:00:00Z",
                "--json-report",
                str(json_path),
                "--markdown-report",
                str(markdown_path),
            ]
        )
        == 0
    )
    report = diagnostic._load_report(json_path)
    contract.validate_report(report)
    assert [row["resolution"] for row in report["grids"]] == list(contract.GRID_RESOLUTIONS)
    assert report["anchor"]["forcing_sha256"] == (contract.EXPECTED_ANCHOR_FORCING_SHA256)
    assert report["anchor"]["response_sha256"] == (contract.EXPECTED_ANCHOR_RESPONSE_SHA256)
    assert set(report["claim_boundary"].values()) == {False}
    assert markdown_path.read_text(encoding="utf-8") == contract.render_markdown(report)
