# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — Fixed-Physical Coil-Vacuum Diagnostic Tests
"""Binding, writer, and CLI tests for the CVGC2 diagnostic."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.ida_coil_vacuum_fixed_physical_fixtures import report_fixture
from validation import diagnose_ida_coil_vacuum_grid_convergence as grid_diagnostic
from validation import diagnose_ida_coil_vacuum_fixed_physical_response as diagnostic
from validation import ida_coil_vacuum_fixed_physical_contract as contract
from validation import ida_coil_vacuum_grid_convergence as convergence


def _grid(row: dict[str, Any]) -> convergence.GridResult:
    """Bind one upstream public row to minimal unused private arrays."""
    resolution = int(row["resolution"])
    array = np.zeros((1, 1), dtype=np.float64)
    mask = np.ones((1, 1), dtype=np.bool_)
    return convergence.GridResult(
        resolution=resolution,
        report=copy.deepcopy(row),
        total_forcing_zr=array,
        source_forcing_zr=array,
        source_free_forcing_zr=array,
        total_response_zr=array,
        source_response_zr=array,
        source_free_response_zr=array,
        interior_mask=mask,
        primary_source_mask=mask,
        fixed_source_free_mask=mask,
        plasma_support_mask=mask,
    )


def _execution(upstream: dict[str, Any]) -> grid_diagnostic.GridLadderExecution:
    """Return an execution binding over the exact upstream public rows."""
    return grid_diagnostic.GridLadderExecution(
        environment={"backend": "gpu"},
        source_artifacts={},
        bindings={},
        anchor=copy.deepcopy(upstream["anchor"]),
        coil_manifest=copy.deepcopy(upstream["coil_manifest"]),
        results=tuple(_grid(row) for row in upstream["grids"]),
    )


def test_verify_cvgc1_arrays_accepts_exact_upstream_rows() -> None:
    """Every CVGC2 total field and fixed radius must bind to CVGC1 exactly."""
    upstream = contract.load_upstream_report(diagnostic.ROOT)
    diagnostic._verify_cvgc1_arrays(_execution(upstream), upstream)


@pytest.mark.parametrize(
    ("surface", "message"),
    [
        ("anchor", "129 anchor drifted"),
        ("manifest", "coil manifest drifted"),
        ("forcing", "33 total forcing drifted"),
        ("response", "33 total response drifted"),
        ("radius", "fixed physical radius drifted"),
    ],
)
def test_verify_cvgc1_arrays_rejects_every_binding_drift(
    surface: str,
    message: str,
) -> None:
    """Anchor, manifest, field, and mask-radius drift must fail independently."""
    upstream = contract.load_upstream_report(diagnostic.ROOT)
    execution = _execution(upstream)
    if surface == "anchor":
        execution.anchor["forcing_sha256"] = "0" * 64
    elif surface == "manifest":
        execution.coil_manifest["parent_count"] = 17
    elif surface == "forcing":
        execution.results[0].report["forcing_partition"]["total"]["field_sha256"] = "0" * 64
    elif surface == "response":
        execution.results[0].report["response_partition"]["total"]["field_sha256"] = "0" * 64
    elif surface == "radius":
        execution.results[0].report["masks"]["fixed_physical_radius_m"] = 0.2
    else:
        raise AssertionError(f"unhandled surface {surface}")
    with pytest.raises(ValueError, match=message):
        diagnostic._verify_cvgc1_arrays(execution, upstream)


def test_source_artifacts_bind_real_files_and_repository_probe() -> None:
    """Executed source provenance must name and hash every CVGC2 module."""
    artifacts = diagnostic._source_artifacts()
    assert set(artifacts) == {*contract.SOURCE_PATHS, "repository"}
    for name, path in contract.SOURCE_PATHS.items():
        assert artifacts[name] == {
            "path": path,
            "sha256": grid_diagnostic._file_sha256(diagnostic.ROOT / path),
        }
    assert artifacts["repository"]["path"] == "."


def test_run_diagnostic_routes_exact_execution_into_fixed_partition_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestrator must preserve execution bindings through the CVGC2 builder."""
    upstream = contract.load_upstream_report(diagnostic.ROOT)
    execution = _execution(upstream)
    fixed_rows = [
        SimpleNamespace(report={"resolution": row.resolution}) for row in execution.results
    ]
    captured: dict[str, Any] = {}

    monkeypatch.setattr(contract, "load_upstream_report", lambda root: upstream)
    monkeypatch.setattr(grid_diagnostic, "execute_grid_ladder", lambda: execution)
    monkeypatch.setattr(diagnostic, "build_fixed_physical_grid", lambda row: fixed_rows.pop(0))
    monkeypatch.setattr(
        diagnostic,
        "build_fixed_physical_convergence",
        lambda rows: {"count": len(rows)},
    )
    monkeypatch.setattr(diagnostic, "_source_artifacts", lambda: {"repository": {}})
    monkeypatch.setattr(
        contract,
        "build_report",
        lambda **kwargs: captured.update(kwargs) or {"result": "built"},
    )

    assert diagnostic.run_diagnostic(generated_at="2026-07-26T03:30:00Z") == {"result": "built"}
    assert captured["generated_at"] == "2026-07-26T03:30:00Z"
    assert captured["grids"] == [
        {"resolution": 33},
        {"resolution": 65},
        {"resolution": 129},
        {"resolution": 257},
    ]
    assert captured["convergence"] == {"count": 4}
    assert set(captured["execution_binding"]) == {
        "anchor_sha256",
        "coil_manifest_sha256",
        "source_artifacts_sha256",
    }


def test_writer_and_cli_round_trip_validated_json_and_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI must emit self-validating evidence and exact output digests."""
    report = report_fixture()
    output = tmp_path / "report.json"
    markdown = tmp_path / "report.md"
    monkeypatch.setattr(diagnostic, "run_diagnostic", lambda *, generated_at: report)

    assert (
        diagnostic.main(
            [
                "--generated-at",
                "2026-07-26T03:30:00Z",
                "--output",
                str(output),
                "--markdown-output",
                str(markdown),
                "--check",
            ]
        )
        == 0
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    contract.validate_report(written)
    assert markdown.read_text(encoding="utf-8") == contract.render_markdown(report)
    result = json.loads(capsys.readouterr().out)
    assert result["payload_sha256"] == report["payload_sha256"]
    assert result["json_sha256"] == diagnostic._file_sha256(output)


def test_cli_without_check_writes_and_reports_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The normal CLI path must not require the optional re-read check."""
    report = report_fixture()
    output = tmp_path / "report.json"
    markdown = tmp_path / "report.md"
    monkeypatch.setattr(diagnostic, "run_diagnostic", lambda *, generated_at: report)
    assert (
        diagnostic.main(
            [
                "--generated-at",
                "2026-07-26T03:30:00Z",
                "--output",
                str(output),
                "--markdown-output",
                str(markdown),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["payload_sha256"] == report["payload_sha256"]


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("nonobject", "must remain a JSON object"),
        ("payload", "payload drifted"),
        ("markdown", "must end with a newline"),
    ],
)
def test_cli_check_rejects_written_output_drift(
    case: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-write object, payload, and Markdown corruption must fail independently."""
    report = report_fixture()
    output = tmp_path / "report.json"
    markdown = tmp_path / "report.md"
    monkeypatch.setattr(diagnostic, "run_diagnostic", lambda *, generated_at: report)

    def drifted_writer(
        value: dict[str, Any],
        *,
        output: Path,
        markdown_output: Path,
    ) -> None:
        written: object = value
        markdown_text = "evidence\n"
        if case == "nonobject":
            written = []
        elif case == "payload":
            written = copy.deepcopy(value)
            cast_written = written
            if not isinstance(cast_written, dict):
                raise AssertionError("payload case must remain an object")
            cast_written["payload_sha256"] = "0" * 64
        elif case == "markdown":
            markdown_text = "evidence"
        else:
            raise AssertionError(f"unhandled case {case}")
        output.write_text(json.dumps(written), encoding="utf-8")
        markdown_output.write_text(markdown_text, encoding="utf-8")

    monkeypatch.setattr(diagnostic, "write_report", drifted_writer)
    if case in {"payload", "markdown"}:
        monkeypatch.setattr(contract, "validate_report", lambda value: None)
    with pytest.raises(ValueError, match=message):
        diagnostic.main(
            [
                "--generated-at",
                "2026-07-26T03:30:00Z",
                "--output",
                str(output),
                "--markdown-output",
                str(markdown),
                "--check",
            ]
        )
