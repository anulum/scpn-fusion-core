# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — ψ_N Gate Attribution Generator Tests
"""Contract tests for the ψ_N gate attribution page generator.

Exercises report validation, the NaN-aware percentage formatter, the
counts-table and full Markdown renderers, and every ``main`` mode
(write, drift-check pass/stale/missing, relative-path resolution, and the
outside-repo report-path fallback), plus the ``__main__`` guard.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

from tools import generate_psi_gate_attribution as tool
from tools.generate_psi_gate_attribution import (
    REPORT_SCHEMA,
    _counts_table,
    _fmt_pct,
    load_report,
    main,
    render_markdown,
)


def _row(
    file: str,
    *,
    psi: float = 0.42,
    adapted: float = 0.31,
    residual: str = "plasma_vacuum_source_mismatch",
    solver: str = "free_boundary_vacuum",
    action: str = "acquire_external_coil_currents",
    blocker: str = "external_coil_currents_missing_from_geqdsk",
) -> dict[str, Any]:
    """Build a single attribution row carrying every required field."""
    return {
        "file": file,
        "machine": "DIII-D",
        "psi_rmse_norm": psi,
        "adapted_profile_psi_rmse_norm": adapted,
        "source_domain_residual_class": residual,
        "source_domain_required_solver_mode": solver,
        "source_domain_next_action": action,
        "free_boundary_reconstruction_blocker": blocker,
    }


def _report(*, passes: bool = False) -> dict[str, Any]:
    """Build a schema-valid two-row gate report payload."""
    rows = [
        _row("g150960.eqdsk", psi=0.42),
        # NaN adapted lane exercises the "not applicable" formatter branch and
        # the row-sort places this file after the first alphabetically.
        _row("g999999.eqdsk", psi=float("nan"), adapted=float("nan")),
    ]
    return {
        "schema_version": REPORT_SCHEMA,
        "threshold": 0.02,
        "benchmark_scope": "efit_style_psi_reconstruction",
        "count": 2,
        "count_by_machine": {"DIII-D": 2},
        "pass_count": 0,
        "adapted_profile_pass_count": 0,
        "passes": passes,
        "free_boundary_reconstruction_blocker_counts": {
            "external_coil_currents_missing_from_geqdsk": 2,
        },
        "source_domain_residual_class_counts": {
            "plasma_vacuum_source_mismatch": 2,
        },
        "source_domain_required_solver_mode_counts": {
            "free_boundary_vacuum": 2,
        },
        "rows": rows,
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    """Serialise a report payload (allowing NaN) to ``path``."""
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestLoadReport:
    """Validation of the gate report payload."""

    def test_loads_valid_report(self, tmp_path: Path) -> None:
        """A schema-valid report round-trips unchanged."""
        report_path = tmp_path / "gate.json"
        _write_report(report_path, _report())
        loaded = load_report(report_path)
        assert loaded["schema_version"] == REPORT_SCHEMA
        assert len(loaded["rows"]) == 2

    def test_rejects_non_object_payload(self, tmp_path: Path) -> None:
        """A top-level JSON array is rejected."""
        report_path = tmp_path / "gate.json"
        report_path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a JSON object"):
            load_report(report_path)

    def test_rejects_foreign_schema(self, tmp_path: Path) -> None:
        """An unexpected schema version is rejected verbatim."""
        report_path = tmp_path / "gate.json"
        payload = _report()
        payload["schema_version"] = "efit-nrmse-benchmark.v1"
        _write_report(report_path, payload)
        with pytest.raises(ValueError, match="unexpected gate report schema"):
            load_report(report_path)

    def test_rejects_missing_rows(self, tmp_path: Path) -> None:
        """A report without a rows list is rejected."""
        report_path = tmp_path / "gate.json"
        payload = _report()
        del payload["rows"]
        _write_report(report_path, payload)
        with pytest.raises(ValueError, match="non-empty 'rows' list"):
            load_report(report_path)

    def test_rejects_empty_rows(self, tmp_path: Path) -> None:
        """A report with an empty rows list is rejected."""
        report_path = tmp_path / "gate.json"
        payload = _report()
        payload["rows"] = []
        _write_report(report_path, payload)
        with pytest.raises(ValueError, match="non-empty 'rows' list"):
            load_report(report_path)

    def test_rejects_non_object_row(self, tmp_path: Path) -> None:
        """A non-object row entry is rejected with its index."""
        report_path = tmp_path / "gate.json"
        payload = _report()
        payload["rows"] = ["not-an-object"]
        _write_report(report_path, payload)
        with pytest.raises(ValueError, match=r"rows\[0\] must be an object"):
            load_report(report_path)

    def test_rejects_row_missing_fields(self, tmp_path: Path) -> None:
        """A row missing attribution fields is rejected listing them."""
        report_path = tmp_path / "gate.json"
        payload = _report()
        payload["rows"] = [{"file": "g150960.eqdsk"}]
        _write_report(report_path, payload)
        with pytest.raises(ValueError, match="missing attribution fields"):
            load_report(report_path)


class TestFmtPct:
    """NaN-aware percentage formatter."""

    def test_formats_finite_fraction(self) -> None:
        """A finite fraction renders as a one-decimal percentage."""
        assert _fmt_pct(0.0234) == "2.3%"

    def test_nan_renders_em_dash(self) -> None:
        """NaN renders as an em dash (lane not applicable)."""
        assert _fmt_pct(float("nan")) == "—"


class TestCountsTable:
    """Sorted counts-table fragment."""

    def test_renders_sorted_rows(self) -> None:
        """Counts render sorted by key under a section heading."""
        lines = _counts_table("Blockers", {"b_class": 1, "a_class": 3})
        assert lines[1] == "### Blockers"
        # Sorted ascending: a_class precedes b_class.
        assert lines[-2] == "| `a_class` | 3 |"
        assert lines[-1] == "| `b_class` | 1 |"


class TestRenderMarkdown:
    """Full Markdown document renderer."""

    def test_fail_verdict_and_sorted_rows(self) -> None:
        """A failing gate renders the fail-closed verdict and sorted rows."""
        markdown = render_markdown(_report(passes=False), "validation/reports/gate.json")
        assert "FAIL (fail-closed by design)" in markdown
        assert "# ψ_N Reconstruction Gate — Per-File Attribution" in markdown
        # NaN psi lane renders an em dash in the per-file table.
        assert "| `g999999.eqdsk` | — |" in markdown
        # Rows are sorted by file: g150960 precedes g999999.
        assert markdown.index("g150960.eqdsk") < markdown.index("g999999.eqdsk")
        assert markdown.endswith("\n")

    def test_pass_verdict(self) -> None:
        """A passing gate renders the PASS verdict."""
        markdown = render_markdown(_report(passes=True), "validation/reports/gate.json")
        assert "Gate verdict: `PASS`" in markdown


class TestMain:
    """CLI entry point across write and check modes."""

    def test_write_mode_creates_page(self, tmp_path: Path) -> None:
        """Write mode renders the page and returns 0."""
        report_path = tmp_path / "gate.json"
        _write_report(report_path, _report())
        output_path = tmp_path / "nested" / "PSI_GATE_ATTRIBUTION.md"
        rc = main(["--report", str(report_path), "--output", str(output_path)])
        assert rc == 0
        assert output_path.exists()
        assert "ψ_N Reconstruction Gate" in output_path.read_text(encoding="utf-8")

    def test_check_mode_up_to_date(self, tmp_path: Path) -> None:
        """Check mode returns 0 when the page already matches."""
        report_path = tmp_path / "gate.json"
        _write_report(report_path, _report())
        output_path = tmp_path / "PSI_GATE_ATTRIBUTION.md"
        assert main(["--report", str(report_path), "--output", str(output_path)]) == 0
        assert main(["--report", str(report_path), "--output", str(output_path), "--check"]) == 0

    def test_check_mode_missing_page(self, tmp_path: Path) -> None:
        """Check mode returns 1 when the page is absent."""
        report_path = tmp_path / "gate.json"
        _write_report(report_path, _report())
        output_path = tmp_path / "absent.md"
        rc = main(["--report", str(report_path), "--output", str(output_path), "--check"])
        assert rc == 1

    def test_check_mode_stale_page(self, tmp_path: Path) -> None:
        """Check mode returns 1 when the page is stale."""
        report_path = tmp_path / "gate.json"
        _write_report(report_path, _report())
        output_path = tmp_path / "stale.md"
        output_path.write_text("obsolete content\n", encoding="utf-8")
        rc = main(["--report", str(report_path), "--output", str(output_path), "--check"])
        assert rc == 1

    def test_relative_paths_resolve_against_repo_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Relative report/output paths resolve against the repo root."""
        # Point REPO_ROOT at a temp tree so relative resolution is observable
        # and the report path is inside the (patched) root — exercising the
        # relative_to success branch for the header line.
        monkeypatch.setattr(tool, "REPO_ROOT", tmp_path)
        rel_report = Path("gate.json")
        _write_report(tmp_path / rel_report, _report())
        rc = main(["--report", "gate.json", "--output", "out.md"])
        assert rc == 0
        assert (tmp_path / "out.md").exists()

    def test_outside_repo_report_path_falls_back_to_posix(self, tmp_path: Path) -> None:
        """An absolute report path outside the repo uses its full posix path."""
        report_path = tmp_path / "gate.json"
        _write_report(report_path, _report())
        output_path = tmp_path / "out.md"
        rc = main(["--report", str(report_path), "--output", str(output_path)])
        assert rc == 0
        # tmp_path is outside REPO_ROOT → header carries the absolute posix path.
        assert report_path.as_posix() in output_path.read_text(encoding="utf-8")


def test_module_guard_runs_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Executing the module as ``__main__`` runs ``main`` and exits 0."""
    report_path = tmp_path / "gate.json"
    _write_report(report_path, _report())
    output_path = tmp_path / "out.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_psi_gate_attribution",
            "--report",
            str(report_path),
            "--output",
            str(output_path),
        ],
    )
    # Drop the already-imported module so runpy re-executes it fresh under the
    # ``__main__`` name without a "found in sys.modules" RuntimeWarning.
    monkeypatch.delitem(sys.modules, "tools.generate_psi_gate_attribution", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("tools.generate_psi_gate_attribution", run_name="__main__")
    assert excinfo.value.code == 0
    assert output_path.exists()
