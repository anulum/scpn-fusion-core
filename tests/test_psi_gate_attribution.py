# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Psi Gate Attribution Page Tests
"""Tests for the ψ_N gate per-file attribution page generator (F-1)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "generate_psi_gate_attribution.py"
REPORT = ROOT / "validation" / "reports" / "psi_efit_nrmse_benchmark.json"
OUTPUT = ROOT / "docs" / "PSI_GATE_ATTRIBUTION.md"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("generate_psi_gate_attribution", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _report_payload() -> dict[str, Any]:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


class TestCommittedPage:
    """The committed attribution page tracks the committed gate report."""

    def test_committed_page_is_up_to_date(self) -> None:
        module = _load_module()
        assert module.main(["--check"]) == 0

    def test_render_carries_honest_boundary_language(self) -> None:
        module = _load_module()
        report = module.load_report(REPORT)
        rendered = module.render_markdown(
            report, "validation/reports/psi_efit_nrmse_benchmark.json"
        )
        assert "intentionally reported failing" in rendered
        assert "EFIT-grade inverse reconstruction is **not** claimed" in rendered
        assert "external_coil_currents_missing_from_geqdsk" in rendered

    def test_render_lists_every_gate_file(self) -> None:
        module = _load_module()
        report = module.load_report(REPORT)
        rendered = module.render_markdown(report, "r.json")
        for row in report["rows"]:
            assert f"`{row['file']}`" in rendered

    def test_render_is_deterministic(self) -> None:
        module = _load_module()
        report = module.load_report(REPORT)
        assert module.render_markdown(report, "r.json") == module.render_markdown(report, "r.json")

    def test_nan_percentages_render_as_dash(self) -> None:
        module = _load_module()
        assert module._fmt_pct(float("nan")) == "—"
        assert module._fmt_pct(0.05) == "5.0%"


class TestReportValidation:
    """Loading fails closed on malformed gate reports."""

    def test_rejects_foreign_schema(self, tmp_path: Path) -> None:
        module = _load_module()
        payload = _report_payload()
        payload["schema_version"] = "other.schema"
        target = tmp_path / "report.json"
        target.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="unexpected gate report schema"):
            module.load_report(target)

    def test_rejects_non_object_root(self, tmp_path: Path) -> None:
        module = _load_module()
        target = tmp_path / "report.json"
        target.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a JSON object"):
            module.load_report(target)

    def test_rejects_empty_rows(self, tmp_path: Path) -> None:
        module = _load_module()
        payload = _report_payload()
        payload["rows"] = []
        target = tmp_path / "report.json"
        target.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty 'rows' list"):
            module.load_report(target)

    def test_rejects_row_missing_attribution_field(self, tmp_path: Path) -> None:
        module = _load_module()
        payload = _report_payload()
        del payload["rows"][0]["source_domain_next_action"]
        target = tmp_path / "report.json"
        target.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="missing attribution fields"):
            module.load_report(target)

    def test_rejects_non_object_row(self, tmp_path: Path) -> None:
        module = _load_module()
        payload = _report_payload()
        payload["rows"][0] = "not-an-object"
        target = tmp_path / "report.json"
        target.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="rows\\[0\\] must be an object"):
            module.load_report(target)


class TestCliModes:
    """CLI write and check modes behave fail-closed."""

    def test_write_mode_creates_output(self, tmp_path: Path) -> None:
        module = _load_module()
        output = tmp_path / "attribution.md"
        assert module.main(["--output", str(output)]) == 0
        assert "Per-File Attribution" in output.read_text(encoding="utf-8")

    def test_check_mode_fails_on_missing_output(self, tmp_path: Path) -> None:
        module = _load_module()
        assert module.main(["--output", str(tmp_path / "missing.md"), "--check"]) == 1

    def test_check_mode_fails_on_stale_output(self, tmp_path: Path) -> None:
        module = _load_module()
        output = tmp_path / "attribution.md"
        assert module.main(["--output", str(output)]) == 0
        output.write_text(output.read_text(encoding="utf-8") + "\nedit\n", encoding="utf-8")
        assert module.main(["--output", str(output), "--check"]) == 1

    def test_relative_report_path_resolves_against_repo_root(self, tmp_path: Path) -> None:
        module = _load_module()
        rc = module.main(
            [
                "--report",
                "validation/reports/psi_efit_nrmse_benchmark.json",
                "--output",
                str(tmp_path / "attribution.md"),
            ]
        )
        assert rc == 0
