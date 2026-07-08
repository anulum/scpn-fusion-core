# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FAIR Validation Pack Export Tests
"""Tests for local FAIR validation pack export and readiness drift checks."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "export_zenodo_dataset.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("export_zenodo_dataset", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


class TestCommittedPackDefinitions:
    """The committed FAIR pack definitions resolve to tracked evidence files."""

    def test_committed_pack_definitions_have_three_ready_packs(self) -> None:
        module = _load_module()
        manifests = module.build_all_manifests(ROOT)

        assert len(manifests) == 3
        assert {manifest.definition.pack_id for manifest in manifests} == {
            "safety_traceability",
            "surrogate_uq_cards",
            "inverse_equilibrium_attribution",
        }
        for manifest in manifests:
            assert manifest.file_count >= 4
            assert manifest.total_bytes > 0
            assert all(len(record.sha256) == 64 for record in manifest.files)

    def test_readiness_report_accepts_local_pack_structure_only(self) -> None:
        module = _load_module()
        manifests = module.build_all_manifests(ROOT)
        report = module.build_readiness_report(
            manifests=manifests,
            output_dir=ROOT / "artifacts" / "fair_validation_packs",
            root=ROOT,
        )

        assert report["schema"] == "scpn-fusion-core.fair-validation-packs.v1"
        assert report["status"] == "accepted_local_fair_pack_readiness"
        assert report["accepted_local_pack_readiness"] is True
        assert report["doi_publication_ready"] is False
        assert report["pack_count"] == 3
        assert report["publication_blockers"]

    def test_markdown_names_publication_boundary(self) -> None:
        module = _load_module()
        report = module.build_readiness_report(
            manifests=module.build_all_manifests(ROOT),
            output_dir=ROOT / "artifacts" / "fair_validation_packs",
            root=ROOT,
        )
        rendered = module.render_readiness_markdown(report)

        assert "# FAIR Validation Pack Readiness" in rendered
        assert "DOI publication remains owner-gated" in rendered
        assert "`safety_traceability`" in rendered

    def test_blocked_report_names_local_pack_count_gap(self) -> None:
        module = _load_module()
        manifest = module.build_pack_manifest(module.PACKS[0], root=ROOT)
        report = module.build_readiness_report(
            manifests=(manifest,),
            output_dir=ROOT / "artifacts" / "fair_validation_packs",
            root=ROOT,
        )
        rendered = module.render_readiness_markdown(report)

        assert report["status"] == "blocked_local_pack_readiness"
        assert "at least 3 local packs required" in rendered
        assert "## Local blockers" in rendered

    def test_outside_repo_relative_path_is_rejected(self, tmp_path: Path) -> None:
        module = _load_module()

        with pytest.raises(ValueError, match="outside repository root"):
            module._repo_relative(tmp_path / "outside.txt", ROOT)

    def test_absolute_pack_input_is_rejected(self) -> None:
        module = _load_module()
        definition = module.PackDefinition(
            pack_id="absolute",
            title="Absolute Path Pack",
            description="Invalid absolute path fixture.",
            license_id="AGPL-3.0-or-later",
            files=(str((ROOT / "README.md").resolve()),),
        )

        with pytest.raises(ValueError, match="inside the repository"):
            module.build_pack_manifest(definition, root=ROOT)

    def test_malformed_readiness_payloads_fail_closed(self) -> None:
        module = _load_module()
        report = module.build_readiness_report(
            manifests=module.build_all_manifests(ROOT),
            output_dir=ROOT / "artifacts" / "fair_validation_packs",
            root=ROOT,
        )

        bad_packs = dict(report)
        bad_packs["packs"] = {}
        with pytest.raises(ValueError, match="packs must be a list"):
            module.render_readiness_markdown(bad_packs)

        bad_publication = dict(report)
        bad_publication["publication_blockers"] = "owner"
        with pytest.raises(ValueError, match="publication_blockers must be a list"):
            module.render_readiness_markdown(bad_publication)

        bad_local = dict(report)
        bad_local["local_blockers"] = "none"
        with pytest.raises(ValueError, match="local_blockers must be a list"):
            module.render_readiness_markdown(bad_local)

        bad_row = dict(report)
        bad_row["packs"] = ["not-an-object"]
        with pytest.raises(ValueError, match="pack rows must be objects"):
            module.render_readiness_markdown(bad_row)


class TestExportAndCheckModes:
    """CLI-facing export and drift-check paths work on real pack inputs."""

    def test_export_creates_pack_directories_and_readiness_reports(self, tmp_path: Path) -> None:
        module = _load_module()
        output_dir = tmp_path / "packs"
        json_report = tmp_path / "fair_validation_packs.json"
        md_report = tmp_path / "fair_validation_packs.md"

        rc = module.run_export(
            output_dir=output_dir,
            json_report=json_report,
            md_report=md_report,
        )

        assert rc == 0
        report = _json_payload(json_report)
        assert report["accepted_local_pack_readiness"] is True
        for pack_id in {
            "safety_traceability",
            "surrogate_uq_cards",
            "inverse_equilibrium_attribution",
        }:
            pack_dir = output_dir / pack_id
            assert (pack_dir / "README.md").is_file()
            manifest = _json_payload(pack_dir / "pack_manifest.json")
            assert manifest["pack_id"] == pack_id
            assert manifest["file_count"] > 0
            for row in manifest["files"]:
                assert (pack_dir / "files" / row["path"]).is_file()
        assert "# FAIR Validation Pack Readiness" in md_report.read_text(encoding="utf-8")

    def test_export_replaces_stale_pack_directory(self, tmp_path: Path) -> None:
        module = _load_module()
        output_dir = tmp_path / "packs"
        stale_file = output_dir / "safety_traceability" / "stale.txt"
        stale_file.parent.mkdir(parents=True)
        stale_file.write_text("stale", encoding="utf-8")

        assert module.run_export(output_dir=output_dir) == 0

        assert not stale_file.exists()
        assert (output_dir / "safety_traceability" / "pack_manifest.json").is_file()

    def test_check_mode_accepts_fresh_reports_without_reexporting_packs(
        self, tmp_path: Path
    ) -> None:
        module = _load_module()
        output_dir = tmp_path / "packs"
        json_report = tmp_path / "fair_validation_packs.json"
        md_report = tmp_path / "fair_validation_packs.md"

        assert (
            module.run_export(
                output_dir=output_dir,
                json_report=json_report,
                md_report=md_report,
            )
            == 0
        )
        shutil.rmtree(output_dir)

        assert (
            module.run_export(
                output_dir=output_dir,
                json_report=json_report,
                md_report=md_report,
                check=True,
                export_files=False,
            )
            == 0
        )
        assert not output_dir.exists()

    def test_check_mode_rejects_stale_json_report(self, tmp_path: Path) -> None:
        module = _load_module()
        output_dir = tmp_path / "packs"
        json_report = tmp_path / "fair_validation_packs.json"
        md_report = tmp_path / "fair_validation_packs.md"

        assert (
            module.run_export(
                output_dir=output_dir,
                json_report=json_report,
                md_report=md_report,
            )
            == 0
        )
        payload = _json_payload(json_report)
        payload["pack_count"] = 0
        json_report.write_text(json.dumps(payload), encoding="utf-8")

        assert (
            module.run_export(
                output_dir=output_dir,
                json_report=json_report,
                md_report=md_report,
                check=True,
                export_files=False,
            )
            == 1
        )

    def test_check_mode_rejects_stale_markdown_report(self, tmp_path: Path) -> None:
        module = _load_module()
        output_dir = tmp_path / "packs"
        json_report = tmp_path / "fair_validation_packs.json"
        md_report = tmp_path / "fair_validation_packs.md"

        assert (
            module.run_export(
                output_dir=output_dir,
                json_report=json_report,
                md_report=md_report,
            )
            == 0
        )
        md_report.write_text(md_report.read_text(encoding="utf-8") + "\nstale\n", encoding="utf-8")

        assert (
            module.run_export(
                output_dir=output_dir,
                json_report=json_report,
                md_report=md_report,
                check=True,
                export_files=False,
            )
            == 1
        )

    def test_main_writes_reports_with_cli_paths(self, tmp_path: Path) -> None:
        module = _load_module()
        output_dir = tmp_path / "packs"
        json_report = tmp_path / "report.json"
        md_report = tmp_path / "report.md"

        assert (
            module.main(
                [
                    "--output-dir",
                    str(output_dir),
                    "--json-report",
                    str(json_report),
                    "--md-report",
                    str(md_report),
                    "--no-export",
                ]
            )
            == 0
        )
        assert _json_payload(json_report)["pack_count"] == 3
        assert not output_dir.exists()

    def test_missing_pack_input_fails_closed(self) -> None:
        module = _load_module()
        definition = module.PackDefinition(
            pack_id="bad",
            title="Bad Pack",
            description="Missing file fixture.",
            license_id="AGPL-3.0-or-later",
            files=("validation/no_such_file.json",),
        )

        with pytest.raises(FileNotFoundError, match="pack input missing"):
            module.build_pack_manifest(definition, root=ROOT)
