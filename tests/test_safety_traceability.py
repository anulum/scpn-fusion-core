# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Safety Traceability Matrix Tests
"""Tests for the safety traceability matrix generator and guard (T-2)."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "generate_safety_traceability.py"
MANIFEST = ROOT / "validation" / "safety_traceability.json"
OUTPUT = ROOT / "docs" / "SAFETY_TRACEABILITY_MATRIX.md"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("generate_safety_traceability", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclass field resolution requires the module to be importable by name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest_payload() -> dict[str, Any]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_manifest(tmp_path: Path, payload: dict[str, Any]) -> Path:
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


class TestCommittedManifest:
    """The committed manifest is fully traceable and the doc is fresh."""

    def test_committed_manifest_verifies_clean(self) -> None:
        module = _load_module()
        manifest = module.load_manifest(MANIFEST)
        assert module.verify_manifest(manifest, ROOT) == []

    def test_committed_matrix_is_up_to_date(self) -> None:
        module = _load_module()
        assert module.main(["--check"]) == 0

    def test_every_requirement_has_tests_and_implementation(self) -> None:
        module = _load_module()
        manifest = module.load_manifest(MANIFEST)
        for requirement in manifest.requirements:
            assert len(requirement.implementation) >= 1
            assert len(requirement.tests) >= 1
            assert len(requirement.hazard_ids) >= 1

    def test_language_boundary_disclaims_certification(self) -> None:
        module = _load_module()
        manifest = module.load_manifest(MANIFEST)
        assert "No IEC 61508 compliance, certification, or SIL claim" in (
            manifest.language_boundary
        )

    def test_rendered_matrix_contains_all_ids_and_boundary(self) -> None:
        module = _load_module()
        manifest = module.load_manifest(MANIFEST)
        rendered = module.render_markdown(manifest, "validation/safety_traceability.json")
        for requirement in manifest.requirements:
            assert f"### `{requirement.requirement_id}`" in rendered
        for hazard in manifest.hazards:
            assert hazard.hazard_id in rendered
        assert "No IEC 61508 compliance, certification, or SIL claim" in rendered

    def test_rendering_is_deterministic(self) -> None:
        module = _load_module()
        manifest = module.load_manifest(MANIFEST)
        first = module.render_markdown(manifest, "m.json")
        second = module.render_markdown(manifest, "m.json")
        assert first == second


class TestManifestSchemaValidation:
    """Structural validation fails closed on malformed manifests."""

    def test_rejects_foreign_schema(self, tmp_path: Path) -> None:
        module = _load_module()
        payload = _manifest_payload()
        payload["schema"] = "other.schema"
        with pytest.raises(ValueError, match="unexpected manifest schema"):
            module.load_manifest(_write_manifest(tmp_path, payload))

    def test_rejects_non_object_root(self, tmp_path: Path) -> None:
        module = _load_module()
        target = tmp_path / "manifest.json"
        target.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a JSON object"):
            module.load_manifest(target)

    def test_rejects_duplicate_requirement_id(self, tmp_path: Path) -> None:
        module = _load_module()
        payload = _manifest_payload()
        payload["requirements"].append(dict(payload["requirements"][0]))
        with pytest.raises(ValueError, match="Duplicate requirement id"):
            module.load_manifest(_write_manifest(tmp_path, payload))

    def test_rejects_duplicate_hazard_id(self, tmp_path: Path) -> None:
        module = _load_module()
        payload = _manifest_payload()
        payload["hazards"].append(dict(payload["hazards"][0]))
        with pytest.raises(ValueError, match="Duplicate hazard id"):
            module.load_manifest(_write_manifest(tmp_path, payload))

    def test_rejects_empty_requirements(self, tmp_path: Path) -> None:
        module = _load_module()
        payload = _manifest_payload()
        payload["requirements"] = []
        with pytest.raises(ValueError, match="requirements must be a non-empty list"):
            module.load_manifest(_write_manifest(tmp_path, payload))

    def test_rejects_empty_hazards(self, tmp_path: Path) -> None:
        module = _load_module()
        payload = _manifest_payload()
        payload["hazards"] = []
        with pytest.raises(ValueError, match="hazards must be a non-empty list"):
            module.load_manifest(_write_manifest(tmp_path, payload))

    def test_rejects_requirement_without_tests(self, tmp_path: Path) -> None:
        module = _load_module()
        payload = _manifest_payload()
        payload["requirements"][0]["tests"] = []
        with pytest.raises(ValueError, match="at least 1 entries"):
            module.load_manifest(_write_manifest(tmp_path, payload))

    def test_rejects_missing_statement(self, tmp_path: Path) -> None:
        module = _load_module()
        payload = _manifest_payload()
        payload["requirements"][0]["statement"] = ""
        with pytest.raises(ValueError, match="statement must be a non-empty string"):
            module.load_manifest(_write_manifest(tmp_path, payload))

    def test_rejects_non_object_public_claim(self, tmp_path: Path) -> None:
        module = _load_module()
        payload = _manifest_payload()
        payload["requirements"][0]["public_claim"] = "README.md"
        with pytest.raises(ValueError, match="public_claim must be an object"):
            module.load_manifest(_write_manifest(tmp_path, payload))

    def test_rejects_non_list_anchor_field(self, tmp_path: Path) -> None:
        module = _load_module()
        payload = _manifest_payload()
        payload["requirements"][0]["implementation"] = "not-a-list"
        with pytest.raises(ValueError, match="must be a list"):
            module.load_manifest(_write_manifest(tmp_path, payload))

    def test_rejects_non_object_requirement(self, tmp_path: Path) -> None:
        module = _load_module()
        payload = _manifest_payload()
        payload["requirements"][0] = "not-an-object"
        with pytest.raises(ValueError, match=re.escape("requirements[0] must be an object")):
            module.load_manifest(_write_manifest(tmp_path, payload))

    def test_rejects_non_object_hazard(self, tmp_path: Path) -> None:
        module = _load_module()
        payload = _manifest_payload()
        payload["hazards"][0] = "not-an-object"
        with pytest.raises(ValueError, match=re.escape("hazards[0] must be an object")):
            module.load_manifest(_write_manifest(tmp_path, payload))


class TestAnchorVerification:
    """Anchor verification catches removed or renamed linked entities."""

    def _verify_with_mutation(self, mutate: Any) -> list[str]:
        module = _load_module()
        payload = _manifest_payload()
        mutate(payload)
        target = ROOT / "validation" / "_tmp_traceability_test.json"
        try:
            target.write_text(json.dumps(payload), encoding="utf-8")
            manifest = module.load_manifest(target)
            return list(module.verify_manifest(manifest, ROOT))
        finally:
            target.unlink(missing_ok=True)

    def test_detects_missing_test_function(self) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            payload["requirements"][0]["tests"] = [
                "tests/test_safety_interlocks.py::test_this_function_does_not_exist"
            ]

        errors = self._verify_with_mutation(mutate)
        assert any("symbol not found" in error for error in errors)

    def test_detects_missing_lean_theorem(self) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            payload["requirements"][0]["proofs"] = [
                "scpn-fusion-lean/PIDBoundedOutput.lean::nonexistent_theorem"
            ]

        errors = self._verify_with_mutation(mutate)
        assert any("nonexistent_theorem" in error for error in errors)

    def test_detects_missing_implementation_symbol(self) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            payload["requirements"][0]["implementation"] = [
                "src/scpn_fusion/scpn/safety_interlocks.py::NoSuchRuntime"
            ]

        errors = self._verify_with_mutation(mutate)
        assert any("NoSuchRuntime" in error for error in errors)

    def test_detects_missing_file(self) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            payload["requirements"][0]["evidence"] = ["validation/reports/does_not_exist.json"]

        errors = self._verify_with_mutation(mutate)
        assert any("file not found" in error for error in errors)

    def test_detects_missing_public_claim_pattern(self) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            payload["requirements"][0]["public_claim"] = {
                "file": "README.md",
                "pattern": "this claim text is definitely not in the readme",
            }

        errors = self._verify_with_mutation(mutate)
        assert any("public claim pattern not found" in error for error in errors)

    def test_detects_missing_public_claim_file(self) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            payload["requirements"][0]["public_claim"] = {
                "file": "docs/NO_SUCH_DOC.md",
                "pattern": "anything",
            }

        errors = self._verify_with_mutation(mutate)
        assert any("public claim file not found" in error for error in errors)

    def test_detects_unknown_hazard_reference(self) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            payload["requirements"][0]["hazard_ids"] = ["H-UNDECLARED"]

        errors = self._verify_with_mutation(mutate)
        assert any("unknown hazard id: H-UNDECLARED" in error for error in errors)

    def test_detects_unreferenced_hazard(self) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            payload["hazards"].append({"id": "H-ORPHAN", "description": "Never referenced."})

        errors = self._verify_with_mutation(mutate)
        assert any("H-ORPHAN is not referenced" in error for error in errors)

    def test_detects_requirement_without_hazards(self) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            payload["requirements"][0]["hazard_ids"] = []

        errors = self._verify_with_mutation(mutate)
        assert any("must reference at least one hazard" in error for error in errors)

    def test_rejects_symbol_anchor_on_unsupported_suffix(self) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            payload["requirements"][0]["evidence"] = ["README.md::some_symbol"]

        errors = self._verify_with_mutation(mutate)
        assert any("symbol anchors unsupported" in error for error in errors)


class TestCliModes:
    """CLI write and check modes behave fail-closed."""

    def test_write_mode_creates_output(self, tmp_path: Path) -> None:
        module = _load_module()
        output = tmp_path / "matrix.md"
        assert module.main(["--output", str(output)]) == 0
        assert "# Safety Traceability Matrix" in output.read_text(encoding="utf-8")

    def test_check_mode_fails_on_missing_output(self, tmp_path: Path) -> None:
        module = _load_module()
        assert module.main(["--output", str(tmp_path / "missing.md"), "--check"]) == 1

    def test_check_mode_fails_on_stale_output(self, tmp_path: Path) -> None:
        module = _load_module()
        output = tmp_path / "matrix.md"
        assert module.main(["--output", str(output)]) == 0
        output.write_text(output.read_text(encoding="utf-8") + "\nmanual edit\n", encoding="utf-8")
        assert module.main(["--output", str(output), "--check"]) == 1

    def test_check_mode_fails_on_broken_anchor(self, tmp_path: Path) -> None:
        module = _load_module()
        payload = _manifest_payload()
        payload["requirements"][0]["tests"] = [
            "tests/test_safety_interlocks.py::test_removed_from_suite"
        ]
        manifest = _write_manifest(tmp_path, payload)
        output = tmp_path / "matrix.md"
        assert module.main(["--manifest", str(manifest), "--output", str(output)]) == 1
        assert not output.exists()

    def test_relative_paths_resolve_against_repo_root(self, tmp_path: Path) -> None:
        module = _load_module()
        assert (
            module.main(
                [
                    "--manifest",
                    "validation/safety_traceability.json",
                    "--output",
                    str(tmp_path / "matrix.md"),
                ]
            )
            == 0
        )


class TestAnchorParsing:
    """Anchor label round-trips and symbol parsing edge cases."""

    def test_anchor_label_with_symbol(self) -> None:
        module = _load_module()
        anchor = module.Anchor(file="a/b.py", symbol="thing")
        assert anchor.label == "a/b.py::thing"

    def test_anchor_label_without_symbol(self) -> None:
        module = _load_module()
        anchor = module.Anchor(file="a/b.json", symbol=None)
        assert anchor.label == "a/b.json"

    def test_empty_symbol_after_separator_rejected(self, tmp_path: Path) -> None:
        module = _load_module()
        payload = _manifest_payload()
        payload["requirements"][0]["implementation"] = ["src/scpn_fusion/scpn/contracts.py::"]
        with pytest.raises(ValueError, match="non-empty string"):
            module.load_manifest(_write_manifest(tmp_path, payload))
