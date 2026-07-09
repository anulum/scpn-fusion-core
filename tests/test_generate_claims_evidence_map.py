# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# ----------------------------------------------------------------------
# SCPN Fusion Core -- Claims Evidence Map Generator Tests
# ----------------------------------------------------------------------
"""Tests for tools/generate_claims_evidence_map.py."""

from __future__ import annotations

import importlib.util
import json
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "generate_claims_evidence_map.py"
SPEC = importlib.util.spec_from_file_location("tools.generate_claims_evidence_map", MODULE_PATH)
assert SPEC and SPEC.loader
claims_map = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = claims_map
SPEC.loader.exec_module(claims_map)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _manifest_path(tmp_path: Path, claims: list[dict[str, object]]) -> Path:
    path = tmp_path / "claims_manifest.json"
    _write_json(path, {"claims": claims})
    return path


def _minimal_claim(**overrides: object) -> dict[str, object]:
    claim: dict[str, object] = {
        "id": "claim-a",
        "source_file": "README.md",
        "source_pattern": "SCPN",
        "evidence_files": ["RESULTS.md"],
        "evidence_patterns": [],
    }
    claim.update(overrides)
    return claim


def test_render_markdown_includes_summary_and_claim_ids() -> None:
    """The repository manifest renders headline summary and detail sections."""
    manifest = ROOT / "validation" / "claims_manifest.json"
    claims = claims_map.load_manifest(manifest)
    rendered = claims_map.render_markdown(claims, manifest_path="validation/claims_manifest.json")

    assert "# Claims Evidence Map" in rendered
    assert "## Summary" in rendered
    assert "readme_control_latency_scope_claim" in rendered
    assert "results_real_shot_overall_pass" in rendered


def test_render_markdown_escapes_table_cells_and_empty_evidence(tmp_path: Path) -> None:
    """Markdown rendering escapes cell delimiters and renders empty evidence blocks."""
    manifest = _manifest_path(
        tmp_path,
        [
            _minimal_claim(
                id="claim|`a`",
                source_file="docs/claim|source.md",
                source_pattern="pattern|`value`",
                evidence_files=[],
                evidence_patterns=[{"file": "evidence|file.md", "pattern": "value|`present`"}],
            )
        ],
    )

    claims = claims_map.load_manifest(manifest)
    rendered = claims_map.render_markdown(claims, manifest_path=manifest.as_posix())

    assert "`claim\\|\\`a\\``" in rendered
    assert "`docs/claim|source.md`" in rendered
    assert "- (none)" in rendered
    assert "| `evidence\\|file.md` | `value\\|\\`present\\`` |" in rendered


def test_check_mode_reports_stale_output(tmp_path: Path) -> None:
    """Check mode returns a failure code when the generated map is stale."""
    manifest = _manifest_path(tmp_path, [_minimal_claim()])
    output = tmp_path / "CLAIMS_EVIDENCE_MAP.md"
    output.write_text("# stale\n", encoding="utf-8")

    rc = claims_map.main(["--manifest", str(manifest), "--output", str(output), "--check"])

    assert rc == 1


def test_check_mode_reports_missing_output(tmp_path: Path) -> None:
    """Check mode returns a failure code when the output file is absent."""
    manifest = _manifest_path(tmp_path, [_minimal_claim()])
    output = tmp_path / "missing.md"

    rc = claims_map.main(["--manifest", str(manifest), "--output", str(output), "--check"])

    assert rc == 1


def test_check_mode_passes_when_output_matches(tmp_path: Path) -> None:
    """Check mode returns success for current generated output."""
    manifest = _manifest_path(tmp_path, [_minimal_claim()])
    output = tmp_path / "CLAIMS_EVIDENCE_MAP.md"

    claims = claims_map.load_manifest(manifest)
    rendered = claims_map.render_markdown(claims, manifest_path=manifest.as_posix())
    output.write_text(rendered, encoding="utf-8")

    rc = claims_map.main(["--manifest", str(manifest), "--output", str(output), "--check"])

    assert rc == 0


def test_write_mode_creates_parent_directory_and_output(tmp_path: Path) -> None:
    """Write mode creates parent directories and writes the generated map."""
    manifest = _manifest_path(tmp_path, [_minimal_claim()])
    output = tmp_path / "nested" / "CLAIMS_EVIDENCE_MAP.md"

    rc = claims_map.main(["--manifest", str(manifest), "--output", str(output)])

    assert rc == 0
    assert output.read_text(encoding="utf-8").startswith("# Claims Evidence Map\n")


def test_script_entrypoint_exits_with_main_return_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The executable entrypoint delegates through ``main`` and exits with its code."""
    manifest = _manifest_path(tmp_path, [_minimal_claim()])
    output = tmp_path / "CLAIMS_EVIDENCE_MAP.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(MODULE_PATH), "--manifest", str(manifest), "--output", str(output)],
    )

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(MODULE_PATH), run_name="__main__")

    assert exc_info.value.code == 0
    assert output.exists()


def test_main_accepts_repo_relative_manifest_and_output(tmp_path: Path) -> None:
    """Relative CLI paths resolve from the repository root."""
    output = tmp_path / "CLAIMS_EVIDENCE_MAP.md"

    rc = claims_map.main(
        [
            "--manifest",
            "validation/claims_manifest.json",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    assert "`validation/claims_manifest.json`" in output.read_text(encoding="utf-8")


def test_main_resolves_relative_output_from_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relative output paths resolve from the configured repository root."""
    validation = tmp_path / "validation"
    validation.mkdir()
    _write_json(validation / "claims_manifest.json", {"claims": [_minimal_claim()]})
    monkeypatch.setattr(claims_map, "REPO_ROOT", tmp_path)

    rc = claims_map.main(
        [
            "--manifest",
            "validation/claims_manifest.json",
            "--output",
            "docs/internal/CLAIMS_EVIDENCE_MAP.md",
        ]
    )

    output = tmp_path / "docs" / "internal" / "CLAIMS_EVIDENCE_MAP.md"
    assert rc == 0
    assert output.exists()


def test_claims_manifest_defaults_optional_evidence_fields(tmp_path: Path) -> None:
    """Omitted optional evidence fields default to empty tuples."""
    manifest = _manifest_path(
        tmp_path,
        [
            {
                "id": "minimal",
                "source_file": "source.md",
                "source_pattern": "claim",
            }
        ],
    )

    claims = claims_map.load_manifest(manifest)

    assert claims[0].evidence_files == ()
    assert claims[0].evidence_patterns == ()


def test_claims_manifest_accepts_null_evidence_patterns(tmp_path: Path) -> None:
    """A manifest ``null`` evidence_patterns field is treated as empty."""
    manifest = _manifest_path(
        tmp_path,
        [_minimal_claim(evidence_patterns=None)],
    )

    claims = claims_map.load_manifest(manifest)

    assert claims[0].evidence_patterns == ()


def test_claims_manifest_schema_validation(tmp_path: Path) -> None:
    """Manifest loading rejects malformed claim contract shapes."""
    manifest = tmp_path / "claims_manifest.json"
    invalid_payloads: list[tuple[object, str]] = [
        ([], "JSON object"),
        ({}, "non-empty 'claims'"),
        ({"claims": []}, "non-empty 'claims'"),
        ({"claims": ["bad"]}, "must be an object"),
        ({"claims": [{"id": "", "source_file": "a", "source_pattern": "b"}]}, "non-empty"),
        ({"claims": [{"id": "x", "source_file": "", "source_pattern": "b"}]}, "non-empty"),
        ({"claims": [{"id": "x", "source_file": "a", "source_pattern": ""}]}, "non-empty"),
        (
            {
                "claims": [
                    {"id": "dup", "source_file": "a", "source_pattern": "b"},
                    {"id": "dup", "source_file": "c", "source_pattern": "d"},
                ]
            },
            "Duplicate claim id",
        ),
        (
            {"claims": [_minimal_claim(evidence_files="bad")]},
            "list of strings",
        ),
        (
            {"claims": [_minimal_claim(evidence_files=["ok", ""])]},
            "non-empty string",
        ),
        (
            {"claims": [_minimal_claim(evidence_patterns="bad")]},
            "evidence_patterns must be a list",
        ),
        (
            {"claims": [_minimal_claim(evidence_patterns=["bad"])]},
            "must be an object",
        ),
        (
            {"claims": [_minimal_claim(evidence_patterns=[{"file": "", "pattern": "x"}])]},
            "non-empty string",
        ),
        (
            {"claims": [_minimal_claim(evidence_patterns=[{"file": "x", "pattern": ""}])]},
            "non-empty string",
        ),
    ]

    for payload, message in invalid_payloads:
        _write_json(manifest, payload)
        with pytest.raises(ValueError, match=message):
            claims_map.load_manifest(manifest)
