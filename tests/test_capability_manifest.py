# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core capability manifest tests

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_tool() -> Any:
    tool_path = _repo_root() / "tools" / "capability_manifest.py"
    spec = importlib.util.spec_from_file_location("capability_manifest", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_scans_fusion_core_capability_surfaces() -> None:
    tool = _load_tool()

    manifest = tool.build_capability_manifest(_repo_root())

    assert manifest["schema_version"] == tool.CAPABILITY_MANIFEST_SCHEMA_VERSION
    assert manifest["project_label"] == "SCPN Fusion Core"
    assert manifest["generated_from"]["config"] == "tools/capability_manifest.toml"
    assert manifest["project"]["name"] == "scpn-fusion"
    assert manifest["project"]["readme"] == "README.md"
    assert manifest["counts"]["public_api_exports"] == len(manifest["package_exports"])
    assert manifest["counts"]["python_capability_source_modules"] == len(
        manifest["capabilities"]["python_source_modules"]
    )
    assert manifest["counts"]["python_capability_classes"] == len(
        manifest["capabilities"]["python_classes"]
    )
    assert manifest["counts"]["rust_workspace_crates"] == len(
        manifest["capabilities"]["rust_workspace_crates"]
    )
    assert "full" in manifest["packaging"]["optional_extras"]
    assert ".github/workflows/ci.yml" in manifest["quality_gates"]["github_workflows"]
    assert "tests/test_hypothesis_properties.py" in manifest["quality_gates"]["test_files"]
    assert "docs/internal/AUDIT_INDEX.md" not in manifest["documentation"]["public_pages"]


def test_manifest_validation_rejects_count_drift() -> None:
    tool = _load_tool()
    manifest = tool.build_capability_manifest(_repo_root())
    manifest["counts"]["python_capability_classes"] += 1

    report = tool.validate_manifest(manifest)

    assert not report["passed"]
    assert "counts.python_capability_classes does not match list length" in report["errors"]


def test_generated_outputs_are_current() -> None:
    tool = _load_tool()

    tool.assert_outputs_current(_repo_root())


def test_readme_snapshot_matches_generated_markdown() -> None:
    tool = _load_tool()
    config = tool.load_config(_repo_root())
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")

    block = (
        readme.split(config.readme_marker_start, maxsplit=1)[1]
        .split(
            config.readme_marker_end,
            maxsplit=1,
        )[0]
        .strip()
    )

    assert (
        block == tool.render_markdown_snapshot(tool.build_capability_manifest(_repo_root())).strip()
    )


def test_readme_does_not_shadow_generated_inventory_with_manual_counts() -> None:
    tool = _load_tool()
    config = tool.load_config(_repo_root())
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    before, rest = readme.split(config.readme_marker_start, maxsplit=1)
    _block, after = rest.split(config.readme_marker_end, maxsplit=1)
    public_text = before + after

    stale_patterns = [
        r"\b\d[\d,]*\s+Python modules\b",
        r"\b\d[\d,]*\s+Python source files\b",
        r"\b\d[\d,]*\s+Rust crates\b",
        r"\b\d[\d,]*\s+tests\b",
        r"\bTests-\d",
    ]

    for pattern in stale_patterns:
        assert re.search(pattern, public_text) is None, pattern


def test_readme_exposes_monthly_and_all_time_download_badges() -> None:
    """Lock README package download badges to the published package name."""
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")

    assert "[![PyPI Downloads](https://img.shields.io/pypi/dm/scpn-fusion.svg)]" in readme
    assert "(https://pypi.org/project/scpn-fusion/)" in readme
    assert "[![All-time Downloads](https://static.pepy.tech/badge/scpn-fusion)]" in readme
    assert "(https://pepy.tech/project/scpn-fusion)" in readme


def test_manifest_validation_rejects_schema_and_type_drift() -> None:
    tool = _load_tool()
    manifest = tool.build_capability_manifest(_repo_root())
    manifest["schema_version"] = "wrong"
    manifest["counts"]["rust_workspace_crates"] = "11"
    del manifest["capabilities"]["rust_workspace_crates"]

    report = tool.validate_manifest(manifest)

    assert not report["passed"]
    assert "schema_version mismatch" in report["errors"]
    assert "counts.rust_workspace_crates must be a non-negative integer" in report["errors"]
    assert "capabilities list missing for count rust_workspace_crates" in report["errors"]


def test_manifest_declares_committed_json_schema_contract() -> None:
    tool = _load_tool()
    manifest = tool.build_capability_manifest(_repo_root())
    schema_path = _repo_root() / "schemas" / "capability_manifest.schema.json"

    assert manifest["generated_from"]["schema"] == "schemas/capability_manifest.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert (
        schema["$id"]
        == "https://anulum.github.io/scpn-fusion-core/schemas/capability_manifest.schema.json"
    )
    assert (
        schema["properties"]["schema_version"]["const"] == tool.CAPABILITY_MANIFEST_SCHEMA_VERSION
    )
    assert set(schema["properties"]["counts"]["required"]) == set(manifest["counts"])
    assert set(schema["properties"]["capabilities"]["required"]) == set(manifest["capabilities"])


def test_generator_defaults_are_fusion_core_specific() -> None:
    tool = _load_tool()

    config = tool.load_config(_repo_root(), Path("missing_capability_manifest.toml"))

    assert config.project_label == "SCPN Fusion Core"
    assert config.package_root == Path("src/scpn_fusion")
    assert config.capability_sources == (
        Path("src/scpn_fusion/core"),
        Path("src/scpn_fusion/control"),
        Path("src/scpn_fusion/phase"),
        Path("src/scpn_fusion/scpn"),
        Path("src/scpn_fusion/diagnostics"),
        Path("src/scpn_fusion/engineering"),
        Path("src/scpn_fusion/nuclear"),
        Path("src/scpn_fusion/io"),
        Path("src/scpn_fusion/hpc"),
        Path("src/scpn_fusion/ui"),
    )
    assert config.capability_docs == Path("docs")
    assert config.rust_workspace == Path("scpn-fusion-rs")


def test_public_generator_files_do_not_retain_source_repository_defaults() -> None:
    checked_paths = [
        _repo_root() / "tools" / "capability_manifest.py",
        _repo_root() / "tools" / "capability_manifest.toml",
        _repo_root() / "docs" / "_generated" / "capability_manifest.json",
        _repo_root() / "docs" / "_generated" / "capability_snapshot.md",
        _repo_root() / "schemas" / "capability_manifest.schema.json",
    ]

    forbidden = ("SC-NeuroCore", "src/sc_neurocore", "pyo3_neurons.rs")
    for path in checked_paths:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{token} leaked into {path.relative_to(_repo_root())}"


def test_cli_uses_portable_multi_root_config_and_refreshes_readme() -> None:
    tool_path = _repo_root() / "tools" / "capability_manifest.py"
    with _tempdir() as repo:
        _write_portable_fixture(repo)

        result = subprocess.run(
            [
                sys.executable,
                str(tool_path),
                "--repo",
                str(repo),
                "--config",
                "tools/capability_manifest.toml",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        manifest = json.loads(
            (repo / "docs/_generated/capability_manifest.json").read_text(encoding="utf-8")
        )
        readme = (repo / "README.md").read_text(encoding="utf-8")

        assert manifest["project_label"] == "Portable Fusion Project"
        assert manifest["counts"]["python_capability_source_modules"] == 2
        assert manifest["counts"]["python_capability_classes"] == 2
        assert manifest["counts"]["rust_workspace_crates"] == 2
        assert "Portable Fusion Project Capability Inventory" in readme
        assert "| Python capability source modules | 2 |" in readme


@contextmanager
def _tempdir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


def _write_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_portable_fixture(repo: Path) -> None:
    _write_file(
        repo / "pyproject.toml",
        "\n".join(
            [
                "[project]",
                'name = "portable-fusion-project"',
                'version = "1.2.3"',
                'requires-python = ">=3.10"',
                'readme = "README.md"',
                'license = "AGPL-3.0-or-later"',
                "",
                "[project.optional-dependencies]",
                'full = ["numpy"]',
                "",
            ]
        ),
    )
    _write_file(
        repo / "README.md",
        "\n".join(
            [
                "# Portable Fusion Project",
                "",
                "<!-- capability-snapshot:start -->",
                "stale",
                "<!-- capability-snapshot:end -->",
                "",
            ]
        ),
    )
    _write_file(repo / "src/portable_fusion/__init__.py", '__all__ = ["CoreModel"]\n')
    _write_file(repo / "src/portable_fusion/core/equilibrium.py", "class CoreModel:\n    pass\n")
    _write_file(repo / "src/portable_fusion/control/replay.py", "class ReplayModel:\n    pass\n")
    _write_file(repo / "docs/internal/private.md", "# Private\n")
    _write_file(repo / "docs/public.md", "# Public\n")
    _write_file(repo / "tests/test_portable.py", "def test_portable() -> None:\n    assert True\n")
    _write_file(repo / ".github/workflows/ci.yml", "name: CI\non: [push]\njobs: {}\n")
    _write_file(repo / "fusion-rs/Cargo.toml", '[workspace]\nmembers = ["a", "b"]\n')
    _write_file(repo / "fusion-rs/a/Cargo.toml", '[package]\nname = "a"\nversion = "0.1.0"\n')
    _write_file(repo / "fusion-rs/b/Cargo.toml", '[package]\nname = "b"\nversion = "0.1.0"\n')
    _write_file(
        repo / "tools/capability_manifest.toml",
        "\n".join(
            [
                'project_label = "Portable Fusion Project"',
                'schema_version = "capability-manifest.v1"',
                'exclude_doc_parts = ["internal", "_generated"]',
                "",
                "[paths]",
                'json_output = "docs/_generated/capability_manifest.json"',
                'markdown_output = "docs/_generated/capability_snapshot.md"',
                'package_root = "src/portable_fusion"',
                'capability_sources = ["src/portable_fusion/core", "src/portable_fusion/control"]',
                'capability_docs = "docs"',
                'tests_root = "tests"',
                'docs_root = "docs"',
                'workflows_root = ".github/workflows"',
                'rust_workspace = "fusion-rs"',
                "",
                "[readme]",
                'path = "README.md"',
                'marker_start = "<!-- capability-snapshot:start -->"',
                'marker_end = "<!-- capability-snapshot:end -->"',
                "",
            ]
        ),
    )
