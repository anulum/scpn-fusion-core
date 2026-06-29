# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core capability manifest tests

"""Tests for capability manifest generation, drift checks, and README refreshes."""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import runpy
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest


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


def _public_readme_text_without_generated_inventory() -> str:
    """Return README text outside the generated capability inventory block."""
    tool = _load_tool()
    config = tool.load_config(_repo_root())
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    before, rest = readme.split(config.readme_marker_start, maxsplit=1)
    _block, after = rest.split(config.readme_marker_end, maxsplit=1)
    return before + after


def _itpa_hmode_counts() -> tuple[int, int]:
    """Return live ITPA H-mode shot and machine counts from the bundled CSV."""
    csv_path = _repo_root() / "validation" / "reference_data" / "itpa" / "hmode_confinement.csv"
    machines: set[str] = set()
    shot_count = 0
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            shot_count += 1
            machine = row["machine"].strip()
            assert machine
            machines.add(machine)
    return shot_count, len(machines)


def test_manifest_scans_fusion_core_capability_surfaces() -> None:
    """Verify the generated manifest reflects live Fusion Core surfaces."""
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
    """Verify manifest validation rejects mismatched capability counts."""
    tool = _load_tool()
    manifest = tool.build_capability_manifest(_repo_root())
    manifest["counts"]["python_capability_classes"] += 1

    report = tool.validate_manifest(manifest)

    assert not report["passed"]
    assert "counts.python_capability_classes does not match list length" in report["errors"]


def test_generated_outputs_are_current() -> None:
    """Verify generated manifest outputs are synchronized with source files."""
    tool = _load_tool()

    tool.assert_outputs_current(_repo_root())


def test_readme_snapshot_matches_generated_markdown() -> None:
    """Verify the README inventory block matches generated Markdown exactly."""
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
    """Verify README prose does not duplicate generated inventory counts."""
    public_text = _public_readme_text_without_generated_inventory()

    stale_patterns = [
        r"\b\d[\d,]*\s+Python modules\b",
        r"\b\d[\d,]*\s+Python source files\b",
        r"\b\d[\d,]*\s+Rust crates\b",
        r"\b\d[\d,]*\s+tests\b",
        r"\bTests-\d",
    ]

    for pattern in stale_patterns:
        assert re.search(pattern, public_text) is None, pattern


def test_public_docs_do_not_carry_known_stale_inventory_counts() -> None:
    """Verify public docs do not retain manual inventory counts outside generated output."""
    checked_docs = {
        "README.md": _public_readme_text_without_generated_inventory(),
        "docs/ARCHITECTURE.md": (_repo_root() / "docs" / "ARCHITECTURE.md").read_text(
            encoding="utf-8"
        ),
    }
    stale_tokens = (
        "Architecture (234 modules)",
        "Python package (234 source files)",
        "3,817 test functions",
        "263 hardening tasks",
        "| Python source files | 236 |",
        "| Python lines of code | 65,664 |",
        "| Test functions | 3,815 |",
        "| Validation scripts | 74 |",
        "| CI jobs | 24 |",
        "(≈294 modules)",
        "validation/reports/ (137 JSON+MD)",
        "Module counts by subpackage",
        "### 6.4 Control (`control/`, 71 modules)",
    )

    for doc_name, text in checked_docs.items():
        for token in stale_tokens:
            assert token not in text, f"{token!r} remains in {doc_name}"


def test_public_docs_match_itpa_hmode_dataset_counts() -> None:
    """Verify public ITPA H-mode count claims match the bundled dataset."""
    shot_count, machine_count = _itpa_hmode_counts()
    assert (shot_count, machine_count) == (53, 24)

    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    benchmarks = (_repo_root() / "docs" / "BENCHMARKS.md").read_text(encoding="utf-8")
    validation_status = (_repo_root() / "docs" / "PHYSICS_VALIDATION_STATUS.md").read_text(
        encoding="utf-8"
    )
    sphinx_validation = (
        _repo_root() / "docs" / "sphinx" / "userguide" / "validation.rst"
    ).read_text(encoding="utf-8")
    sphinx_notebooks = (_repo_root() / "docs" / "sphinx" / "notebooks.rst").read_text(
        encoding="utf-8"
    )

    assert f"{shot_count} shots / {machine_count} machines" in readme
    assert f"{shot_count} shots from {machine_count} machines" in readme
    assert (
        f"ITPA H-mode confinement database ({shot_count} shots, {machine_count} machines)"
        in benchmarks
    )
    assert f"{shot_count} shots across {machine_count} machines" in validation_status
    assert f"{shot_count} shots, {machine_count} machines" in sphinx_validation
    assert f"{shot_count} shots, {machine_count} machines" in sphinx_notebooks

    stale_count_claims = (
        "20 entries, 10 machines",
        "20-shot subset",
        "20 entries",
        "10 machines",
        "20 machines",
    )
    for doc_name, text in {
        "docs/BENCHMARKS.md": benchmarks,
        "docs/PHYSICS_VALIDATION_STATUS.md": validation_status,
        "docs/sphinx/userguide/validation.rst": sphinx_validation,
        "docs/sphinx/notebooks.rst": sphinx_notebooks,
    }.items():
        for token in stale_count_claims:
            assert token not in text, f"{token!r} remains in {doc_name}"


def test_readme_exposes_monthly_and_all_time_download_badges() -> None:
    """Lock README package download badges to the published package name."""
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")

    assert "[![PyPI Downloads](https://img.shields.io/pypi/dm/scpn-fusion.svg)]" in readme
    assert "(https://pypi.org/project/scpn-fusion/)" in readme
    assert "[![All-time Downloads](https://static.pepy.tech/badge/scpn-fusion)]" in readme
    assert "(https://pepy.tech/project/scpn-fusion)" in readme


def test_manifest_validation_rejects_schema_and_type_drift() -> None:
    """Verify schema and type drift are reported as validation errors."""
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

    manifest["counts"] = []
    report = tool.validate_manifest(manifest)

    assert not report["passed"]
    assert "counts must be an object" in report["errors"]


def test_manifest_declares_committed_json_schema_contract() -> None:
    """Verify the manifest points at the committed JSON schema contract."""
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
    """Verify built-in generator defaults target Fusion Core paths."""
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
    """Verify copied generator assets do not retain source-repo defaults."""
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
    """Verify the CLI respects portable config and workspace-member crates."""
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


def test_refresh_helpers_write_outputs_and_reject_missing_readme_markers() -> None:
    """Verify output writers and README marker validation."""
    tool = _load_tool()
    with _tempdir() as repo:
        _write_portable_fixture(repo)
        config = tool.load_config(repo, Path("tools/capability_manifest.toml"))
        manifest = tool.build_capability_manifest(repo, config)

        json_path, markdown_path = tool.write_outputs(
            manifest,
            json_output=repo / "custom/manifest.json",
            markdown_output=repo / "custom/snapshot.md",
        )
        readme_path = tool.refresh_readme_block(repo, "fresh snapshot", config=config)

        assert json.loads(json_path.read_text(encoding="utf-8"))["project_label"]
        assert "Portable Fusion Project Capability Inventory" in markdown_path.read_text(
            encoding="utf-8"
        )
        assert "fresh snapshot" in readme_path.read_text(encoding="utf-8")

        readme_path.write_text("# Missing markers\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="missing capability snapshot markers"):
            tool.refresh_readme_block(repo, "fresh snapshot", config=config)


def test_output_drift_checks_report_missing_stale_and_readme_errors() -> None:
    """Verify generated-output drift checks report each stale surface."""
    tool = _load_tool()
    with _tempdir() as repo:
        _write_portable_fixture(repo)
        config = tool.load_config(repo, Path("tools/capability_manifest.toml"))

        with pytest.raises(RuntimeError) as missing:
            tool.assert_outputs_current(repo, config=config)
        assert "missing generated manifest" in str(missing.value)
        assert "missing generated snapshot" in str(missing.value)
        assert "stale README capability block" in str(missing.value)

        tool.refresh_outputs(repo, config=config)
        tool.assert_outputs_current(repo, config=config)

        manifest_path = repo / config.json_output
        manifest_path.write_text("{}\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="stale generated manifest"):
            tool.assert_outputs_current(repo, config=config)

        tool.refresh_outputs(repo, config=config)
        snapshot_path = repo / config.markdown_output
        snapshot_path.write_text("stale\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="stale generated snapshot"):
            tool.assert_outputs_current(repo, config=config)

        tool.refresh_outputs(repo, config=config)
        (repo / config.readme_path).write_text("# Missing markers\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="stale README capability block"):
            tool.assert_outputs_current(repo, config=config)
        tool.assert_outputs_current(repo, config=config, check_readme=False)


def test_inventory_helper_fallbacks_cover_optional_and_missing_surfaces() -> None:
    """Verify helper fallbacks for absent roots, invalid config, and Cargo layouts."""
    tool = _load_tool()
    with _tempdir() as repo, _tempdir() as external:
        assert (
            tool._relative_config_path(external / "config.toml", repo) == external / "config.toml"
        )
        assert tool._configured_paths(
            {"roots": "src/one"}, key="roots", default=(Path("src"),)
        ) == (Path("src/one"),)
        assert tool._configured_paths({"roots": 7}, key="roots", default=(Path("src"),)) == (
            Path("src"),
        )
        assert tool._public_exports(repo / "missing.py") == []
        assert tool._literal_string_list("not-ast") == []
        assert tool._python_capability_sources((repo / "missing",), repo=repo) == []
        assert tool._python_capability_classes((repo / "missing",), repo=repo) == []
        assert tool._rust_workspace_crates(repo / "missing-rs", repo=repo) == []
        (repo / "empty-rs").mkdir()
        assert tool._rust_workspace_crates(repo / "empty-rs", repo=repo) == []
        assert tool._project_extras({"project": {"optional-dependencies": []}}) == []
        assert tool._workflow_files(repo / "missing-workflows", repo=repo) == []
        assert tool._python_files(repo / "missing-tests", repo=repo) == []
        assert tool._markdown_docs(repo / "missing-docs", repo=repo, exclude_parts=()) == []
        assert not tool._readme_block_matches(
            repo / "missing-readme.md",
            "expected",
            config=_minimal_config(repo),
        )

        _write_file(repo / "pkg/__init__.py", "__all__ = 'not-list'\n")
        _write_file(repo / "pkg/no_exports.py", "VALUE = 1\n")
        assert tool._public_exports(repo / "pkg/__init__.py") == []
        assert tool._public_exports(repo / "pkg/no_exports.py") == []

        _write_file(repo / "docs/public.md", "# Public\n")
        _write_file(repo / "docs/internal/private.md", "# Private\n")
        assert tool._markdown_docs(
            repo / "docs",
            repo=repo,
            exclude_parts=("internal",),
            display_root=repo / "docs",
        ) == ["public.md"]

        _write_file(repo / "fallback-rs/Cargo.toml", "[workspace]\nmembers = 'not-a-list'\n")
        _write_file(
            repo / "fallback-rs/crate-a/Cargo.toml",
            "[package]\nname = 'crate-a'\nversion = '0.1.0'\n",
        )
        assert tool._rust_workspace_crates(repo / "fallback-rs", repo=repo) == [
            {"name": "crate-a", "path": "fallback-rs/crate-a"}
        ]

        _write_file(
            repo / "member-rs/Cargo.toml",
            "[workspace]\nmembers = ['crate-a', 7, 'crate-b/Cargo.toml']\n",
        )
        _write_file(
            repo / "member-rs/crate-a/Cargo.toml",
            "[package]\nname = 'crate-a'\nversion = '0.1.0'\n",
        )
        _write_file(
            repo / "member-rs/crate-b/Cargo.toml",
            "[package]\nname = 'crate-b'\nversion = '0.1.0'\n",
        )
        assert tool._rust_workspace_crates(repo / "member-rs", repo=repo) == [
            {"name": "crate-a", "path": "member-rs/crate-a"},
            {"name": "crate-b", "path": "member-rs/crate-b"},
        ]


def test_cli_modes_validate_check_generate_and_script_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify in-process CLI modes and the script entrypoint."""
    tool = _load_tool()
    with _tempdir() as repo:
        _write_portable_fixture(repo)
        common_args = ["--repo", str(repo), "--config", "tools/capability_manifest.toml"]

        assert tool.main([*common_args, "--no-readme"]) == 0
        stdout = capsys.readouterr().out
        assert "Wrote" in stdout
        assert "Refreshed" not in stdout

        assert tool.main(common_args) == 0
        assert "Refreshed" in capsys.readouterr().out

        manifest_path = repo / "docs/_generated/capability_manifest.json"
        assert tool.main([*common_args, "--validate", str(manifest_path)]) == 0
        assert '"passed": true' in capsys.readouterr().out

        invalid_manifest = repo / "invalid.json"
        invalid_manifest.write_text('{"schema_version": "wrong"}\n', encoding="utf-8")
        assert tool.main([*common_args, "--validate", str(invalid_manifest)]) == 1
        assert "schema_version mismatch" in capsys.readouterr().out

        assert tool.main([*common_args, "--check", "--no-readme"]) == 0
        manifest_path.write_text("{}\n", encoding="utf-8")
        assert tool.main([*common_args, "--check", "--no-readme"]) == 1
        assert "stale generated manifest" in capsys.readouterr().err

        tool.refresh_outputs(
            repo,
            config=tool.load_config(repo, Path("tools/capability_manifest.toml")),
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["capability_manifest.py", *common_args, "--check"],
        )
        with pytest.raises(SystemExit) as exit_info:
            runpy.run_path(str(_repo_root() / "tools/capability_manifest.py"), run_name="__main__")
        assert exit_info.value.code == 0


@contextmanager
def _tempdir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


def _write_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _minimal_config(repo: Path) -> Any:
    tool = _load_tool()
    _write_portable_fixture(repo)
    return tool.load_config(repo, Path("tools/capability_manifest.toml"))


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
        repo / "fusion-rs/fuzz/Cargo.toml",
        '[package]\nname = "fuzz-targets"\nversion = "0.1.0"\n',
    )
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
