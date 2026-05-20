#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Generate the public capability manifest from repository sources.

The manifest is intentionally derived from static files instead of imported
modules. That keeps it deterministic in CI and avoids optional dependency
side effects while still giving README, docs, and release tooling one source
of truth for public capability counts.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised by Python-version matrix in CI.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


CAPABILITY_MANIFEST_SCHEMA_VERSION = "capability-manifest.v1"
DEFAULT_JSON_OUTPUT = Path("docs/_generated/capability_manifest.json")
DEFAULT_MARKDOWN_OUTPUT = Path("docs/_generated/capability_snapshot.md")
DEFAULT_CONFIG = Path("tools/capability_manifest.toml")
DEFAULT_SCHEMA_PATH = Path("schemas/capability_manifest.schema.json")
DEFAULT_README = Path("README.md")
DEFAULT_MARKER_START = "<!-- capability-snapshot:start -->"
DEFAULT_MARKER_END = "<!-- capability-snapshot:end -->"
DEFAULT_PROJECT_LABEL = "SCPN Fusion Core"
DEFAULT_PACKAGE_ROOT = Path("src/scpn_fusion")
DEFAULT_CAPABILITY_SOURCES = (
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
DEFAULT_CAPABILITY_DOCS = Path("docs")
DEFAULT_TESTS_ROOT = Path("tests")
DEFAULT_DOCS_ROOT = Path("docs")
DEFAULT_WORKFLOWS_ROOT = Path(".github/workflows")
DEFAULT_RUST_WORKSPACE = Path("scpn-fusion-rs")


def _default_labels() -> dict[str, str]:
    """Return stable public labels for README and docs snapshots."""

    return {
        "version": "Package version",
        "public_api_exports": "Public API exports",
        "python_capability_source_modules": "Python capability source modules",
        "python_capability_classes": "Python capability classes",
        "capability_documentation_pages": "Capability documentation pages",
        "rust_workspace_crates": "Rust workspace crates",
        "optional_extras": "Optional extras",
        "test_files": "Python test files",
        "public_documentation_pages": "Public documentation pages",
        "github_workflows": "GitHub Actions workflows",
    }


@dataclass(frozen=True)
class CapabilityManifestConfig:
    """Portable configuration for repository capability inventory."""

    project_label: str
    schema_version: str
    json_output: Path
    markdown_output: Path
    readme_path: Path
    readme_marker_start: str
    readme_marker_end: str
    package_root: Path
    capability_sources: tuple[Path, ...]
    capability_docs: Path
    tests_root: Path
    docs_root: Path
    workflows_root: Path
    rust_workspace: Path
    exclude_doc_parts: tuple[str, ...]
    labels: dict[str, str]
    source_path: Path | None


@dataclass(frozen=True)
class CapabilityPaths:
    """Repository paths scanned by the manifest builder."""

    repo: Path
    pyproject: Path
    package_root: Path
    capability_roots: tuple[Path, ...]
    capability_docs_root: Path
    tests_root: Path
    docs_root: Path
    workflows_root: Path
    rust_workspace_root: Path


def load_config(repo: Path, config_path: Path | None = None) -> CapabilityManifestConfig:
    """Load repository capability manifest configuration."""

    repo = repo.resolve()
    raw: dict[str, Any] = {}
    path = repo / (config_path or DEFAULT_CONFIG)
    if path.exists():
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    paths = raw.get("paths", {})
    readme = raw.get("readme", {})
    labels = _default_labels()
    labels.update({str(key): str(value) for key, value in raw.get("labels", {}).items()})
    return CapabilityManifestConfig(
        project_label=str(raw.get("project_label", DEFAULT_PROJECT_LABEL)),
        schema_version=str(raw.get("schema_version", CAPABILITY_MANIFEST_SCHEMA_VERSION)),
        json_output=Path(paths.get("json_output", DEFAULT_JSON_OUTPUT.as_posix())),
        markdown_output=Path(paths.get("markdown_output", DEFAULT_MARKDOWN_OUTPUT.as_posix())),
        readme_path=Path(readme.get("path", DEFAULT_README.as_posix())),
        readme_marker_start=str(readme.get("marker_start", DEFAULT_MARKER_START)),
        readme_marker_end=str(readme.get("marker_end", DEFAULT_MARKER_END)),
        package_root=Path(paths.get("package_root", DEFAULT_PACKAGE_ROOT.as_posix())),
        capability_sources=_configured_paths(
            paths,
            key="capability_sources",
            default=DEFAULT_CAPABILITY_SOURCES,
        ),
        capability_docs=Path(paths.get("capability_docs", DEFAULT_CAPABILITY_DOCS.as_posix())),
        tests_root=Path(paths.get("tests_root", DEFAULT_TESTS_ROOT.as_posix())),
        docs_root=Path(paths.get("docs_root", DEFAULT_DOCS_ROOT.as_posix())),
        workflows_root=Path(paths.get("workflows_root", DEFAULT_WORKFLOWS_ROOT.as_posix())),
        rust_workspace=Path(paths.get("rust_workspace", DEFAULT_RUST_WORKSPACE.as_posix())),
        exclude_doc_parts=tuple(
            str(part) for part in raw.get("exclude_doc_parts", ["internal", "_generated"])
        ),
        labels=labels,
        source_path=_relative_config_path(path, repo) if path.exists() else None,
    )


def _relative_config_path(path: Path, repo: Path) -> Path:
    try:
        return path.resolve().relative_to(repo)
    except ValueError:
        return path.resolve()


def _configured_paths(
    paths: dict[str, Any],
    *,
    key: str,
    default: tuple[Path, ...],
) -> tuple[Path, ...]:
    """Read a path or list of paths from TOML."""

    raw = paths.get(key, [item.as_posix() for item in default])
    if isinstance(raw, str):
        return (Path(raw),)
    if isinstance(raw, list):
        return tuple(Path(str(item)) for item in raw)
    return default


def capability_paths(repo: Path, config: CapabilityManifestConfig) -> CapabilityPaths:
    """Return canonical manifest scan roots."""

    return CapabilityPaths(
        repo=repo,
        pyproject=repo / "pyproject.toml",
        package_root=repo / config.package_root,
        capability_roots=tuple(repo / root for root in config.capability_sources),
        capability_docs_root=repo / config.capability_docs,
        tests_root=repo / config.tests_root,
        docs_root=repo / config.docs_root,
        workflows_root=repo / config.workflows_root,
        rust_workspace_root=repo / config.rust_workspace,
    )


def build_capability_manifest(
    repo: Path, config: CapabilityManifestConfig | None = None
) -> dict[str, Any]:
    """Build a deterministic capability manifest for public surfaces."""

    repo = repo.resolve()
    config = config or load_config(repo)
    paths = capability_paths(repo, config)
    pyproject = _load_pyproject(paths.pyproject)
    public_exports = _public_exports(paths.package_root / "__init__.py")
    python_sources = _python_capability_sources(paths.capability_roots, repo=repo)
    python_classes = _python_capability_classes(paths.capability_roots, repo=repo)
    rust_crates = _rust_workspace_crates(paths.rust_workspace_root, repo=repo)
    extras = _project_extras(pyproject)
    package_data_key = paths.package_root.name
    workflows = _workflow_files(paths.workflows_root, repo=repo)
    tests = _python_files(paths.tests_root, repo=repo)
    docs_pages = _markdown_docs(paths.docs_root, repo=repo, exclude_parts=config.exclude_doc_parts)
    capability_docs = _markdown_docs(
        paths.capability_docs_root,
        repo=repo,
        exclude_parts=config.exclude_doc_parts,
    )

    return {
        "SPDX-License-Identifier": "AGPL-3.0-or-later",
        "schema_version": config.schema_version,
        "project_label": config.project_label,
        "generated_from": {
            "config": str(config.source_path)
            if config.source_path is not None
            else "built-in defaults",
            "generator": "tools/capability_manifest.py",
            "schema": DEFAULT_SCHEMA_PATH.as_posix(),
        },
        "project": {
            "name": pyproject["project"]["name"],
            "version": pyproject["project"]["version"],
            "requires_python": pyproject["project"]["requires-python"],
            "readme": pyproject["project"]["readme"],
            "license": pyproject["project"]["license"],
        },
        "labels": config.labels,
        "counts": {
            "public_api_exports": len(public_exports),
            "python_capability_source_modules": len(python_sources),
            "python_capability_classes": len(python_classes),
            "capability_documentation_pages": len(capability_docs),
            "rust_workspace_crates": len(rust_crates),
            "optional_extras": len(extras),
            "test_files": len(tests),
            "public_documentation_pages": len(docs_pages),
            "github_workflows": len(workflows),
        },
        "package_exports": public_exports,
        "capabilities": {
            "python_source_modules": python_sources,
            "python_classes": python_classes,
            "documentation_pages": capability_docs,
            "rust_workspace_crates": rust_crates,
        },
        "packaging": {
            "optional_extras": extras,
            "shipped_package_data": pyproject.get("tool", {})
            .get("setuptools", {})
            .get("package-data", {})
            .get(package_data_key, []),
        },
        "quality_gates": {
            "test_files": tests,
            "github_workflows": workflows,
        },
        "documentation": {
            "public_pages": docs_pages,
        },
        "evidence_boundary": (
            "Counts are file-system and static-source inventory only; benchmark, "
            "coverage, hardware, and scientific-fidelity claims remain governed by "
            "their dedicated evidence artifacts."
        ),
    }


def render_markdown_snapshot(manifest: dict[str, Any]) -> str:
    """Render a compact public snapshot for README and PyPI reuse."""

    counts = manifest["counts"]
    project = manifest["project"]
    labels = manifest.get("labels", _default_labels())
    rows = [
        (labels["version"], project["version"]),
        (labels["public_api_exports"], counts["public_api_exports"]),
        (
            labels["python_capability_source_modules"],
            counts["python_capability_source_modules"],
        ),
        (labels["python_capability_classes"], counts["python_capability_classes"]),
        (
            labels["capability_documentation_pages"],
            counts["capability_documentation_pages"],
        ),
        (labels["rust_workspace_crates"], counts["rust_workspace_crates"]),
        (labels["optional_extras"], counts["optional_extras"]),
        (labels["test_files"], counts["test_files"]),
        (labels["public_documentation_pages"], counts["public_documentation_pages"]),
        (labels["github_workflows"], counts["github_workflows"]),
    ]
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "<!-- Generated by tools/capability_manifest.py; do not edit counts by hand. -->",
        "",
        f"### {manifest.get('project_label', 'Project')} Capability Inventory",
        "",
        "| Surface | Current inventory |",
        "|---|---:|",
    ]
    lines.extend(f"| {label} | {value} |" for label, value in rows)
    lines.extend(
        [
            "",
            (
                "Evidence boundary: this snapshot is a static inventory. Performance, "
                "coverage, hardware, and scientific-fidelity claims require their own "
                "committed evidence artifacts."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def refresh_readme_block(
    repo: Path,
    snapshot: str,
    *,
    config: CapabilityManifestConfig,
) -> Path:
    """Refresh the README block bounded by configured markers."""

    readme_path = repo / config.readme_path
    text = readme_path.read_text(encoding="utf-8")
    start = config.readme_marker_start
    end = config.readme_marker_end
    if start not in text or end not in text:
        raise RuntimeError(f"{config.readme_path} is missing capability snapshot markers")
    before, rest = text.split(start, maxsplit=1)
    _old, after = rest.split(end, maxsplit=1)
    replacement = f"{start}\n{snapshot.rstrip()}\n{end}"
    readme_path.write_text(before + replacement + after, encoding="utf-8")
    return readme_path


def write_outputs(
    manifest: dict[str, Any],
    *,
    json_output: Path,
    markdown_output: Path,
) -> tuple[Path, Path]:
    """Write deterministic JSON and Markdown outputs."""

    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_output.write_text(render_markdown_snapshot(manifest), encoding="utf-8")
    return json_output, markdown_output


def refresh_outputs(
    repo: Path,
    *,
    config: CapabilityManifestConfig,
    json_output: Path | None = None,
    markdown_output: Path | None = None,
    update_readme: bool = True,
) -> tuple[Path, Path, Path | None]:
    """Regenerate JSON, Markdown, and optionally the README snapshot."""

    manifest = build_capability_manifest(repo, config)
    json_path, markdown_path = write_outputs(
        manifest,
        json_output=repo / (json_output or config.json_output),
        markdown_output=repo / (markdown_output or config.markdown_output),
    )
    readme_path = None
    if update_readme:
        readme_path = refresh_readme_block(
            repo,
            render_markdown_snapshot(manifest),
            config=config,
        )
    return json_path, markdown_path, readme_path


def validate_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a capability manifest payload."""

    errors: list[str] = []
    if payload.get("schema_version") != CAPABILITY_MANIFEST_SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    for key in ("project", "counts", "package_exports", "capabilities", "packaging"):
        if key not in payload:
            errors.append(f"missing top-level key: {key}")
    counts = payload.get("counts", {})
    if not isinstance(counts, dict):
        errors.append("counts must be an object")
    else:
        for key, value in counts.items():
            if not isinstance(value, int) or value < 0:
                errors.append(f"counts.{key} must be a non-negative integer")
    capabilities = payload.get("capabilities", {})
    if isinstance(capabilities, dict) and isinstance(counts, dict):
        _check_count(
            errors,
            counts,
            "python_capability_source_modules",
            capabilities.get("python_source_modules"),
        )
        _check_count(
            errors, counts, "python_capability_classes", capabilities.get("python_classes")
        )
        _check_count(
            errors,
            counts,
            "capability_documentation_pages",
            capabilities.get("documentation_pages"),
        )
        _check_count(
            errors,
            counts,
            "rust_workspace_crates",
            capabilities.get("rust_workspace_crates"),
        )
    return {"passed": not errors, "errors": errors}


def assert_outputs_current(
    repo: Path,
    *,
    config: CapabilityManifestConfig | None = None,
    json_output: Path | None = None,
    markdown_output: Path | None = None,
    check_readme: bool = True,
) -> None:
    """Raise if tracked generated outputs drift from current sources."""

    config = config or load_config(repo)
    manifest = build_capability_manifest(repo, config)
    expected_json = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    expected_markdown = render_markdown_snapshot(manifest)
    json_path = repo / (json_output or config.json_output)
    markdown_path = repo / (markdown_output or config.markdown_output)
    errors: list[str] = []
    if not json_path.exists():
        errors.append(f"missing generated manifest: {json_path.relative_to(repo)}")
    elif json_path.read_text(encoding="utf-8") != expected_json:
        errors.append(f"stale generated manifest: {json_path.relative_to(repo)}")
    if not markdown_path.exists():
        errors.append(f"missing generated snapshot: {markdown_path.relative_to(repo)}")
    elif markdown_path.read_text(encoding="utf-8") != expected_markdown:
        errors.append(f"stale generated snapshot: {markdown_path.relative_to(repo)}")
    if check_readme:
        readme_path = repo / config.readme_path
        if not _readme_block_matches(readme_path, expected_markdown, config=config):
            errors.append(f"stale README capability block: {config.readme_path}")
    if errors:
        raise RuntimeError("; ".join(errors))


def _check_count(
    errors: list[str],
    counts: dict[str, Any],
    key: str,
    values: Any,
) -> None:
    if not isinstance(values, list):
        errors.append(f"capabilities list missing for count {key}")
        return
    if counts.get(key) != len(values):
        errors.append(f"counts.{key} does not match list length")


def _load_pyproject(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _public_exports(init_path: Path) -> list[str]:
    if not init_path.exists():
        return []
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return sorted(_literal_string_list(node.value))
    return []


def _literal_string_list(node: ast.AST) -> list[str]:
    if not isinstance(node, ast.List):
        return []
    values: list[str] = []
    for item in node.elts:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            values.append(item.value)
    return values


def _python_capability_sources(roots: tuple[Path, ...], *, repo: Path) -> list[str]:
    rows: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        rows.update(
            _rel(path, repo) for path in sorted(root.rglob("*.py")) if path.name != "__init__.py"
        )
    return sorted(rows)


def _python_capability_classes(roots: tuple[Path, ...], *, repo: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_paths: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            resolved = path.resolve()
            if path.name == "__init__.py" or resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                    rows.append({"name": node.name, "path": _rel(path, repo)})
    return sorted(rows, key=lambda row: (row["name"], row["path"]))


def _rust_workspace_crates(root: Path, *, repo: Path) -> list[dict[str, str]]:
    if not root.exists():
        return []
    crates: list[dict[str, str]] = []
    for manifest in sorted(root.rglob("Cargo.toml")):
        if manifest == root / "Cargo.toml":
            continue
        payload = tomllib.loads(manifest.read_text(encoding="utf-8"))
        package = payload.get("package", {})
        name = package.get("name")
        if isinstance(name, str) and name:
            crates.append({"name": name, "path": _rel(manifest.parent, repo)})
    return sorted(crates, key=lambda row: (row["name"], row["path"]))


def _project_extras(pyproject: dict[str, Any]) -> list[str]:
    extras = pyproject.get("project", {}).get("optional-dependencies", {})
    if not isinstance(extras, dict):
        return []
    return sorted(str(name) for name in extras)


def _workflow_files(workflows_root: Path, *, repo: Path) -> list[str]:
    if not workflows_root.exists():
        return []
    return [
        _rel(path, repo)
        for path in sorted(list(workflows_root.glob("*.yml")) + list(workflows_root.glob("*.yaml")))
    ]


def _python_files(root: Path, *, repo: Path) -> list[str]:
    if not root.exists():
        return []
    return [_rel(path, repo) for path in sorted(root.rglob("*.py"))]


def _markdown_docs(
    root: Path,
    *,
    repo: Path,
    exclude_parts: tuple[str, ...],
    display_root: Path | None = None,
) -> list[str]:
    if not root.exists():
        return []
    relative_root = display_root or repo
    return [
        _rel(path, relative_root)
        for path in sorted(root.rglob("*.md"))
        if not set(path.relative_to(root).parts).intersection(exclude_parts)
    ]


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _readme_block_matches(
    readme_path: Path,
    expected_markdown: str,
    *,
    config: CapabilityManifestConfig,
) -> bool:
    if not readme_path.exists():
        return False
    text = readme_path.read_text(encoding="utf-8")
    start = config.readme_marker_start
    end = config.readme_marker_end
    if start not in text or end not in text:
        return False
    block = text.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0].strip()
    return block == expected_markdown.strip()


def main(argv: Iterable[str] | None = None) -> int:
    """Run capability manifest generation, validation, or drift checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--no-readme", action="store_true")
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo = args.repo.resolve()
    config = load_config(repo, args.config)
    if args.validate is not None:
        report = validate_manifest(json.loads(args.validate.read_text(encoding="utf-8")))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    if args.check:
        try:
            assert_outputs_current(
                repo,
                config=config,
                json_output=args.output,
                markdown_output=args.markdown_output,
                check_readme=not args.no_readme,
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return 0

    json_path, markdown_path, readme_path = refresh_outputs(
        repo,
        config=config,
        json_output=args.output,
        markdown_output=args.markdown_output,
        update_readme=not args.no_readme,
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    if readme_path is not None:
        print(f"Refreshed {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
