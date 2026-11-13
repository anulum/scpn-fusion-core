# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Packaging guardrails for dependency tiering."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml
from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
MAKEFILE = ROOT / "Makefile"
DOCS_WORKFLOW = ROOT / ".github" / "workflows" / "docs.yml"
FULL_REQUIREMENTS_IN = ROOT / "requirements" / "full.in"


def _extract_array_block(content: str, key: str) -> list[str]:
    marker = f"{key} = ["
    start = content.find(marker)
    if start < 0:
        return []
    i = start + len(marker)
    end = content.find("]", i)
    if end < 0:
        return []
    block = content[i:end]
    return [line.strip().strip('",') for line in block.splitlines() if line.strip()]


def test_core_dependencies_do_not_include_heavy_optional_stacks() -> None:
    content = PYPROJECT.read_text(encoding="utf-8")
    deps = _extract_array_block(content, "dependencies")
    joined = "\n".join(deps)
    assert "streamlit" not in joined
    assert "jax" not in joined
    assert "jaxlib" not in joined
    assert "gymnasium" not in joined


def test_optional_dependency_groups_expose_ui_ml_rl_and_full() -> None:
    content = PYPROJECT.read_text(encoding="utf-8")
    assert "ui = [" in content
    assert "ml = [" in content
    assert "rl = [" in content
    assert "full = [" in content
    assert '"streamlit<2"' in content
    assert '"jax>=0.7.1,<0.8"' in content
    assert '"jaxlib>=0.7.1,<0.8"' in content
    assert '"gymnasium>=1.0.0,<2"' in content


def test_every_published_dependency_has_an_upper_bound() -> None:
    """Every base and optional requirement in the release metadata caps its version.

    An unbounded specifier lets a future major release of a dependency reach
    users of an already published version; exact pins and compatible-release
    clauses count as bounded.
    """
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    declared = [("dependencies", spec) for spec in project["dependencies"]]
    for extra, specs in project["optional-dependencies"].items():
        declared += [(extra, spec) for spec in specs]
    assert declared

    unbounded = []
    for group, spec in declared:
        operators = {clause.operator for clause in Requirement(spec).specifier}
        if not operators & {"<", "<=", "==", "===", "~="}:
            unbounded.append(f"{group}: {spec}")
    assert unbounded == []


def test_full_lock_source_covers_declared_heavy_extras() -> None:
    pyproject = PYPROJECT.read_text(encoding="utf-8")
    full_requirements = FULL_REQUIREMENTS_IN.read_text(encoding="utf-8")

    for dependency in (
        "streamlit",
        "freegs>=0.6",
        "omas>=0.70",
        "maturin>=1.14.1,<2.0",
    ):
        assert dependency in pyproject
        assert dependency in full_requirements


def test_docs_workflow_rebuilds_when_the_lock_it_installs_changes() -> None:
    """Every requirements file the docs jobs install is in the push path filter.

    A lock that a path-filtered workflow installs but does not watch can change
    on main without that workflow ever building against it.
    """
    text = DOCS_WORKFLOW.read_text(encoding="utf-8")
    document = yaml.load(text, Loader=yaml.BaseLoader)
    watched = set(document["on"]["push"]["paths"])
    installed = set(re.findall(r"-r (requirements/[A-Za-z0-9_.-]+\.txt)", text))
    assert installed == {"requirements/docs.txt"}
    for lock in installed:
        assert lock in watched
        assert lock.removesuffix(".txt") + ".in" in watched


def test_root_docs_targets_match_native_docs_ci_contract() -> None:
    makefile = MAKEFILE.read_text(encoding="utf-8")
    workflow = DOCS_WORKFLOW.read_text(encoding="utf-8")

    assert "mkdocs" not in makefile.lower()
    assert "docs-build:" in makefile
    assert "PYTHONPATH=src $(SPHINXBUILD) -W -b html $(DOCS_SOURCE) $(DOCS_BUILD)" in makefile
    assert "python tools/run_native_docs.py --language python" in workflow
