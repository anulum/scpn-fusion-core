# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Packaging guardrails for dependency tiering."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"


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
    assert '"streamlit"' in content
    assert '"jax>=0.4.20"' in content
    assert '"jaxlib>=0.4.20"' in content
    assert '"gymnasium>=1.0.0"' in content
