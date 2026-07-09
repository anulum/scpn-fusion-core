# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — CI Workflow Reference Tests
"""Tests for Python file references in the GitHub Actions workflow."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
PYTHON_PATH_RE = re.compile(r"\b(?:tests|tools|validation)/[A-Za-z0-9_./-]+\.py\b")


def _workflow_text() -> str:
    """Return the tracked CI workflow text."""
    return CI_WORKFLOW.read_text(encoding="utf-8")


def test_ci_workflow_python_file_references_exist() -> None:
    """All Python paths referenced by the CI workflow exist in the checkout."""
    references = sorted(set(PYTHON_PATH_RE.findall(_workflow_text())))

    assert references
    assert [path for path in references if not (ROOT / path).exists()] == []


def test_ci_workflow_uses_current_torax_gate() -> None:
    """The workflow uses the current real-TORAX parity gate and test module."""
    workflow = _workflow_text()

    assert "validation/benchmark_vs_torax.py" not in workflow
    assert "tests/test_benchmark_vs_torax.py" not in workflow
    assert "validation/benchmark_torax_real_parity.py --check" in workflow
    assert "validation/benchmark_torax_real_parity.py --output artifacts/torax_benchmark.json" in workflow
    assert "tests/test_torax_real_parity.py" in workflow
