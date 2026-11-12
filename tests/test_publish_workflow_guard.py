# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — PyPI publish workflow guard tests
"""The PyPI publish workflow publishes only from a tag or on an explicit request."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
PUBLISH_WORKFLOW = ROOT / ".github" / "workflows" / "publish.yml"

EXPECTED_PUBLISH_CONDITION = (
    "(github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')) || "
    "(github.event_name == 'workflow_dispatch' && inputs.publish_pypi == true "
    "&& startsWith(inputs.ref, 'v'))"
)


def _workflow() -> dict[str, Any]:
    document = yaml.load(PUBLISH_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(document, dict)
    return document


def _normalised(expression: str) -> str:
    return " ".join(expression.split())


def test_pypi_publish_job_is_guarded_by_tag_push_or_explicit_tag_dispatch() -> None:
    """Publish on a pushed v* tag, or on a manual run that asks for it with a v* tag ref."""
    job = _workflow()["jobs"]["publish-pypi"]
    assert _normalised(job["if"]) == EXPECTED_PUBLISH_CONDITION
    assert job["needs"] == "build"
    assert job["environment"]["name"] == "pypi"


def test_manual_publish_input_defaults_to_build_only() -> None:
    """The manual publish switch exists and is off unless set for a run."""
    inputs = _workflow()["on"]["workflow_dispatch"]["inputs"]
    publish_pypi = inputs["publish_pypi"]
    assert publish_pypi["type"] == "boolean"
    assert publish_pypi["default"] == "false"
    assert publish_pypi["required"] == "false"
    assert inputs["ref"]["default"] == ""


def test_tag_push_trigger_and_dependent_jobs_follow_the_guard() -> None:
    """Tag pushes still trigger the workflow; parity runs only after a publish."""
    workflow = _workflow()
    assert workflow["on"]["push"]["tags"] == ["v*"]
    assert workflow["jobs"]["verify-pypi-parity"]["needs"] == ["publish-pypi"]
    testpypi = workflow["jobs"]["publish-testpypi"]
    assert _normalised(testpypi["if"]) == (
        "${{ github.event_name == 'workflow_dispatch' && inputs.publish_testpypi }}"
    )
