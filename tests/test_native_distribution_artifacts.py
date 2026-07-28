# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — native distribution publication guard tests
"""Tests for the fail-closed native artifact publication guard."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tools.verify_native_distribution_artifacts import (
    ArtifactVerificationError,
    TOML_MODULE_NAME,
    main,
    verify_native_distribution_artifacts,
)


def _candidate(tmp_path: Path) -> tuple[Path, Path]:
    metadata = tmp_path / "pyproject.toml"
    metadata.write_text(
        '[project]\nname = "scpn-fusion-rs"\nversion = "4.0.0"\n',
        encoding="utf-8",
    )
    artifacts = tmp_path / "dist"
    artifacts.mkdir()
    (artifacts / "scpn_fusion_rs-4.0.0.tar.gz").touch()
    (artifacts / "scpn_fusion_rs-4.0.0-cp312-cp312-manylinux.whl").touch()
    return artifacts, metadata


def test_toml_loader_matches_supported_python_runtime() -> None:
    """Select the standard or backported TOML loader for the runtime."""
    expected = "tomllib" if sys.version_info >= (3, 11) else "tomli"

    assert expected == TOML_MODULE_NAME


def test_matching_release_tag_is_accepted(tmp_path: Path) -> None:
    """Accept a tagged push whose tag matches project metadata."""
    artifacts, metadata = _candidate(tmp_path)

    result = verify_native_distribution_artifacts(
        artifact_dir=artifacts,
        metadata_path=metadata,
        event_name="push",
        ref_name="v4.0.0",
    )

    assert result == "Validated 2 scpn-fusion-rs 4.0.0 artifacts"


def test_mismatched_release_tag_fails_closed(tmp_path: Path) -> None:
    """Reject a tagged push whose tag and metadata version differ."""
    artifacts, metadata = _candidate(tmp_path)

    with pytest.raises(ArtifactVerificationError, match="Tag/metadata mismatch"):
        verify_native_distribution_artifacts(
            artifact_dir=artifacts,
            metadata_path=metadata,
            event_name="push",
            ref_name="v4.1.0",
        )


def test_manual_main_publication_is_accepted(tmp_path: Path) -> None:
    """Accept a manual publication dispatched from main."""
    artifacts, metadata = _candidate(tmp_path)

    result = verify_native_distribution_artifacts(
        artifact_dir=artifacts,
        metadata_path=metadata,
        event_name="workflow_dispatch",
        ref_name="main",
    )

    assert result == "Validated 2 scpn-fusion-rs 4.0.0 artifacts"


def test_manual_non_main_publication_fails_closed(tmp_path: Path) -> None:
    """Reject a manual publication dispatched outside main."""
    artifacts, metadata = _candidate(tmp_path)

    with pytest.raises(ArtifactVerificationError, match="must run from 'main'"):
        verify_native_distribution_artifacts(
            artifact_dir=artifacts,
            metadata_path=metadata,
            event_name="workflow_dispatch",
            ref_name="release-candidate",
        )


def test_mixed_distribution_cohort_fails_closed(tmp_path: Path) -> None:
    """Reject artifacts belonging to a different distribution identity."""
    artifacts, metadata = _candidate(tmp_path)
    (artifacts / "scpn_fusion-4.0.0.tar.gz").touch()

    with pytest.raises(ArtifactVerificationError, match="Unexpected distribution"):
        verify_native_distribution_artifacts(
            artifact_dir=artifacts,
            metadata_path=metadata,
            event_name="workflow_dispatch",
            ref_name="main",
        )


def test_same_prefix_non_distribution_file_fails_closed(tmp_path: Path) -> None:
    """Reject unexpected files even when their prefix resembles the package."""
    artifacts, metadata = _candidate(tmp_path)
    (artifacts / "scpn_fusion_rs-4.0.0-unexpected.txt").touch()

    with pytest.raises(ArtifactVerificationError, match="Unexpected distribution"):
        verify_native_distribution_artifacts(
            artifact_dir=artifacts,
            metadata_path=metadata,
            event_name="workflow_dispatch",
            ref_name="main",
        )


@pytest.mark.parametrize(
    "metadata_text, expected_message",
    [
        ("[build-system]\n", "Missing project.version"),
        ('project = "invalid"\n', "Missing project.version"),
        ('[project]\nversion = ""\n', "Invalid project.version"),
        ("[project]\nversion = 4\n", "Invalid project.version"),
    ],
)
def test_invalid_project_metadata_fails_closed(
    tmp_path: Path,
    metadata_text: str,
    expected_message: str,
) -> None:
    """Reject absent, malformed, empty, and non-string project versions."""
    artifacts, metadata = _candidate(tmp_path)
    metadata.write_text(metadata_text, encoding="utf-8")

    with pytest.raises(ArtifactVerificationError, match=expected_message):
        verify_native_distribution_artifacts(
            artifact_dir=artifacts,
            metadata_path=metadata,
            event_name="workflow_dispatch",
            ref_name="main",
        )


def test_unsupported_publication_event_fails_closed(tmp_path: Path) -> None:
    """Reject events other than tagged pushes and manual dispatches."""
    artifacts, metadata = _candidate(tmp_path)

    with pytest.raises(ArtifactVerificationError, match="Unsupported publication event"):
        verify_native_distribution_artifacts(
            artifact_dir=artifacts,
            metadata_path=metadata,
            event_name="pull_request",
            ref_name="main",
        )


def test_empty_artifact_directory_fails_closed(tmp_path: Path) -> None:
    """Reject a publication cohort that contains no downloaded artifacts."""
    artifacts, metadata = _candidate(tmp_path)
    for artifact in artifacts.iterdir():
        artifact.unlink()

    with pytest.raises(ArtifactVerificationError, match="No distribution artifacts"):
        verify_native_distribution_artifacts(
            artifact_dir=artifacts,
            metadata_path=metadata,
            event_name="workflow_dispatch",
            ref_name="main",
        )


@pytest.mark.parametrize(
    "missing_name, expected_counts",
    [
        ("scpn_fusion_rs-4.0.0.tar.gz", "0 sdist\\(s\\), 1 wheel\\(s\\)"),
        (
            "scpn_fusion_rs-4.0.0-cp312-cp312-manylinux.whl",
            "1 sdist\\(s\\), 0 wheel\\(s\\)",
        ),
    ],
)
def test_incomplete_artifact_cohort_fails_closed(
    tmp_path: Path,
    missing_name: str,
    expected_counts: str,
) -> None:
    """Require exactly one source archive and at least one native wheel."""
    artifacts, metadata = _candidate(tmp_path)
    (artifacts / missing_name).unlink()

    with pytest.raises(ArtifactVerificationError, match=expected_counts):
        verify_native_distribution_artifacts(
            artifact_dir=artifacts,
            metadata_path=metadata,
            event_name="workflow_dispatch",
            ref_name="main",
        )


def test_cli_reports_validated_artifact_cohort(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exercise the workflow-facing CLI with a valid temporary cohort."""
    artifacts, metadata = _candidate(tmp_path)

    exit_code = main(
        [
            "--artifact-dir",
            str(artifacts),
            "--metadata",
            str(metadata),
            "--event-name",
            "workflow_dispatch",
            "--ref-name",
            "main",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == "Validated 2 scpn-fusion-rs 4.0.0 artifacts\n"
