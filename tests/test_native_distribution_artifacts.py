# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — native distribution publication guard tests
"""Tests for the fail-closed native artifact publication guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.verify_native_distribution_artifacts import (
    ArtifactVerificationError,
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


def test_matching_release_tag_is_accepted(tmp_path: Path) -> None:
    artifacts, metadata = _candidate(tmp_path)

    result = verify_native_distribution_artifacts(
        artifact_dir=artifacts,
        metadata_path=metadata,
        event_name="push",
        ref_name="v4.0.0",
    )

    assert result == "Validated 2 scpn-fusion-rs 4.0.0 artifacts"


def test_mismatched_release_tag_fails_closed(tmp_path: Path) -> None:
    artifacts, metadata = _candidate(tmp_path)

    with pytest.raises(ArtifactVerificationError, match="Tag/metadata mismatch"):
        verify_native_distribution_artifacts(
            artifact_dir=artifacts,
            metadata_path=metadata,
            event_name="push",
            ref_name="v4.1.0",
        )


def test_manual_main_publication_is_accepted(tmp_path: Path) -> None:
    artifacts, metadata = _candidate(tmp_path)

    result = verify_native_distribution_artifacts(
        artifact_dir=artifacts,
        metadata_path=metadata,
        event_name="workflow_dispatch",
        ref_name="main",
    )

    assert result == "Validated 2 scpn-fusion-rs 4.0.0 artifacts"


def test_manual_non_main_publication_fails_closed(tmp_path: Path) -> None:
    artifacts, metadata = _candidate(tmp_path)

    with pytest.raises(ArtifactVerificationError, match="must run from 'main'"):
        verify_native_distribution_artifacts(
            artifact_dir=artifacts,
            metadata_path=metadata,
            event_name="workflow_dispatch",
            ref_name="release-candidate",
        )


def test_mixed_distribution_cohort_fails_closed(tmp_path: Path) -> None:
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
    artifacts, metadata = _candidate(tmp_path)
    (artifacts / "scpn_fusion_rs-4.0.0-unexpected.txt").touch()

    with pytest.raises(ArtifactVerificationError, match="Unexpected distribution"):
        verify_native_distribution_artifacts(
            artifact_dir=artifacts,
            metadata_path=metadata,
            event_name="workflow_dispatch",
            ref_name="main",
        )
