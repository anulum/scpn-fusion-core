# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — native distribution publication guard
"""Fail-closed verification for ``scpn-fusion-rs`` publication artifacts."""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast


class _TomlLoader(Protocol):
    """Describe the TOML loader shared by supported Python versions."""

    def loads(self, data: str, /) -> dict[str, object]:
        """Parse one TOML document."""


TOML_MODULE_NAME = "tomllib" if sys.version_info >= (3, 11) else "tomli"
tomllib = cast(_TomlLoader, importlib.import_module(TOML_MODULE_NAME))


class ArtifactVerificationError(ValueError):
    """Raised when native artifacts do not match their publication context."""


def _expected_version(metadata_path: Path) -> str:
    metadata = tomllib.loads(metadata_path.read_text(encoding="utf-8"))
    project = metadata.get("project")
    if not isinstance(project, dict) or "version" not in project:
        raise ArtifactVerificationError(f"Missing project.version in {metadata_path}")
    version: object = project["version"]
    if not isinstance(version, str) or not version:
        raise ArtifactVerificationError(f"Invalid project.version in {metadata_path}: {version!r}")
    return version


class _Arguments(argparse.Namespace):
    """Typed command-line arguments for the publication guard."""

    artifact_dir: Path
    metadata: Path
    event_name: str
    ref_name: str


def verify_native_distribution_artifacts(
    *,
    artifact_dir: Path,
    metadata_path: Path,
    event_name: str,
    ref_name: str,
) -> str:
    """Verify identity, version, artifact cohort, and triggering Git ref."""
    version = _expected_version(metadata_path)
    if event_name == "push":
        expected_ref = f"v{version}"
        if ref_name != expected_ref:
            raise ArtifactVerificationError(
                f"Tag/metadata mismatch: ref {ref_name!r}, expected {expected_ref!r}"
            )
    elif event_name == "workflow_dispatch":
        if ref_name != "main":
            raise ArtifactVerificationError(
                f"Manual publication must run from 'main', got {ref_name!r}"
            )
    else:
        raise ArtifactVerificationError(f"Unsupported publication event {event_name!r}")

    artifacts = sorted(path for path in artifact_dir.iterdir() if path.is_file())
    if not artifacts:
        raise ArtifactVerificationError("No distribution artifacts were downloaded")

    expected_sdist = f"scpn_fusion_rs-{version}.tar.gz"
    expected_wheel = re.compile(
        rf"scpn_fusion_rs-{re.escape(version)}"
        r"(?:-[0-9][A-Za-z0-9_]*)?"
        r"-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+\.whl"
    )
    unexpected = [
        path.name
        for path in artifacts
        if path.name != expected_sdist and expected_wheel.fullmatch(path.name) is None
    ]
    if unexpected:
        raise ArtifactVerificationError(f"Unexpected distribution artifact(s): {unexpected}")

    sdists = [path for path in artifacts if path.name == expected_sdist]
    wheels = [path for path in artifacts if expected_wheel.fullmatch(path.name) is not None]
    if len(sdists) != 1 or not wheels:
        raise ArtifactVerificationError(
            "Expected one native sdist and at least one wheel; "
            f"found {len(sdists)} sdist(s), {len(wheels)} wheel(s)"
        )

    return f"Validated {len(artifacts)} scpn-fusion-rs {version} artifacts"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--ref-name", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run native artifact verification from command-line arguments.

    Parameters
    ----------
    argv:
        Optional argument sequence excluding the program name.

    Returns
    -------
    int
        Zero after the artifact cohort passes all publication checks.
    """
    args = _parser().parse_args(argv, namespace=_Arguments())
    print(
        verify_native_distribution_artifacts(
            artifact_dir=args.artifact_dir,
            metadata_path=args.metadata,
            event_name=args.event_name,
            ref_name=args.ref_name,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
