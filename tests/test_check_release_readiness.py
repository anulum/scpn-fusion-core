# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Release Readiness Guard Tests
"""Contract tests for the release-readiness markdown validation tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import check_release_readiness as tool
from tools.check_release_readiness import (
    REQUIRED_ITEMS,
    _normalize,
    _parse_check_items,
    _parse_readiness_state,
    _parse_release_version,
    check_release_readiness,
    main,
)


def _valid_readiness(version: str = "v3.5.0", state: str = "ready") -> str:
    """Build a readiness markdown with every required item checked."""
    lines = [f"Release Version: `{version}`", f"Readiness State: `{state}`", ""]
    lines.extend(f"- [x] {item}" for item in REQUIRED_ITEMS)
    return "\n".join(lines) + "\n"


class TestParsers:
    """Low-level readiness-field parsers."""

    def test_normalize_collapses_whitespace(self) -> None:
        """Normalisation lowercases and collapses runs of whitespace."""
        assert _normalize("  A   B\tC ") == "a b c"

    def test_parse_release_version_present_and_absent(self) -> None:
        """The release version is extracted when present, else None."""
        assert _parse_release_version("Release Version: `v1.2.3`") == "v1.2.3"
        assert _parse_release_version("no version here") is None

    def test_parse_readiness_state_present_and_absent(self) -> None:
        """The readiness state is extracted when present, else None."""
        assert _parse_readiness_state("Readiness State: `ready`") == "ready"
        assert _parse_readiness_state("nothing") is None

    def test_parse_check_items_tracks_checked_flag(self) -> None:
        """Check items map their normalised label to the checked flag."""
        items = _parse_check_items("- [x] Done thing\n- [ ] Pending thing")
        assert items["done thing"] is True
        assert items["pending thing"] is False


class TestCheckReleaseReadiness:
    """The structural readiness validator."""

    def test_valid_readiness_has_no_errors(self, tmp_path: Path) -> None:
        """A fully-checked, ready, version-matched file yields no errors."""
        path = tmp_path / "R.md"
        path.write_text(_valid_readiness(), encoding="utf-8")
        errors = check_release_readiness(path, expected_version="v3.5.0", require_ready_state=True)
        assert errors == []

    def test_missing_file_is_reported(self, tmp_path: Path) -> None:
        """A missing readiness file is reported as a single error."""
        errors = check_release_readiness(
            tmp_path / "absent.md", expected_version=None, require_ready_state=True
        )
        assert len(errors) == 1
        assert "missing" in errors[0].lower()

    def test_missing_version_line(self, tmp_path: Path) -> None:
        """A readiness file without a version line is flagged."""
        path = tmp_path / "R.md"
        body = _valid_readiness().replace("Release Version: `v3.5.0`", "")
        path.write_text(body, encoding="utf-8")
        errors = check_release_readiness(path, expected_version=None, require_ready_state=True)
        assert any("Release Version" in e for e in errors)

    def test_version_mismatch(self, tmp_path: Path) -> None:
        """A version differing from the expected value is flagged."""
        path = tmp_path / "R.md"
        path.write_text(_valid_readiness(version="v3.4.0"), encoding="utf-8")
        errors = check_release_readiness(path, expected_version="v3.5.0", require_ready_state=True)
        assert any("mismatch" in e for e in errors)

    def test_missing_and_not_ready_state(self, tmp_path: Path) -> None:
        """A missing or non-ready state is flagged when readiness is required."""
        missing = tmp_path / "missing.md"
        missing.write_text(
            _valid_readiness().replace("Readiness State: `ready`", ""), encoding="utf-8"
        )
        assert any(
            "Readiness State" in e
            for e in check_release_readiness(
                missing, expected_version=None, require_ready_state=True
            )
        )

        not_ready = tmp_path / "notready.md"
        not_ready.write_text(_valid_readiness(state="draft"), encoding="utf-8")
        assert any(
            "must be 'ready'" in e
            for e in check_release_readiness(
                not_ready, expected_version=None, require_ready_state=True
            )
        )

    def test_allow_not_ready_skips_state_checks(self, tmp_path: Path) -> None:
        """When readiness is not required, the state is not validated."""
        path = tmp_path / "R.md"
        path.write_text(_valid_readiness(state="draft"), encoding="utf-8")
        errors = check_release_readiness(path, expected_version=None, require_ready_state=False)
        assert errors == []

    def test_missing_item_and_unchecked_item(self, tmp_path: Path) -> None:
        """A dropped item is missing and a `- [ ]` item is unchecked."""
        first = REQUIRED_ITEMS[0]
        second = REQUIRED_ITEMS[1]
        body = _valid_readiness()
        body = body.replace(f"- [x] {first}\n", "")
        body = body.replace(f"- [x] {second}", f"- [ ] {second}")
        path = tmp_path / "R.md"
        path.write_text(body, encoding="utf-8")
        errors = check_release_readiness(path, expected_version=None, require_ready_state=True)
        assert any("Missing readiness item" in e for e in errors)
        assert any("not checked" in e for e in errors)


class TestMain:
    """The command-line entry point."""

    def test_main_passes_on_valid_relative_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A valid file addressed relatively under the repo root passes."""
        monkeypatch.setattr(tool, "REPO_ROOT", tmp_path)
        (tmp_path / "R.md").write_text(_valid_readiness(), encoding="utf-8")
        rc = main(["--readiness-file", "R.md", "--expected-version", "v3.5.0"])
        assert rc == 0
        assert "passed" in capsys.readouterr().out

    def test_main_fails_on_invalid_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An invalid readiness file makes the CLI return a non-zero code."""
        path = tmp_path / "R.md"
        path.write_text(_valid_readiness(state="draft"), encoding="utf-8")
        rc = main(["--readiness-file", str(path)])
        assert rc == 1
        assert "FAILED" in capsys.readouterr().out
