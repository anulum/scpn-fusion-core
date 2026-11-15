# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — tests of the linear-MHD acceptance record
"""The linear-MHD acceptance record names its source revision, source digests and host.

The scored run itself takes many minutes and is replaced here by a fixed record; the
criteria are exercised by ``validation/linear_mhd_reduced_model_acceptance.py``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from validation import linear_mhd_reduced_model_acceptance as acceptance

ROOT = Path(__file__).resolve().parents[1]


def _scored(passed: int) -> dict[str, object]:
    return {"criteria": {}, "passed": passed, "total": 1}


@pytest.mark.parametrize(("passed", "code"), [(1, 0), (0, 1)])
def test_record_binds_revision_sources_and_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, passed: int, code: int
) -> None:
    """The written record carries the checkout's revision and the digests of its sources."""
    monkeypatch.setattr(acceptance, "run", lambda: _scored(passed))
    output = tmp_path / "acceptance.json"

    assert acceptance.main(["--output", str(output)]) == code

    origin = json.loads(output.read_text(encoding="utf-8"))["provenance"]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    assert origin["source_revision"] == head
    assert set(origin["source_sha256"]) == set(acceptance.SOURCE_PATHS)
    for relative, digest in origin["source_sha256"].items():
        assert digest == hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    assert set(origin["threads"]) == set(acceptance.THREAD_VARIABLES)
    assert origin["blas"]["name"] and origin["blas"]["version"]
    assert isinstance(origin["cpu_models"], list)
    assert origin["elapsed_s"] >= 0.0


def test_record_is_printed_without_an_output_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without ``--output`` the same record, provenance included, goes to standard output."""
    monkeypatch.setattr(acceptance, "run", lambda: _scored(1))

    assert acceptance.main([]) == 0

    record = json.loads(capsys.readouterr().out)
    assert record["passed"] == 1
    assert record["provenance"]["numpy"]
