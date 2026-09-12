# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — real PROCESS integration fixture
"""Generate genuine PROCESS output once per dedicated integration session."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from validation.process_reference_run import run_process_reference, verify_process_source


@pytest.fixture(scope="session")
def process_case() -> Path:
    """Resolve the upstream case before subprocesses change working directory."""
    return Path(os.environ["PROCESS_TEST_CASE"]).resolve(strict=True)


@pytest.fixture(scope="session")
def reference_run(
    process_case: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, dict[str, Any]]:
    """Execute the actual source-verified model; never replace it with mock output."""
    directory = tmp_path_factory.mktemp("process-parent") / "run"
    return directory, run_process_reference(process_case, directory, equality_tolerance=1e-8)


@pytest.fixture()
def actual_mfile(reference_run: tuple[Path, dict[str, Any]]) -> Path:
    """Expose the actual output of this session's upstream execution."""
    return reference_run[0] / "MFILE.DAT"


@pytest.fixture(scope="session")
def actual_scan_mfile(process_case: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a two-point field scan through PROCESS's public input-file API.

    This is an explicitly modified example, not the pinned reference case.
    Scan selector 17 follows upstream examples/scan.ex.py (peak toroidal field).
    """
    source = verify_process_source()
    directory = tmp_path_factory.mktemp("process-scan")
    case = directory / "IN.DAT"
    case.write_bytes(process_case.read_bytes() + b"\nnsweep = 17\nisweep = 2\nsweep = 10.5, 10.4\n")
    with (directory / "execution.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from process.main import SingleRun; "
                "SingleRun(sys.argv[1], filepath_out=sys.argv[1]).run()",
                str(case),
            ],
            check=True,
            timeout=240,
            cwd=directory,
            env={
                **os.environ,
                "OPENBLAS_NUM_THREADS": "2",
                "NUMBA_NUM_THREADS": "2",
                "MPLBACKEND": "Agg",
            },
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    assert verify_process_source() == source
    return directory / "MFILE.DAT"
