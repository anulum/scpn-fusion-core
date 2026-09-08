# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — actual PROCESS artifact tests
"""Exercise public diagnostics using the retained output of a real PROCESS run."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from validation.process_power_report import read_process_power_report

CONSTRAINTS = (
    1,
    2,
    5,
    8,
    9,
    13,
    15,
    30,
    16,
    24,
    25,
    26,
    27,
    33,
    34,
    35,
    36,
    60,
    62,
    65,
    72,
    81,
    68,
    31,
    32,
)


def test_convergence_does_not_hide_actual_physical_errors(actual_mfile: Path) -> None:
    """The real upstream case converges while violating power and design checks."""
    report = read_process_power_report(
        actual_mfile,
        expected_sha256=hashlib.sha256(actual_mfile.read_bytes()).hexdigest(),
        expected_constraints=CONSTRAINTS,
        equality_tolerance=1e-8,
    )
    row = report["scans"][0]
    assert len(report["scans"]) == 1
    assert row["checks"]["solver_converged"]
    assert row["checks"]["equality_norm_within_tolerance"]
    assert not row["checks"]["upstream_has_no_errors"]
    assert not row["checks"]["recorded_power_balances_within_tolerance"]
    assert set(row["violated_constraints"]) == {"ineq_con016", "ineq_con072", "ineq_con068"}
    assert row["checks"]["electrical_accounting_closes"]
    assert not report["actionable"] and not report["evidence_claimed"]
    assert not report["provenance_verified"]


def test_changed_artifact_cannot_reuse_original_digest(actual_mfile: Path, tmp_path: Path) -> None:
    """Reject altered actual output before invoking a permissive upstream parser."""
    original = actual_mfile.read_bytes()
    changed = tmp_path / "MFILE.DAT"
    changed.write_bytes(original + b"\n# modified\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        read_process_power_report(
            changed,
            expected_sha256=hashlib.sha256(original).hexdigest(),
            expected_constraints=CONSTRAINTS,
            equality_tolerance=1e-8,
        )


def test_missing_declared_constraint_is_not_silently_ignored(actual_mfile: Path) -> None:
    """The caller's input-case constraint contract must match the artifact."""
    with pytest.raises(ValueError, match="active constraints"):
        read_process_power_report(
            actual_mfile,
            expected_sha256=hashlib.sha256(actual_mfile.read_bytes()).hexdigest(),
            expected_constraints=CONSTRAINTS + (999,),
            equality_tolerance=1e-8,
        )


def test_actual_scan_retains_both_points(actual_scan_mfile: Path) -> None:
    """Read every point produced by an actual two-point upstream field scan."""
    report = read_process_power_report(
        actual_scan_mfile,
        expected_sha256=hashlib.sha256(actual_scan_mfile.read_bytes()).hexdigest(),
        expected_constraints=CONSTRAINTS,
        equality_tolerance=1e-8,
    )
    assert [row["scan"] for row in report["scans"]] == [1, 2]
    actual_balances = [
        float(line.split()[2])
        for line in actual_scan_mfile.read_text().splitlines()
        if "(p_plasma_imbalance_mw)" in line
    ]
    assert [row["values"]["p_plasma_imbalance_mw"] for row in report["scans"]] == actual_balances
    assert not report["actionable"] and not report["federated"]
    assert not report["evidence_claimed"]


def test_truncated_final_scan_is_rejected(actual_scan_mfile: Path, tmp_path: Path) -> None:
    """A final scan header without diagnostics must not silently drop that point."""
    lines = actual_scan_mfile.read_text().splitlines(keepends=True)
    headers = [i for i, line in enumerate(lines) if "(iscan)" in line]
    assert len(headers) == 2
    changed = tmp_path / "MFILE.DAT"
    changed.write_text("".join(lines[: headers[1] + 1]))
    with pytest.raises(ValueError, match="scan"):
        read_process_power_report(
            changed,
            expected_sha256=hashlib.sha256(changed.read_bytes()).hexdigest(),
            expected_constraints=CONSTRAINTS,
            equality_tolerance=1e-8,
        )


@pytest.mark.parametrize("scan_label", ["1", "3", "nan", "1.5"])
def test_invalid_scan_labels_are_rejected(
    actual_scan_mfile: Path, tmp_path: Path, scan_label: str
) -> None:
    """Reject duplicate, skipped, nonfinite and fractional scan identities."""
    lines = actual_scan_mfile.read_text().splitlines(keepends=True)
    headers = [i for i, line in enumerate(lines) if "(iscan)" in line]
    index = headers[1]
    words = lines[index].split()
    words[2] = scan_label
    lines[index] = " ".join(words) + "\n"
    changed = tmp_path / "MFILE.DAT"
    changed.write_text("".join(lines))
    with pytest.raises(ValueError, match="scan labels"):
        read_process_power_report(
            changed,
            expected_sha256=hashlib.sha256(changed.read_bytes()).hexdigest(),
            expected_constraints=CONSTRAINTS,
            equality_tolerance=1e-8,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("error_status", None, "missing required"),
        ("p_plant_electric_net_mw", "nan", "finite"),
        ("p_plasma_imbalance_mw", "inf", "finite"),
        ("ifail", "1.5", "integers"),
    ],
)
def test_corrupt_diagnostics_are_rejected(
    actual_mfile: Path, tmp_path: Path, field: str, replacement: str | None, message: str
) -> None:
    """Refuse missing and invalid diagnostics in a real single-run artifact."""
    lines = actual_mfile.read_text().splitlines(keepends=True)
    indices = [i for i, line in enumerate(lines) if f"({field})" in line]
    assert indices
    for index in indices:
        words = lines[index].split()
        if replacement is None:
            lines[index] = ""
        else:
            words[2] = replacement
            lines[index] = " ".join(words) + "\n"
    changed = tmp_path / "MFILE.DAT"
    changed.write_text("".join(lines))
    with pytest.raises(ValueError, match=message):
        read_process_power_report(
            changed,
            expected_sha256=hashlib.sha256(changed.read_bytes()).hexdigest(),
            expected_constraints=CONSTRAINTS,
            equality_tolerance=1e-8,
        )


def test_missing_first_scan_field_cannot_borrow_later_value(
    actual_scan_mfile: Path, tmp_path: Path
) -> None:
    """Keep a missing first-point error status from shifting the second into it."""
    lines = actual_scan_mfile.read_text().splitlines(keepends=True)
    indices = [i for i, line in enumerate(lines) if "(error_status)" in line]
    assert len(indices) == 2
    lines[indices[0]] = ""
    changed = tmp_path / "MFILE.DAT"
    changed.write_text("".join(lines))
    with pytest.raises(ValueError, match="scan is missing required"):
        read_process_power_report(
            changed,
            expected_sha256=hashlib.sha256(changed.read_bytes()).hexdigest(),
            expected_constraints=CONSTRAINTS,
            equality_tolerance=1e-8,
        )


def test_conflicting_repeated_power_value_is_rejected(actual_mfile: Path, tmp_path: Path) -> None:
    """Do not let upstream's first-value preference hide conflicting accounting."""
    lines = actual_mfile.read_text().splitlines(keepends=True)
    indices = [i for i, line in enumerate(lines) if "(p_plant_electric_net_mw)" in line]
    assert len(indices) > 1
    index = indices[-1]
    words = lines[index].split()
    words[2] = str(float(words[2]) + 1.0)
    lines[index] = " ".join(words) + "\n"
    changed = tmp_path / "MFILE.DAT"
    changed.write_text("".join(lines))
    with pytest.raises(ValueError, match="conflicting diagnostic values"):
        read_process_power_report(
            changed,
            expected_sha256=hashlib.sha256(changed.read_bytes()).hexdigest(),
            expected_constraints=CONSTRAINTS,
            equality_tolerance=1e-8,
        )


def test_finite_powers_cannot_overflow_accounting_output(
    actual_mfile: Path, tmp_path: Path
) -> None:
    """Reject nonfinite arithmetic even when each recorded input power is finite."""
    lines = actual_mfile.read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        words = line.split()
        if "(p_plant_electric_gross_mw)" in line:
            words[2] = "1e308"
        elif "(p_plant_electric_recirc_mw)" in line:
            words[2] = "-1e308"
        else:
            continue
        lines[index] = " ".join(words) + "\n"
    changed = tmp_path / "MFILE.DAT"
    changed.write_text("".join(lines))
    with pytest.raises(ValueError, match="accounting.*finite"):
        read_process_power_report(
            changed,
            expected_sha256=hashlib.sha256(changed.read_bytes()).hexdigest(),
            expected_constraints=CONSTRAINTS,
            equality_tolerance=1e-8,
        )


@pytest.mark.parametrize("field", ["ineq_con016", "ineq_con068", "ineq_con072", "eq_con001"])
@pytest.mark.parametrize("replacement", ["0.1", "nan", "inf", "-inf"])
@pytest.mark.parametrize("before", [True, False])
def test_constraint_repeats_must_be_finite_and_consistent(
    actual_mfile: Path, tmp_path: Path, field: str, replacement: str, before: bool
) -> None:
    """Neither duplicate ordering may hide an invalid recorded constraint value."""
    lines = actual_mfile.read_text().splitlines(keepends=True)
    index = next(i for i, line in enumerate(lines) if f"({field})" in line)
    words = lines[index].split()
    assert float(words[2]) != float(replacement)
    words[2] = replacement
    lines.insert(index if before else index + 1, " ".join(words) + "\n")
    changed = tmp_path / "MFILE.DAT"
    changed.write_text("".join(lines))
    with pytest.raises(ValueError, match="finite|conflicting"):
        read_process_power_report(
            changed,
            expected_sha256=hashlib.sha256(changed.read_bytes()).hexdigest(),
            expected_constraints=CONSTRAINTS,
            equality_tolerance=1e-8,
        )


def test_identical_constraint_repeats_preserve_violations(
    actual_scan_mfile: Path, tmp_path: Path
) -> None:
    """Accept genuine equal repeats without losing any original constraint violation."""
    original = actual_scan_mfile.read_bytes()
    baseline = read_process_power_report(
        actual_scan_mfile,
        expected_sha256=hashlib.sha256(original).hexdigest(),
        expected_constraints=CONSTRAINTS,
        equality_tolerance=1e-8,
    )
    lines = original.decode().splitlines(keepends=True)
    duplicated = []
    for line in lines:
        duplicated.append(line)
        if any(
            f"({field})" in line
            for field in ["ineq_con016", "ineq_con068", "ineq_con072", "eq_con001"]
        ):
            duplicated.append(line)
    changed = tmp_path / "MFILE.DAT"
    changed.write_text("".join(duplicated))
    report = read_process_power_report(
        changed,
        expected_sha256=hashlib.sha256(changed.read_bytes()).hexdigest(),
        expected_constraints=CONSTRAINTS,
        equality_tolerance=1e-8,
    )
    assert report["scans"] == baseline["scans"]
    assert not report["scans"][0]["checks"]["all_declared_inequalities_satisfied"]


@pytest.mark.parametrize("scan", [1, 2])
@pytest.mark.parametrize(("field", "value"), [("ineq_con016", "0.1"), ("eq_con001", "nan")])
def test_constraint_conflict_is_rejected_in_each_scan(
    actual_scan_mfile: Path, tmp_path: Path, scan: int, field: str, value: str
) -> None:
    """A contradiction in either real scan point must reject the whole artifact."""
    lines = actual_scan_mfile.read_text().splitlines(keepends=True)
    boundaries = [i for i, line in enumerate(lines) if "(iscan)" in line] + [len(lines)]
    indices = [i for i in range(boundaries[scan - 1], boundaries[scan]) if f"({field})" in lines[i]]
    assert len(indices) >= 2
    index = indices[0] if field.startswith("ineq") else indices[-1]
    words = lines[index].split()
    words[2] = value
    lines[index] = " ".join(words) + "\n"
    changed = tmp_path / "MFILE.DAT"
    changed.write_text("".join(lines))
    with pytest.raises(ValueError, match="finite|conflicting"):
        read_process_power_report(
            changed,
            expected_sha256=hashlib.sha256(changed.read_bytes()).hexdigest(),
            expected_constraints=CONSTRAINTS,
            equality_tolerance=1e-8,
        )


@pytest.mark.parametrize(
    ("digest", "constraints", "tolerance", "message"),
    [
        ("invalid", CONSTRAINTS, 1e-8, "SHA256"),
        (None, CONSTRAINTS, 0.0, "tolerance"),
        (None, CONSTRAINTS, float("nan"), "tolerance"),
        (None, (), 1e-8, "unique positive IDs"),
        (None, (1, 1), 1e-8, "unique positive IDs"),
        (None, (True,), 1e-8, "unique positive IDs"),
        (None, (1000,), 1e-8, "unique positive IDs"),
    ],
)
def test_invalid_caller_contract_rejects_actual_artifact(
    actual_mfile: Path,
    digest: str | None,
    constraints: tuple[int, ...],
    tolerance: float,
    message: str,
) -> None:
    """Invalid custody, tolerance or constraint declarations cannot enter parsing."""
    with pytest.raises(ValueError, match=message):
        read_process_power_report(
            actual_mfile,
            expected_sha256=digest
            if digest is not None
            else hashlib.sha256(actual_mfile.read_bytes()).hexdigest(),
            expected_constraints=constraints,
            equality_tolerance=tolerance,
        )


def test_diagnostic_before_first_scan_refuses(actual_scan_mfile: Path, tmp_path: Path) -> None:
    """A diagnostic before scan one cannot be silently assigned to that point."""
    lines = actual_scan_mfile.read_text().splitlines(keepends=True)
    header = next(i for i, line in enumerate(lines) if "(iscan)" in line)
    diagnostic = next(line for line in lines if "(ifail)" in line)
    lines.insert(header, diagnostic)
    changed = tmp_path / "MFILE.DAT"
    changed.write_text("".join(lines))
    with pytest.raises(ValueError, match="precede the first scan"):
        read_process_power_report(
            changed,
            expected_sha256=hashlib.sha256(changed.read_bytes()).hexdigest(),
            expected_constraints=CONSTRAINTS,
            equality_tolerance=1e-8,
        )


def test_upstream_early_terminator_cannot_truncate_scan_values(
    actual_scan_mfile: Path, tmp_path: Path
) -> None:
    """Reject a parser stop marker that hides fields still present in the raw file."""
    lines = actual_scan_mfile.read_text().splitlines(keepends=True)
    statuses = [i for i, line in enumerate(lines) if "(error_status)" in line]
    assert len(statuses) == 2
    lines.insert(statuses[1], "***\n")
    changed = tmp_path / "MFILE.DAT"
    changed.write_text("".join(lines))
    with pytest.raises(ValueError, match="scan counts"):
        read_process_power_report(
            changed,
            expected_sha256=hashlib.sha256(changed.read_bytes()).hexdigest(),
            expected_constraints=CONSTRAINTS,
            equality_tolerance=1e-8,
        )


def test_artifact_replacement_during_parsing_is_refused(actual_mfile: Path, tmp_path: Path) -> None:
    """Exercise successive filesystem versions through the real upstream parser.

    A FIFO EOF boundary orders replacement without sleeps or patched readers.
    The parser receives a regular file with unchanged diagnostics but changed bytes.
    The separate interpreter bounds any unexpected FIFO deadlock.
    """
    code = r"""
import hashlib
import os
import sys
import threading
from pathlib import Path
from validation.process_power_report import read_process_power_report

raw = Path(sys.argv[1]).read_bytes()
path = Path(sys.argv[2])
os.mkfifo(path)
def replace_between_reads():
    with path.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        path.unlink()
        path.write_bytes(raw + b"\n# changed custody\n")
writer = threading.Thread(target=replace_between_reads, daemon=True)
writer.start()
try:
    read_process_power_report(
        path, expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_constraints=tuple(map(int, sys.argv[3].split(","))),
        equality_tolerance=1e-8,
    )
except ValueError as error:
    assert str(error) == "PROCESS artifact changed during parsing", str(error)
    print(str(error))
else:
    raise AssertionError("Changed artifact was accepted")
writer.join(timeout=1)
assert not writer.is_alive()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(actual_mfile),
            str(tmp_path / "MFILE.DAT"),
            ",".join(map(str, CONSTRAINTS)),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "artifact changed during parsing" in result.stdout
