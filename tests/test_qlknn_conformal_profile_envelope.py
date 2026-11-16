# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — tests of the frozen-coefficient profile envelope
"""The profile envelope on a synthetic certificate and a shortened transport run.

A zero-weight network predicts the same turbulent diffusivity everywhere, so the neural
steady state is cheap. The step counts are shortened, so the steady-state gate may fail
here; the tests check that the gates report what the runs did.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools import generate_surrogate_conformal_certificates as certificates
from validation import qlknn_conformal_profile_envelope as envelope

ROWS = 40


def _weights(path: Path) -> None:
    np.savez(
        path,
        input_mean=np.zeros(10),
        input_std=np.ones(10),
        output_scale=np.ones(3),
        version=np.asarray(1),
        log_transform=np.asarray(0),
        gb_scale=np.asarray(0),
        gated=np.asarray(0),
        w1=np.zeros((10, 4)),
        b1=np.zeros(4),
        w2=np.zeros((4, 3)),
        b2=np.zeros(3),
    )


def _split(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = np.ones((ROWS, 10))
    x[:, 0] = np.linspace(0.1, 0.9, ROWS)
    x[:, 1], x[:, 2], x[:, 7] = 2.0, 3.0, 2.0
    y = rng.uniform(0.1, 4.0, (ROWS, 3))
    y[: ROWS // 2] = 0.0
    np.savez(path, X=x, Y=y, regimes=np.zeros(ROWS, dtype=np.int32))


@pytest.fixture
def certificate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    weights, calibration, holdout = (tmp_path / f"{n}.npz" for n in ("w", "cal", "test"))
    _weights(weights)
    _split(calibration, 1)
    _split(holdout, 2)
    payload = certificates.build_certificate(
        weights_path=weights, calibration_path=calibration, holdout_path=holdout, alpha=0.2
    )
    path = tmp_path / "certificate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(envelope, "STEPS", 40)
    monkeypatch.setattr(envelope, "CHECK_STEPS", 5)
    return path


def test_bounded_coefficients_move_clip_and_recombine_as_the_closure_does() -> None:
    base = np.array([0.1, 0.2])
    turbulent = {"chi_e": np.array([1.0, 0.5]), "chi_i": np.array([2.0, 0.0]), "D_e": np.zeros(2)}
    half = {"chi_e": np.array([0.5, 1.0]), "chi_i": np.ones(2), "D_e": np.full(2, 0.3)}

    chi_e, chi_i, d_n = envelope.bounded_coefficients(base, turbulent, half, -1)
    assert chi_e.tolist() == pytest.approx([0.6, 0.2])
    assert chi_i.tolist() == pytest.approx([1.1, 0.2])
    assert d_n.tolist() == pytest.approx([0.01, 0.02])

    chi_e, chi_i, d_n = envelope.bounded_coefficients(base, turbulent, half, 1)
    assert chi_e.tolist() == pytest.approx([1.6, 1.7])
    assert d_n.tolist() == pytest.approx([0.3, 0.3])


def test_report_runs_every_envelope_and_records_its_gates(certificate: Path) -> None:
    report = envelope.build_report(certificate)

    assert report["case"]["p_aux_mw"] == envelope.P_AUX_MW
    assert set(report["envelopes"]) == {"certificate", "normalised_prediction"}
    for variant in report["envelopes"].values():
        assert (
            variant["lower_turbulence"]["T_core_keV"] >= variant["upper_turbulence"]["T_core_keV"]
        )
    expected = {
        "neural_run_reached_steady_state": report["steady_state_relative_change"] < 1e-3,
        "nominal_frozen_run_reproduces_neural_steady_state": (
            report["nominal_reproduction_relative_error"] < 1e-3
        ),
    }
    assert report["checks"] == expected
    assert report["status"] == ("pass" if all(expected.values()) else "fail")
    assert len(report["source_revision"]) == 40


def test_report_refuses_power_where_a_pedestal_modifier_acts(certificate: Path) -> None:
    with pytest.raises(ValueError, match="no pedestal modifier acts"):
        envelope.build_report(certificate, p_aux_mw=40.0)


def test_report_refuses_weights_that_differ_from_the_certificate(
    certificate: Path, tmp_path: Path
) -> None:
    np.savez(tmp_path / "w.npz", w1=np.ones((2, 2)))
    with pytest.raises(ValueError, match="weights differ"):
        envelope.build_report(certificate)


def test_command_line_writes_both_reports(certificate: Path, tmp_path: Path) -> None:
    output, markdown = tmp_path / "env.json", tmp_path / "env.md"
    arguments = ["--certificate", str(certificate), "--output", str(output)]

    code = envelope.main([*arguments, "--markdown", str(markdown)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert code == (0 if report["status"] == "pass" else 1)
    text = markdown.read_text(encoding="utf-8")
    assert text.splitlines()[:7] == list(envelope.MARKDOWN_HEADER)
    assert "| certificate | lower |" in text and "| normalised_prediction | upper |" in text


def test_script_entry_point_runs_from_the_repository_root() -> None:
    """Run the script directly: it must put the repository on its own path to import tools."""
    root = Path(envelope.__file__).resolve().parents[1]
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    completed = subprocess.run(
        [sys.executable, "validation/qlknn_conformal_profile_envelope.py", "--help"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--p-aux" in completed.stdout


def test_import_puts_the_repository_on_the_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Imported from outside the repository, the module puts the root on the path itself."""
    root = str(envelope.ROOT)
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != root])
    importlib.reload(envelope)
    assert sys.path[0] == root
