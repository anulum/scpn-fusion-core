# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — tests of the self-consistent QLKNN conformal profile envelope
"""The self-consistent envelope on a synthetic certificate and shortened transport runs.

A zero-weight network predicts no turbulent diffusivity, so every solve is cheap. The step
limits are shortened, so the convergence gates may fail here; the tests check that the
gates report what the runs did, and test the iteration itself with stand-in closures.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools import generate_surrogate_conformal_certificates as certificates
from validation import qlknn_conformal_profile_envelope as frozen
from validation import qlknn_conformal_self_consistent_envelope as study

ROWS = 40
P_AUX = 20.0


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
def certificate(tmp_path: Path) -> Path:
    weights, calibration, holdout = (tmp_path / f"{n}.npz" for n in ("w", "cal", "test"))
    _weights(weights)
    _split(calibration, 1)
    _split(holdout, 2)
    payload = certificates.build_certificate(
        weights_path=weights, calibration_path=calibration, holdout_path=holdout, alpha=0.2
    )
    path = tmp_path / "certificate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _frozen_report(intervals: study.Intervals) -> dict[str, Any]:
    end = {"T_core_keV": 1.0, "T_avg_keV": 0.5, "tau_e_s": 0.1}
    return {
        "weights_sha256": intervals.weights_sha256,
        "certificate_radius_m2_s": dict(intervals.radius),
        "normalised_prediction_quantile": dict(intervals.normalised_quantile),
        "envelopes": {
            variant: {"lower_turbulence": end, "upper_turbulence": end}
            for variant in study.VARIANTS
        },
    }


@pytest.fixture
def frozen_report(certificate: Path, tmp_path: Path) -> Path:
    path = tmp_path / "frozen.json"
    report = _frozen_report(study.load_intervals(certificate))
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


@pytest.fixture
def short_runs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(study, "MAX_STEPS", 6)
    monkeypatch.setattr(frozen, "STEPS", 6)
    monkeypatch.setattr(frozen, "CHECK_STEPS", 2)
    yield


def _intervals() -> study.Intervals:
    return study.Intervals(
        alpha=0.1,
        weights=Path("w.npz"),
        weights_sha256="a" * 64,
        radius={"chi_e": 2.0, "chi_i": 3.0, "D_e": 0.5},
        normalised_quantile={"chi_e": 0.5, "chi_i": 0.25, "D_e": 1.0},
    )


def test_half_widths_follow_each_variant() -> None:
    turbulent = {name: np.array([0.0, -4.0]) for name in frozen.CHANNELS}

    constant = study.half_widths("certificate", turbulent, _intervals())
    scaled = study.half_widths("normalised_prediction", turbulent, _intervals())

    assert constant["chi_i"].tolist() == [3.0, 3.0]
    assert scaled["chi_e"].tolist() == pytest.approx([1.0, 3.0])
    assert scaled["D_e"].tolist() == pytest.approx([0.5, 4.5])
    with pytest.raises(ValueError, match="unknown interval variant"):
        study.half_widths("mondrian", turbulent, _intervals())


def test_residual_and_relaxation() -> None:
    applied, target = np.array([3.0, 4.0]), np.array([3.0, 9.0])

    assert study.relative_residual(target, applied) == pytest.approx(1.0)
    assert study.relax(applied, target, 0.2).tolist() == pytest.approx([3.0, 5.0])


def test_widths_match_checks_the_digest_and_every_width() -> None:
    intervals = _intervals()
    report = _frozen_report(intervals)
    assert study.widths_match(intervals, report)

    report["normalised_prediction_quantile"]["D_e"] = 1.0 + 1e-9
    assert not study.widths_match(intervals, report)
    assert not study.widths_match(intervals, {**_frozen_report(intervals), "weights_sha256": ""})


def test_closure_target_recovers_the_neural_diffusivity_at_zero_width(certificate: Path) -> None:
    """With zero half-width the bounded target equals the coefficients of the real backend."""
    intervals = study.load_intervals(certificate)
    solver = study.make_solver(frozen.CONFIG_PATH, intervals.weights)

    target, widths = study.closure_target(solver, P_AUX, "certificate", 0, intervals)

    neural = np.stack([solver.chi_e, solver.chi_i, solver.D_n])
    assert solver.transport_backend == "neural_transport"
    assert np.array_equal(target, neural)
    assert widths["chi_e"][0] == intervals.radius["chi_e"]


def test_run_stops_once_the_target_stops_moving(
    certificate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intervals = study.load_intervals(certificate)

    def constant(
        solver: Any, p_aux: float, variant: str, sign: int, given: study.Intervals
    ) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        size = solver.rho.size
        return np.ones((3, size)), {name: np.full(size, 0.5) for name in frozen.CHANNELS}

    monkeypatch.setattr(study, "closure_target", constant)
    run = study.self_consistent_run(frozen.CONFIG_PATH, P_AUX, intervals, "certificate", 1)

    assert run["steps"] == 2
    assert run["converged"] and run["final_relative_residual"] == 0.0
    assert run["median_half_width_m2_s"] == {name: 0.5 for name in frozen.CHANNELS}
    assert run["T_core_keV"] > 0.0


def test_run_reports_the_step_limit_without_convergence(
    certificate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intervals = study.load_intervals(certificate)
    calls: list[int] = []

    def alternating(
        solver: Any, p_aux: float, variant: str, sign: int, given: study.Intervals
    ) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        calls.append(sign)
        size = solver.rho.size
        level = 1.0 + (len(calls) % 2)
        return np.full((3, size), level), {name: np.zeros(size) for name in frozen.CHANNELS}

    monkeypatch.setattr(study, "MAX_STEPS", 3)
    monkeypatch.setattr(study, "closure_target", alternating)
    run = study.self_consistent_run(frozen.CONFIG_PATH, P_AUX, intervals, "certificate", -1)

    assert run["steps"] == 3 and len(calls) == 3
    # Targets 2, 1, 2: the first is applied as is, the second relaxes it to 1.995, and the
    # third then differs by 0.005 / 1.995.
    assert not run["converged"]
    assert run["final_relative_residual"] == pytest.approx(0.005 / 1.995)


def test_report_runs_every_end_and_records_its_gates(
    certificate: Path, frozen_report: Path, short_runs: None
) -> None:
    report = study.build_report(certificate, frozen_report_path=frozen_report)

    assert set(report["envelopes"]) == set(study.VARIANTS)
    runs = [
        report["nominal"],
        report["nominal_at_check_relaxation"],
        *(run for ends in report["envelopes"].values() for run in ends.values()),
    ]
    assert all(run["steps"] <= study.MAX_STEPS for run in runs)
    expected = {
        "widths_match_frozen_report": True,
        "neural_run_reached_steady_state": report["neural_steady_state_relative_change"] < 1e-3,
        "nominal_reproduces_neural_steady_state": (
            report["nominal_reproduction_relative_error"] < 1e-3
        ),
        "nominal_fixed_point_independent_of_relaxation": (
            report["relaxation_independence_relative_difference"] < 1e-8
        ),
        "every_run_converged": all(run["converged"] for run in runs),
    }
    assert report["checks"] == expected
    assert report["status"] == ("pass" if all(expected.values()) else "fail")
    assert report["frozen_counterpart"]["report_sha256"] == (
        hashlib.sha256(frozen_report.read_bytes()).hexdigest()
    )
    assert set(report["source_sha256"]) == set(study.SOURCE_PATHS)
    assert len(report["source_revision"]) == 40


def test_report_refuses_power_where_a_pedestal_modifier_acts(certificate: Path) -> None:
    with pytest.raises(ValueError, match="no pedestal modifier acts"):
        study.build_report(certificate, p_aux_mw=40.0)


def test_intervals_refuse_weights_that_differ_from_the_certificate(
    certificate: Path, tmp_path: Path
) -> None:
    np.savez(tmp_path / "w.npz", w1=np.ones((2, 2)))
    with pytest.raises(ValueError, match="weights differ"):
        study.load_intervals(certificate)


def test_command_line_writes_both_reports(
    certificate: Path, frozen_report: Path, tmp_path: Path, short_runs: None
) -> None:
    output, markdown = tmp_path / "sc.json", tmp_path / "sc.md"
    arguments = ["--certificate", str(certificate), "--frozen-report", str(frozen_report)]

    code = study.main([*arguments, "--output", str(output), "--markdown", str(markdown)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert code == (0 if report["status"] == "pass" else 1)
    text = markdown.read_text(encoding="utf-8")
    assert text.splitlines()[:7] == list(study.MARKDOWN_HEADER)
    assert "| certificate | lower |" in text and "| normalised_prediction | upper |" in text
    assert "Failed checks: " in text


def test_script_entry_point_runs_from_the_repository_root() -> None:
    """Run the script directly: it must put the repository on its own path to import tools."""
    root = Path(study.__file__).resolve().parents[1]
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    completed = subprocess.run(
        [sys.executable, "validation/qlknn_conformal_self_consistent_envelope.py", "--help"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--frozen-report" in completed.stdout


def test_import_puts_the_repository_on_the_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Imported from outside the repository, the module puts the root on the path itself."""
    root = str(study.ROOT)
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != root])
    importlib.reload(study)
    assert sys.path[0] == root
