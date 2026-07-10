# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Gyro-Bohm Calibration Tests
"""Contract tests for the gyro-Bohm transport-coefficient calibration.

Covers CSV loading, the Chang-Hinton neoclassical and gyro-Bohm diffusivity
scalars (including their degenerate-input guards), the single-shot tau_E
predictor, the RMSE/log-RMSE/objective metrics, the bootstrap uncertainty
estimator, and the full ``calibrate`` routine across both the passing and
failing RMSE-gate branches and both verbosity modes.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools.calibrate_gyro_bohm import (
    ITPA_CSV,
    ShotRecord,
    bootstrap_uncertainty,
    calibrate,
    chang_hinton_chi_scalar,
    compute_log_rmse,
    compute_rmse,
    gyro_bohm_chi_scalar,
    load_itpa_csv,
    objective,
    predict_tau_e,
)

_CSV_HEADER = (
    "machine,shot,Ip_MA,BT_T,ne19_1e19m3,Ploss_MW,R_m,a_m,kappa,delta,M_AMU,tau_E_s,H98y2,source"
)


_DEFAULT_SHOT = ShotRecord(
    machine="ITER",
    shot="design",
    Ip_MA=15.0,
    BT_T=5.3,
    ne19=10.1,
    Ploss_MW=87.0,
    R_m=6.2,
    a_m=2.0,
    kappa=1.7,
    delta=0.33,
    M_AMU=2.5,
    tau_E_s=3.7,
    H98y2=1.0,
    source="test",
)


def _shot(**overrides: Any) -> ShotRecord:
    """Build a representative ITER-like shot record with field overrides."""
    return replace(_DEFAULT_SHOT, **overrides)


def _csv_row(shot: ShotRecord) -> str:
    """Render a ShotRecord back to a CSV data row."""
    return (
        f"{shot.machine},{shot.shot},{shot.Ip_MA},{shot.BT_T},{shot.ne19},"
        f"{shot.Ploss_MW},{shot.R_m},{shot.a_m},{shot.kappa},{shot.delta},"
        f"{shot.M_AMU},{shot.tau_E_s},{shot.H98y2},{shot.source}"
    )


def _write_csv(path: Path, shots: list[ShotRecord]) -> None:
    """Write a set of shot records to an ITPA-format CSV."""
    lines = [_CSV_HEADER, *(_csv_row(s) for s in shots)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestLoadItpaCsv:
    """CSV parsing into ShotRecord objects."""

    def test_parses_rows(self, tmp_path: Path) -> None:
        """A two-row CSV parses into two typed records."""
        csv_path = tmp_path / "shots.csv"
        _write_csv(csv_path, [_shot(), _shot(machine="JET", shot="42")])
        records = load_itpa_csv(csv_path)
        assert len(records) == 2
        assert records[0].machine == "ITER"
        assert records[1].shot == "42"
        assert isinstance(records[0].Ip_MA, float)

    def test_reference_csv_loads(self) -> None:
        """The bundled ITPA reference CSV loads without error."""
        records = load_itpa_csv(ITPA_CSV)
        assert len(records) > 10


class TestChangHintonChi:
    """Chang-Hinton neoclassical diffusivity scalar."""

    def test_normal_evaluation_positive(self) -> None:
        """A physical operating point yields a positive, finite chi."""
        chi = chang_hinton_chi_scalar(0.5, 5.0, 5.0, 3.0, 6.2, 2.0, 5.3)
        assert chi > 0.0
        assert np.isfinite(chi)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"rho_norm": 0.0},
            {"T_i_keV": 0.0},
            {"n_e_19": 0.0},
            {"q": 0.0},
        ],
    )
    def test_degenerate_inputs_floor(self, kwargs: dict[str, float]) -> None:
        """Non-physical inputs return the 0.01 floor."""
        args = {"rho_norm": 0.5, "T_i_keV": 5.0, "n_e_19": 5.0, "q": 3.0}
        args.update(kwargs)
        assert chang_hinton_chi_scalar(R0=6.2, a=2.0, B0=5.3, **args) == 0.01

    def test_vanishing_epsilon_floor(self) -> None:
        """A vanishing inverse aspect ratio returns the 0.01 floor."""
        # rho_norm * a / R0 < 1e-6 with all other inputs physical.
        assert chang_hinton_chi_scalar(1e-9, 5.0, 5.0, 3.0, 6.2, 2.0, 5.3) == 0.01


class TestGyroBohmChi:
    """Gyro-Bohm anomalous diffusivity scalar."""

    def test_normal_evaluation_positive(self) -> None:
        """A physical operating point yields a positive, finite chi."""
        chi = gyro_bohm_chi_scalar(2.0, 5.0, 5.0, 3.0, 6.2, 2.0, 5.3)
        assert chi > 0.0
        assert np.isfinite(chi)

    def test_low_temperature_floor(self) -> None:
        """Near-zero temperatures are floored before the scaling."""
        chi = gyro_bohm_chi_scalar(2.0, 0.0, 0.0, 3.0, 6.2, 2.0, 5.3)
        assert chi >= 0.01


class TestPredictTauE:
    """Single-shot confinement-time prediction."""

    def test_positive_prediction(self) -> None:
        """A representative shot predicts a positive confinement time."""
        tau = predict_tau_e(_shot(), c_gB=0.08)
        assert tau > 0.0

    def test_scales_with_coefficient(self) -> None:
        """Larger c_gB predicts a longer confinement time (monotone prefactor)."""
        low = predict_tau_e(_shot(), c_gB=0.05)
        high = predict_tau_e(_shot(), c_gB=0.20)
        assert high > low


class TestMetrics:
    """RMSE, log-RMSE and the optimiser objective."""

    def test_rmse_non_negative(self) -> None:
        """RMSE across shots is non-negative."""
        shots = [_shot(), _shot(machine="JET", tau_E_s=0.5)]
        assert compute_rmse(0.08, shots) >= 0.0

    def test_log_rmse_non_negative(self) -> None:
        """Log-RMSE across shots is non-negative."""
        shots = [_shot(), _shot(machine="JET", tau_E_s=0.5)]
        assert compute_log_rmse(0.08, shots) >= 0.0

    def test_objective_matches_rmse_in_log_space(self) -> None:
        """The objective evaluates RMSE at 10**log_c_gB."""
        shots = [_shot()]
        assert objective(np.log10(0.08), shots) == pytest.approx(compute_rmse(0.08, shots))


class TestBootstrapUncertainty:
    """Bootstrap 1-sigma uncertainty estimate."""

    def test_returns_finite_non_negative(self) -> None:
        """A small bootstrap run yields a finite, non-negative sigma."""
        shots = [_shot(), _shot(machine="JET", tau_E_s=0.5), _shot(machine="DIII-D")]
        sigma = bootstrap_uncertainty(shots, c_gB_best=0.08, n_bootstrap=4, rng_seed=1)
        assert sigma >= 0.0
        assert np.isfinite(sigma)


class TestCalibrate:
    """Full calibration routine and RMSE gate."""

    def test_single_shot_passes_gate_verbose(self, tmp_path: Path) -> None:
        """A single-shot fit drives RMSE→0 and reports a passing gate."""
        csv_path = tmp_path / "one.csv"
        _write_csv(csv_path, [_shot()])
        out = tmp_path / "out" / "coeff.json"
        result = calibrate(csv_path=csv_path, output_path=out, verbose=True)
        assert result["n_shots"] == 1
        assert result["rmse_relative"] < 0.15
        assert out.exists()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["c_gB"] == result["c_gB"]
        assert payload["name"] == "gyro_bohm_transport_coefficient"

    def test_reference_data_fails_gate_quiet(self, tmp_path: Path) -> None:
        """The full ITPA reference set exceeds the 15% gate (fail branch)."""
        out = tmp_path / "coeff.json"
        result = calibrate(output_path=out, verbose=False)
        assert result["rmse_relative"] >= 0.15
        assert result["n_shots"] > 10
        assert result["c_gB_uncertainty"] >= 0.0
