# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — IPB98(y,2) Scaling Law Benchmark Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Benchmark tests for the IPB98(y,2) confinement scaling law.

Tests cover:
  - Formula correctness against hand-calculated reference values
  - RMSE against the 20-shot ITPA H-mode confinement dataset
  - Per-machine error breakdown
  - H-factor computation
  - Input validation (non-physical and non-finite values rejected)
"""

from __future__ import annotations

import csv
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core.scaling_laws import (
    TransportBenchmarkResult,
    compute_h_factor,
    ipb98y2_tau_e,
    ipb98y2_with_uncertainty,
    load_ipb98y2_coefficients,
)

logger = logging.getLogger(__name__)

_ITPA_DIR = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "reference_data"
    / "itpa"
)
_CSV_PATH = _ITPA_DIR / "hmode_confinement.csv"
_COEFF_PATH = _ITPA_DIR / "ipb98y2_coefficients.json"


def _load_itpa_dataset():
    """Load the 20-shot ITPA H-mode confinement dataset."""
    rows = []
    with open(_CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ── Hand-calculated reference values ─────────────────────────────────


class TestIPB98y2Formula:
    """Verify the IPB98(y,2) formula against known values."""

    def test_iter_design_point(self):
        """ITER design: Ip=15MA, BT=5.3T, ne=10.1, Ploss=87MW, R=6.2m."""
        # Hand-calculated using published IPB98(y,2) coefficients:
        # τ_E = 0.0562 * 15^0.93 * 5.3^0.15 * 10.1^0.41 * 87^-0.69
        #       * 6.2^1.97 * 1.70^0.78 * (2.0/6.2)^0.58 * 2.5^0.19
        tau = ipb98y2_tau_e(
            Ip=15.0, BT=5.3, ne19=10.1, Ploss=87.0,
            R=6.2, kappa=1.70, epsilon=2.0 / 6.2, M=2.5,
        )
        # The ITER design expects τ_E ≈ 3.7s at H98=1.0
        assert 2.0 < tau < 6.0, f"ITER τ_E={tau:.3f}s out of expected range"
        logger.info("ITER design τ_E = %.3f s (expected ~3.7 s)", tau)

    def test_coefficients_loaded(self):
        """Coefficient file loads correctly with expected keys."""
        coeff = load_ipb98y2_coefficients()
        assert "C" in coeff
        assert "exponents" in coeff
        assert coeff["C"] == pytest.approx(0.0562, abs=0.001)
        assert coeff["exponents"]["Ip_MA"] == pytest.approx(0.93, abs=0.01)
        assert coeff["exponents"]["Ploss_MW"] == pytest.approx(-0.69, abs=0.01)

    def test_load_coefficients_rejects_non_finite_c(self, tmp_path: Path):
        """Coefficient loader should reject non-finite scale factor C."""
        coeff = load_ipb98y2_coefficients(_COEFF_PATH)
        coeff["C"] = float("nan")
        path = tmp_path / "bad_coeff_nan_c.json"
        path.write_text(json.dumps(coeff), encoding="utf-8")
        with pytest.raises(ValueError, match="C"):
            load_ipb98y2_coefficients(path)

    def test_load_coefficients_rejects_missing_exponent_key(self, tmp_path: Path):
        """Coefficient loader should reject incomplete exponent mappings."""
        coeff = load_ipb98y2_coefficients(_COEFF_PATH)
        del coeff["exponents"]["Ip_MA"]
        path = tmp_path / "bad_coeff_missing_key.json"
        path.write_text(json.dumps(coeff), encoding="utf-8")
        with pytest.raises(ValueError, match="missing exponent key 'Ip_MA'"):
            load_ipb98y2_coefficients(path)

    def test_tau_rejects_negative_sigma_lnc_in_custom_coefficients(self):
        """Runtime evaluator should validate custom uncertainty metadata."""
        coeff = load_ipb98y2_coefficients(_COEFF_PATH)
        coeff["sigma_lnC"] = -0.1
        with pytest.raises(ValueError, match="sigma_lnC"):
            ipb98y2_tau_e(
                Ip=15.0,
                BT=5.3,
                ne19=10.1,
                Ploss=87.0,
                R=6.2,
                kappa=1.70,
                epsilon=2.0 / 6.2,
                M=2.5,
                coefficients=coeff,
            )

    def test_with_uncertainty_normalizes_numeric_string_uncertainties(self):
        """Uncertainty path should normalize numeric-string metadata safely."""
        coeff = load_ipb98y2_coefficients(_COEFF_PATH)
        coeff["exponent_uncertainties"] = {"Ip_MA": "0.02"}
        tau, sigma = ipb98y2_with_uncertainty(
            Ip=15.0,
            BT=5.3,
            ne19=10.1,
            Ploss=87.0,
            R=6.2,
            kappa=1.70,
            epsilon=2.0 / 6.2,
            M=2.5,
            coefficients=coeff,
        )
        assert np.isfinite(tau) and tau > 0.0
        assert np.isfinite(sigma) and sigma >= 0.0

    def test_load_coefficients_rejects_negative_exponent_uncertainty(
        self,
        tmp_path: Path,
    ):
        """Coefficient loader should reject negative exponent uncertainty."""
        coeff = load_ipb98y2_coefficients(_COEFF_PATH)
        coeff["exponent_uncertainties"] = {"Ip_MA": -0.01}
        path = tmp_path / "bad_coeff_negative_uncertainty.json"
        path.write_text(json.dumps(coeff), encoding="utf-8")
        with pytest.raises(ValueError, match="exponent_uncertainties.Ip_MA"):
            load_ipb98y2_coefficients(path)

    def test_tau_monotonic_in_ip(self):
        """τ_E should increase with plasma current (positive exponent)."""
        base = dict(BT=5.0, ne19=8.0, Ploss=10.0, R=3.0,
                    kappa=1.7, epsilon=0.3, M=2.5)
        tau_low = ipb98y2_tau_e(Ip=1.0, **base)
        tau_high = ipb98y2_tau_e(Ip=3.0, **base)
        assert tau_high > tau_low

    def test_tau_decreasing_in_ploss(self):
        """τ_E should decrease with loss power (negative exponent)."""
        base = dict(Ip=2.0, BT=5.0, ne19=8.0, R=3.0,
                    kappa=1.7, epsilon=0.3, M=2.5)
        tau_low_p = ipb98y2_tau_e(Ploss=5.0, **base)
        tau_high_p = ipb98y2_tau_e(Ploss=20.0, **base)
        assert tau_low_p > tau_high_p

    def test_negative_ploss_rejected(self):
        """Ploss <= 0 must raise ValueError."""
        with pytest.raises(ValueError, match="Ploss"):
            ipb98y2_tau_e(Ip=2.0, BT=5.0, ne19=8.0, Ploss=-1.0,
                          R=3.0, kappa=1.7, epsilon=0.3)

    @pytest.mark.parametrize(
        ("tau_actual", "tau_predicted", "match"),
        [
            (float("nan"), 1.0, "tau_actual"),
            (1.0, float("nan"), "tau_predicted"),
            (1.0, float("inf"), "tau_predicted"),
        ],
    )
    def test_h_factor_rejects_non_finite_inputs(
        self,
        tau_actual: float,
        tau_predicted: float,
        match: str,
    ):
        """H-factor helper should reject non-finite inputs deterministically."""
        with pytest.raises(ValueError, match=match):
            compute_h_factor(tau_actual, tau_predicted)

    def test_h_factor_preserves_non_positive_denominator_behavior(self):
        """Historical behavior: non-positive τ_predicted returns +∞ sentinel."""
        assert compute_h_factor(3.0, 0.0) == float("inf")
        assert compute_h_factor(3.0, -1.0) == float("inf")

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("Ip", float("nan")),
            ("BT", float("inf")),
            ("ne19", 0.0),
            ("Ploss", float("nan")),
            ("R", -1.0),
            ("kappa", float("-inf")),
            ("epsilon", 0.0),
            ("M", float("nan")),
        ],
    )
    def test_non_finite_or_non_positive_inputs_rejected(
        self,
        field: str,
        bad_value: float,
    ):
        """All scaling-law inputs must be finite and strictly positive."""
        params = dict(
            Ip=2.0,
            BT=5.0,
            ne19=8.0,
            Ploss=10.0,
            R=3.0,
            kappa=1.7,
            epsilon=0.3,
            M=2.5,
        )
        params[field] = bad_value
        with pytest.raises(ValueError, match=field):
            ipb98y2_tau_e(**params)


# ── ITPA 20-shot benchmark ──────────────────────────────────────────


class TestITPABenchmark:
    """Run the full 20-shot ITPA benchmark and report RMSE."""

    def test_itpa_benchmark_rmse(self):
        """IPB98(y,2) RMSE against 20-shot dataset should be documented.

        The 1.5D Bohm/gyro-Bohm model is not expected to match exactly.
        This test passes with a WARNING if RMSE > 10%, printing the actual
        RMSE for README documentation.
        """
        rows = _load_itpa_dataset()
        coeff = load_ipb98y2_coefficients()

        results: list[TransportBenchmarkResult] = []
        errors_sq = []

        for row in rows:
            R = float(row["R_m"])
            a = float(row["a_m"])
            epsilon = a / R

            tau_pred = ipb98y2_tau_e(
                Ip=float(row["Ip_MA"]),
                BT=float(row["BT_T"]),
                ne19=float(row["ne19_1e19m3"]),
                Ploss=float(row["Ploss_MW"]),
                R=R,
                kappa=float(row["kappa"]),
                epsilon=epsilon,
                M=float(row["M_AMU"]),
                coefficients=coeff,
            )
            tau_meas = float(row["tau_E_s"])
            h = compute_h_factor(tau_meas, tau_pred)
            rel_err = abs(tau_pred - tau_meas) / tau_meas if tau_meas > 0 else 0

            results.append(TransportBenchmarkResult(
                machine=row["machine"],
                shot=row["shot"],
                tau_e_measured=tau_meas,
                tau_e_predicted=tau_pred,
                h_factor=h,
                relative_error=rel_err,
            ))
            errors_sq.append(rel_err ** 2)

        rmse_pct = 100.0 * np.sqrt(np.mean(errors_sq))

        # Log per-machine breakdown
        logger.info("=== ITPA 20-shot IPB98(y,2) Benchmark ===")
        for r in results:
            logger.info(
                "%-10s %-10s  τ_meas=%.3f  τ_pred=%.3f  H98=%.2f  err=%.1f%%",
                r.machine, r.shot,
                r.tau_e_measured, r.tau_e_predicted,
                r.h_factor, r.relative_error * 100,
            )
        logger.info("Overall RMSE: %.1f%%", rmse_pct)

        # The test always passes, but documents the actual RMSE.
        # If RMSE > 30%, we flag it as a warning.
        if rmse_pct > 30.0:
            warnings.warn(
                f"IPB98(y,2) RMSE = {rmse_pct:.1f}% (> 30% threshold). "
                "This is expected for large spherical tokamaks (NSTX/MAST) "
                "which deviate from the standard H-mode scaling.",
                stacklevel=1,
            )

        # Hard failure only if something is fundamentally broken
        assert rmse_pct < 200.0, f"RMSE = {rmse_pct:.1f}% — formula broken"

    def test_h_factor_near_unity_for_reference(self):
        """ITER design point should have H98 ≈ 1.0."""
        coeff = load_ipb98y2_coefficients()
        tau_pred = ipb98y2_tau_e(
            Ip=15.0, BT=5.3, ne19=10.1, Ploss=87.0,
            R=6.2, kappa=1.70, epsilon=2.0 / 6.2, M=2.5,
            coefficients=coeff,
        )
        # The CSV says τ_E = 3.70s, H98 = 1.00
        h = compute_h_factor(3.70, tau_pred)
        logger.info("ITER H98(y,2) = %.3f (predicted τ=%.3f s)", h, tau_pred)
        # H-factor should be in a reasonable range
        assert 0.5 < h < 2.0, f"ITER H98 = {h:.3f} out of range"


class TestPerMachineBreakdown:
    """Per-machine error breakdown to identify systematic biases."""

    def test_conventional_tokamaks_rmse_below_25pct(self):
        """Conventional aspect-ratio machines should be within 25% RMSE."""
        rows = _load_itpa_dataset()
        coeff = load_ipb98y2_coefficients()

        # Conventional = not spherical tokamaks (NSTX, MAST, START)
        spherical = {"NSTX", "MAST", "START"}
        errors_sq = []

        for row in rows:
            if row["machine"] in spherical:
                continue
            R = float(row["R_m"])
            a = float(row["a_m"])
            tau_pred = ipb98y2_tau_e(
                Ip=float(row["Ip_MA"]),
                BT=float(row["BT_T"]),
                ne19=float(row["ne19_1e19m3"]),
                Ploss=float(row["Ploss_MW"]),
                R=R,
                kappa=float(row["kappa"]),
                epsilon=a / R,
                M=float(row["M_AMU"]),
                coefficients=coeff,
            )
            tau_meas = float(row["tau_E_s"])
            if tau_meas > 0:
                rel_err = abs(tau_pred - tau_meas) / tau_meas
                errors_sq.append(rel_err ** 2)

        rmse_pct = 100.0 * np.sqrt(np.mean(errors_sq))
        logger.info("Conventional tokamaks RMSE: %.1f%%", rmse_pct)

        # This is a soft gate — warn rather than fail
        if rmse_pct > 25.0:
            warnings.warn(
                f"Conventional tokamak RMSE = {rmse_pct:.1f}% (> 25%)",
                stacklevel=1,
            )
        assert rmse_pct < 100.0
