# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Test Full-Chain Uncertainty Quantification
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Tests for the equilibrium -> transport -> fusion power full-chain
Monte Carlo uncertainty propagation added to
``scpn_fusion.core.uncertainty``.
"""

import json

import numpy as np
import pytest

from scpn_fusion.core.uncertainty import (
    EquilibriumUncertainty,
    FullChainUQResult,
    PlasmaScenario,
    TransportUncertainty,
    quantify_full_chain,
    quantify_uncertainty,
    summarize_uq,
)


# ── Reference scenarios ──────────────────────────────────────────────

ITER_SCENARIO = PlasmaScenario(
    I_p=15.0,
    B_t=5.3,
    P_heat=50.0,
    n_e=10.1,
    R=6.2,
    A=3.1,
    kappa=1.7,
    M=2.5,
)

SPARC_SCENARIO = PlasmaScenario(
    I_p=8.7,
    B_t=12.2,
    P_heat=25.0,
    n_e=3.1,
    R=1.85,
    A=3.6,
    kappa=1.97,
    M=2.5,
)


# ── Tests ─────────────────────────────────────────────────────────────


class TestFullChainReturnShape:

    def test_full_chain_returns_expected_keys(self):
        """All band fields must be present on FullChainUQResult."""
        r = quantify_full_chain(ITER_SCENARIO, n_samples=200, seed=42)
        assert isinstance(r, FullChainUQResult)

        # Scalar central / sigma fields
        for attr in ("tau_E", "P_fusion", "Q",
                     "tau_E_sigma", "P_fusion_sigma", "Q_sigma",
                     "n_samples"):
            assert hasattr(r, attr), f"Missing attribute: {attr}"

        # Band arrays [5%, 50%, 95%] — length 3
        for attr in ("psi_nrmse_bands", "tau_E_bands",
                     "P_fusion_bands", "Q_bands", "beta_N_bands"):
            arr = getattr(r, attr)
            assert isinstance(arr, np.ndarray), f"{attr} is not ndarray"
            assert len(arr) == 3, f"{attr} length {len(arr)}, expected 3"

        # Legacy percentile arrays [5, 25, 50, 75, 95] — length 5
        for attr in ("tau_E_percentiles", "P_fusion_percentiles",
                     "Q_percentiles"):
            arr = getattr(r, attr)
            assert isinstance(arr, np.ndarray)
            assert len(arr) == 5


class TestITERScenario:

    def test_full_chain_iter_scenario(self):
        """ITER scenario: Q median 5-40, tau_E 2-8 s.

        The simplified power model (quadratic in tau_E, calibrated to
        ITER reference point) combined with the log-normal chi
        perturbation produces a median Q around 19 and tau_E around
        5 s.  The wide Q range accounts for the log-normal transport
        uncertainty broadening the upper tail.
        """
        r = quantify_full_chain(ITER_SCENARIO, n_samples=3000, seed=7)
        assert 5.0 <= r.Q <= 40.0, f"ITER Q median = {r.Q:.2f}"
        assert 2.0 <= r.tau_E <= 8.0, f"ITER tau_E median = {r.tau_E:.2f} s"


class TestSPARCScenario:

    def test_full_chain_sparc_scenario(self):
        """SPARC scenario: positive Q and tau_E, with tau_E < ITER.

        SPARC's compact, high-field design (R=1.85 m, B=12.2 T) yields
        a short confinement time in the IPB98 scaling.  The simplified
        fusion power model (P ~ n^2 tau^2 R^3) under-predicts absolute
        Q for compact machines because it lacks explicit temperature
        dependence.  We check structural physics: Q > 0 and tau_E
        shorter than ITER.
        """
        r_sparc = quantify_full_chain(SPARC_SCENARIO, n_samples=3000, seed=7)
        r_iter = quantify_full_chain(ITER_SCENARIO, n_samples=3000, seed=7)
        assert r_sparc.Q > 0.0, f"SPARC Q median = {r_sparc.Q:.6f}"
        assert r_sparc.tau_E > 0.0, f"SPARC tau_E = {r_sparc.tau_E:.4f} s"
        assert r_sparc.tau_E < r_iter.tau_E, (
            f"SPARC tau_E ({r_sparc.tau_E:.3f}) should be shorter than "
            f"ITER tau_E ({r_iter.tau_E:.3f})"
        )


class TestBandOrdering:

    def test_full_chain_bands_ordered(self):
        """5th percentile < 50th < 95th for all band quantities."""
        r = quantify_full_chain(ITER_SCENARIO, n_samples=3000, seed=99)
        for name in ("psi_nrmse_bands", "tau_E_bands",
                     "P_fusion_bands", "Q_bands", "beta_N_bands"):
            bands = getattr(r, name)
            assert bands[0] <= bands[1] + 1e-12, (
                f"{name}: 5th ({bands[0]}) > 50th ({bands[1]})"
            )
            assert bands[1] <= bands[2] + 1e-12, (
                f"{name}: 50th ({bands[1]}) > 95th ({bands[2]})"
            )


class TestReproducibility:

    def test_full_chain_reproducible(self):
        """Same seed must produce identical results."""
        r1 = quantify_full_chain(ITER_SCENARIO, n_samples=500, seed=42)
        r2 = quantify_full_chain(ITER_SCENARIO, n_samples=500, seed=42)
        assert r1.tau_E == r2.tau_E
        assert r1.Q == r2.Q
        assert r1.P_fusion == r2.P_fusion
        np.testing.assert_array_equal(r1.tau_E_bands, r2.tau_E_bands)
        np.testing.assert_array_equal(r1.beta_N_bands, r2.beta_N_bands)


class TestSampleConsistency:

    def test_full_chain_more_samples_consistent(self):
        """Sigma from 500 and 5000 samples should be within 50% of each other."""
        r_small = quantify_full_chain(ITER_SCENARIO, n_samples=500, seed=1)
        r_large = quantify_full_chain(ITER_SCENARIO, n_samples=5000, seed=1)
        ratio = r_small.tau_E_sigma / r_large.tau_E_sigma
        assert 0.5 < ratio < 2.0, f"tau_E sigma ratio = {ratio:.3f}"


class TestInputValidation:

    @pytest.mark.parametrize("n_samples", [0, -5, 2.5, "100", True])
    def test_invalid_n_samples_rejected(self, n_samples):
        with pytest.raises(ValueError, match="n_samples"):
            quantify_full_chain(ITER_SCENARIO, n_samples=n_samples, seed=1)

    @pytest.mark.parametrize("seed", [-1, 3.5, True, "11"])
    def test_invalid_seed_rejected(self, seed):
        with pytest.raises(ValueError, match="seed"):
            quantify_full_chain(ITER_SCENARIO, n_samples=64, seed=seed)

    @pytest.mark.parametrize(
        ("kwargs", "field"),
        [
            ({"chi_gB_sigma": -0.01}, "chi_gB_sigma"),
            ({"pedestal_sigma": -0.01}, "pedestal_sigma"),
            ({"boundary_sigma": -0.01}, "boundary_sigma"),
            ({"chi_gB_sigma": np.nan}, "chi_gB_sigma"),
            ({"pedestal_sigma": np.inf}, "pedestal_sigma"),
            ({"boundary_sigma": -np.inf}, "boundary_sigma"),
        ],
    )
    def test_invalid_sigma_inputs_rejected(self, kwargs, field):
        with pytest.raises(ValueError, match=field):
            quantify_full_chain(ITER_SCENARIO, n_samples=64, seed=1, **kwargs)

    def test_invalid_scenario_rejected(self):
        bad = PlasmaScenario(
            I_p=15.0, B_t=5.3, P_heat=0.0, n_e=10.1,
            R=6.2, A=3.1, kappa=1.7, M=2.5,
        )
        with pytest.raises(ValueError, match="scenario\\.P_heat"):
            quantify_full_chain(bad, n_samples=64, seed=1)


class TestChiUncertaintyEffect:

    def test_chi_uncertainty_widens_tau_e_band(self):
        """Larger chi_gB_sigma should produce a wider tau_E band."""
        r_narrow = quantify_full_chain(
            ITER_SCENARIO, n_samples=3000, seed=10, chi_gB_sigma=0.1,
        )
        r_wide = quantify_full_chain(
            ITER_SCENARIO, n_samples=3000, seed=10, chi_gB_sigma=0.6,
        )
        width_narrow = r_narrow.tau_E_bands[2] - r_narrow.tau_E_bands[0]
        width_wide = r_wide.tau_E_bands[2] - r_wide.tau_E_bands[0]
        assert width_wide > width_narrow, (
            f"Expected wider band with chi_sigma=0.6 ({width_wide:.3f}) "
            f"vs 0.1 ({width_narrow:.3f})"
        )


class TestBoundaryUncertaintyEffect:

    def test_boundary_uncertainty_affects_psi(self):
        """Larger boundary_sigma should widen the psi_nrmse band."""
        r_tight = quantify_full_chain(
            ITER_SCENARIO, n_samples=3000, seed=20, boundary_sigma=0.005,
        )
        r_loose = quantify_full_chain(
            ITER_SCENARIO, n_samples=3000, seed=20, boundary_sigma=0.08,
        )
        width_tight = r_tight.psi_nrmse_bands[2] - r_tight.psi_nrmse_bands[0]
        width_loose = r_loose.psi_nrmse_bands[2] - r_loose.psi_nrmse_bands[0]
        assert width_loose > width_tight, (
            f"psi_nrmse band width: boundary_sigma=0.08 ({width_loose:.5f}) "
            f"should be wider than 0.005 ({width_tight:.5f})"
        )


class TestSummarizeUQ:

    def test_summarize_uq_json_serializable(self):
        """summarize_uq() output must round-trip through json.dumps."""
        r = quantify_full_chain(ITER_SCENARIO, n_samples=200, seed=1)
        d = summarize_uq(r)
        # Must not raise
        text = json.dumps(d)
        assert isinstance(text, str)
        # Round-trip
        reloaded = json.loads(text)
        assert reloaded["n_samples"] == 200
        assert "central" in reloaded
        assert "sigma" in reloaded
        assert "bands_5_50_95" in reloaded
        for key in ("psi_nrmse", "tau_E_s", "P_fusion_MW", "Q", "beta_N"):
            assert key in reloaded["bands_5_50_95"], f"Missing band key: {key}"


class TestBackwardCompatibility:

    def test_basic_uq_still_works(self):
        """Existing quantify_uncertainty() must remain unchanged."""
        r = quantify_uncertainty(ITER_SCENARIO, n_samples=500, seed=42)
        # Must return UQResult (not FullChainUQResult)
        assert type(r).__name__ == "UQResult"
        assert r.tau_E > 0
        assert r.Q > 0
        assert r.tau_E_sigma > 0
        assert len(r.tau_E_percentiles) == 5
        assert r.n_samples == 500


class TestDataclassDefaults:

    def test_equilibrium_uncertainty_defaults(self):
        """EquilibriumUncertainty should construct with sensible defaults."""
        eq = EquilibriumUncertainty()
        assert eq.psi_nrmse_mean == 0.0
        assert eq.psi_nrmse_sigma == 0.01
        assert eq.R_axis_sigma == 0.02
        assert eq.Z_axis_sigma == 0.01

    def test_transport_uncertainty_defaults(self):
        """TransportUncertainty should construct with sensible defaults."""
        tr = TransportUncertainty()
        assert tr.chi_gB_factor_sigma == 0.3
        assert tr.pedestal_height_sigma == 0.2
