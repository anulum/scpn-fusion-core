# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Transport physics consistency tests for the elongated
# toroidal volume element (FUS-C transport lane).
"""Volume-element parity, scaling, and config-lookup tests for the transport
runtime physics mixin.

Targets the kappa volume-scaling fix tracked as ``B.5`` in
``docs/internal/TODO.md``. The reference identity used throughout this file is
the analytic volume of a torus with an elliptical cross-section:

``V_torus = 2 * pi^2 * R_0 * a_minor^2 * kappa``.

The transport solver discretises that integral over the normalised flux
coordinate ``rho in [0, 1]`` as
``dV(rho) = 4 * pi^2 * R_0 * kappa * a^2 * rho * drho``,
so a midpoint-style Riemann sum over the cached ``dV`` array converges to
``V_torus`` as the grid refines.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from scpn_fusion.core.integrated_transport_solver_runtime_physics import (
    TransportSolverRuntimePhysicsMixin,
    _extract_elongation,
)


class _VolumeOnlyRuntime(TransportSolverRuntimePhysicsMixin):
    """Minimal mixin host that exposes the volume-element kernel for testing.

    The transport solver is large; instantiating it for a unit test on a single
    private kernel is wasteful. The mixin only reads ``self.cfg``, ``self.rho``
    and ``self.drho`` for the volume element, so a hand-rolled host is faithful.
    """

    def __init__(
        self,
        *,
        nr: int = 64,
        r_min: float = 4.2,
        r_max: float = 8.2,
        kappa: float | None = 1.7,
        kappa_section: str = "physics",
    ) -> None:
        self.nr = int(nr)
        self.rho = np.linspace(0.0, 1.0, self.nr, dtype=np.float64)
        self.drho = float(self.rho[1] - self.rho[0])
        cfg: dict[str, object] = {"dimensions": {"R_min": r_min, "R_max": r_max}}
        if kappa is not None:
            cfg[kappa_section] = {"kappa": float(kappa)}
        self.cfg = cfg


def _analytic_elongated_torus_volume(*, r_min: float, r_max: float, kappa: float) -> float:
    """Return ``V = 2 * pi^2 * R_0 * a^2 * kappa`` for the given bounds."""
    r0 = 0.5 * (r_min + r_max)
    a = 0.5 * (r_max - r_min)
    return 2.0 * np.pi**2 * r0 * a**2 * kappa


# ── 1. Analytic parity ────────────────────────────────────────────────────


def test_volume_element_matches_elongated_torus_analytic_iter_like() -> None:
    """The summed dV approximates the analytic V at ITER-like geometry.

    With a 64-cell rho grid the midpoint convergence error is O(1/nr) ~ 1.5%,
    so the tolerance is 2% and the refined-grid test below pins down better.
    """
    host = _VolumeOnlyRuntime(nr=64, r_min=4.2, r_max=8.2, kappa=1.7)
    expected = _analytic_elongated_torus_volume(r_min=4.2, r_max=8.2, kappa=1.7)
    measured = float(np.sum(host._rho_volume_element()))
    assert measured == pytest.approx(expected, rel=2.0e-2)


def test_volume_element_recovers_circular_limit_when_kappa_unity() -> None:
    """kappa = 1 must reproduce ``V = 2 pi^2 R0 a^2`` (circular torus)."""
    host = _VolumeOnlyRuntime(nr=128, r_min=4.2, r_max=8.2, kappa=1.0)
    expected = _analytic_elongated_torus_volume(r_min=4.2, r_max=8.2, kappa=1.0)
    measured = float(np.sum(host._rho_volume_element()))
    assert measured == pytest.approx(expected, rel=1.0e-2)


def test_volume_element_converges_to_analytic_under_refinement() -> None:
    """Doubling the radial resolution at least halves the relative error.

    The discretisation is a midpoint sum of a smooth linear-in-rho integrand,
    which converges as O(1/nr). The tolerance ``0.6`` leaves headroom for the
    finite ``drho`` boundary contribution at ``rho = 0``.
    """
    coarse = _VolumeOnlyRuntime(nr=32, kappa=1.7)
    fine = _VolumeOnlyRuntime(nr=256, kappa=1.7)
    expected = _analytic_elongated_torus_volume(r_min=4.2, r_max=8.2, kappa=1.7)
    err_coarse = abs(float(np.sum(coarse._rho_volume_element())) - expected) / expected
    err_fine = abs(float(np.sum(fine._rho_volume_element())) - expected) / expected
    assert err_fine < 0.6 * err_coarse
    assert err_fine < 5.0e-3


# ── 2. Scaling laws ───────────────────────────────────────────────────────


def test_volume_element_scales_linearly_with_kappa() -> None:
    """Sum(dV)(kappa) / Sum(dV)(1) = kappa over a realistic elongation sweep."""
    baseline_host = _VolumeOnlyRuntime(nr=128, kappa=1.0)
    baseline_volume = float(np.sum(baseline_host._rho_volume_element()))
    for kappa in (1.0, 1.2, 1.5, 1.7, 1.97, 2.5):
        host = _VolumeOnlyRuntime(nr=128, kappa=kappa)
        volume = float(np.sum(host._rho_volume_element()))
        assert volume / baseline_volume == pytest.approx(kappa, rel=1.0e-12)


def test_volume_element_scales_linearly_with_drho() -> None:
    """The cached array is a per-cell volume, so element-wise ``dV/drho``
    is independent of grid refinement (up to a global drho factor).
    """
    host_a = _VolumeOnlyRuntime(nr=32, kappa=1.7)
    host_b = _VolumeOnlyRuntime(nr=128, kappa=1.7)
    # ``dV / (drho * rho)`` is the elongation-prefactor 4 pi^2 R0 kappa a^2,
    # which depends only on geometry. Compare at the matching rho midpoint.
    pref_a = host_a._rho_volume_element()[16] / (host_a.drho * host_a.rho[16])
    pref_b = host_b._rho_volume_element()[64] / (host_b.drho * host_b.rho[64])
    assert pref_a == pytest.approx(pref_b, rel=1.0e-12)


# ── 3. Config-key fallback chain ──────────────────────────────────────────


@pytest.mark.parametrize("section", ["physics", "shape", "target", "geometry"])
def test_elongation_lookup_supports_all_four_canonical_sections(section: str) -> None:
    """Every documented kappa location must be consulted in priority order."""
    host = _VolumeOnlyRuntime(kappa=1.7, kappa_section=section)
    assert _extract_elongation(host.cfg) == 1.7


def test_elongation_priority_prefers_physics_over_other_sections() -> None:
    """If multiple sections specify kappa, ``physics`` wins."""
    host = _VolumeOnlyRuntime(kappa=None)
    host.cfg["physics"] = {"kappa": 1.7}
    host.cfg["shape"] = {"kappa": 1.0}
    host.cfg["target"] = {"kappa": 2.5}
    assert _extract_elongation(host.cfg) == 1.7


def test_elongation_falls_back_to_unity_when_missing_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing kappa is recoverable, but the user must see a warning."""
    host = _VolumeOnlyRuntime(kappa=None)
    with caplog.at_level(
        logging.WARNING, logger="scpn_fusion.core.integrated_transport_solver_runtime_physics"
    ):
        kappa = _extract_elongation(host.cfg)
    assert kappa == 1.0
    assert any("kappa" in rec.message for rec in caplog.records)


@pytest.mark.parametrize("bad_value", [None, "not-a-number", float("nan"), 0.0, -1.0])
def test_elongation_rejects_invalid_values_and_falls_back(bad_value: object) -> None:
    """Non-finite, non-positive, and non-numeric values must be ignored."""
    cfg: dict[str, object] = {
        "dimensions": {"R_min": 4.2, "R_max": 8.2},
        "physics": {"kappa": bad_value},
    }
    assert _extract_elongation(cfg) == 1.0


# ── 4. Cache discipline ───────────────────────────────────────────────────


def test_volume_element_is_cached_after_first_call() -> None:
    """Subsequent calls must return the cached array (identity, not a copy)."""
    host = _VolumeOnlyRuntime()
    first = host._rho_volume_element()
    second = host._rho_volume_element()
    assert first is second


def test_volume_element_cache_invalidation_via_attribute_reset() -> None:
    """Clearing ``_dV_cache`` allows the kernel to recompute with a new kappa."""
    host = _VolumeOnlyRuntime(kappa=1.0)
    v_circular = float(np.sum(host._rho_volume_element()))
    host.cfg["physics"] = {"kappa": 1.7}
    host._dV_cache = None
    v_elongated = float(np.sum(host._rho_volume_element()))
    assert v_elongated / v_circular == pytest.approx(1.7, rel=1.0e-12)


# ── 5. Physics impact on derived quantities ───────────────────────────────


def test_kappa_fix_propagates_to_stored_thermal_energy_estimate() -> None:
    """Stored thermal energy W = (3/2) integral(n_e (T_i + T_e) dV) scales with kappa.

    The transport solver's confinement-time diagnostic depends on this quantity;
    under the prior circular-cross-section bug, ITER-class plasmas (kappa~1.7)
    were under-counted by ~70% in stored energy and therefore in tau_E. This
    test pins that the kappa factor flows through the integral verbatim.
    """
    e_keV_J = 1.602176634e-16

    def stored_energy(host: _VolumeOnlyRuntime) -> float:
        ne_m3 = 5.0e19 * np.ones(host.nr, dtype=np.float64)
        t_keV = 10.0 * np.ones(host.nr, dtype=np.float64)
        dV = host._rho_volume_element()
        return float(1.5 * np.sum(ne_m3 * (t_keV + t_keV) * e_keV_J * dV))

    w_circular = stored_energy(_VolumeOnlyRuntime(kappa=1.0))
    w_iter = stored_energy(_VolumeOnlyRuntime(kappa=1.7))
    assert w_iter / w_circular == pytest.approx(1.7, rel=1.0e-12)


# ── 6. Error paths ────────────────────────────────────────────────────────


def test_volume_element_raises_on_invalid_major_radius() -> None:
    """Degenerate ``R_min == R_max`` must be rejected as a config error."""
    host = _VolumeOnlyRuntime(r_min=4.2, r_max=4.2, kappa=1.7)
    with pytest.raises(Exception, match="minor radius"):
        host._rho_volume_element()


def test_volume_element_raises_on_negative_radius_extent() -> None:
    """Inverted ``R_max < R_min`` must be rejected before producing negative dV."""
    host = _VolumeOnlyRuntime(r_min=8.2, r_max=4.2, kappa=1.7)
    with pytest.raises(Exception, match="minor radius"):
        host._rho_volume_element()
