# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Ideal-Ballooning Reference Runner Tests
"""Hermetic tests for the pyrokinetics ballooning reference runner.

The boundary logic is exercised with injected synthetic growth-rate functions;
the pyrokinetics-dependent paths run against a fake ``pyrokinetics`` module put
on ``sys.modules``, so no heavy eigenvalue solve (and no pyrokinetics install)
is needed.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from validation import ballooning_reference_runner as brr


# --- pure boundary logic --------------------------------------------------


def _step_gamma(alpha_c: float):
    """Growth rate: stable below alpha_c, unstable above (single boundary)."""
    return lambda alpha: alpha - alpha_c


def test_find_alpha_crit_locates_single_boundary():
    result = brr.find_alpha_crit(_step_gamma(1.3), alpha_max=8.0)
    assert result["alpha_crit"] == pytest.approx(1.3, abs=1e-2)
    assert result["second_stability_access"] is False
    assert result["gamma_max"] > 0.0
    assert result["evaluations"] > 0


def test_find_alpha_crit_reports_second_stability_when_stable_throughout():
    # Always stable (gamma < 0 for every alpha): second-stability access.
    result = brr.find_alpha_crit(lambda alpha: -1.0, alpha_max=8.0)
    assert result["alpha_crit"] is None
    assert result["second_stability_access"] is True
    assert result["gamma_max"] == pytest.approx(-1.0)


def test_find_alpha_crit_returns_first_boundary_of_an_unstable_band():
    # Unstable only in a band [2, 4]; the finder must return the lower edge.
    def banded(alpha: float) -> float:
        return 1.0 if 2.0 <= alpha <= 4.0 else -1.0

    result = brr.find_alpha_crit(banded, alpha_max=8.0, coarse_points=33)
    assert result["alpha_crit"] == pytest.approx(2.0, abs=0.3)
    assert result["second_stability_access"] is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha_max": 0.0},
        {"coarse_points": 1},
        {"bisect_iterations": 0},
    ],
)
def test_find_alpha_crit_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        brr.find_alpha_crit(_step_gamma(1.0), **kwargs)


def test_crit_non_decreasing_skips_none_and_flags_regressions():
    up = [{"alpha_crit": 0.4}, {"alpha_crit": None}, {"alpha_crit": 0.6}, {"alpha_crit": 1.2}]
    down = [{"alpha_crit": 0.6}, {"alpha_crit": 0.4}]
    assert brr._crit_non_decreasing(up) is True
    assert brr._crit_non_decreasing(down) is False


def test_build_report_assembles_schema_and_flags_monotonicity():
    diiid = [{"shat": 1.0, "alpha_crit": 1.5, "second_stability_access": False}]
    circular = [{"shat": 0.5, "alpha_crit": 0.4}, {"shat": 1.0, "alpha_crit": 0.6}]
    report = brr.build_report(diiid, circular, {"code": "pyrokinetics"})
    assert report["schema"] == brr.ARTIFACT_SCHEMA
    assert report["diiid_alpha_crit"] == diiid
    assert report["circular_self_check"]["alpha_crit_non_decreasing_in_shear"] is True
    assert report["scope"]["kind"].startswith("infinite_n_ideal_ballooning")


def test_self_sha256_matches_file_digest():
    import hashlib

    expected = hashlib.sha256(Path(brr.__file__).read_bytes()).hexdigest()
    assert brr._self_sha256() == expected


def test_pyrokinetics_version_reads_installed_metadata(monkeypatch):
    import importlib.metadata as md

    monkeypatch.setattr(md, "version", lambda name: "9.9.9" if name == "pyrokinetics" else "x")
    assert brr._pyrokinetics_version() == "9.9.9"


# --- fake pyrokinetics for the integration paths --------------------------


class _FakeUnit:
    def __rmul__(self, value):
        return _FakeQty(float(value), self)

    __mul__ = __rmul__


class _FakeQty:
    def __init__(self, magnitude, units):
        self.magnitude = magnitude
        self.units = units


class _FakeLocalGeometry:
    def __init__(self):
        self.q = 2.0
        self.shat = 1.0
        self.kappa = 1.0
        self.delta = 0.0
        self.s_kappa = 0.0
        self.s_delta = 0.0
        self.shift = 0.0
        self.rho = _FakeQty(0.5, _FakeUnit())
        self.Rmaj = _FakeQty(3.0, _FakeUnit())
        self.beta_prime = _FakeQty(-0.04, _FakeUnit())


class _FakePyro:
    def __init__(self, gk_file=None, gk_code=None):
        self.local_geometry = _FakeLocalGeometry()


class _FakeDiagnostics:
    """Synthetic solver: alpha_crit set by shear so the grid is monotonic."""

    def __init__(self, pyro):
        self.pyro = pyro

    def ideal_ballooning_solver(self, theta0: float = 0.0):
        lg = self.pyro.local_geometry
        alpha = -float(lg.beta_prime.magnitude) * float(lg.q) ** 2 * float(lg.Rmaj.magnitude)
        alpha_c = 0.4 + 0.3 * float(lg.shat)  # rises with shear
        return _FakeQty(alpha - alpha_c, _FakeUnit())


@pytest.fixture
def fake_pyrokinetics(monkeypatch, tmp_path):
    pk = types.ModuleType("pyrokinetics")
    pk.__version__ = "0.0.0-fake"
    pk.Pyro = _FakePyro
    pk.template_dir = tmp_path
    diag = types.ModuleType("pyrokinetics.diagnostics")
    diag.Diagnostics = _FakeDiagnostics
    pk.diagnostics = diag
    monkeypatch.setitem(sys.modules, "pyrokinetics", pk)
    monkeypatch.setitem(sys.modules, "pyrokinetics.diagnostics", diag)
    monkeypatch.setattr(brr, "_pyrokinetics_version", lambda: "0.0.0-fake")
    return pk


def test_pyrokinetics_gamma_factory_maps_drive_to_growth_rate(fake_pyrokinetics):
    gamma_fn = brr._pyrokinetics_gamma_factory(brr.CIRCULAR_SHAPING, shat=1.0)
    # alpha_c = 0.4 + 0.3*1.0 = 0.7; gamma(alpha) = alpha - 0.7.
    assert gamma_fn(0.3) == pytest.approx(-0.4, abs=1e-6)
    assert gamma_fn(1.0) == pytest.approx(0.3, abs=1e-6)


def test_scan_grid_produces_one_row_per_shear(fake_pyrokinetics):
    rows = brr._scan_grid(brr.CIRCULAR_SHAPING, (0.5, 1.0, 2.0), alpha_max=5.0)
    assert [r["shat"] for r in rows] == [0.5, 1.0, 2.0]
    crits = [r["alpha_crit"] for r in rows]
    assert crits == sorted(crits)  # monotonic in shear


def test_main_writes_artifact_with_provenance(fake_pyrokinetics, tmp_path):
    out = tmp_path / "alpha_crit.json"
    rc = brr.main(["--output", str(out), "--diiid-alpha-max", "8.0", "--circular-alpha-max", "5.0"])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["schema"] == brr.ARTIFACT_SCHEMA
    assert payload["provenance"]["code"] == "pyrokinetics"
    assert payload["provenance"]["licence"] == "LGPL-3.0-or-later"
    assert payload["provenance"]["pyrokinetics_version"] == "0.0.0-fake"
    assert len(payload["diiid_alpha_crit"]) == len(brr.SHEAR_GRID)
    assert payload["circular_self_check"]["alpha_crit_non_decreasing_in_shear"] is True


def test_main_fails_closed_when_circular_self_check_not_monotonic(
    fake_pyrokinetics, tmp_path, monkeypatch
):
    # Force a non-monotonic circular table: alpha_c decreasing with shear.
    def bad_solver(self, theta0: float = 0.0):
        lg = self.pyro.local_geometry
        alpha = -float(lg.beta_prime.magnitude) * float(lg.q) ** 2 * float(lg.Rmaj.magnitude)
        alpha_c = 2.0 - 0.5 * float(lg.shat)  # falls with shear
        return _FakeQty(alpha - alpha_c, _FakeUnit())

    monkeypatch.setattr(
        fake_pyrokinetics.diagnostics.Diagnostics, "ideal_ballooning_solver", bad_solver
    )
    out = tmp_path / "bad.json"
    rc = brr.main(["--output", str(out)])
    assert rc == 1
    assert not out.exists()
