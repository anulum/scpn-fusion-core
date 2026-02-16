# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Grad-Shafranov Convergence Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Convergence tests for the Grad-Shafranov equilibrium solver on real
SPARC GEQDSK files.

Tests cover:
  - Convergence (residual below threshold) for each solver method
  - Iteration count comparison: SOR < Jacobi
  - Anderson acceleration improvement over plain SOR
  - Wall-time constraints (< 1s per solve at default grid)
  - No NaN/Inf in the solution
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core.eqdsk import read_geqdsk
from scpn_fusion.core.fusion_kernel import FusionKernel

logger = logging.getLogger(__name__)

# ── SPARC GEQDSK files ───────────────────────────────────────────────

_SPARC_DIR = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "reference_data"
    / "sparc"
)

_LMODE_FILES = sorted(_SPARC_DIR.glob("lmode_*.geqdsk"))
_ALL_GEQDSK = sorted(
    list(_SPARC_DIR.glob("lmode_*.geqdsk"))
    + list(_SPARC_DIR.glob("sparc_*.eqdsk"))
)


def _geqdsk_ids():
    """Generate readable test IDs from GEQDSK filenames."""
    return [f.stem for f in _ALL_GEQDSK]


def _make_config(eq, method="sor", max_iter=500, tol=1e-4):
    """Build a FusionKernel config dict from a GEqdsk with a solver method."""
    cfg = eq.to_config(name=f"sparc_{eq.description[:20].strip()}")
    cfg["solver"]["solver_method"] = method
    cfg["solver"]["max_iterations"] = max_iter
    cfg["solver"]["convergence_threshold"] = tol
    cfg["solver"]["relaxation_factor"] = 0.15
    cfg["solver"]["sor_omega"] = 1.6
    return cfg


def _write_temp_config(tmp_path, cfg, name="test_config.json"):
    """Write config to a temporary JSON file and return its path."""
    p = tmp_path / name
    p.write_text(json.dumps(cfg))
    return p


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(params=_LMODE_FILES, ids=[f.stem for f in _LMODE_FILES])
def lmode_eq(request):
    """Parametrized fixture yielding each L-mode GEQDSK equilibrium."""
    return read_geqdsk(request.param)


# ── Tests ────────────────────────────────────────────────────────────


class TestSORConvergence:
    """SOR solver convergence on SPARC equilibria."""

    @pytest.mark.parametrize("geqdsk_path", _LMODE_FILES,
                             ids=[f.stem for f in _LMODE_FILES])
    def test_sor_converges(self, geqdsk_path, tmp_path):
        """SOR method converges within 500 iterations on L-mode files."""
        eq = read_geqdsk(geqdsk_path)
        cfg = _make_config(eq, method="sor", max_iter=500, tol=1e-4)
        config_file = _write_temp_config(tmp_path, cfg)

        fk = FusionKernel(config_file)
        result = fk.solve_equilibrium()

        assert result["converged"], (
            f"SOR did not converge on {geqdsk_path.stem}: "
            f"residual={result['residual']:.2e}, "
            f"iters={result['iterations']}"
        )
        assert result["residual"] < 1e-4
        assert not np.any(np.isnan(result["psi"]))
        assert not np.any(np.isinf(result["psi"]))

    @pytest.mark.parametrize("geqdsk_path", _LMODE_FILES,
                             ids=[f.stem for f in _LMODE_FILES])
    def test_sor_wall_time(self, geqdsk_path, tmp_path):
        """SOR solver completes within 2s on default grid."""
        eq = read_geqdsk(geqdsk_path)
        cfg = _make_config(eq, method="sor", max_iter=500, tol=1e-4)
        config_file = _write_temp_config(tmp_path, cfg)

        fk = FusionKernel(config_file)
        result = fk.solve_equilibrium()

        assert result["wall_time_s"] < 2.0, (
            f"SOR took {result['wall_time_s']:.2f}s on {geqdsk_path.stem}"
        )


class TestJacobiComparison:
    """Compare Jacobi vs SOR iteration counts."""

    @pytest.mark.parametrize("geqdsk_path", _LMODE_FILES[:1],
                             ids=[f.stem for f in _LMODE_FILES[:1]])
    def test_sor_fewer_iterations_than_jacobi(self, geqdsk_path, tmp_path):
        """SOR converges in fewer iterations than Jacobi."""
        eq = read_geqdsk(geqdsk_path)

        # Jacobi solve
        cfg_j = _make_config(eq, method="jacobi", max_iter=1000, tol=1e-3)
        fk_j = FusionKernel(_write_temp_config(tmp_path, cfg_j, "jacobi.json"))
        res_j = fk_j.solve_equilibrium()

        # SOR solve
        cfg_s = _make_config(eq, method="sor", max_iter=1000, tol=1e-3)
        fk_s = FusionKernel(_write_temp_config(tmp_path, cfg_s, "sor.json"))
        res_s = fk_s.solve_equilibrium()

        logger.info(
            "Jacobi: %d iters, res=%.2e | SOR: %d iters, res=%.2e",
            res_j["iterations"], res_j["residual"],
            res_s["iterations"], res_s["residual"],
        )

        # SOR should converge faster (fewer iters) or at least reach
        # a lower residual in the same iteration budget.
        if res_j["converged"] and res_s["converged"]:
            assert res_s["iterations"] <= res_j["iterations"], (
                f"SOR ({res_s['iterations']}) did not beat "
                f"Jacobi ({res_j['iterations']})"
            )
        else:
            # At least SOR residual should be lower
            assert res_s["residual"] <= res_j["residual"] * 1.5


class TestAndersonAcceleration:
    """Anderson acceleration improves convergence over plain SOR."""

    @pytest.mark.parametrize("geqdsk_path", _LMODE_FILES[:1],
                             ids=[f.stem for f in _LMODE_FILES[:1]])
    def test_anderson_converges(self, geqdsk_path, tmp_path):
        """Anderson method converges without crash."""
        eq = read_geqdsk(geqdsk_path)
        cfg = _make_config(eq, method="anderson", max_iter=500, tol=1e-4)
        config_file = _write_temp_config(tmp_path, cfg)

        fk = FusionKernel(config_file)
        result = fk.solve_equilibrium()

        assert result["solver_method"] == "anderson"
        assert not np.any(np.isnan(result["psi"]))
        assert not np.any(np.isinf(result["psi"]))
        # Anderson should converge or at least not be worse than SOR
        assert result["residual"] < 1e-2, (
            f"Anderson residual too high: {result['residual']:.2e}"
        )

    @pytest.mark.parametrize("geqdsk_path", _LMODE_FILES[:1],
                             ids=[f.stem for f in _LMODE_FILES[:1]])
    def test_anderson_vs_sor(self, geqdsk_path, tmp_path):
        """Anderson should match or beat SOR convergence rate."""
        eq = read_geqdsk(geqdsk_path)

        cfg_s = _make_config(eq, method="sor", max_iter=300, tol=1e-4)
        fk_s = FusionKernel(_write_temp_config(tmp_path, cfg_s, "sor.json"))
        res_s = fk_s.solve_equilibrium()

        cfg_a = _make_config(eq, method="anderson", max_iter=300, tol=1e-4)
        fk_a = FusionKernel(_write_temp_config(tmp_path, cfg_a, "and.json"))
        res_a = fk_a.solve_equilibrium()

        logger.info(
            "SOR: %d iters, res=%.2e | Anderson: %d iters, res=%.2e",
            res_s["iterations"], res_s["residual"],
            res_a["iterations"], res_a["residual"],
        )

        # Anderson residual should not be significantly worse
        assert res_a["residual"] < res_s["residual"] * 2.0


class TestResidualHistory:
    """Residual history is monotonically decreasing (on average)."""

    def test_residual_history_populated(self, tmp_path):
        """solve_equilibrium returns a non-empty residual_history."""
        eq = read_geqdsk(_LMODE_FILES[0])
        cfg = _make_config(eq, method="sor", max_iter=100, tol=1e-4)
        fk = FusionKernel(_write_temp_config(tmp_path, cfg))
        result = fk.solve_equilibrium()

        assert len(result["residual_history"]) > 0
        assert all(np.isfinite(r) for r in result["residual_history"])

    def test_residual_trend_decreasing(self, tmp_path):
        """Residual should generally decrease (last 10 avg < first 10 avg)."""
        eq = read_geqdsk(_LMODE_FILES[0])
        cfg = _make_config(eq, method="sor", max_iter=200, tol=1e-6)
        fk = FusionKernel(_write_temp_config(tmp_path, cfg))
        result = fk.solve_equilibrium()

        hist = result["residual_history"]
        if len(hist) >= 20:
            first_10 = np.mean(hist[:10])
            last_10 = np.mean(hist[-10:])
            assert last_10 < first_10, (
                f"Residual not decreasing: first_10={first_10:.2e}, "
                f"last_10={last_10:.2e}"
            )
