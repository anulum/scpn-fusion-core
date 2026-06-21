# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Global Design Scanner Tests
"""Tests for the global reactor design-space explorer, scan, and Pareto analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest

from scpn_fusion.core.global_design_scanner import GlobalDesignExplorer


def _explorer(tmp_path: Path) -> GlobalDesignExplorer:
    """Build an explorer against a dummy config file."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"dummy": true}')
    return GlobalDesignExplorer(str(cfg))


class TestValidators:
    """Static finite/bounds validators on the explorer."""

    def test_require_finite_positive_ok(self) -> None:
        """A positive finite value passes the validator unchanged."""
        assert GlobalDesignExplorer._require_finite_positive("x", 3.0) == 3.0

    def test_require_finite_positive_rejects_zero(self) -> None:
        """A zero value is rejected by the positive validator."""
        with pytest.raises(ValueError):
            GlobalDesignExplorer._require_finite_positive("x", 0.0)

    def test_require_finite_positive_rejects_nan(self) -> None:
        """A non-finite value is rejected by the positive validator."""
        with pytest.raises(ValueError):
            GlobalDesignExplorer._require_finite_positive("x", float("nan"))

    def test_validate_bounds_ok(self) -> None:
        """A correctly-ordered bounds tuple is accepted."""
        lo, hi = GlobalDesignExplorer._validate_bounds("x", (1.0, 5.0))
        assert lo == 1.0
        assert hi == 5.0

    def test_validate_bounds_rejects_inverted(self) -> None:
        """An inverted bounds tuple is rejected."""
        with pytest.raises(ValueError):
            GlobalDesignExplorer._validate_bounds("x", (5.0, 1.0))

    def test_validate_bounds_rejects_equal(self) -> None:
        """A degenerate (equal-endpoint) bounds tuple is rejected."""
        with pytest.raises(ValueError):
            GlobalDesignExplorer._validate_bounds("x", (1.0, 1.0))

    def test_validate_bounds_rejects_wrong_length(self) -> None:
        """A bounds tuple of the wrong length is rejected."""
        with pytest.raises(ValueError):
            GlobalDesignExplorer._validate_bounds("x", (1.0,))  # type: ignore[arg-type]


class TestEvaluateDesign:
    """Single-point design evaluation."""

    def test_evaluate_returns_dict(self, tmp_path: Path) -> None:
        """Evaluating a valid design returns the full metric dictionary."""
        explorer = _explorer(tmp_path)
        result = explorer.evaluate_design(R_maj=3.0, B_field=10.0, I_plasma=5.0)
        assert isinstance(result, dict)
        assert "R" in result
        assert "B" in result
        assert "Cost" in result
        assert result["Model_Regime"] == "physics_scaling_surrogate"

    def test_evaluate_rejects_nonpositive(self, tmp_path: Path) -> None:
        """A non-positive design parameter is rejected."""
        explorer = _explorer(tmp_path)
        with pytest.raises(ValueError):
            explorer.evaluate_design(R_maj=0.0, B_field=10.0, I_plasma=5.0)


class TestConstructorContracts:
    """Explorer constructor input contracts."""

    def test_constructor_rejects_invalid_zeff_cap(self, tmp_path: Path) -> None:
        """An out-of-range effective-charge cap is rejected."""
        cfg = tmp_path / "cfg.json"
        cfg.write_text('{"dummy": true}')
        with pytest.raises(ValueError, match="zeff_cap"):
            GlobalDesignExplorer(str(cfg), zeff_cap=1.2)


class TestRunScan:
    """Design-space sampling and validation."""

    def test_run_scan_returns_valid_designs(self, tmp_path: Path) -> None:
        """A seeded scan returns a dataframe of physically-valid designs."""
        explorer = _explorer(tmp_path)
        df = explorer.run_scan(n_samples=200, seed=7)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert {"R", "B", "Q", "Wall_Load", "Cost"}.issubset(df.columns)

    def test_run_scan_returns_empty_frame_when_all_discarded(self, tmp_path: Path) -> None:
        """An unreachable safety floor discards every sample into an empty frame."""
        explorer = _explorer(tmp_path)
        df = explorer.run_scan(n_samples=20, seed=1, q95_min=1000.0)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "Cost" in df.columns

    def test_run_scan_rejects_bad_sample_count(self, tmp_path: Path) -> None:
        """A non-positive or boolean sample count is rejected."""
        explorer = _explorer(tmp_path)
        with pytest.raises(ValueError, match="n_samples"):
            explorer.run_scan(n_samples=0)
        with pytest.raises(ValueError, match="n_samples"):
            explorer.run_scan(n_samples=True)

    def test_run_scan_rejects_low_safety_factor(self, tmp_path: Path) -> None:
        """A safety-factor floor below the MHD margin is rejected."""
        explorer = _explorer(tmp_path)
        with pytest.raises(ValueError, match="q95_min"):
            explorer.run_scan(n_samples=10, q95_min=0.5)

    def test_run_compact_scan_runs(self, tmp_path: Path) -> None:
        """The compact-reactor envelope scan returns a dataframe."""
        explorer = _explorer(tmp_path)
        df = explorer.run_compact_scan(n_samples=200, seed=3)
        assert isinstance(df, pd.DataFrame)


class TestAnalyzePareto:
    """Pareto-region analysis and rendering."""

    def test_analyze_pareto_reports_no_viable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An all-non-viable frame reports no viable reactors and returns early."""
        explorer = _explorer(tmp_path)
        df = pd.DataFrame({"Q": [1.0], "Wall_Load": [9.0], "Constraint_OK": [False]})
        explorer.analyze_pareto(df)
        assert "No viable reactors" in capsys.readouterr().out

    def test_analyze_pareto_renders_optimal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A frame with a viable design prints the optimum and renders safely."""
        import matplotlib.pyplot as plt

        saved: list[str] = []
        monkeypatch.setattr(plt, "savefig", lambda path, *a, **k: saved.append(str(path)))
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)

        explorer = _explorer(tmp_path)
        df = pd.DataFrame(
            {
                "R": [6.0, 3.0],
                "B": [12.0, 8.0],
                "Ip": [15.0, 10.0],
                "P_fus": [800.0, 400.0],
                "Q": [12.0, 1.0],
                "Wall_Load": [2.0, 9.0],
                "Constraint_OK": [True, False],
                "Cost": [100.0, 50.0],
                "Div_Load_Baseline": [20.0, 30.0],
                "Div_Load_Optimized": [8.0, 12.0],
                "Div_Load": [8.0, 12.0],
                "Zeff_Est": [1.6, 2.0],
                "B_peak_HTS_T": [20.0, 14.0],
            }
        )
        explorer.analyze_pareto(df)
        assert saved == ["Global_Design_Pareto.png"]
