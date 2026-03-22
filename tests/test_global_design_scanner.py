# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# SCPN Fusion Core — Global Design Scanner Tests
from __future__ import annotations

import pytest

from scpn_fusion.core.global_design_scanner import GlobalDesignExplorer


class TestValidators:
    def test_require_finite_positive_ok(self):
        assert GlobalDesignExplorer._require_finite_positive("x", 3.0) == 3.0

    def test_require_finite_positive_rejects_zero(self):
        with pytest.raises(ValueError):
            GlobalDesignExplorer._require_finite_positive("x", 0.0)

    def test_require_finite_positive_rejects_nan(self):
        with pytest.raises(ValueError):
            GlobalDesignExplorer._require_finite_positive("x", float("nan"))

    def test_validate_bounds_ok(self):
        lo, hi = GlobalDesignExplorer._validate_bounds("x", (1.0, 5.0))
        assert lo == 1.0
        assert hi == 5.0

    def test_validate_bounds_rejects_inverted(self):
        with pytest.raises(ValueError):
            GlobalDesignExplorer._validate_bounds("x", (5.0, 1.0))

    def test_validate_bounds_rejects_equal(self):
        with pytest.raises(ValueError):
            GlobalDesignExplorer._validate_bounds("x", (1.0, 1.0))

    def test_validate_bounds_rejects_wrong_length(self):
        with pytest.raises(ValueError):
            GlobalDesignExplorer._validate_bounds("x", (1.0,))


class TestEvaluateDesign:
    def test_evaluate_returns_dict(self, tmp_path):
        cfg = tmp_path / "cfg.json"
        cfg.write_text('{"dummy": true}')
        explorer = GlobalDesignExplorer(str(cfg))
        result = explorer.evaluate_design(R_maj=3.0, B_field=10.0, I_plasma=5.0)
        assert isinstance(result, dict)
        assert "R" in result
        assert "B" in result
        assert "Cost" in result

    def test_evaluate_rejects_nonpositive(self, tmp_path):
        cfg = tmp_path / "cfg.json"
        cfg.write_text('{"dummy": true}')
        explorer = GlobalDesignExplorer(str(cfg))
        with pytest.raises(ValueError):
            explorer.evaluate_design(R_maj=0, B_field=10, I_plasma=5)
