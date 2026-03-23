# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# SCPN Fusion Core — Coverage Batch 11 (passing only)
from __future__ import annotations

import numpy as np


class TestGKCorrector:
    def test_init(self):
        from scpn_fusion.core.gk_corrector import GKCorrector

        gc = GKCorrector(nr=50)
        assert gc.nr == 50


class TestGKScheduler:
    def test_init(self):
        from scpn_fusion.core.gk_scheduler import GKScheduler

        gs = GKScheduler()
        assert gs._step == 0


class TestOODDetector:
    def test_init(self):
        from scpn_fusion.core.gk_ood_detector import OODDetector

        det = OODDetector()
        assert det is not None


class TestOnlineLearner:
    def test_init(self):
        from scpn_fusion.core.gk_online_learner import OnlineLearner

        ol = OnlineLearner()
        assert len(ol.buffer) == 0


class TestVerificationReport:
    def test_init(self):
        from scpn_fusion.core.gk_verification_report import VerificationReport

        vr = VerificationReport()
        assert vr is not None


class TestIMASCommon:
    def test_missing_keys(self):
        from scpn_fusion.io.imas_connector_common import _missing_required_keys

        assert _missing_required_keys({"a": 1}, ("a", "b")) == ["b"]


class TestSolModel:
    def test_two_point_sol(self):
        from scpn_fusion.core.sol_model import TwoPointSOL

        sol = TwoPointSOL(R0=6.2, a=2.0, q95=3.0, B_pol=0.5, kappa=1.7)
        assert sol is not None
