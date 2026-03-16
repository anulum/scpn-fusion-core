# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Phase Dynamics Package
"""
Paper 27 Knm/UPDE engine + Kuramoto-Sakaguchi phase reduction.

Implements the generalized Kuramoto-Sakaguchi mean-field model with an
exogenous global field driver ζ sin(Ψ − θ), per the reviewer request
referencing arXiv:2004.06344 and SCPN Paper 27.
"""

from scpn_fusion.phase.adaptive_knm import (
    AdaptiveKnmConfig,
    AdaptiveKnmEngine,
    DiagnosticSnapshot,
)
from scpn_fusion.phase.knm import KnmSpec, build_knm_paper27
from scpn_fusion.phase.kuramoto import (
    GlobalPsiDriver,
    kuramoto_sakaguchi_step,
    lyapunov_exponent,
    lyapunov_v,
    order_parameter,
    wrap_phase,
)
from scpn_fusion.phase.lyapunov_guard import LyapunovGuard
from scpn_fusion.phase.plasma_knm import (
    OMEGA_PLASMA_8,
    PLASMA_LAYER_NAMES,
    build_knm_plasma,
    build_knm_plasma_from_config,
    plasma_omega,
)
from scpn_fusion.phase.realtime_monitor import RealtimeMonitor, TrajectoryRecorder
from scpn_fusion.phase.upde import UPDESystem
from scpn_fusion.phase.ws_phase_stream import PhaseStreamServer

__all__ = [
    "AdaptiveKnmConfig",
    "AdaptiveKnmEngine",
    "DiagnosticSnapshot",
    "kuramoto_sakaguchi_step",
    "order_parameter",
    "wrap_phase",
    "lyapunov_v",
    "lyapunov_exponent",
    "GlobalPsiDriver",
    "KnmSpec",
    "build_knm_paper27",
    "build_knm_plasma",
    "build_knm_plasma_from_config",
    "plasma_omega",
    "OMEGA_PLASMA_8",
    "PLASMA_LAYER_NAMES",
    "UPDESystem",
    "LyapunovGuard",
    "RealtimeMonitor",
    "TrajectoryRecorder",
    "PhaseStreamServer",
]
