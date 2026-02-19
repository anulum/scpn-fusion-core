# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Neuro-Symbolic Logic Compiler
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
SCPN Neuro-Symbolic Logic Compiler
===================================

Translates Stochastic Petri Nets into sc_neurocore matrix operations.

Packet A — ``structure.StochasticPetriNet``
    Pure-Python graph builder with sparse W_in / W_out matrices.

Packet B — ``compiler.FusionCompiler`` / ``compiler.CompiledNet``
    Compiles the net into LIF neurons + packed-bitstream weight tensors
    ready for stochastic or float-path execution.
"""

from .structure import StochasticPetriNet
from .compiler import FusionCompiler, CompiledNet
from .contracts import (
    ControlObservation,
    ControlAction,
    ControlTargets,
    ControlScales,
    SafetyContract,
    DEFAULT_SAFETY_CONTRACTS,
    extract_features,
    decode_actions,
    verify_safety_contracts,
)
from .artifact import Artifact, load_artifact, save_artifact
from .controller import NeuroSymbolicController
from .safety_interlocks import (
    CONTROL_TRANSITIONS,
    SAFETY_CHANNELS,
    SafetyInterlockRuntime,
    SafetyLimits,
    build_safety_net,
    safety_tokens_from_state,
)

__all__ = [
    # Packet A
    "StochasticPetriNet",
    # Packet B
    "FusionCompiler",
    "CompiledNet",
    # Packet C — contracts
    "ControlObservation",
    "ControlAction",
    "ControlTargets",
    "ControlScales",
    "SafetyContract",
    "DEFAULT_SAFETY_CONTRACTS",
    "extract_features",
    "decode_actions",
    "verify_safety_contracts",
    # Packet C — safety interlocks
    "SAFETY_CHANNELS",
    "CONTROL_TRANSITIONS",
    "SafetyLimits",
    "SafetyInterlockRuntime",
    "build_safety_net",
    "safety_tokens_from_state",
    # Packet C — artifact
    "Artifact",
    "load_artifact",
    "save_artifact",
    # Packet C — controller
    "NeuroSymbolicController",
]
