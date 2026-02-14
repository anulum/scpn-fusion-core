# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Vertical Position Control Petri Net
# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Vertical position control Petri net for tokamak plasma stabilization.

Topology: 8 places, 7 transitions
- Encodes error sign routing (positive/negative displacement)
- Encodes error magnitude classification (large/small correction)
- Stochastic arc weights map error magnitude to firing probability

Formal properties (to be verified):
1. BOUNDED: Every place has at most 1 token at any time
2. LIVE: From any reachable marking, T_read can eventually fire
3. MUTEX: P_error_pos and P_error_neg are never simultaneously marked
4. RETURN: Every token eventually returns to P_idle
5. DETERMINISTIC ROUTING: Exactly one of T_pos/T_neg fires per cycle

Places
------
P_idle          controller waiting for measurement (initial token = 1.0)
P_measured      measurement received
P_error_pos     Z > 0 (plasma displaced upward)
P_error_neg     Z < 0 (plasma displaced downward)
P_error_large   |Z| > 2 mm (needs strong correction)
P_error_small   |Z| <= 2 mm (fine correction sufficient)
P_actuating     control signal being computed
P_applied       control signal sent to coil

Transitions
-----------
T_read      P_idle -> P_measured
T_pos       P_measured -> P_error_pos  (fires when Z > 0)
T_neg       P_measured -> P_error_neg  (fires when Z <= 0)
T_large     P_error_pos/P_error_neg -> P_error_large
T_small     P_error_pos/P_error_neg -> P_error_small
T_compute   P_error_large/P_error_small -> P_actuating
T_apply     P_actuating -> P_applied -> P_idle
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .structure import StochasticPetriNet
from .compiler import FusionCompiler, CompiledNet


# ── Place & Transition Name Constants ────────────────────────────────────────

PLACES = [
    "P_idle",
    "P_measured",
    "P_error_pos",
    "P_error_neg",
    "P_error_large",
    "P_error_small",
    "P_actuating",
    "P_applied",
]

TRANSITIONS = [
    "T_read",
    "T_pos",
    "T_neg",
    "T_large",
    "T_small",
    "T_compute",
    "T_apply",
]


# ── Default control parameters ──────────────────────────────────────────────

# Error threshold in metres (2 mm) separating large vs small corrections.
_DEFAULT_ERROR_THRESHOLD_M = 0.002

# Scaling factor for large-error arc weight (aggressive correction).
_DEFAULT_ERROR_SCALE_LARGE = 0.9

# Scaling factor for small-error arc weight (fine correction).
_DEFAULT_ERROR_SCALE_SMALL = 0.4

# Output gain scaling for converting marking difference to control signal.
_DEFAULT_GAIN_SCALE = 1.0


# ── VerticalControlNet ──────────────────────────────────────────────────────


class VerticalControlNet:
    """Vertical position control Petri net for tokamak plasma Z-axis.

    Builds a ``StochasticPetriNet`` with 8 places and 7 transitions that
    route a vertical displacement measurement through error-sign routing
    and error-magnitude classification, producing a corrective control
    signal for the poloidal field coil.

    Parameters
    ----------
    error_scale_large : float
        Arc weight for the large-error path.  Higher values produce
        stronger actuation when |Z| > 2 mm.  Default 0.9.
    error_scale_small : float
        Arc weight for the small-error path.  Default 0.4.
    gain_scale : float
        Multiplier applied to the decoded control signal.  Default 1.0.
    error_threshold_m : float
        Boundary (metres) between large and small error classification.
        Default 0.002 (2 mm).
    transition_threshold : float
        Firing threshold for all transitions.  Default 0.3.
    """

    def __init__(
        self,
        *,
        error_scale_large: float = _DEFAULT_ERROR_SCALE_LARGE,
        error_scale_small: float = _DEFAULT_ERROR_SCALE_SMALL,
        gain_scale: float = _DEFAULT_GAIN_SCALE,
        error_threshold_m: float = _DEFAULT_ERROR_THRESHOLD_M,
        transition_threshold: float = 0.3,
    ) -> None:
        self.error_scale_large = float(error_scale_large)
        self.error_scale_small = float(error_scale_small)
        self.gain_scale = float(gain_scale)
        self.error_threshold_m = float(error_threshold_m)
        self.transition_threshold = float(transition_threshold)

        self._net: StochasticPetriNet | None = None
        self._compiled_net: CompiledNet | None = None

    # ── Net Construction ────────────────────────────────────────────────

    def create_net(self) -> StochasticPetriNet:
        """Build and return the 8-place, 7-transition control net.

        The net is not compiled yet; call :meth:`compile` to produce
        matrices.

        Returns
        -------
        StochasticPetriNet
            Configured net with all places, transitions, and arcs.
        """
        net = StochasticPetriNet()

        # -- Places -------------------------------------------------------
        net.add_place("P_idle", initial_tokens=1.0)
        net.add_place("P_measured", initial_tokens=0.0)
        net.add_place("P_error_pos", initial_tokens=0.0)
        net.add_place("P_error_neg", initial_tokens=0.0)
        net.add_place("P_error_large", initial_tokens=0.0)
        net.add_place("P_error_small", initial_tokens=0.0)
        net.add_place("P_actuating", initial_tokens=0.0)
        net.add_place("P_applied", initial_tokens=0.0)

        th = self.transition_threshold

        # -- Transitions --------------------------------------------------
        net.add_transition("T_read", threshold=th)
        net.add_transition("T_pos", threshold=th)
        net.add_transition("T_neg", threshold=th)
        net.add_transition("T_large", threshold=th)
        net.add_transition("T_small", threshold=th)
        net.add_transition("T_compute", threshold=th)
        net.add_transition("T_apply", threshold=th)

        # -- Arcs ---------------------------------------------------------
        # T_read: P_idle -> P_measured
        net.add_arc("P_idle", "T_read", weight=1.0)
        net.add_arc("T_read", "P_measured", weight=1.0)

        # T_pos: P_measured -> P_error_pos  (fires when Z > 0)
        net.add_arc("P_measured", "T_pos", weight=1.0)
        net.add_arc("T_pos", "P_error_pos", weight=1.0)

        # T_neg: P_measured -> P_error_neg  (fires when Z <= 0)
        net.add_arc("P_measured", "T_neg", weight=1.0)
        net.add_arc("T_neg", "P_error_neg", weight=1.0)

        # T_large: P_error_pos or P_error_neg -> P_error_large
        #   Both error-sign places feed into T_large; whichever has tokens
        #   activates T_large with the large-error arc weight.
        net.add_arc("P_error_pos", "T_large", weight=self.error_scale_large)
        net.add_arc("P_error_neg", "T_large", weight=self.error_scale_large)
        net.add_arc("T_large", "P_error_large", weight=1.0)

        # T_small: P_error_pos or P_error_neg -> P_error_small
        net.add_arc("P_error_pos", "T_small", weight=self.error_scale_small)
        net.add_arc("P_error_neg", "T_small", weight=self.error_scale_small)
        net.add_arc("T_small", "P_error_small", weight=1.0)

        # T_compute: P_error_large or P_error_small -> P_actuating
        net.add_arc("P_error_large", "T_compute", weight=1.0)
        net.add_arc("P_error_small", "T_compute", weight=1.0)
        net.add_arc("T_compute", "P_actuating", weight=1.0)

        # T_apply: P_actuating -> P_applied, and also return to P_idle
        net.add_arc("P_actuating", "T_apply", weight=1.0)
        net.add_arc("T_apply", "P_applied", weight=1.0)
        net.add_arc("T_apply", "P_idle", weight=1.0)

        self._net = net
        return net

    # ── Compilation ─────────────────────────────────────────────────────

    def compile(
        self,
        bitstream_length: int = 1024,
        seed: int = 42,
        firing_mode: str = "fractional",
    ) -> CompiledNet:
        """Compile the control net into matrix form.

        Parameters
        ----------
        bitstream_length : int
            Number of bits per stochastic stream.
        seed : int
            Base RNG seed.
        firing_mode : str
            ``"binary"`` or ``"fractional"`` (default ``"fractional"``
            for graded control output).

        Returns
        -------
        CompiledNet
            Compiled weight matrices, thresholds, and initial marking.
        """
        if self._net is None:
            self.create_net()
        assert self._net is not None

        compiler = FusionCompiler(bitstream_length=bitstream_length, seed=seed)
        self._compiled_net = compiler.compile(
            self._net,
            firing_mode=firing_mode,
        )
        return self._compiled_net

    # ── Measurement Encoding ────────────────────────────────────────────

    def encode_measurement(self, z: float) -> Dict[str, float]:
        """Map a vertical displacement *z* (metres) to place markings.

        The measurement is decomposed into:
        - **Sign routing**: token goes to ``P_error_pos`` (z > 0) or
          ``P_error_neg`` (z <= 0).
        - **Magnitude classification**: token goes to ``P_error_large``
          (|z| > error_threshold) or ``P_error_small`` (|z| <= threshold),
          with strength proportional to |z|.

        Returns a dict of place-name -> token value (all in [0, 1]).
        All places not listed are implicitly 0.

        Parameters
        ----------
        z : float
            Vertical displacement in metres.  Positive = upward.

        Returns
        -------
        dict[str, float]
            Marking vector as a sparse dict.
        """
        marking: Dict[str, float] = {p: 0.0 for p in PLACES}

        abs_z = abs(z)

        # Error magnitude as a token density in [0, 1].
        # Use a saturating map: density = tanh(|z| / error_threshold).
        # This ensures values stay in [0, 1] and gives near-linear
        # response for small errors with saturation for large ones.
        if self.error_threshold_m > 0:
            density = math.tanh(abs_z / self.error_threshold_m)
        else:
            density = 1.0 if abs_z > 0 else 0.0

        # Sign routing: exactly one of pos/neg gets the token.
        if z > 0:
            marking["P_error_pos"] = density
        else:
            marking["P_error_neg"] = density

        # Magnitude classification
        if abs_z > self.error_threshold_m:
            marking["P_error_large"] = density
        else:
            marking["P_error_small"] = density

        return marking

    # ── Output Decoding ─────────────────────────────────────────────────

    def decode_output(
        self,
        marking: Dict[str, float],
        z_positive: bool,
    ) -> float:
        """Decode the current net marking into a control signal.

        The control law opposes the displacement: positive Z produces
        a negative control signal (push plasma down) and vice versa.

        The signal magnitude is derived from the actuation-stage token
        densities, with larger-error paths producing stronger output.

        Parameters
        ----------
        marking : dict[str, float]
            Current place marking (place name -> token density).
        z_positive : bool
            True if the original measurement had Z > 0.

        Returns
        -------
        float
            Control signal u (in arbitrary units, sign opposes Z).
        """
        # Combine actuation signals from both magnitude paths.
        large_tok = marking.get("P_error_large", 0.0)
        small_tok = marking.get("P_error_small", 0.0)
        actuating = marking.get("P_actuating", 0.0)
        applied = marking.get("P_applied", 0.0)

        # Weighted combination: large errors contribute more.
        magnitude = (
            self.error_scale_large * large_tok
            + self.error_scale_small * small_tok
            + actuating
            + applied
        )

        u = self.gain_scale * magnitude

        # Oppose the displacement direction.
        if z_positive:
            u = -u
        return u
