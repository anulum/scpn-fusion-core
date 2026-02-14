# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
"""SNN controller compiled from vertical position Petri net.

The control loop:
1. Read sensor -> encode Z as place marking (error sign + magnitude)
2. Forward through W_in (places -> transitions firing probabilities)
3. Apply threshold (stochastic firing)
4. Forward through W_out (transitions -> new place marking)
5. Decode output marking -> control signal u

The stochastic firing naturally implements proportional control:
higher error -> higher firing probability -> stronger correction.
This is NOT learned; it is a property of the compilation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from .vertical_control_net import VerticalControlNet, PLACES
from .compiler import CompiledNet, FusionCompiler

logger = logging.getLogger(__name__)

# ---- sc_neurocore import (graceful fallback) --------------------------------

_HAS_SC_NEUROCORE = False

try:
    from sc_neurocore import SCDenseLayer

    _HAS_SC_NEUROCORE = True
    logger.info("sc_neurocore detected -- SCDenseLayer path available.")
except ImportError:
    logger.warning(
        "sc_neurocore not installed -- using numpy float-path only."
    )

FloatArray = NDArray[np.float64]


class VerticalSNNController:
    """SNN controller compiled from the vertical position Petri net.

    Wraps a :class:`VerticalControlNet`, compiles it to weight matrices,
    and runs a forward pass each control tick to produce a scalar
    control signal ``u`` that opposes vertical displacement.

    The interface mirrors :class:`PIDController`: call
    ``compute(z_measured, dz_measured)`` each tick.

    Parameters
    ----------
    vcn : VerticalControlNet
        Pre-configured control net (will be compiled internally).
    bitstream_length : int
        Number of bits per stochastic stream.  Default 1024.
    batch_size : int
        Number of SC time-steps to run per forward pass when using the
        SCDenseLayer backend.  Default 10.
    seed : int
        Base RNG seed for reproducibility.
    force_numpy : bool
        If True, skip the SC-NeuroCore path even when available.
    """

    def __init__(
        self,
        vcn: VerticalControlNet,
        *,
        bitstream_length: int = 1024,
        batch_size: int = 10,
        seed: int = 42,
        force_numpy: bool = False,
    ) -> None:
        self._vcn = vcn
        self._bitstream_length = int(bitstream_length)
        self._batch_size = int(batch_size)
        self._seed = int(seed)
        self._force_numpy = bool(force_numpy)

        # Compile the net to get W_in, W_out, thresholds, initial marking.
        self._compiled: CompiledNet = vcn.compile(
            bitstream_length=self._bitstream_length,
            seed=self._seed,
            firing_mode="fractional",
        )

        # Cache dense matrices for the numpy forward path.
        self._W_in: FloatArray = self._compiled.W_in.copy()    # (nT, nP)
        self._W_out: FloatArray = self._compiled.W_out.copy()   # (nP, nT)
        self._thresholds: FloatArray = self._compiled.thresholds.copy()
        self._firing_margin: float = self._compiled.firing_margin

        self._n_places: int = self._compiled.n_places
        self._n_transitions: int = self._compiled.n_transitions
        self._place_names: List[str] = self._compiled.place_names

        # Live state: current marking vector.
        self._marking: FloatArray = self._compiled.initial_marking.copy()

        # Determine backend.
        self._use_sc = _HAS_SC_NEUROCORE and not self._force_numpy
        if self._use_sc:
            logger.info("VerticalSNNController: using SCDenseLayer backend.")
        else:
            logger.info("VerticalSNNController: using NumPy float-path backend.")

    # ---- Public API ---------------------------------------------------------

    def compute(self, z_measured: float, dz_measured: float) -> float:
        """Compute a scalar control signal opposing vertical displacement.

        Parameters
        ----------
        z_measured : float
            Vertical displacement in metres.  Positive = upward.
        dz_measured : float
            Vertical velocity in m/s (currently unused; kept for
            interface compatibility with PIDController).

        Returns
        -------
        float
            Control signal u.  Sign opposes displacement.
        """
        # 1. Encode measurement into place marking dict.
        encoded: Dict[str, float] = self._vcn.encode_measurement(z_measured)

        # Inject encoded tokens into the marking vector.
        marking = self._marking.copy()
        for i, pname in enumerate(self._place_names):
            if pname in encoded and encoded[pname] > 0.0:
                marking[i] = encoded[pname]

        # 2. Forward through W_in: activations = W_in @ marking.
        if self._use_sc:
            new_marking = self._forward_sc(marking)
        else:
            new_marking = self._forward_numpy(marking)

        # 3. Update internal state.
        self._marking = new_marking

        # 4. Decode marking into a control signal.
        marking_dict: Dict[str, float] = {
            self._place_names[i]: float(new_marking[i])
            for i in range(self._n_places)
        }
        u = self._vcn.decode_output(marking_dict, z_positive=(z_measured > 0))
        return float(u)

    def reset(self) -> None:
        """Reset internal state to the initial marking."""
        self._marking = self._compiled.initial_marking.copy()

    @property
    def marking(self) -> List[float]:
        """Current marking vector as a list of floats."""
        return self._marking.tolist()

    @property
    def backend_name(self) -> str:
        """Return the active backend name."""
        return "sc_neurocore" if self._use_sc else "numpy"

    # ---- NumPy float-path forward -------------------------------------------

    def _forward_numpy(self, marking: FloatArray) -> FloatArray:
        """NumPy forward pass: W_in -> threshold -> W_out.

        Steps
        -----
        1. activations = clip(W_in @ marking, 0, 1)
        2. firing = fractional threshold
        3. new_marking = clip(W_out @ firing, 0, 1)
        """
        # W_in is (nT, nP), marking is (nP,) -> activations is (nT,).
        activations = np.clip(self._W_in @ marking, 0.0, 1.0)

        # Fractional firing: f = clip((a - threshold) / margin, 0, 1).
        margin = max(self._firing_margin, 1e-12)
        firing = np.clip(
            (activations - self._thresholds) / margin, 0.0, 1.0
        )

        # W_out is (nP, nT), firing is (nT,) -> new_marking is (nP,).
        new_marking = np.clip(self._W_out @ firing, 0.0, 1.0)
        return new_marking

    # ---- SC-NeuroCore forward -----------------------------------------------

    def _forward_sc(self, marking: FloatArray) -> FloatArray:
        """SC-NeuroCore forward pass via SCDenseLayer.

        Uses SCDenseLayer to perform the stochastic matrix-vector
        product W_in @ marking, applies threshold, then uses a second
        SCDenseLayer for W_out @ firing.
        """
        # Step 1: W_in @ marking  (nT, nP) @ (nP,) -> (nT,)
        activations = self._sc_dense_forward(
            self._W_in, marking, layer_seed=self._seed
        )
        activations = np.clip(activations, 0.0, 1.0)

        # Step 2: Fractional threshold.
        margin = max(self._firing_margin, 1e-12)
        firing = np.clip(
            (activations - self._thresholds) / margin, 0.0, 1.0
        )

        # Step 3: W_out @ firing  (nP, nT) @ (nT,) -> (nP,)
        new_marking = self._sc_dense_forward(
            self._W_out, firing, layer_seed=self._seed + 10000
        )
        new_marking = np.clip(new_marking, 0.0, 1.0)
        return new_marking

    def _sc_dense_forward(
        self,
        W: FloatArray,
        inputs: FloatArray,
        layer_seed: int,
    ) -> FloatArray:
        """Perform a single SC dense forward pass for one matrix.

        Creates an SCDenseLayer per output neuron (row of W),
        runs it for ``batch_size`` steps, and returns the mean
        firing rate as the output activation.

        Parameters
        ----------
        W : (n_out, n_in) weight matrix with values in [0, 1].
        inputs : (n_in,) input vector with values in [0, 1].
        layer_seed : base seed for this layer.

        Returns
        -------
        output : (n_out,) float64 activation vector.
        """
        n_out, n_in = W.shape
        output = np.zeros(n_out, dtype=np.float64)

        for i in range(n_out):
            weights_row = W[i, :].tolist()
            inputs_list = inputs.tolist()

            # SCDenseLayer expects all values non-negative.
            # Clamp to [0, 1].
            safe_inputs = [max(0.0, min(1.0, v)) for v in inputs_list]
            safe_weights = [max(0.0, min(1.0, v)) for v in weights_row]

            layer = SCDenseLayer(
                n_neurons=1,
                x_inputs=safe_inputs,
                weight_values=safe_weights,
                x_min=0.0,
                x_max=1.0,
                w_min=0.0,
                w_max=1.0,
                length=self._bitstream_length,
                base_seed=layer_seed + i,
            )
            layer.run(self._batch_size)
            trains = layer.get_spike_trains()  # (1, T)
            # Mean firing rate as output activation.
            output[i] = float(np.mean(trains))

        return output
