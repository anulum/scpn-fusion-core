# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Neuro-Symbolic Logic Compiler
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Packet B — FusionCompiler and CompiledNet.

Compiles a ``StochasticPetriNet`` into sc_neurocore artifacts:
    - One ``StochasticLIFNeuron`` per transition (pure threshold comparator).
    - Pre-packed uint64 weight bitstreams for AND+popcount forward pass.
    - Float-path fallback when sc_neurocore is not installed.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from scpn_fusion import __version__ as PACKAGE_VERSION
from .structure import StochasticPetriNet

logger = logging.getLogger(__name__)

# ── sc_neurocore import (graceful fallback) ──────────────────────────────────

_HAS_SC_NEUROCORE = False

try:
    from sc_neurocore import StochasticLIFNeuron
    from sc_neurocore import generate_bernoulli_bitstream
    from sc_neurocore import bitstream_to_probability
    from sc_neurocore import RNG as _SC_RNG
    from sc_neurocore.accel.vector_ops import pack_bitstream, vec_and, vec_popcount

    _HAS_SC_NEUROCORE = True
    logger.info("sc_neurocore detected — stochastic path enabled.")
except ImportError:
    logger.warning(
        "sc_neurocore not installed — using numpy float-path only."
    )

# ── Helpers ──────────────────────────────────────────────────────────────────


def _resolve_git_sha() -> str:
    """Resolve a short git SHA for artifact metadata."""
    for key in ("SCPN_GIT_SHA", "GITHUB_SHA", "CI_COMMIT_SHA"):
        value = os.environ.get(key, "").strip()
        if value:
            return value[:7]

    try:
        repo_root = Path(__file__).resolve().parents[3]
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        sha = result.stdout.strip()
        if sha:
            return sha[:7]
    except Exception:
        pass

    return "0000000"


def _encode_weight_matrix_packed(
    W: np.ndarray,
    bitstream_length: int,
    seed: int,
) -> np.ndarray:
    """Encode each element of *W* as a packed uint64 bitstream.

    Parameters
    ----------
    W : (R, C) float64 array with values in [0, 1].
    bitstream_length : Number of bits per stream.
    seed : Base seed (incremented per element for independence).

    Returns
    -------
    packed : (R, C, n_words) uint64 array.
    """
    R, C = W.shape
    n_words = int(np.ceil(bitstream_length / 64))
    packed = np.zeros((R, C, n_words), dtype=np.uint64)

    rng_seed = seed
    for r in range(R):
        for c in range(C):
            p = float(np.clip(W[r, c], 0.0, 1.0))
            rng = _SC_RNG(rng_seed)
            bits = generate_bernoulli_bitstream(p, bitstream_length, rng=rng)
            packed[r, c, :] = pack_bitstream(bits)
            rng_seed += 1

    return packed


# ── CompiledNet ──────────────────────────────────────────────────────────────


@dataclass
class CompiledNet:
    """Compiled Petri Net ready for sc_neurocore execution.

    Holds both the dense float matrices (for validation / fallback) and
    pre-packed uint64 weight bitstreams (for the stochastic path).
    """

    # Topology
    n_places: int
    n_transitions: int
    place_names: List[str]
    transition_names: List[str]

    # Dense weight matrices (float path)
    W_in: np.ndarray          # (nT, nP)
    W_out: np.ndarray         # (nP, nT)

    # Pre-packed weight bitstreams (stochastic path) — None if no sc_neurocore
    W_in_packed: Optional[np.ndarray] = None   # (nT, nP, n_words) uint64
    W_out_packed: Optional[np.ndarray] = None  # (nP, nT, n_words) uint64

    # LIF neurons (one per transition) — empty list if no sc_neurocore
    neurons: List = field(default_factory=list)

    # Config
    bitstream_length: int = 1024
    thresholds: np.ndarray = field(default_factory=lambda: np.array([]))
    initial_marking: np.ndarray = field(default_factory=lambda: np.array([]))
    seed: int = 42
    firing_mode: str = "binary"
    firing_margin: float = 0.05

    @property
    def has_stochastic_path(self) -> bool:
        return self.W_in_packed is not None

    # ── Forward passes ───────────────────────────────────────────────────

    def dense_forward(
        self,
        W_packed: np.ndarray,
        input_probs: np.ndarray,
    ) -> np.ndarray:
        """Stochastic matrix-vector product via AND + popcount.

        Parameters
        ----------
        W_packed : (n_out, n_in, n_words) uint64 — pre-packed weight bitstreams.
        input_probs : (n_in,) float64 — input probabilities in [0, 1].

        Returns
        -------
        output : (n_out,) float64 — stochastic estimate of W @ input_probs.
        """
        if not _HAS_SC_NEUROCORE:
            raise RuntimeError(
                "dense_forward requires sc_neurocore.  "
                "Use dense_forward_float for the numpy fallback."
            )

        n_out, n_in, n_words = W_packed.shape
        output = np.zeros(n_out, dtype=np.float64)

        # Encode each input probability as a packed bitstream
        input_packed = np.zeros((n_in, n_words), dtype=np.uint64)
        rng_seed = self.seed + 1_000_000  # offset from weight seeds
        for j in range(n_in):
            p = float(np.clip(input_probs[j], 0.0, 1.0))
            rng = _SC_RNG(rng_seed + j)
            bits = generate_bernoulli_bitstream(p, self.bitstream_length, rng=rng)
            input_packed[j, :] = pack_bitstream(bits)

        # For each output row: AND each weight stream with input stream, sum
        for i in range(n_out):
            total_ones = 0
            for j in range(n_in):
                anded = vec_and(W_packed[i, j, :], input_packed[j, :])
                total_ones += int(vec_popcount(anded))
            # Normalize: sum of products, each product ≈ w_ij * x_j
            # Max possible ones per AND = bitstream_length, there are n_in
            # terms, but we want the sum not the average, so divide only by L.
            output[i] = total_ones / self.bitstream_length

        return output

    def dense_forward_float(
        self,
        W: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Float-path validation: simple ``W @ inputs``."""
        return W @ inputs

    def lif_fire(self, currents: np.ndarray) -> np.ndarray:
        """Run LIF threshold detection on all transitions.

        Binary mode: ``f_t = 1 if current >= threshold else 0``
        Fractional mode: ``f_t = clip((current - threshold) / margin, 0, 1)``

        Parameters
        ----------
        currents : (n_transitions,) float64 — weighted-sum activations.

        Returns
        -------
        fired : (n_transitions,) float64 vector.
            Binary mode → values in {0.0, 1.0}.
            Fractional mode → values in [0.0, 1.0].
        """
        if self.firing_mode == "fractional":
            margin = max(self.firing_margin, 1e-12)
            raw = (currents - self.thresholds) / margin
            return np.clip(raw, 0.0, 1.0)

        # Binary mode
        if self.neurons:
            fired = np.zeros(self.n_transitions, dtype=np.float64)
            for i, neuron in enumerate(self.neurons):
                neuron.reset_state()
                fired[i] = float(neuron.step(float(currents[i])))
            return fired
        else:
            return (currents >= self.thresholds).astype(np.float64)

    # ── Convenience ──────────────────────────────────────────────────────

    def summary(self) -> str:
        mode = "stochastic" if self.has_stochastic_path else "float-only"
        return (
            f"CompiledNet  P={self.n_places}  T={self.n_transitions}  "
            f"L={self.bitstream_length}  mode={mode}"
        )

    # ── Artifact export ──────────────────────────────────────────────────

    def export_artifact(
        self,
        name: str = "controller",
        dt_control_s: float = 0.001,
        readout_config: Optional[Dict[str, Any]] = None,
        injection_config: Optional[List[Dict[str, Any]]] = None,
    ) -> "artifact_mod.Artifact":
        """Build an ``Artifact`` from compiled state + user-provided config.

        Parameters
        ----------
        name : artifact name.
        dt_control_s : control tick period (s).
        readout_config : dict with ``actions``, ``gains``, ``abs_max``,
            ``slew_per_s`` lists.  Required for a complete artifact.
        injection_config : list of place-injection dicts.
        """
        from . import artifact as artifact_mod

        n_words = int(math.ceil(self.bitstream_length / 64))

        meta = artifact_mod.ArtifactMeta(
            artifact_version=artifact_mod.ARTIFACT_SCHEMA_VERSION,
            name=name,
            dt_control_s=dt_control_s,
            stream_length=self.bitstream_length,
            fixed_point=artifact_mod.FixedPoint(
                data_width=16, fraction_bits=10, signed=False
            ),
            firing_mode=self.firing_mode,
            seed_policy=artifact_mod.SeedPolicy(
                id="default", hash_fn="splitmix64", rng_family="xoshiro256++"
            ),
            created_utc=datetime.now(timezone.utc).isoformat(),
            compiler=artifact_mod.CompilerInfo(
                name="FusionCompiler",
                version=PACKAGE_VERSION,
                git_sha=_resolve_git_sha(),
            ),
        )

        places = [
            artifact_mod.PlaceSpec(id=i, name=n)
            for i, n in enumerate(self.place_names)
        ]
        transitions = [
            artifact_mod.TransitionSpec(
                id=i,
                name=n,
                threshold=float(self.thresholds[i]),
                margin=self.firing_margin if self.firing_mode == "fractional" else None,
            )
            for i, n in enumerate(self.transition_names)
        ]
        topology = artifact_mod.Topology(places=places, transitions=transitions)

        w_in_mat = artifact_mod.WeightMatrix(
            shape=[self.n_transitions, self.n_places],
            data=self.W_in.ravel().tolist(),
        )
        w_out_mat = artifact_mod.WeightMatrix(
            shape=[self.n_places, self.n_transitions],
            data=self.W_out.ravel().tolist(),
        )
        weights = artifact_mod.Weights(w_in=w_in_mat, w_out=w_out_mat)

        # Readout
        rc = readout_config or {}
        actions_raw = rc.get("actions", [])
        actions = [
            artifact_mod.ActionReadout(
                id=a.get("id", i),
                name=a["name"],
                pos_place=a["pos_place"],
                neg_place=a["neg_place"],
            )
            for i, a in enumerate(actions_raw)
        ]
        readout = artifact_mod.Readout(
            actions=actions,
            gains=rc.get("gains", [1.0] * len(actions)),
            abs_max=rc.get("abs_max", [1e4] * len(actions)),
            slew_per_s=rc.get("slew_per_s", [1e6] * len(actions)),
        )

        # Injections
        injections = [
            artifact_mod.PlaceInjection(
                place_id=inj["place_id"],
                source=inj["source"],
                scale=inj.get("scale", 1.0),
                offset=inj.get("offset", 0.0),
                clamp_0_1=inj.get("clamp_0_1", True),
            )
            for inj in (injection_config or [])
        ]
        initial_state = artifact_mod.InitialState(
            marking=self.initial_marking.tolist(),
            place_injections=injections,
        )

        return artifact_mod.Artifact(
            meta=meta,
            topology=topology,
            weights=weights,
            readout=readout,
            initial_state=initial_state,
        )


# ── FusionCompiler ───────────────────────────────────────────────────────────


class FusionCompiler:
    """Compiles a ``StochasticPetriNet`` into a ``CompiledNet``.

    Parameters
    ----------
    bitstream_length : Number of bits per stochastic stream (default 1024).
    seed : Base RNG seed for reproducibility.
    """

    def __init__(self, bitstream_length: int = 1024, seed: int = 42) -> None:
        if bitstream_length < 64:
            raise ValueError("bitstream_length must be >= 64")
        self.bitstream_length = bitstream_length
        self.seed = seed

    def compile(
        self,
        net: StochasticPetriNet,
        firing_mode: str = "binary",
        firing_margin: float = 0.05,
    ) -> CompiledNet:
        """Compile the Petri Net into sc_neurocore artifacts.

        Parameters
        ----------
        net : compiled ``StochasticPetriNet``.
        firing_mode : ``"binary"`` (default) or ``"fractional"``.
        firing_margin : margin for fractional firing (ignored in binary mode).

        Steps:
            1. Extract dense W_in (nT x nP) and W_out (nP x nT).
            2. Create one LIF neuron per transition (pure threshold comparator).
            3. Pre-encode weight matrices as packed uint64 bitstreams.
            4. Return ``CompiledNet`` with all artifacts.
        """
        if firing_mode not in ("binary", "fractional"):
            raise ValueError(
                f"firing_mode must be 'binary' or 'fractional', got '{firing_mode}'"
            )
        if not net.is_compiled:
            net.compile()

        # 1. Dense matrices
        W_in = net.W_in.toarray()    # (nT, nP)
        W_out = net.W_out.toarray()  # (nP, nT)

        thresholds = net.get_thresholds()
        initial_marking = net.get_initial_marking()

        # 2. LIF neurons (one per transition)
        neurons: list = []
        if _HAS_SC_NEUROCORE:
            for t_idx in range(net.n_transitions):
                neuron = StochasticLIFNeuron(
                    v_rest=0.0,
                    v_reset=0.0,
                    v_threshold=float(thresholds[t_idx]),
                    tau_mem=1e6,       # Effectively no leak
                    dt=1.0,
                    noise_std=0.0,     # Deterministic threshold
                    resistance=1.0,    # Current passes through unchanged
                    refractory_period=0,
                    seed=self.seed + t_idx,
                )
                neurons.append(neuron)

        # 3. Pre-encode weight bitstreams
        W_in_packed: np.ndarray | None = None
        W_out_packed: np.ndarray | None = None

        if _HAS_SC_NEUROCORE:
            W_in_packed = _encode_weight_matrix_packed(
                W_in, self.bitstream_length, seed=self.seed
            )
            W_out_packed = _encode_weight_matrix_packed(
                W_out, self.bitstream_length, seed=self.seed + W_in.size
            )

        # 4. Assemble
        return CompiledNet(
            n_places=net.n_places,
            n_transitions=net.n_transitions,
            place_names=net.place_names,
            transition_names=net.transition_names,
            W_in=W_in,
            W_out=W_out,
            W_in_packed=W_in_packed,
            W_out_packed=W_out_packed,
            neurons=neurons,
            bitstream_length=self.bitstream_length,
            thresholds=thresholds,
            initial_marking=initial_marking,
            seed=self.seed,
            firing_mode=firing_mode,
            firing_margin=firing_margin,
        )
