# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Embedded NeuroCore Compatibility Layer
"""
Embedded ``sc_neurocore``-compatible primitives for standalone SCPN-Fusion-Core.

This module provides the minimal runtime surface consumed by fusion control:

- ``StochasticLIFNeuron``
- ``RNG``
- ``generate_bernoulli_bitstream``
- ``pack_bitstream``, ``vec_and``, ``vec_popcount``
- ``QuantumEntropySource``

The API is intentionally aligned with the historical optional dependency so
existing SCPN/control code can run without requiring external package fetches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
UInt64Array = NDArray[np.uint64]

SC_NEUROCORE_AVAILABLE = True
SC_NEUROCORE_BACKEND = "embedded"


class RNG:
    """Small compatibility wrapper for deterministic random streams."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))

    def random(self, size: Optional[int | tuple[int, ...]] = None) -> Any:
        return self._rng.random(size=size)

    def normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Optional[int | tuple[int, ...]] = None,
    ) -> Any:
        return self._rng.normal(loc=loc, scale=scale, size=size)


def generate_bernoulli_bitstream(
    p: float,
    length: int,
    *,
    rng: Optional[RNG] = None,
) -> NDArray[np.uint8]:
    """Generate a Bernoulli bitstream of length ``length`` with mean ``p``."""
    if int(length) < 1:
        raise ValueError("length must be >= 1")
    p_f = float(np.clip(float(p), 0.0, 1.0))
    r = rng if rng is not None else RNG(0)
    values = np.asarray(r.random(size=int(length)), dtype=np.float64)
    return (values < p_f).astype(np.uint8)


def pack_bitstream(bitstream: NDArray[Any]) -> UInt64Array:
    """Pack ``{0,1}`` bitstream into little-endian ``uint64`` words."""
    bits = np.asarray(bitstream, dtype=np.uint8).reshape(-1)
    n_words = int(np.ceil(bits.size / 64))
    if n_words == 0:
        return np.zeros(0, dtype=np.uint64)
    pad = n_words * 64 - bits.size
    if pad > 0:
        bits = np.pad(bits, (0, pad), mode="constant")
    words_bits = bits.reshape(n_words, 64).astype(np.uint64)
    shifts = np.arange(64, dtype=np.uint64).reshape(1, 64)
    return np.sum(words_bits << shifts, axis=1, dtype=np.uint64)


def vec_and(a_packed: UInt64Array, b_packed: UInt64Array) -> UInt64Array:
    return np.bitwise_and(
        np.asarray(a_packed, dtype=np.uint64),
        np.asarray(b_packed, dtype=np.uint64),
    )


def vec_popcount(packed: UInt64Array) -> int:
    arr = np.asarray(packed, dtype=np.uint64).reshape(-1)
    if hasattr(np, "bit_count"):
        return int(np.bit_count(arr).sum(dtype=np.uint64))
    # NumPy < 2 fallback that is independent of Python's int.bit_count availability.
    byte_view = arr.view(np.uint8)
    return int(np.unpackbits(byte_view, bitorder="little").sum(dtype=np.uint64))


@dataclass
class QuantumEntropySource:
    """Compatibility entropy source used by the cybernetic controller path."""

    n_qubits: int = 4
    seed: int = 1337

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(int(self.seed))

    def random(self) -> float:
        return float(self._rng.random())

    def normal(self, loc: float = 0.0, scale: float = 1.0) -> float:
        return float(self._rng.normal(loc=loc, scale=scale))


class StochasticLIFNeuron:
    """Minimal stochastic LIF neuron aligned with historical sc_neurocore API."""

    def __init__(
        self,
        *,
        v_rest: float = 0.0,
        v_reset: float = 0.0,
        v_threshold: float = 1.0,
        tau_mem: float = 10.0,
        dt: float = 1.0,
        noise_std: float = 0.0,
        resistance: float = 1.0,
        refractory_period: int = 0,
        seed: int = 0,
        entropy_source: Optional[QuantumEntropySource] = None,
    ) -> None:
        self.v_rest = float(v_rest)
        self.v_reset = float(v_reset)
        self.v_threshold = float(v_threshold)
        self.tau_mem = float(tau_mem)
        self.dt = float(dt)
        self.noise_std = float(noise_std)
        self.resistance = float(resistance)
        self.refractory_period = int(refractory_period)
        self.seed = int(seed)
        self.entropy_source = entropy_source
        # Compiler path uses very large tau_mem and expects single-step thresholding.
        self._comparator_mode = self.tau_mem >= 1e5

        self.v = self.v_rest
        self._refractory_count = 0
        self._rng = np.random.default_rng(self.seed)

    def _sample_noise(self) -> float:
        if self.noise_std <= 0.0:
            return 0.0
        if self.entropy_source is not None and hasattr(self.entropy_source, "normal"):
            return float(self.entropy_source.normal(0.0, self.noise_std))
        return float(self._rng.normal(0.0, self.noise_std))

    def reset_state(self) -> None:
        """Reset internal membrane/refractory state before a deterministic step."""
        self.v = self.v_rest
        self._refractory_count = 0

    def step(self, input_current: float) -> bool:
        if self._refractory_count > 0:
            self._refractory_count -= 1
            return False

        if self._comparator_mode:
            sample = self.v_rest + self.resistance * float(input_current) + self._sample_noise()
            if sample >= self.v_threshold:
                if self.refractory_period > 0:
                    self._refractory_count = self.refractory_period
                self.v = self.v_reset
                return True
            self.v = self.v_rest
            return False

        drive = self.resistance * float(input_current)
        leak = -(self.v - self.v_rest)
        self.v += (self.dt / max(self.tau_mem, 1e-12)) * (leak + drive) + self._sample_noise()

        if self.v >= self.v_threshold:
            self.v = self.v_reset
            if self.refractory_period > 0:
                self._refractory_count = self.refractory_period
            return True
        return False
