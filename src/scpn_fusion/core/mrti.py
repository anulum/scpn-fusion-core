# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — MRTI Growth Spectrum
"""Magneto-Rayleigh-Taylor instability growth and spectrum tracking.

This module implements the analytical MIF/FRC MRTI growth-rate contract used by
the current work lane. It is a deterministic physics surface, not a surrogate:
the growth rate is evaluated from the supplied acceleration, perpendicular
field, density, and resolved mode spectrum. Coupling to a supplied-current
pulsed-compression trajectory consumes the accepted FUS-C.6 state history and
projects signed radial acceleration onto the MRTI interface-normal convention.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

MU_0 = 4.0 * np.pi * 1.0e-7
FloatArray: TypeAlias = NDArray[np.float64]


class PulsedCompressionLike(Protocol):
    """Minimum state contract required for MRTI/compression coupling."""

    t_s: float
    R_s_m: float
    dR_s_dt_m_s: float
    B_ext_T: float


def _as_float_array(
    name: str, values: FloatArray | list[float] | tuple[float, ...] | float
) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(FloatArray, array)


def _require_finite(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_positive(name: str, value: float) -> float:
    result = _require_finite(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def mrti_growth_rate(
    k: FloatArray | list[float] | tuple[float, ...] | float,
    a_eff: float,
    B_perp: float = 0.0,
    rho_kg_m3: float = 1.0e-3,
) -> FloatArray:
    """Return MRTI growth rate in ``s^-1`` for each mode ``k``.

    The evaluated contract is

    ``gamma^2 = k a_eff - k^2 B_perp^2 / (mu0 rho)``.

    Negative radicands are clipped to zero, representing magnetic-tension
    stabilization for the supplied mode. ``k`` is in ``m^-1``, ``a_eff`` is in
    ``m s^-2``, ``B_perp`` is in tesla, and ``rho_kg_m3`` is in ``kg m^-3``.
    """

    k_modes = _as_float_array("k", k)
    if np.any(k_modes < 0.0):
        raise ValueError("k must be non-negative")
    acceleration = _require_finite("a_eff", a_eff)
    b_perp = _require_finite("B_perp", B_perp)
    density = _require_positive("rho_kg_m3", rho_kg_m3)

    radicand = k_modes * acceleration - (k_modes * k_modes * b_perp * b_perp) / (MU_0 * density)
    return cast(FloatArray, np.sqrt(np.maximum(radicand, 0.0)))


def effective_acceleration_from_radius_rate(
    time_s: FloatArray | list[float] | tuple[float, ...],
    d_radius_dt_m_s: FloatArray | list[float] | tuple[float, ...],
    *,
    smoothing_window: int = 1,
) -> FloatArray:
    """Estimate ``d²R_s/dt²`` from a separatrix radial-speed history.

    The helper performs finite differences on the supplied time grid and can
    apply an odd-width edge-padded moving average. It is a coupling adapter for
    future pulsed-compression trajectories; it does not replay or replace the
    missing full Hall-MHD compression solver.
    """

    t = _as_float_array("time_s", time_s)
    velocity = _as_float_array("d_radius_dt_m_s", d_radius_dt_m_s)
    if t.ndim != 1 or velocity.ndim != 1:
        raise ValueError("time_s and d_radius_dt_m_s must be one-dimensional")
    if t.shape != velocity.shape:
        raise ValueError("time_s and d_radius_dt_m_s must have identical shape")
    if t.size < 2:
        raise ValueError("at least two samples are required")
    if not np.all(np.diff(t) > 0.0):
        raise ValueError("time_s must be strictly increasing")
    if smoothing_window < 1 or smoothing_window % 2 == 0:
        raise ValueError("smoothing_window must be a positive odd integer")
    if smoothing_window > t.size:
        raise ValueError("smoothing_window cannot exceed the number of samples")

    edge_order: Literal[1, 2] = 2 if t.size >= 3 else 1
    acceleration = cast(FloatArray, np.gradient(velocity, t, edge_order=edge_order))
    if smoothing_window == 1:
        return acceleration

    pad = smoothing_window // 2
    kernel = np.ones(smoothing_window, dtype=np.float64) / float(smoothing_window)
    padded = np.pad(acceleration, pad_width=pad, mode="edge")
    return cast(FloatArray, np.convolve(padded, kernel, mode="valid"))


def effective_acceleration_from_pulsed_compression(
    states: Sequence[PulsedCompressionLike],
    *,
    smoothing_window: int = 1,
    radial_projection_sign: float = -1.0,
) -> FloatArray:
    """Project a FUS-C.6 pulsed-compression trajectory into MRTI acceleration.

    The supplied-current pulsed-compression contract stores radius in the
    outward radial coordinate. During compression, an inward MRTI-normal
    acceleration therefore has the opposite sign of ``d²R_s/dt²``. The default
    ``radial_projection_sign=-1`` maps inward radial acceleration to positive
    ``a_eff`` for the analytical MRTI growth-rate contract. Set it explicitly
    to ``+1`` when the caller's MRTI normal follows the outward radius.
    """

    if len(states) < 2:
        raise ValueError("at least two pulsed-compression states are required")
    projection = _require_finite("radial_projection_sign", radial_projection_sign)
    if projection == 0.0:
        raise ValueError("radial_projection_sign must be non-zero")

    time_s = np.asarray([state.t_s for state in states], dtype=np.float64)
    radius_m = np.asarray([state.R_s_m for state in states], dtype=np.float64)
    speed_m_s = np.asarray([state.dR_s_dt_m_s for state in states], dtype=np.float64)
    field_t = np.asarray([state.B_ext_T for state in states], dtype=np.float64)
    if np.any(radius_m <= 0.0):
        raise ValueError("pulsed-compression radii must be positive")
    if not np.all(np.isfinite(field_t)):
        raise ValueError("pulsed-compression fields must be finite")

    signed_acceleration = effective_acceleration_from_radius_rate(
        time_s,
        speed_m_s,
        smoothing_window=smoothing_window,
    )
    return cast(FloatArray, projection * signed_acceleration)


@dataclass(frozen=True)
class MRTISpectrumState:
    """Immutable MRTI spectrum state after a tracker step."""

    t_s: float
    k_modes_m_inv: FloatArray
    amplitudes_m: FloatArray
    growth_rates_s_inv: FloatArray
    fastest_growing_k_m_inv: float
    max_amplitude_m: float
    saturation_warning: bool
    time_of_breach_s: float | None


class MRTISpectrumTracker:
    """Track exponential MRTI mode growth over a resolved wavenumber spectrum."""

    def __init__(
        self,
        *,
        k_max_m_inv: float | None = None,
        n_modes: int = 64,
        k_modes_m_inv: FloatArray | list[float] | tuple[float, ...] | None = None,
        initial_perturbation_m: float = 1.0e-9,
        rho_kg_m3: float = 1.0e-3,
        saturation_threshold_m: float = 1.0e-3,
    ) -> None:
        if k_modes_m_inv is None:
            if k_max_m_inv is None:
                raise ValueError("k_max_m_inv is required when k_modes_m_inv is not supplied")
            k_max = _require_positive("k_max_m_inv", k_max_m_inv)
            if n_modes < 2:
                raise ValueError("n_modes must be at least 2")
            modes = np.linspace(k_max / float(n_modes), k_max, n_modes, dtype=np.float64)
        else:
            modes = _as_float_array("k_modes_m_inv", k_modes_m_inv)
            if modes.ndim != 1:
                raise ValueError("k_modes_m_inv must be one-dimensional")
            if modes.size < 2:
                raise ValueError("at least two MRTI modes are required")
            if np.any(modes < 0.0):
                raise ValueError("k_modes_m_inv must be non-negative")
            if not np.all(np.diff(modes) > 0.0):
                raise ValueError("k_modes_m_inv must be strictly increasing")

        initial = _require_positive("initial_perturbation_m", initial_perturbation_m)
        self._rho_kg_m3 = _require_positive("rho_kg_m3", rho_kg_m3)
        self._saturation_threshold_m = _require_positive(
            "saturation_threshold_m", saturation_threshold_m
        )
        self._k_modes_m_inv = modes.copy()
        self._amplitudes_m = np.full(self._k_modes_m_inv.shape, initial, dtype=np.float64)
        self._growth_rates_s_inv = np.zeros(self._k_modes_m_inv.shape, dtype=np.float64)
        self._t_s = 0.0
        self._time_of_breach_s: float | None = None

    @property
    def k_modes_m_inv(self) -> FloatArray:
        """Return a copy of the tracked mode spectrum in ``m^-1``."""

        return self._k_modes_m_inv.copy()

    @property
    def amplitudes_m(self) -> FloatArray:
        """Return a copy of current perturbation amplitudes in metres."""

        return self._amplitudes_m.copy()

    def saturation_threshold_breached(self, threshold_m: float | None = None) -> bool:
        """Return whether the maximum perturbation amplitude crossed threshold."""

        threshold = (
            self._saturation_threshold_m
            if threshold_m is None
            else _require_positive("threshold_m", threshold_m)
        )
        return bool(np.max(self._amplitudes_m) >= threshold)

    def state(self) -> MRTISpectrumState:
        """Return an immutable snapshot of the current spectrum state."""

        fastest_index = int(np.argmax(self._growth_rates_s_inv))
        max_amplitude = float(np.max(self._amplitudes_m))
        return MRTISpectrumState(
            t_s=self._t_s,
            k_modes_m_inv=self._k_modes_m_inv.copy(),
            amplitudes_m=self._amplitudes_m.copy(),
            growth_rates_s_inv=self._growth_rates_s_inv.copy(),
            fastest_growing_k_m_inv=float(self._k_modes_m_inv[fastest_index]),
            max_amplitude_m=max_amplitude,
            saturation_warning=max_amplitude >= self._saturation_threshold_m,
            time_of_breach_s=self._time_of_breach_s,
        )

    def step(self, dt_s: float, a_eff_m_s2: float, B_perp_t: float = 0.0) -> MRTISpectrumState:
        """Advance amplitudes by ``dt_s`` using frozen-coefficient exponential growth."""

        dt = _require_positive("dt_s", dt_s)
        self._growth_rates_s_inv = mrti_growth_rate(
            self._k_modes_m_inv,
            a_eff_m_s2,
            B_perp_t,
            self._rho_kg_m3,
        )
        exponent = np.clip(self._growth_rates_s_inv * dt, 0.0, 700.0)
        self._amplitudes_m = cast(FloatArray, self._amplitudes_m * np.exp(exponent))
        self._t_s += dt
        if self._time_of_breach_s is None and self.saturation_threshold_breached():
            self._time_of_breach_s = self._t_s
        return self.state()


def track_mrti_from_pulsed_compression(
    states: Sequence[PulsedCompressionLike],
    tracker: MRTISpectrumTracker,
    *,
    smoothing_window: int = 1,
    radial_projection_sign: float = -1.0,
    b_perp_scale: float = 1.0,
) -> tuple[MRTISpectrumState, ...]:
    """Advance an MRTI tracker over a supplied FUS-C.6 compression trajectory.

    Each interval uses the finite-difference acceleration at the interval end
    and the endpoint external field as the stabilising perpendicular field. The
    function returns one MRTI state per trajectory interval and raises on
    malformed trajectories instead of fabricating coupled evidence.
    """

    if len(states) < 2:
        raise ValueError("at least two pulsed-compression states are required")
    field_scale = _require_finite("b_perp_scale", b_perp_scale)
    if field_scale < 0.0:
        raise ValueError("b_perp_scale must be non-negative")
    accelerations = effective_acceleration_from_pulsed_compression(
        states,
        smoothing_window=smoothing_window,
        radial_projection_sign=radial_projection_sign,
    )
    snapshots: list[MRTISpectrumState] = []
    for index in range(1, len(states)):
        dt_s = _require_positive("trajectory dt_s", states[index].t_s - states[index - 1].t_s)
        snapshots.append(
            tracker.step(
                dt_s,
                float(accelerations[index]),
                B_perp_t=float(states[index].B_ext_T) * field_scale,
            )
        )
    return tuple(snapshots)
