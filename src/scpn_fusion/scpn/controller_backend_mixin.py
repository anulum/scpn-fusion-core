# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Controller Backend Runtime Mixins
"""Runtime backend helpers extracted from controller monolith."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from .contracts import _seed64
from scpn_fusion.fallback_telemetry import record_fallback_event


def _controller_module() -> Any:
    import scpn_fusion.scpn.controller as controller_mod

    return controller_mod


class NeuroSymbolicControllerBackendMixin:
    def _dense_activations(self, marking: np.ndarray) -> np.ndarray:
        controller_mod = _controller_module()
        if self._runtime_backend == "rust" and controller_mod._HAS_RUST_SCPN_RUNTIME:
            try:
                rust_dense_activations = controller_mod._rust_dense_activations
                if rust_dense_activations is None:
                    raise RuntimeError("Rust runtime reports availability without dense kernel.")
                out = rust_dense_activations(self._W_in, marking)
                return np.asarray(out, dtype=np.float64)
            except Exception as exc:  # pragma: no cover - depends on Rust runtime failures
                record_fallback_event(
                    "scpn_controller",
                    "rust_dense_activation_failed",
                    context={"error": exc.__class__.__name__},
                )
                self._runtime_backend = "numpy"
        self._tmp_activations[:] = self._W_in @ marking
        return self._tmp_activations

    def _marking_update(
        self,
        marking: np.ndarray,
        firing: np.ndarray,
        out: np.ndarray,
    ) -> np.ndarray:
        controller_mod = _controller_module()
        if self._runtime_backend == "rust" and controller_mod._HAS_RUST_SCPN_RUNTIME:
            try:
                rust_marking_update = controller_mod._rust_marking_update
                if rust_marking_update is None:
                    raise RuntimeError("Rust runtime reports availability without marking kernel.")
                rust_out = rust_marking_update(marking, self._W_in, self._W_out, firing)
                np.copyto(out, np.asarray(rust_out, dtype=np.float64))
                return out
            except Exception as exc:  # pragma: no cover - depends on Rust runtime failures
                record_fallback_event(
                    "scpn_controller",
                    "rust_marking_update_failed",
                    context={"error": exc.__class__.__name__},
                )
                self._runtime_backend = "numpy"
        self._tmp_consumption[:] = self._W_in_t @ firing
        self._tmp_production[:] = self._W_out @ firing
        out[:] = marking
        out -= self._tmp_consumption
        out += self._tmp_production
        np.clip(out, 0.0, 1.0, out=out)
        return out

    def _oracle_step(self, marking: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Float-path Petri step."""
        a = self._dense_activations(marking)
        if self._firing_mode == "fractional":
            margins = np.maximum(self._margins, 1e-12)
            f = np.clip((a - self._thresholds) / margins, 0.0, 1.0)
        else:
            f = (a >= self._thresholds).astype(np.float64)

        f_timed, self._oracle_cursor = self._apply_transition_timing(
            f, self._oracle_pending, self._oracle_cursor
        )
        m2 = self._marking_update(marking, f_timed, self._tmp_marking_oracle)
        return f_timed, m2

    def _sc_step(self, marking: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Deterministic stochastic path with optional bit-flip fault injection."""
        a = self._dense_activations(marking)

        if self._firing_mode == "fractional":
            margins = np.maximum(self._margins, 1e-12)
            p_fire = np.clip((a - self._thresholds) / margins, 0.0, 1.0)
        else:
            if self._sc_binary_margin > 0.0:
                p_fire = np.clip(
                    0.5 + 0.5 * ((a - self._thresholds) / self._sc_binary_margin),
                    0.0,
                    1.0,
                )
            else:
                p_fire = (a >= self._thresholds).astype(np.float64)

        if self._sc_n_passes <= 1 or (
            self._firing_mode == "binary" and self._sc_binary_margin <= 0.0
        ):
            f = p_fire
            rng = None
        else:
            sample_seed = _seed64(self.seed_base, f"sc_step:{int(k)}")
            controller_mod = _controller_module()
            use_rust_sampler = (
                self._runtime_backend == "rust"
                and controller_mod._HAS_RUST_SCPN_RUNTIME
                and controller_mod._rust_sample_firing is not None
            )
            if use_rust_sampler:
                try:
                    sampler = controller_mod._rust_sample_firing
                    if sampler is None:
                        raise RuntimeError("rust sampler unavailable")
                    sampled = sampler(
                        p_fire,
                        int(self._sc_n_passes),
                        int(sample_seed),
                        bool(self._sc_antithetic),
                    )
                    f = np.asarray(sampled, dtype=np.float64)
                    if self._sc_bitflip_rate > 0.0:
                        rng = np.random.default_rng(_seed64(self.seed_base, f"sc_flip:{int(k)}"))
                    else:
                        rng = None
                except Exception as exc:  # pragma: no cover - depends on Rust runtime failures
                    record_fallback_event(
                        "scpn_controller",
                        "rust_sample_firing_failed",
                        context={"error": exc.__class__.__name__},
                    )
                    use_rust_sampler = False

            if not use_rust_sampler:
                rng = np.random.default_rng(sample_seed)
                counts = self._tmp_sc_counts
                if self._sc_antithetic and self._sc_n_passes >= 2:
                    n_pairs = (self._sc_n_passes + 1) // 2
                    counts.fill(0)
                    if self._nT <= self._sc_antithetic_chunk_size:
                        base = rng.random((n_pairs, self._nT))
                        low_hits = np.sum(base < p_fire[None, :], axis=0, dtype=np.int64)
                        if self._sc_n_passes % 2 == 0:
                            high_hits = np.sum(
                                base > (1.0 - p_fire)[None, :], axis=0, dtype=np.int64
                            )
                        else:
                            high_hits = np.sum(
                                base[:-1, :] > (1.0 - p_fire)[None, :],
                                axis=0,
                                dtype=np.int64,
                            )
                        counts[:] = np.asarray(low_hits + high_hits, dtype=np.int64)
                    else:
                        for start in range(0, self._nT, self._sc_antithetic_chunk_size):
                            end = min(start + self._sc_antithetic_chunk_size, self._nT)
                            p_chunk = p_fire[start:end]
                            base = rng.random((n_pairs, end - start))
                            low_hits = np.sum(
                                base < p_chunk[None, :],
                                axis=0,
                                dtype=np.int64,
                            )
                            if self._sc_n_passes % 2 == 0:
                                high_hits = np.sum(
                                    base > (1.0 - p_chunk)[None, :],
                                    axis=0,
                                    dtype=np.int64,
                                )
                            else:
                                high_hits = np.sum(
                                    base[:-1, :] > (1.0 - p_chunk)[None, :],
                                    axis=0,
                                    dtype=np.int64,
                                )
                            counts[start:end] = np.asarray(low_hits + high_hits, dtype=np.int64)
                else:
                    np.copyto(
                        counts,
                        np.asarray(
                            rng.binomial(self._sc_n_passes, p_fire, size=self._nT),
                            dtype=np.int64,
                        ),
                    )
                f = counts.astype(np.float64) / float(self._sc_n_passes)

        if self._sc_bitflip_rate > 0.0:
            if rng is None:
                rng = np.random.default_rng(_seed64(self.seed_base, f"sc_flip:{int(k)}"))
            f = self._apply_bit_flip_faults(f, rng)

        f_timed, self._sc_cursor = self._apply_transition_timing(
            f, self._sc_pending, self._sc_cursor
        )
        m2 = self._marking_update(marking, f_timed, self._tmp_marking_sc)
        if self._sc_bitflip_rate > 0.0:
            if rng is None:
                rng = np.random.default_rng(_seed64(self.seed_base, f"sc_flip:{int(k)}"))
            m2 = self._apply_bit_flip_faults(m2, rng)

        return f_timed, m2

    def _apply_transition_timing(
        self,
        desired_firing: np.ndarray,
        pending: np.ndarray,
        cursor: int,
    ) -> Tuple[np.ndarray, int]:
        desired = np.asarray(np.clip(desired_firing, 0.0, 1.0), dtype=np.float64)
        if self._max_delay_ticks <= 0:
            return desired, cursor

        fired_now = np.asarray(pending[cursor], dtype=np.float64).copy()
        pending[cursor, :] = 0.0

        if self._delay_immediate_idx.size:
            idx = self._delay_immediate_idx
            fired_now[idx] = np.clip(fired_now[idx] + desired[idx], 0.0, 1.0)

        if self._delay_delayed_idx.size:
            np.add(cursor, self._delay_delayed_offsets, out=self._tmp_delay_slots)
            self._tmp_delay_slots %= pending.shape[0]
            idx = self._delay_delayed_idx
            pending[self._tmp_delay_slots, idx] = np.clip(
                pending[self._tmp_delay_slots, idx] + desired[idx], 0.0, 1.0
            )

        next_cursor = (cursor + 1) % pending.shape[0]
        return fired_now, next_cursor
