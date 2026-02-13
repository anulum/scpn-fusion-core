# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Neuro-Symbolic Logic Compiler
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Neuro-Symbolic Controller — oracle + SC dual paths.

Loads a ``.scpnctl.json`` artifact and provides deterministic
``step(obs, k) → ControlAction`` with JSONL logging.
"""

from __future__ import annotations

import json
import time
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from .artifact import Artifact
from .contracts import (
    ControlAction,
    FeatureAxisSpec,
    ControlScales,
    ControlTargets,
    _seed64,
)

FloatArray = NDArray[np.float64]

_HAS_RUST_SCPN_RUNTIME = False
_rust_dense_activations: Optional[Callable[[FloatArray, FloatArray], object]] = None
_rust_marking_update: Optional[
    Callable[[FloatArray, FloatArray, FloatArray, FloatArray], object]
] = None
_rust_sample_firing: Optional[
    Callable[[FloatArray, int, int, bool], object]
] = None

try:
    from scpn_fusion_rs import (  # type: ignore[import-not-found]
        scpn_dense_activations as _rust_dense_activations_impl,
        scpn_marking_update as _rust_marking_update_impl,
        scpn_sample_firing as _rust_sample_firing_impl,
    )

    _rust_dense_activations = _rust_dense_activations_impl
    _rust_marking_update = _rust_marking_update_impl
    _rust_sample_firing = _rust_sample_firing_impl
    _HAS_RUST_SCPN_RUNTIME = True
except Exception:
    _HAS_RUST_SCPN_RUNTIME = False


class NeuroSymbolicController:
    """Reference controller with oracle float and stochastic paths.

    Parameters
    ----------
    artifact : loaded ``.scpnctl.json`` artifact.
    seed_base : 64-bit base seed for deterministic stochastic execution.
    targets : control setpoint targets.
    scales : normalisation scales.
    """

    def __init__(
        self,
        artifact: Artifact,
        seed_base: int,
        targets: ControlTargets,
        scales: ControlScales,
        sc_n_passes: int = 8,
        sc_bitflip_rate: float = 0.0,
        sc_binary_margin: Optional[float] = None,
        sc_antithetic: bool = True,
        enable_oracle_diagnostics: bool = True,
        feature_axes: Optional[Sequence[FeatureAxisSpec]] = None,
        runtime_profile: str = "adaptive",
        runtime_backend: str = "auto",
        rust_backend_min_problem_size: int = 4096,
    ) -> None:
        self.artifact = artifact
        self.seed_base = int(seed_base)
        self.targets = targets
        self.scales = scales
        self._sc_n_passes = max(int(sc_n_passes), 1)
        self._sc_bitflip_rate = float(np.clip(sc_bitflip_rate, 0.0, 1.0))
        self._runtime_profile = runtime_profile.strip().lower()
        if self._runtime_profile not in {"adaptive", "deterministic"}:
            raise ValueError("runtime_profile must be 'adaptive' or 'deterministic'")
        self._sc_antithetic = bool(sc_antithetic)
        self._enable_oracle_diagnostics = bool(enable_oracle_diagnostics)
        self._feature_axes = list(feature_axes) if feature_axes is not None else None
        self._runtime_backend_request = runtime_backend.strip().lower()
        if self._runtime_backend_request not in {"auto", "numpy", "rust"}:
            raise ValueError("runtime_backend must be 'auto', 'numpy', or 'rust'")
        self._rust_backend_min_problem_size = max(int(rust_backend_min_problem_size), 1)

        if self._feature_axes is not None:
            axes = list(self._feature_axes)
        else:
            axes = [
                FeatureAxisSpec(
                    obs_key="R_axis_m",
                    target=self.targets.R_target_m,
                    scale=self.scales.R_scale_m,
                    pos_key="x_R_pos",
                    neg_key="x_R_neg",
                ),
                FeatureAxisSpec(
                    obs_key="Z_axis_m",
                    target=self.targets.Z_target_m,
                    scale=self.scales.Z_scale_m,
                    pos_key="x_Z_pos",
                    neg_key="x_Z_neg",
                ),
            ]
        self._feature_axes_effective = axes
        self._axis_count = len(axes)
        self._axis_obs_keys = [axis.obs_key for axis in axes]
        self._axis_targets = np.asarray(
            [axis.target for axis in axes], dtype=np.float64
        )
        self._axis_scales = np.asarray(
            [
                axis.scale if abs(axis.scale) > 1e-12 else 1e-12
                for axis in axes
            ],
            dtype=np.float64,
        )
        self._axis_pos_keys = [axis.pos_key for axis in axes]
        self._axis_neg_keys = [axis.neg_key for axis in axes]
        self._empty = np.zeros(0, dtype=np.float64)
        self._tmp_obs_vals = np.zeros(self._axis_count, dtype=np.float64)
        self._tmp_feature_err = np.zeros(self._axis_count, dtype=np.float64)
        self._tmp_feature_pos = np.zeros(self._axis_count, dtype=np.float64)
        self._tmp_feature_neg = np.zeros(self._axis_count, dtype=np.float64)

        # Flatten weight matrices for fast indexing
        self._w_in = artifact.weights.w_in.data[:]
        self._w_out = artifact.weights.w_out.data[:]
        self._nP = artifact.nP
        self._nT = artifact.nT
        self._W_in = np.asarray(self._w_in, dtype=np.float64).reshape(self._nT, self._nP)
        self._W_out = np.asarray(self._w_out, dtype=np.float64).reshape(self._nP, self._nT)
        self._W_in_t = self._W_in.T
        self._tmp_activations = np.zeros(self._nT, dtype=np.float64)
        self._tmp_consumption = np.zeros(self._nP, dtype=np.float64)
        self._tmp_production = np.zeros(self._nP, dtype=np.float64)
        self._tmp_marking_oracle = np.zeros(self._nP, dtype=np.float64)
        self._tmp_marking_sc = np.zeros(self._nP, dtype=np.float64)
        self._tmp_marking_input = np.zeros(self._nP, dtype=np.float64)
        self._thresholds = np.asarray(
            [tr.threshold for tr in artifact.topology.transitions], dtype=np.float64
        )
        self._delay_ticks = np.asarray(
            [max(int(getattr(tr, "delay_ticks", 0)), 0) for tr in artifact.topology.transitions],
            dtype=np.int64,
        )
        self._max_delay_ticks = int(np.max(self._delay_ticks)) if self._delay_ticks.size else 0
        pending_len = self._max_delay_ticks + 1
        self._oracle_pending = np.zeros((pending_len, self._nT), dtype=np.float64)
        self._sc_pending = np.zeros((pending_len, self._nT), dtype=np.float64)
        self._oracle_cursor = 0
        self._sc_cursor = 0
        self._firing_mode = artifact.meta.firing_mode
        default_margin = float(getattr(artifact.meta, "firing_margin", 0.05) or 0.05)
        self._margins = np.asarray(
            [
                float(
                    ((tr.margin if tr.margin is not None else default_margin) or default_margin)
                )
                for tr in artifact.topology.transitions
            ],
            dtype=np.float64,
        )
        if sc_binary_margin is None:
            if self._runtime_profile == "adaptive":
                self._sc_binary_margin = 0.05
            else:
                self._sc_binary_margin = 0.0
        else:
            self._sc_binary_margin = float(max(0.0, sc_binary_margin))

        problem_size = int(self._nP * self._nT)
        rust_eligible = _HAS_RUST_SCPN_RUNTIME and (
            problem_size >= self._rust_backend_min_problem_size
        )
        if self._runtime_backend_request == "numpy":
            self._runtime_backend = "numpy"
        elif self._runtime_backend_request == "rust":
            self._runtime_backend = "rust" if _HAS_RUST_SCPN_RUNTIME else "numpy"
        else:
            self._runtime_backend = "rust" if rust_eligible else "numpy"
        produced_feature_keys = set(self._axis_pos_keys)
        produced_feature_keys.update(self._axis_neg_keys)
        passthrough_sources: list[str] = []
        for inj in self.artifact.initial_state.place_injections:
            src = inj.source
            if src not in produced_feature_keys and src not in passthrough_sources:
                passthrough_sources.append(src)
        self._passthrough_sources = passthrough_sources
        key_to_axis: Dict[str, Tuple[int, bool]] = {}
        for i, key in enumerate(self._axis_pos_keys):
            key_to_axis[key] = (i, True)
        for i, key in enumerate(self._axis_neg_keys):
            key_to_axis[key] = (i, False)

        # Live state
        self._marking = np.asarray(
            artifact.initial_state.marking, dtype=np.float64
        ).copy()
        injections = artifact.initial_state.place_injections
        self._inj_sources = [inj.source for inj in injections]
        self._inj_count = len(self._inj_sources)
        self._inj_place_ids = np.asarray(
            [inj.place_id for inj in injections], dtype=np.int64
        )
        self._inj_scales = np.asarray(
            [inj.scale for inj in injections], dtype=np.float64
        )
        self._inj_offsets = np.asarray(
            [inj.offset for inj in injections], dtype=np.float64
        )
        self._inj_clamp_mask = np.asarray(
            [bool(inj.clamp_0_1) for inj in injections], dtype=np.bool_
        )
        self._inj_clamp_idx = np.flatnonzero(self._inj_clamp_mask)
        self._inj_has_clamp = bool(self._inj_clamp_idx.size)
        self._inj_source_axis_idx = np.full(self._inj_count, -1, dtype=np.int64)
        self._inj_source_axis_pos = np.zeros(self._inj_count, dtype=np.bool_)
        self._tmp_inj_values = np.zeros(self._inj_count, dtype=np.float64)
        passthrough_pairs: list[Tuple[int, str]] = []
        for i, src in enumerate(self._inj_sources):
            axis_info = key_to_axis.get(src)
            if axis_info is not None:
                axis_idx, is_pos = axis_info
                self._inj_source_axis_idx[i] = int(axis_idx)
                self._inj_source_axis_pos[i] = bool(is_pos)
            else:
                passthrough_pairs.append((i, src))
        self._inj_passthrough_pairs = passthrough_pairs

        self._action_names = [a.name for a in artifact.readout.actions]
        self._action_pos_idx = np.asarray(
            [a.pos_place for a in artifact.readout.actions], dtype=np.int64
        )
        self._action_neg_idx = np.asarray(
            [a.neg_place for a in artifact.readout.actions], dtype=np.int64
        )
        self._action_gains = np.asarray(artifact.readout.gains, dtype=np.float64)
        self._action_abs_max = np.asarray(artifact.readout.abs_max, dtype=np.float64)
        self._action_slew_per_s = np.asarray(
            artifact.readout.slew_per_s, dtype=np.float64
        )
        self._dt = float(artifact.meta.dt_control_s)
        self._prev_actions = np.zeros(len(self._action_names), dtype=np.float64)
        self.last_oracle_firing: List[float] = []
        self.last_sc_firing: List[float] = []
        self.last_oracle_marking: List[float] = self._marking.tolist()
        self.last_sc_marking: List[float] = self._marking.tolist()

    # ── Public API ───────────────────────────────────────────────────────

    def reset(self) -> None:
        """Restore initial marking and zero previous actions."""
        np.copyto(
            self._marking,
            np.asarray(self.artifact.initial_state.marking, dtype=np.float64),
        )
        self._prev_actions.fill(0.0)
        self._oracle_pending.fill(0.0)
        self._sc_pending.fill(0.0)
        self._oracle_cursor = 0
        self._sc_cursor = 0
        self.last_oracle_firing = []
        self.last_sc_firing = []
        self.last_oracle_marking = (
            self._marking.tolist() if self._enable_oracle_diagnostics else []
        )
        self.last_sc_marking = self._marking.tolist()

    @property
    def runtime_backend_name(self) -> str:
        return self._runtime_backend

    @property
    def marking(self) -> List[float]:
        return cast(List[float], self._marking.tolist())

    @marking.setter
    def marking(self, values: Sequence[float]) -> None:
        arr = np.asarray(list(values), dtype=np.float64)
        if arr.shape != (self._nP,):
            raise ValueError(f"marking must have length {self._nP}, got {arr.size}")
        self._marking = np.clip(arr, 0.0, 1.0)

    def step(
        self,
        obs: Mapping[str, float],
        k: int,
        log_path: Optional[str] = None,
    ) -> ControlAction:
        """Execute one control tick.

        Steps:
            1. ``extract_features(obs)`` → 4 unipolar features
            2. ``_inject_places(features)``
            3. ``_oracle_step()`` — float path (optional)
            4. ``_sc_step(k)`` — deterministic stochastic path
            5. ``_decode_actions()`` — gain × differencing, slew + abs clamp
            6. Optional JSONL logging
        """
        t0 = time.perf_counter()

        # 1. Feature extraction (fast compiled mapping)
        pos_vals, neg_vals = self._compute_feature_components(obs)
        feats = (
            self._build_feature_dict(obs, pos_vals, neg_vals)
            if log_path is not None
            else None
        )

        # 2. Inject features into marking
        m = self._tmp_marking_input
        np.copyto(m, self._marking)
        self._inject_places(m, obs, pos_vals, neg_vals)

        # 3. Oracle float path (optional)
        if self._enable_oracle_diagnostics:
            f_oracle, m_oracle = self._oracle_step(m)
        else:
            f_oracle = np.asarray([], dtype=np.float64)
            m_oracle = np.asarray([], dtype=np.float64)

        # 4. Stochastic path
        f_sc, m_sc = self._sc_step(m, k)

        # Diagnostics (used by deterministic benchmark gates)
        self.last_oracle_firing = f_oracle.tolist()
        self.last_sc_firing = f_sc.tolist()
        self.last_oracle_marking = (
            m_oracle.tolist() if self._enable_oracle_diagnostics else []
        )
        self.last_sc_marking = m_sc.tolist()

        # Commit SC state
        np.copyto(self._marking, m_sc)

        # 5. Decode actions
        actions_dict = self._decode_actions(m_sc)

        t1 = time.perf_counter()

        # 6. Optional JSONL logging
        if log_path is not None:
            rec = {
                "k": int(k),
                "obs": dict(obs),
                "features": feats,
                "f_oracle": f_oracle.tolist(),
                "f_sc": f_sc.tolist(),
                "marking": m_sc.tolist(),
                "actions": actions_dict,
                "timing_ms": (t1 - t0) * 1000.0,
            }
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")

        # Build typed result
        result: ControlAction = {
            "dI_PF3_A": actions_dict.get("dI_PF3_A", 0.0),
            "dI_PF_topbot_A": actions_dict.get("dI_PF_topbot_A", 0.0),
        }
        return result

    # ── Internal ─────────────────────────────────────────────────────────

    def _compute_feature_components(
        self, obs_map: Mapping[str, float]
    ) -> Tuple[FloatArray, FloatArray]:
        if self._axis_count == 0:
            return self._empty, self._empty

        obs_vals = self._tmp_obs_vals
        for i, key in enumerate(self._axis_obs_keys):
            if key not in obs_map:
                raise KeyError(
                    f"Missing observation key for feature extraction: {key}"
                )
            obs_vals[i] = float(obs_map[key])
        np.subtract(self._axis_targets, obs_vals, out=self._tmp_feature_err)
        np.divide(self._tmp_feature_err, self._axis_scales, out=self._tmp_feature_err)
        np.clip(self._tmp_feature_err, -1.0, 1.0, out=self._tmp_feature_err)
        np.clip(self._tmp_feature_err, 0.0, 1.0, out=self._tmp_feature_pos)
        np.negative(self._tmp_feature_err, out=self._tmp_feature_neg)
        np.clip(self._tmp_feature_neg, 0.0, 1.0, out=self._tmp_feature_neg)
        return self._tmp_feature_pos, self._tmp_feature_neg

    def _build_feature_dict(
        self, obs_map: Mapping[str, float], pos_vals: FloatArray, neg_vals: FloatArray
    ) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        for i, key in enumerate(self._axis_pos_keys):
            feats[key] = float(pos_vals[i])
        for i, key in enumerate(self._axis_neg_keys):
            feats[key] = float(neg_vals[i])
        for key in self._passthrough_sources:
            if key not in obs_map:
                raise KeyError(f"Missing observation key for passthrough: {key}")
            feats[key] = float(np.clip(obs_map[key], 0.0, 1.0))
        return feats

    def _inject_places(
        self,
        marking: FloatArray,
        obs_map: Mapping[str, float],
        pos_vals: FloatArray,
        neg_vals: FloatArray,
    ) -> None:
        """Write features into a marking vector via place_injections config."""
        if self._inj_count == 0:
            return

        values = self._tmp_inj_values
        values.fill(0.0)
        axis_mask = self._inj_source_axis_idx >= 0
        if np.any(axis_mask):
            axis_indices = self._inj_source_axis_idx[axis_mask]
            axis_pos = self._inj_source_axis_pos[axis_mask]
            values[axis_mask] = np.where(
                axis_pos,
                pos_vals[axis_indices],
                neg_vals[axis_indices],
            )
        for idx, key in self._inj_passthrough_pairs:
            if key not in obs_map:
                raise KeyError(f"Missing observation key for passthrough: {key}")
            values[idx] = float(np.clip(obs_map[key], 0.0, 1.0))

        np.multiply(values, self._inj_scales, out=values)
        values += self._inj_offsets
        if self._inj_has_clamp:
            values[self._inj_clamp_idx] = np.clip(
                values[self._inj_clamp_idx], 0.0, 1.0
            )
        marking[self._inj_place_ids] = values

    def _oracle_step(self, marking: FloatArray) -> Tuple[FloatArray, FloatArray]:
        """Float-path Petri step.

        Returns (firing_vector, next_marking).
        """
        # Activation: a = W_in @ m
        a = self._dense_activations(marking)

        # Firing decision
        if self._firing_mode == "fractional":
            margins = np.maximum(self._margins, 1e-12)
            f = np.clip((a - self._thresholds) / margins, 0.0, 1.0)
        else:
            f = (a >= self._thresholds).astype(np.float64)

        f_timed, self._oracle_cursor = self._apply_transition_timing(
            f, self._oracle_pending, self._oracle_cursor
        )

        # Marking update: m' = clip(m - W_in^T @ f + W_out @ f, 0, 1)
        m2 = self._marking_update(marking, f_timed, self._tmp_marking_oracle)

        return f_timed, m2

    def _sc_step(self, marking: FloatArray, k: int) -> Tuple[FloatArray, FloatArray]:
        """Deterministic stochastic path with optional bit-flip fault injection."""
        a = self._dense_activations(marking)

        if self._firing_mode == "fractional":
            margins = np.maximum(self._margins, 1e-12)
            p_fire = np.clip((a - self._thresholds) / margins, 0.0, 1.0)
        else:
            if self._sc_binary_margin > 0.0:
                # Optional smooth probability around threshold for binary mode.
                p_fire = np.clip(
                    0.5 + 0.5 * ((a - self._thresholds) / self._sc_binary_margin),
                    0.0,
                    1.0,
                )
            else:
                # Binary mode keeps exact threshold semantics for stability.
                p_fire = (a >= self._thresholds).astype(np.float64)

        if self._sc_n_passes <= 1 or (
            self._firing_mode == "binary" and self._sc_binary_margin <= 0.0
        ):
            f = p_fire
            rng = None
        else:
            sample_seed = _seed64(self.seed_base, f"sc_step:{int(k)}")
            if (
                self._runtime_backend == "rust"
                and _HAS_RUST_SCPN_RUNTIME
                and _rust_sample_firing is not None
            ):
                sampled = _rust_sample_firing(
                    p_fire,
                    int(self._sc_n_passes),
                    int(sample_seed),
                    bool(self._sc_antithetic),
                )
                f = np.asarray(sampled, dtype=np.float64)
                if self._sc_bitflip_rate > 0.0:
                    rng = np.random.default_rng(
                        _seed64(self.seed_base, f"sc_flip:{int(k)}")
                    )
                else:
                    rng = None
            else:
                rng = np.random.default_rng(sample_seed)
                if self._sc_antithetic and self._sc_n_passes >= 2:
                    n_pairs = (self._sc_n_passes + 1) // 2
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
                    counts = low_hits + high_hits
                else:
                    counts = np.asarray(
                        rng.binomial(self._sc_n_passes, p_fire), dtype=np.int64
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
            assert rng is not None
            m2 = self._apply_bit_flip_faults(m2, rng)

        return f_timed, m2

    def _apply_transition_timing(
        self,
        desired_firing: FloatArray,
        pending: FloatArray,
        cursor: int,
    ) -> Tuple[FloatArray, int]:
        desired = np.asarray(np.clip(desired_firing, 0.0, 1.0), dtype=np.float64)
        if self._max_delay_ticks <= 0:
            return desired, cursor

        fired_now = np.asarray(pending[cursor], dtype=np.float64).copy()
        pending[cursor, :] = 0.0

        immediate_mask = self._delay_ticks == 0
        if np.any(immediate_mask):
            fired_now[immediate_mask] = np.clip(
                fired_now[immediate_mask] + desired[immediate_mask], 0.0, 1.0
            )

        delayed_idx = np.flatnonzero(self._delay_ticks > 0)
        if delayed_idx.size:
            slots = (cursor + self._delay_ticks[delayed_idx]) % pending.shape[0]
            pending[slots, delayed_idx] = np.clip(
                pending[slots, delayed_idx] + desired[delayed_idx], 0.0, 1.0
            )

        next_cursor = (cursor + 1) % pending.shape[0]
        return fired_now, next_cursor

    def _dense_activations(self, marking: FloatArray) -> FloatArray:
        if self._runtime_backend == "rust" and _HAS_RUST_SCPN_RUNTIME:
            assert _rust_dense_activations is not None
            out = _rust_dense_activations(self._W_in, marking)
            return np.asarray(out, dtype=np.float64)
        self._tmp_activations[:] = self._W_in @ marking
        return self._tmp_activations

    def _marking_update(
        self, marking: FloatArray, firing: FloatArray, out: FloatArray
    ) -> FloatArray:
        if self._runtime_backend == "rust" and _HAS_RUST_SCPN_RUNTIME:
            assert _rust_marking_update is not None
            rust_out = _rust_marking_update(marking, self._W_in, self._W_out, firing)
            np.copyto(out, np.asarray(rust_out, dtype=np.float64))
            return out
        self._tmp_consumption[:] = self._W_in_t @ firing
        self._tmp_production[:] = self._W_out @ firing
        out[:] = marking
        out -= self._tmp_consumption
        out += self._tmp_production
        np.clip(out, 0.0, 1.0, out=out)
        return out

    def _decode_actions(self, marking: FloatArray) -> Dict[str, float]:
        if not self._action_names:
            return {}

        raw = (
            (marking[self._action_pos_idx] - marking[self._action_neg_idx])
            * self._action_gains
        )
        max_delta = self._action_slew_per_s * self._dt
        raw = np.clip(raw, self._prev_actions - max_delta, self._prev_actions + max_delta)
        raw = np.clip(raw, -self._action_abs_max, self._action_abs_max)
        self._prev_actions = np.asarray(raw, dtype=np.float64)
        return {
            name: float(self._prev_actions[i]) for i, name in enumerate(self._action_names)
        }

    def _apply_bit_flip_faults(
        self, values: FloatArray, rng: np.random.Generator
    ) -> FloatArray:
        """Inject bounded deterministic bit-flip faults into float vectors."""
        out = np.asarray(values, dtype=np.float64).copy()
        if self._sc_bitflip_rate <= 0.0 or out.size == 0:
            return out

        flips = rng.random(out.size) < self._sc_bitflip_rate
        if not np.any(flips):
            return out

        raw = out.view(np.uint64)
        flip_idx = np.flatnonzero(flips)
        bits = rng.integers(0, 52, size=flip_idx.size, dtype=np.uint64)
        masks = np.left_shift(np.uint64(1), bits)
        raw[flip_idx] ^= masks

        out = raw.view(np.float64)
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(out, 0.0, 1.0)
