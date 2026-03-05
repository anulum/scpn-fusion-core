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
)
from .controller_backend_mixin import NeuroSymbolicControllerBackendMixin
from .controller_features_mixin import NeuroSymbolicControllerFeaturesMixin
from .controller_runtime_backend import probe_rust_runtime_bindings
from scpn_fusion.fallback_telemetry import record_fallback_event

FloatArray = NDArray[np.float64]

_HAS_RUST_SCPN_RUNTIME = False
_rust_dense_activations: Optional[Callable[[FloatArray, FloatArray], object]] = None
_rust_marking_update: Optional[
    Callable[[FloatArray, FloatArray, FloatArray, FloatArray], object]
] = None
_rust_sample_firing: Optional[
    Callable[[FloatArray, int, int, bool], object]
] = None

(
    _HAS_RUST_SCPN_RUNTIME,
    _rust_dense_activations,
    _rust_marking_update,
    _rust_sample_firing,
) = probe_rust_runtime_bindings()


class NeuroSymbolicController(
    NeuroSymbolicControllerFeaturesMixin,
    NeuroSymbolicControllerBackendMixin,
):
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
        rust_backend_min_problem_size: int = 1,
        sc_antithetic_chunk_size: int = 2048,
    ) -> None:
        def _require_int_ge(name: str, value: object, minimum: int) -> int:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer >= {minimum}.")
            parsed = int(value)
            if parsed < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
            return parsed

        self.artifact = artifact
        self.seed_base = int(seed_base)
        self.targets = targets
        self.scales = scales
        self._sc_n_passes = _require_int_ge("sc_n_passes", sc_n_passes, 1)
        self._sc_bitflip_rate = float(sc_bitflip_rate)
        if (
            not np.isfinite(self._sc_bitflip_rate)
            or self._sc_bitflip_rate < 0.0
            or self._sc_bitflip_rate > 1.0
        ):
            raise ValueError("sc_bitflip_rate must be finite and in [0, 1].")
        self._runtime_profile = runtime_profile.strip().lower()
        if self._runtime_profile not in {"adaptive", "deterministic", "traceable"}:
            raise ValueError(
                "runtime_profile must be 'adaptive', 'deterministic', or 'traceable'"
            )
        self._sc_antithetic = bool(sc_antithetic)
        self._enable_oracle_diagnostics = bool(enable_oracle_diagnostics)
        self._feature_axes = list(feature_axes) if feature_axes is not None else None
        self._runtime_backend_request = runtime_backend.strip().lower()
        if self._runtime_backend_request not in {"auto", "numpy", "rust"}:
            raise ValueError("runtime_backend must be 'auto', 'numpy', or 'rust'")
        self._rust_backend_min_problem_size = _require_int_ge(
            "rust_backend_min_problem_size", rust_backend_min_problem_size, 1
        )
        self._sc_antithetic_chunk_size = _require_int_ge(
            "sc_antithetic_chunk_size", sc_antithetic_chunk_size, 1
        )

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
        self._tmp_sc_counts = np.zeros(self._nT, dtype=np.int64)
        self._thresholds = np.asarray(
            [tr.threshold for tr in artifact.topology.transitions], dtype=np.float64
        )
        self._delay_ticks = np.asarray(
            [max(int(getattr(tr, "delay_ticks", 0)), 0) for tr in artifact.topology.transitions],
            dtype=np.int64,
        )
        self._delay_immediate_idx = np.flatnonzero(self._delay_ticks == 0).astype(
            np.int64, copy=False
        )
        self._delay_delayed_idx = np.flatnonzero(self._delay_ticks > 0).astype(
            np.int64, copy=False
        )
        if self._delay_delayed_idx.size:
            self._delay_delayed_offsets = np.asarray(
                self._delay_ticks[self._delay_delayed_idx], dtype=np.int64
            )
            self._tmp_delay_slots = np.zeros(
                self._delay_delayed_idx.size, dtype=np.int64
            )
        else:
            self._delay_delayed_offsets = np.asarray([], dtype=np.int64)
            self._tmp_delay_slots = np.asarray([], dtype=np.int64)
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
            self._sc_binary_margin = float(sc_binary_margin)
            if not np.isfinite(self._sc_binary_margin) or self._sc_binary_margin < 0.0:
                raise ValueError("sc_binary_margin must be finite and >= 0.")

        problem_size = int(self._nP * self._nT)
        rust_eligible = _HAS_RUST_SCPN_RUNTIME and (
            problem_size >= self._rust_backend_min_problem_size
        )
        if self._runtime_backend_request == "numpy":
            self._runtime_backend = "numpy"
        elif self._runtime_backend_request == "rust":
            if _HAS_RUST_SCPN_RUNTIME:
                self._runtime_backend = "rust"
            else:
                self._runtime_backend = "numpy"
                record_fallback_event(
                    "scpn_controller",
                    "rust_backend_unavailable",
                    context={"runtime_backend_request": "rust"},
                )
        else:
            self._runtime_backend = "rust" if rust_eligible else "numpy"
            if self._runtime_backend == "numpy" and not _HAS_RUST_SCPN_RUNTIME:
                record_fallback_event(
                    "scpn_controller",
                    "auto_backend_numpy_due_to_missing_rust",
                    context={"problem_size": int(problem_size)},
                )
        produced_feature_keys = set(self._axis_pos_keys)
        produced_feature_keys.update(self._axis_neg_keys)
        passthrough_sources: list[str] = []
        for inj in self.artifact.initial_state.place_injections:
            src = inj.source
            if src not in produced_feature_keys and src not in passthrough_sources:
                passthrough_sources.append(src)
        self._passthrough_sources = passthrough_sources
        self._traceable_ready = len(self._passthrough_sources) == 0
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
        self._action_count = len(self._action_names)
        self._dt = float(artifact.meta.dt_control_s)
        self._action_max_delta = self._action_slew_per_s * self._dt
        self._prev_actions = np.zeros(self._action_count, dtype=np.float64)
        self._tmp_actions = np.zeros(self._action_count, dtype=np.float64)
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
    def runtime_profile_name(self) -> str:
        return self._runtime_profile

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

        # Preserve all decoded action channels from the artifact readout.
        return cast(ControlAction, dict(actions_dict))

    def step_traceable(
        self,
        obs_vector: Sequence[float],
        k: int,
        log_path: Optional[str] = None,
    ) -> FloatArray:
        """Execute one control tick from a fixed-order observation vector.

        The vector order is ``self._axis_obs_keys``. This avoids per-step key
        lookups/dict allocation in tight control loops.
        """
        if not self._traceable_ready:
            raise RuntimeError(
                "step_traceable requires axis-only injections (no passthrough sources)"
            )

        t0 = time.perf_counter()
        pos_vals, neg_vals = self._compute_feature_components_vector(obs_vector)

        m = self._tmp_marking_input
        np.copyto(m, self._marking)
        self._inject_places(m, {}, pos_vals, neg_vals)

        if self._enable_oracle_diagnostics:
            f_oracle, m_oracle = self._oracle_step(m)
        else:
            f_oracle = np.asarray([], dtype=np.float64)
            m_oracle = np.asarray([], dtype=np.float64)

        f_sc, m_sc = self._sc_step(m, k)

        self.last_oracle_firing = f_oracle.tolist()
        self.last_sc_firing = f_sc.tolist()
        self.last_oracle_marking = (
            m_oracle.tolist() if self._enable_oracle_diagnostics else []
        )
        self.last_sc_marking = m_sc.tolist()

        np.copyto(self._marking, m_sc)
        actions_vec = np.asarray(self._decode_actions_vector(m_sc), dtype=np.float64).copy()

        t1 = time.perf_counter()
        if log_path is not None:
            obs_payload = {
                key: float(value)
                for key, value in zip(self._axis_obs_keys, obs_vector)
            }
            rec = {
                "k": int(k),
                "obs": obs_payload,
                "features": self._build_feature_dict(obs_payload, pos_vals, neg_vals),
                "f_oracle": f_oracle.tolist(),
                "f_sc": f_sc.tolist(),
                "marking": m_sc.tolist(),
                "actions": {
                    name: float(actions_vec[i]) for i, name in enumerate(self._action_names)
                },
                "timing_ms": (t1 - t0) * 1000.0,
            }
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")

        return actions_vec
