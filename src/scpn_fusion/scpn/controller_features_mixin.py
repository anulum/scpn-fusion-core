# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Neuro-Symbolic Controller Feature Mixins
"""Feature extraction and action-decoding mixins for NeuroSymbolicController."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class NeuroSymbolicControllerFeaturesMixin:
    def _compute_feature_components(
        self, obs_map: Mapping[str, float]
    ) -> Tuple[FloatArray, FloatArray]:
        if self._axis_count == 0:
            return self._empty, self._empty

        obs_vals = self._tmp_obs_vals
        for i, key in enumerate(self._axis_obs_keys):
            if key not in obs_map:
                raise KeyError(f"Missing observation key for feature extraction: {key}")
            obs_vals[i] = float(obs_map[key])
        return self._compute_feature_components_vector(obs_vals)

    def _compute_feature_components_vector(
        self, obs_vector: Sequence[float] | FloatArray
    ) -> Tuple[FloatArray, FloatArray]:
        if self._axis_count == 0:
            return self._empty, self._empty

        obs_vals = np.asarray(obs_vector, dtype=np.float64)
        if obs_vals.shape != (self._axis_count,):
            raise ValueError(f"obs_vector must have length {self._axis_count}, got {obs_vals.size}")
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
            values[self._inj_clamp_idx] = np.clip(values[self._inj_clamp_idx], 0.0, 1.0)
        marking[self._inj_place_ids] = values

    def _decode_actions(self, marking: FloatArray) -> Dict[str, float]:
        actions = self._decode_actions_vector(marking)
        if actions.size == 0:
            return {}
        return {name: float(actions[i]) for i, name in enumerate(self._action_names)}

    def _decode_actions_vector(self, marking: FloatArray) -> FloatArray:
        if self._action_count == 0:
            return self._empty

        raw = self._tmp_actions
        np.subtract(marking[self._action_pos_idx], marking[self._action_neg_idx], out=raw)
        raw *= self._action_gains
        raw = np.clip(
            raw,
            self._prev_actions - self._action_max_delta,
            self._prev_actions + self._action_max_delta,
        )
        np.clip(raw, -self._action_abs_max, self._action_abs_max, out=self._prev_actions)
        return self._prev_actions

    def _apply_bit_flip_faults(self, values: FloatArray, rng: np.random.Generator) -> FloatArray:
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
