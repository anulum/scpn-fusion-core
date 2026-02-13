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
from typing import Dict, List, Optional, Tuple

import numpy as np

from .artifact import Artifact
from .contracts import (
    ActionSpec,
    ControlAction,
    ControlObservation,
    ControlScales,
    ControlTargets,
    _clip01,
    _seed64,
    decode_actions,
    extract_features,
)


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
    ) -> None:
        self.artifact = artifact
        self.seed_base = int(seed_base)
        self.targets = targets
        self.scales = scales
        self._sc_n_passes = max(int(sc_n_passes), 1)
        self._sc_bitflip_rate = float(np.clip(sc_bitflip_rate, 0.0, 1.0))

        # Flatten weight matrices for fast indexing
        self._w_in = artifact.weights.w_in.data[:]
        self._w_out = artifact.weights.w_out.data[:]
        self._nP = artifact.nP
        self._nT = artifact.nT
        self._W_in = np.asarray(self._w_in, dtype=np.float64).reshape(self._nT, self._nP)
        self._W_out = np.asarray(self._w_out, dtype=np.float64).reshape(self._nP, self._nT)
        self._thresholds = np.asarray(
            [tr.threshold for tr in artifact.topology.transitions], dtype=np.float64
        )
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

        # Live state
        self.marking: List[float] = artifact.initial_state.marking[:]
        self._prev_actions: List[float] = [
            0.0 for _ in artifact.readout.actions
        ]
        self.last_oracle_firing: List[float] = []
        self.last_sc_firing: List[float] = []
        self.last_oracle_marking: List[float] = self.marking[:]
        self.last_sc_marking: List[float] = self.marking[:]

        # Build ActionSpec list for decode_actions
        self._action_specs = [
            ActionSpec(
                name=a.name,
                pos_place=a.pos_place,
                neg_place=a.neg_place,
            )
            for a in artifact.readout.actions
        ]

    # ── Public API ───────────────────────────────────────────────────────

    def reset(self) -> None:
        """Restore initial marking and zero previous actions."""
        self.marking = self.artifact.initial_state.marking[:]
        self._prev_actions = [0.0 for _ in self.artifact.readout.actions]
        self.last_oracle_firing = []
        self.last_sc_firing = []
        self.last_oracle_marking = self.marking[:]
        self.last_sc_marking = self.marking[:]

    def step(
        self,
        obs: ControlObservation,
        k: int,
        log_path: Optional[str] = None,
    ) -> ControlAction:
        """Execute one control tick.

        Steps:
            1. ``extract_features(obs)`` → 4 unipolar features
            2. ``_inject_places(features)``
            3. ``_oracle_step()`` — float path
            4. ``_sc_step(k)`` — deterministic stochastic path
            5. ``_decode_actions()`` — gain × differencing, slew + abs clamp
            6. Optional JSONL logging
        """
        t0 = time.perf_counter()

        # 1. Feature extraction
        feats = extract_features(obs, self.targets, self.scales)

        # 2. Inject features into marking
        self._inject_places(feats)

        # 3. Oracle float path
        f_oracle, m_oracle = self._oracle_step()

        # 4. Stochastic path
        f_sc, m_sc = self._sc_step(k)

        # Diagnostics (used by deterministic benchmark gates)
        self.last_oracle_firing = f_oracle[:]
        self.last_sc_firing = f_sc[:]
        self.last_oracle_marking = m_oracle[:]
        self.last_sc_marking = m_sc[:]

        # Commit SC state
        self.marking = m_sc

        # 5. Decode actions
        actions_dict = decode_actions(
            marking=self.marking,
            actions_spec=self._action_specs,
            gains=self.artifact.readout.gains,
            abs_max=self.artifact.readout.abs_max,
            slew_per_s=self.artifact.readout.slew_per_s,
            dt=self.artifact.meta.dt_control_s,
            prev=self._prev_actions,
        )

        t1 = time.perf_counter()

        # 6. Optional JSONL logging
        if log_path is not None:
            rec = {
                "k": int(k),
                "obs": dict(obs),
                "features": feats,
                "f_oracle": f_oracle,
                "f_sc": f_sc,
                "marking": self.marking,
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

    def _inject_places(self, feats: Dict[str, float]) -> None:
        """Write features into marking via place_injections config."""
        for inj in self.artifact.initial_state.place_injections:
            pid = inj.place_id
            v = feats[inj.source] * inj.scale + inj.offset
            if inj.clamp_0_1:
                v = _clip01(v)
            self.marking[pid] = v

    def _oracle_step(self) -> Tuple[List[float], List[float]]:
        """Float-path Petri step.

        Returns (firing_vector, next_marking).
        """
        m = np.asarray(self.marking, dtype=np.float64)

        # Activation: a = W_in @ m
        a = self._W_in @ m

        # Firing decision
        if self._firing_mode == "fractional":
            margins = np.maximum(self._margins, 1e-12)
            f = np.clip((a - self._thresholds) / margins, 0.0, 1.0)
        else:
            f = (a >= self._thresholds).astype(np.float64)

        # Marking update: m' = clip(m - W_in^T @ f + W_out @ f, 0, 1)
        cons = self._W_in.T @ f
        prod = self._W_out @ f
        m2 = np.clip(m - cons + prod, 0.0, 1.0)

        return f.tolist(), m2.tolist()

    def _sc_step(self, k: int) -> Tuple[List[float], List[float]]:
        """Deterministic stochastic path with optional bit-flip fault injection."""
        m = np.asarray(self.marking, dtype=np.float64)
        a = self._W_in @ m

        if self._firing_mode == "fractional":
            margins = np.maximum(self._margins, 1e-12)
            p_fire = np.clip((a - self._thresholds) / margins, 0.0, 1.0)
        else:
            # Binary mode keeps exact threshold semantics for stability.
            p_fire = (a >= self._thresholds).astype(np.float64)

        if self._firing_mode == "binary" or self._sc_n_passes <= 1:
            f = p_fire
            rng = None
        else:
            rng = np.random.default_rng(_seed64(self.seed_base, f"sc_step:{int(k)}"))
            draws = rng.random((self._sc_n_passes, self._nT)) < p_fire[None, :]
            f = draws.mean(axis=0).astype(np.float64)

        if self._sc_bitflip_rate > 0.0:
            if rng is None:
                rng = np.random.default_rng(_seed64(self.seed_base, f"sc_flip:{int(k)}"))
            f = self._apply_bit_flip_faults(f, rng)

        cons = self._W_in.T @ f
        prod = self._W_out @ f
        m2 = np.clip(m - cons + prod, 0.0, 1.0)
        if self._sc_bitflip_rate > 0.0:
            assert rng is not None
            m2 = self._apply_bit_flip_faults(m2, rng)

        return f.tolist(), m2.tolist()

    def _apply_bit_flip_faults(
        self, values: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Inject bounded deterministic bit-flip faults into float vectors."""
        out = np.asarray(values, dtype=np.float64).copy()
        if self._sc_bitflip_rate <= 0.0 or out.size == 0:
            return out

        flips = rng.random(out.size) < self._sc_bitflip_rate
        if not np.any(flips):
            return out

        raw = out.view(np.uint64)
        for idx in np.flatnonzero(flips):
            bit = int(rng.integers(0, 52))
            raw[idx] = np.uint64(raw[idx] ^ (np.uint64(1) << np.uint64(bit)))

        out = raw.view(np.float64)
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(out, 0.0, 1.0)
