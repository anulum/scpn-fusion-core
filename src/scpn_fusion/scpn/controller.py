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

from .artifact import Artifact
from .contracts import (
    ActionSpec,
    ControlAction,
    ControlObservation,
    ControlScales,
    ControlTargets,
    _clip01,
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
    ) -> None:
        self.artifact = artifact
        self.seed_base = int(seed_base)
        self.targets = targets
        self.scales = scales

        # Flatten weight matrices for fast indexing
        self._w_in = artifact.weights.w_in.data[:]
        self._w_out = artifact.weights.w_out.data[:]

        # Live state
        self.marking: List[float] = artifact.initial_state.marking[:]
        self._prev_actions: List[float] = [
            0.0 for _ in artifact.readout.actions
        ]

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
            4. ``_sc_step(k)`` — stochastic path (oracle fallback for now)
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

        # Commit SC state (falls back to oracle until Rust kernel exposed)
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
        nT = self.artifact.nT
        nP = self.artifact.nP

        # Activation: a_t = Σ_p W_in[t, p] × m[p]
        a = [0.0] * nT
        for t in range(nT):
            acc = 0.0
            row_off = t * nP
            for p in range(nP):
                acc += self._w_in[row_off + p] * self.marking[p]
            a[t] = acc

        # Firing decision
        f = [0.0] * nT
        for t, tr in enumerate(self.artifact.topology.transitions):
            thr = tr.threshold
            if self.artifact.meta.firing_mode == "fractional":
                margin = (tr.margin or 0.05) or 0.05
                f[t] = _clip01((a[t] - thr) / margin)
            else:
                f[t] = 1.0 if a[t] >= thr else 0.0

        # Marking update: m' = clip(m - W_in^T @ f + W_out @ f, 0, 1)
        m2 = self.marking[:]
        for p in range(nP):
            cons = 0.0
            for t in range(nT):
                cons += self._w_in[t * nP + p] * f[t]
            prod = 0.0
            row_off = p * nT
            for t in range(nT):
                prod += self._w_out[row_off + t] * f[t]
            m2[p] = _clip01(m2[p] - cons + prod)

        return f, m2

    def _sc_step(self, k: int) -> Tuple[List[float], List[float]]:
        """Stochastic path — falls back to oracle until Rust kernel exposed."""
        return self._oracle_step()
