# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Checkpoint and Resume API
"""
Utilities for saving and restoring simulation and campaign states.

Enables resuming long-running stress campaigns and persisting solver
hot-starts across sessions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_checkpoint(
    path: Path | str,
    solver_state: dict[str, Any],
    episode: int,
    metrics: dict[str, Any],
) -> None:
    """Save simulation state to a JSON checkpoint.

    Parameters
    ----------
    path : Path or str
        Destination file path.
    solver_state : dict
        Current state of the physics solvers (e.g. Psi, profiles).
    episode : int
        Current episode counter.
    metrics : dict
        Accumulated performance metrics.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    def _serializable(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serializable(v) for v in obj]
        return obj

    data = {
        "episode": episode,
        "solver_state": _serializable(solver_state),
        "metrics": _serializable(metrics),
        "timestamp": float(np.datetime64("now").astype(float)),  # Simple epoch
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_checkpoint(path: Path | str) -> tuple[dict[str, Any], int, dict[str, Any]]:
    """Restore simulation state from a JSON checkpoint.

    Parameters
    ----------
    path : Path or str
        Source file path.

    Returns
    -------
    tuple
        (solver_state, episode, metrics)
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert lists back to numpy arrays where appropriate
    def _restore(obj: Any) -> Any:
        if isinstance(obj, list):
            # Heuristic: convert lists of numbers to arrays
            if obj and isinstance(obj[0], (int, float)):
                return np.array(obj)
            return [_restore(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _restore(v) for k, v in obj.items()}
        return obj

    solver_state = _restore(data["solver_state"])
    episode = int(data["episode"])
    metrics = _restore(data["metrics"])

    return solver_state, episode, metrics
