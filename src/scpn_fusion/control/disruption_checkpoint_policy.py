# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Disruption Checkpoint Policy Helpers
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from scpn_fusion.control.disruption_risk_runtime import _require_int
from scpn_fusion.fallback_telemetry import record_fallback_event, snapshot_fallback_telemetry

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - optional dependency path
    torch = None

_DISRUPTION_STRICT_NO_FALLBACK_ENV = "SCPN_DISRUPTION_DISABLE_FALLBACK"
_CHECKPOINT_SHA256_ALLOWLIST_ENV = "SCPN_DISRUPTION_CHECKPOINT_SHA256_ALLOWLIST"
_MAX_CHECKPOINT_PARAMETER_COUNT = 5_000_000
_MAX_CHECKPOINT_BYTES = 128 * 1024 * 1024
_ALLOWED_CHECKPOINT_SUFFIXES = {".pth", ".pt", ".ckpt"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_model_path(default_model_filename: str) -> Path:
    return _repo_root() / "artifacts" / default_model_filename


def _normalize_seq_len(seq_len: Any) -> int:
    return _require_int("seq_len", seq_len, 8)


def _resolve_allow_fallback(allow_fallback: bool) -> bool:
    """Resolve fallback policy from API argument + environment strict mode."""
    if not bool(allow_fallback):
        return False
    raw = os.getenv(_DISRUPTION_STRICT_NO_FALLBACK_ENV, "")
    return raw.strip().lower() not in {"1", "true", "yes", "on"}


def _augment_with_fallback_telemetry(meta: dict[str, Any]) -> dict[str, Any]:
    out = dict(meta)
    out["fallback_telemetry"] = snapshot_fallback_telemetry()
    return out


def _record_recovery_event(
    reason: str,
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record fallback telemetry via a single choke-point for this module."""
    return record_fallback_event(
        "disruption_predictor",
        reason,
        context=context,
    )


def _prepare_signal_window(signal: Any, seq_len: Any) -> np.ndarray:
    seq_len_i = _normalize_seq_len(seq_len)
    flat = np.asarray(signal, dtype=float).reshape(-1)
    if flat.size >= seq_len_i:
        return flat[:seq_len_i]
    return np.pad(flat, (0, seq_len_i - flat.size), mode="edge")


def _parse_checkpoint_sha256_allowlist() -> set[str]:
    """Parse optional checkpoint SHA256 allowlist from environment."""
    raw = os.getenv(_CHECKPOINT_SHA256_ALLOWLIST_ENV, "")
    items: set[str] = set()
    for token in raw.replace(";", ",").split(","):
        digest = token.strip().lower()
        if not digest:
            continue
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(
                f"{_CHECKPOINT_SHA256_ALLOWLIST_ENV} contains invalid SHA256: {digest!r}"
            )
        items.add(digest)
    return items


def _sha256_file(path: Path) -> str:
    """Return lowercase SHA256 digest for a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _safe_torch_checkpoint_load(path: Path) -> Any:
    """Load checkpoint with safest available torch semantics.

    Enforces ``weights_only=True`` and fails closed when unavailable to avoid
    arbitrary object deserialization paths in library runtime.
    """
    if torch is None:
        raise RuntimeError("Torch is required for checkpoint loading.")
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.strip().lower()
    if suffix not in _ALLOWED_CHECKPOINT_SUFFIXES:
        raise RuntimeError(
            "Checkpoint suffix is not allowed by policy: "
            f"{suffix!r} not in {sorted(_ALLOWED_CHECKPOINT_SUFFIXES)}"
        )
    size = int(path.stat().st_size)
    if size <= 0:
        raise RuntimeError("Checkpoint file is empty.")
    if size > _MAX_CHECKPOINT_BYTES:
        _record_recovery_event(
            "checkpoint_oversize_blocked",
            context={"path": str(path), "size_bytes": size},
        )
        raise RuntimeError(
            "Checkpoint file exceeds safety size budget: "
            f"{size} > {_MAX_CHECKPOINT_BYTES} bytes."
        )
    sha_allowlist = _parse_checkpoint_sha256_allowlist()
    if sha_allowlist:
        digest = _sha256_file(path)
        if digest not in sha_allowlist:
            _record_recovery_event(
                "checkpoint_sha256_allowlist_blocked",
                context={"path": str(path), "sha256": digest},
            )
            raise RuntimeError(
                "Checkpoint SHA256 digest is not allowlisted by policy: "
                f"{digest}"
            )
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        _record_recovery_event(
            "legacy_checkpoint_load_blocked",
            context={"path": str(path)},
        )
        raise RuntimeError(
            "Legacy torch checkpoint loading is disabled because weights_only=True "
            "is unavailable. Use a compatible torch version or convert checkpoint "
            "outside library runtime."
        ) from exc


def _validated_checkpoint_state_dict(raw_state: Any) -> dict[str, Any]:
    """Validate loaded checkpoint payload shape and enforce size budget."""
    if not isinstance(raw_state, dict):
        raise ValueError("checkpoint state_dict must be a mapping.")
    total_params = 0
    for key, value in raw_state.items():
        if not isinstance(key, str):
            raise ValueError("checkpoint state_dict keys must be strings.")
        if torch is not None and hasattr(torch, "Tensor") and isinstance(value, torch.Tensor):
            total_params += int(value.numel())
            continue
        if isinstance(value, np.ndarray):
            total_params += int(value.size)
            continue
        raise ValueError("checkpoint state_dict values must be tensor/ndarray.")
    if total_params <= 0:
        raise ValueError("checkpoint state_dict must contain at least one parameter tensor.")
    if total_params > _MAX_CHECKPOINT_PARAMETER_COUNT:
        raise ValueError(
            "checkpoint state_dict parameter count exceeds safety budget: "
            f"{total_params} > {_MAX_CHECKPOINT_PARAMETER_COUNT}"
        )
    return raw_state


__all__ = [
    "_DISRUPTION_STRICT_NO_FALLBACK_ENV",
    "_CHECKPOINT_SHA256_ALLOWLIST_ENV",
    "_MAX_CHECKPOINT_PARAMETER_COUNT",
    "_MAX_CHECKPOINT_BYTES",
    "default_model_path",
    "_normalize_seq_len",
    "_resolve_allow_fallback",
    "_augment_with_fallback_telemetry",
    "_record_recovery_event",
    "_prepare_signal_window",
    "_safe_torch_checkpoint_load",
    "_validated_checkpoint_state_dict",
]
