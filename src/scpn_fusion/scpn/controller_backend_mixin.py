# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Controller Backend Runtime Mixins
# ──────────────────────────────────────────────────────────────────────
"""Runtime backend helpers extracted from controller monolith."""

from __future__ import annotations

from typing import Any

import numpy as np

from scpn_fusion.fallback_telemetry import record_fallback_event


def _controller_module() -> Any:
    import scpn_fusion.scpn.controller as controller_mod

    return controller_mod


class NeuroSymbolicControllerBackendMixin:
    def _dense_activations(self, marking: np.ndarray) -> np.ndarray:
        controller_mod = _controller_module()
        if self._runtime_backend == "rust" and controller_mod._HAS_RUST_SCPN_RUNTIME:
            try:
                assert controller_mod._rust_dense_activations is not None
                out = controller_mod._rust_dense_activations(self._W_in, marking)
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
                assert controller_mod._rust_marking_update is not None
                rust_out = controller_mod._rust_marking_update(
                    marking, self._W_in, self._W_out, firing
                )
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

