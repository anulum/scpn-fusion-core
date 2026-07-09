# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Neuro-Symbolic Runtime Backend Probe
"""Runtime backend probing helpers for optional Rust acceleration."""

from __future__ import annotations

from typing import Callable, Optional, cast

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core import _multi_compat


FloatArray = NDArray[np.float64]


def probe_rust_runtime_bindings() -> tuple[
    bool,
    Optional[Callable[[FloatArray, FloatArray], object]],
    Optional[Callable[[FloatArray, FloatArray, FloatArray, FloatArray], object]],
    Optional[Callable[[FloatArray, int, int, bool], object]],
]:
    """Resolve optional Rust SCPN runtime callables."""
    has_runtime = False
    dense_fn: Optional[Callable[[FloatArray, FloatArray], object]] = None
    update_fn: Optional[Callable[[FloatArray, FloatArray, FloatArray, FloatArray], object]] = None
    sample_fn: Optional[Callable[[FloatArray, int, int, bool], object]] = None
    try:
        dense_fn = cast(
            Callable[[FloatArray, FloatArray], object],
            _multi_compat.dispatch_rust_symbol("scpn_dense_activations"),
        )
        update_fn = cast(
            Callable[[FloatArray, FloatArray, FloatArray, FloatArray], object],
            _multi_compat.dispatch_rust_symbol("scpn_marking_update"),
        )
        sample_fn = cast(
            Callable[[FloatArray, int, int, bool], object],
            _multi_compat.dispatch_rust_symbol("scpn_sample_firing"),
        )
        has_runtime = True
    except (AttributeError, ImportError, RuntimeError, TypeError):
        has_runtime = False
        dense_fn = None
        update_fn = None
        sample_fn = None
    return has_runtime, dense_fn, update_fn, sample_fn


__all__ = ["probe_rust_runtime_bindings", "FloatArray"]
