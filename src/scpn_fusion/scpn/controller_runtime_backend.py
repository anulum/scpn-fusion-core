# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Neuro-Symbolic Runtime Backend Probe
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray


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
        from scpn_fusion_rs import (  # type: ignore[import-not-found,unused-ignore]
            scpn_dense_activations as _dense_impl,
            scpn_marking_update as _update_impl,
            scpn_sample_firing as _sample_impl,
        )

        dense_fn = _dense_impl
        update_fn = _update_impl
        sample_fn = _sample_impl
        has_runtime = True
    except Exception:
        has_runtime = False
    return has_runtime, dense_fn, update_fn, sample_fn


__all__ = ["probe_rust_runtime_bindings", "FloatArray"]
