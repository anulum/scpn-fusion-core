# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Root conftest.py — adds src/ to sys.path so that tests can import
scpn_fusion without installing the package in editable mode.

It also disables JAX's default GPU memory pre-allocation. On a small GPU
(e.g. the 6 GB GTX 1060 in the development rig) JAX otherwise grabs ~75% of
device memory on first use; across the full ~4800-test suite the accumulated
reservations starve cuBLAS handle creation on later JAX tests, surfacing as
``XlaRuntimeError: INTERNAL: No BLAS support in stream`` even though each test
passes in isolation. On-demand allocation keeps the GPU available for the
dedicated GPU tests while preventing that starvation cascade. Both settings are
applied with ``setdefault`` so CI or a developer can override them.

The ``float64_jax`` fixture gives one test float64 JAX and restores the previous
setting afterwards. A test that needs float64 requests it; it must not depend on
another collected module having enabled float64 when it was imported.
"""

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


@pytest.fixture
def float64_jax() -> Iterator[None]:
    """Enable float64 JAX for one test, then restore the previous setting."""
    jax = pytest.importorskip("jax")
    config = cast(Any, jax.config)
    previous = bool(config.read("jax_enable_x64"))
    config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        config.update("jax_enable_x64", previous)
