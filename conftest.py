# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Pytest Configuration
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
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
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
