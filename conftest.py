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
"""

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
