# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — HPC Package Init
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""High-Performance Computing bridge and utilities."""

from .hpc_bridge import HPCBridge, compile_cpp

__all__ = ["HPCBridge", "compile_cpp"]
