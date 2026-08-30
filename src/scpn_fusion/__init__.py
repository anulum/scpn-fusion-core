# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
"""Top-level package exports for SCPN Fusion Core."""

from typing import Any

__version__ = "4.0.0"

__all__ = ["setup_fusion_logging", "__version__"]


def __getattr__(name: str) -> Any:
    """Load optional package exports only when callers request them."""
    if name == "setup_fusion_logging":
        from scpn_fusion.io.logging_config import setup_fusion_logging

        globals()[name] = setup_fusion_logging
        return setup_fusion_logging
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
