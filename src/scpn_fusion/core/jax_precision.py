# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Declared JAX Precision
"""Refuse unavailable precision for explicitly binary64 JAX model surfaces."""

from __future__ import annotations


class JaxPrecisionRefusal(ValueError):
    """Report an unmet model precision requirement before numerical work.

    Parameters
    ----------
    model : str
        Fixed model surface whose declared binary64 operations cannot execute.
    x64_enabled : bool or None, optional
        Actual observed capability flag, or absence of a Boolean observation.

    Attributes
    ----------
    reason_code : str
        Typed capability refusal, without a numerical result or fallback.
    model : str
        Original model surface requiring binary64 support.
    required_dtype : str
        Required real component precision, also used by complex128 models.
    x64_enabled : bool or None
        Actual disabled capability flag, or an unavailable runtime observation.
    """

    reason_code = "unsupported_precision"
    required_dtype = "float64"

    def __init__(self, model: str, *, x64_enabled: bool | None = None) -> None:
        """Retain the fixed model identity without changing runtime settings."""
        super().__init__("Requested JAX model requires enabled float64 support")
        self.model = model
        self.x64_enabled = x64_enabled


def _require_float64(model: str) -> None:
    """Check an explicit binary64 model without changing the application's mode."""
    from jax import config

    enabled = getattr(config, "x64_enabled", None)
    if enabled is not True:
        raise JaxPrecisionRefusal(model, x64_enabled=enabled if type(enabled) is bool else None)
