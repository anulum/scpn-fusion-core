# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TORAX Runtime Public API
"""Stable, TORAX-free caller surface for real TORAX execution."""

from .client import ToraxRuntimeClient
from .contracts import (
    TORAX_OUTCOME_SCHEMA,
    TORAX_REQUEST_SCHEMA,
    ToraxArtifact,
    ToraxClock,
    ToraxConfigBinding,
    ToraxFailureCode,
    ToraxGeometry,
    ToraxProjection,
    ToraxProvenance,
    ToraxRunOutcome,
    ToraxRunRequest,
    ToraxRuntimeError,
    ToraxSignal,
)
from .review import (
    COUPLED_TRANSPORT_SOURCE_SCHEMA,
    MAX_REVIEW_ENVELOPE_BYTES,
    TORAX_REVIEW_SCHEMA,
    ToraxReviewEnvelope,
    build_review_envelope,
    review_envelope_from_bytes,
    review_envelope_sha256,
    review_envelope_to_bytes,
)

__all__ = [
    "TORAX_OUTCOME_SCHEMA",
    "TORAX_REVIEW_SCHEMA",
    "TORAX_REQUEST_SCHEMA",
    "COUPLED_TRANSPORT_SOURCE_SCHEMA",
    "MAX_REVIEW_ENVELOPE_BYTES",
    "ToraxArtifact",
    "ToraxClock",
    "ToraxConfigBinding",
    "ToraxFailureCode",
    "ToraxGeometry",
    "ToraxProjection",
    "ToraxProvenance",
    "ToraxRunOutcome",
    "ToraxRunRequest",
    "ToraxReviewEnvelope",
    "ToraxRuntimeClient",
    "ToraxRuntimeError",
    "ToraxSignal",
    "build_review_envelope",
    "review_envelope_from_bytes",
    "review_envelope_sha256",
    "review_envelope_to_bytes",
]
