# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Studio federation surface
"""The FUSION studio's federation surface on the SCPN-STUDIO platform contract.

Exposes the schema-A capability manifest (verbs, evidence schemas, content digest)
that the Hub ingests for federation. See :mod:`scpn_fusion.studio.manifest` and
:mod:`scpn_fusion.studio.verbs`.
"""

from __future__ import annotations

from .exactness import (
    ComparisonResult,
    ExactnessClass,
    ReproVerdict,
    canonical_value_digest,
    compare_bit_exact,
    compare_tolerance,
    parse_exactness_class,
    reproduce,
)
from .federation import (
    build_architecture_map_extension,
    build_federation_document,
    write_federation_document,
)
from .manifest import build_manifest, declared_surface
from .verbs import FUSION_VERBS, STUDIO_ID, evidence_schemas

__all__ = [
    "FUSION_VERBS",
    "STUDIO_ID",
    "ComparisonResult",
    "ExactnessClass",
    "ReproVerdict",
    "build_architecture_map_extension",
    "build_federation_document",
    "build_manifest",
    "canonical_value_digest",
    "compare_bit_exact",
    "compare_tolerance",
    "declared_surface",
    "evidence_schemas",
    "parse_exactness_class",
    "reproduce",
    "write_federation_document",
]
