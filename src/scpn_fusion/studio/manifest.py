# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Studio capability manifest (schema A)
"""The FUSION studio's capability manifest (schema A) on the platform contract.

Authors FUSION's :class:`scpn_studio_platform.manifest.CapabilityManifest`: the
verbs it advertises (:mod:`scpn_fusion.studio.verbs`), the evidence schemas they
emit, and a content-addressed digest of that declared surface. The digest is
computed with :func:`scpn_studio_platform.manifest.content_digest` over the
canonical JSON of the declared verbs and evidence schemas, so it is reproducible
across checkouts and independent of git state — the digest tracks the declared
*content*, not the working tree.

FUSION does not yet ship a federated Studio UI panel, so ``ui_module`` is
``None``; the panel is added once the Studio UI binding exists.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version

from scpn_studio_platform.manifest import (
    CapabilityManifest,
    TransportProfile,
    content_digest,
)

from .verbs import FUSION_VERBS, STUDIO_ID, evidence_schemas

PLATFORM_SDK_RANGE = ">=0.10,<0.11"
"""The platform SDK SemVer range the studio builds on (matches the ``studio`` extra)."""

PROTOCOL_VERSION = "1"
"""The SYNAPSE wire protocol version the studio pins."""


def _resolve_studio_version() -> str:
    """Return the installed ``scpn-fusion`` version, or a local sentinel.

    Returns
    -------
    str
        The distribution version when installed, otherwise ``"0+unknown"`` so the
        manifest never carries a fabricated version.
    """
    try:
        return version("scpn-fusion")
    except PackageNotFoundError:  # pragma: no cover - only in a non-installed tree
        return "0+unknown"


STUDIO_VERSION = _resolve_studio_version()
"""The FUSION studio version this manifest stamps (the installed package version)."""


def declared_surface() -> dict[str, bytes]:
    """Return the content-addressable declared surface of the FUSION studio.

    The surface is the canonical JSON of each advertised verb plus the evidence
    schema list, keyed by a stable logical path. Hashing the declared *content*
    (not git state) is what makes the digest reproducible across checkouts.

    Returns
    -------
    dict[str, bytes]
        Mapping of logical path to canonical-JSON bytes, suitable for
        :func:`scpn_studio_platform.manifest.content_digest`.
    """
    surface: dict[str, bytes] = {
        f"verb/{verb.name}": json.dumps(
            verb.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        for verb in FUSION_VERBS
    }
    surface["evidence/schemas"] = json.dumps(
        list(evidence_schemas()), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return surface


def build_manifest(*, studio_version: str = STUDIO_VERSION) -> CapabilityManifest:
    """Build the FUSION studio's capability manifest (schema A).

    Parameters
    ----------
    studio_version
        The studio version to stamp; defaults to :data:`STUDIO_VERSION`.

    Returns
    -------
    CapabilityManifest
        The schema-A manifest, with a content digest over :func:`declared_surface`.
    """
    return CapabilityManifest(
        studio=STUDIO_ID,
        studio_version=studio_version,
        platform_sdk=PLATFORM_SDK_RANGE,
        content_digest=content_digest(declared_surface()),
        protocol_version=PROTOCOL_VERSION,
        transport_profile=TransportProfile.LOCAL_FIRST,
        verbs=FUSION_VERBS,
        evidence_types=evidence_schemas(),
        ui_module=None,
    )
