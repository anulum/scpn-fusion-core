# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests
from scpn_fusion.exceptions import FusionCoreError


def test_fusion_core_error_is_exception():
    assert issubclass(FusionCoreError, Exception)


def test_physics_error_inherits_fusion_core_error():
    from scpn_fusion.core.integrated_transport_solver import PhysicsError

    assert issubclass(PhysicsError, FusionCoreError)


def test_artifact_validation_error_inherits_fusion_core_error():
    from scpn_fusion.scpn.artifact import ArtifactValidationError

    assert issubclass(ArtifactValidationError, FusionCoreError)
