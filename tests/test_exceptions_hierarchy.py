# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3
# ──────────────────────────────────────────────────────────────────────
from scpn_fusion.exceptions import FusionCoreError


def test_fusion_core_error_is_exception():
    assert issubclass(FusionCoreError, Exception)


def test_physics_error_inherits_fusion_core_error():
    from scpn_fusion.core.integrated_transport_solver import PhysicsError

    assert issubclass(PhysicsError, FusionCoreError)


def test_artifact_validation_error_inherits_fusion_core_error():
    from scpn_fusion.scpn.artifact import ArtifactValidationError

    assert issubclass(ArtifactValidationError, FusionCoreError)
