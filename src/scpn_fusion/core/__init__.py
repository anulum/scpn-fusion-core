# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Core Package Init
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
try:
    from ._rust_compat import FusionKernel, RUST_BACKEND
except ImportError:
    from .fusion_kernel import FusionKernel
    RUST_BACKEND = False
from .fusion_ignition_sim import FusionBurnPhysics
from .equilibrium_3d import FourierMode3D, VMECStyleEquilibrium3D
from .fieldline_3d import FieldLineTrace3D, FieldLineTracer3D, PoincareSection3D
