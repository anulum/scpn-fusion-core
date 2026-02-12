try:
    from ._rust_compat import FusionKernel, RUST_BACKEND
except ImportError:
    from .fusion_kernel import FusionKernel
    RUST_BACKEND = False
from .fusion_ignition_sim import FusionBurnPhysics
