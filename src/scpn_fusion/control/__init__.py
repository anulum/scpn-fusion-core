# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Control Module
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from .fusion_sota_mpc import run_sota_simulation, ModelPredictiveController
from .fusion_nmpc_jax import get_nmpc_controller, NonlinearMPC
