# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Bio-Holonomic Controller
"""
Bio-Holonomic Controller.

Integrates SCPN Layer 4 (Cellular/Tissue) and Layer 5 (Organismal/Psychoemotional)
adapters from `sc-neurocore` to drive bio-resonant feedback control. This
demonstrates that the core control architecture is not plasma-bound, but
can ingest physiological telemetry (HRV, EEG) to drive clinical interventions
(like VIBRANA bio-acoustics).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    from sc_neurocore.adapters.holonomic.l4_cell import L4_CellularAdapter, L4_HolonomicParameters
    from sc_neurocore.adapters.holonomic.l5_org import L5_HolonomicParameters, L5_OrganismalAdapter

    SC_NEUROCORE_HOLONOMIC_AVAILABLE = True
except ImportError:
    SC_NEUROCORE_HOLONOMIC_AVAILABLE = False
    L4_CellularAdapter = None  # type: ignore[assignment,misc]
    L5_OrganismalAdapter = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class BioTelemetrySnapshot:
    """Per-tick biological telemetry bundle."""

    heart_rate_bpm: float
    eeg_coherence_r: float
    galvanic_skin_response: float


class BioHolonomicController:
    """
    Biological Feedback Controller mapping L4/L5 states to clinical actions.

    Rather than mitigating plasma disruptions, this controller mitigates
    biological decoherence by tracking autonomic tone and triggering
    resonant acoustic interventions.
    """

    def __init__(self, dt_s: float = 0.01, seed: int = 42) -> None:
        self.dt_s = dt_s
        if not SC_NEUROCORE_HOLONOMIC_AVAILABLE:
            raise RuntimeError(
                "sc-neurocore holonomic adapters are required for BioHolonomicController. "
                "Ensure sc-neurocore is installed and JAX is available."
            )

        # Initialize L4 and L5 adapters natively
        self.l4_adapter = L4_CellularAdapter(L4_HolonomicParameters(), seed=seed)
        self.l5_adapter = L5_OrganismalAdapter(L5_HolonomicParameters(), seed=seed + 1)

    def step(self, telemetry: BioTelemetrySnapshot) -> Dict[str, Any]:
        """
        Advances the bio-controller one tick using incoming telemetry.
        """
        # 1. Modulate L4 (Cellular/Tissue) based on empirical EEG coherence
        # Higher global coherence drives stronger low-level cellular synchronization
        self.l4_adapter.params.k_coupling = 0.3 * (1.0 + telemetry.eeg_coherence_r)

        # Advance L4 Holonomic Dynamics
        l4_bitstreams = self.l4_adapter.step_jax(self.dt_s)
        l4_metrics = self.l4_adapter.get_metrics()

        # 2. Advance L5 (Organismal) using L4 output as bottom-up drive
        # The organism's autonomic tone shifts based on the cellular synchronization
        self.l5_adapter.step_jax(self.dt_s, inputs=l4_bitstreams)
        l5_metrics = self.l5_adapter.get_metrics()

        # 3. Compute Control Actions based on L5 Autonomic/Emotional state
        # For example, if HRV coherence drops, trigger the VIBRANA acoustic array
        hrv_coherence = l5_metrics["hrv_coherence_r5"]
        valence = l5_metrics["emotional_valence"]

        vibrana_intensity = 0.0
        # If the Strange Loop detects a shift toward sympathetic dominance/decoherence:
        if hrv_coherence < 0.4:
            vibrana_intensity = min((0.4 - hrv_coherence) * 2.5, 1.0)

        return {
            "l4_order_parameter": l4_metrics["order_parameter"],
            "l4_avalanche_density": l4_metrics["avalanche_density"],
            "l5_hrv_coherence": hrv_coherence,
            "l5_emotional_valence": valence,
            "actuator_vibrana_intensity": vibrana_intensity,
        }
