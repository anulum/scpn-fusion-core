# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — IO Package Init
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Data-interop adapters (IMAS/IDS and related exchange helpers)."""

from .imas_connector import (
    ids_to_digital_twin_summary,
    validate_ids_payload,
    digital_twin_summary_to_ids,
)
