# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — UI Package Init
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Streamlit dashboards and UI utilities."""

from .dashboard_launcher import main as launch_dashboard
from .dashboard_generator import DashboardGenerator, run_dashboard

__all__ = ["launch_dashboard", "DashboardGenerator", "run_dashboard"]
