# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — UI Package Init
"""Streamlit dashboards and UI utilities."""

from .dashboard_launcher import main as launch_dashboard
from .dashboard_generator import DashboardGenerator, run_dashboard

__all__ = ["launch_dashboard", "DashboardGenerator", "run_dashboard"]
