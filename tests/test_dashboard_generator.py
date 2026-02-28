# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Dashboard Generator Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from scpn_fusion.ui.dashboard_generator import DashboardGenerator


class _DummyKernel:
    def __init__(self) -> None:
        self.cfg = {"dimensions": {"R_min": 1.0, "R_max": 5.0, "Z_min": -2.0, "Z_max": 2.0}}
        self.R = np.linspace(1.0, 5.0, 60)
        self.Z = np.linspace(-2.0, 2.0, 60)
        self.NR = self.R.size
        self.NZ = self.Z.size
        self.dR = float(self.R[1] - self.R[0])
        self.dZ = float(self.Z[1] - self.Z[0])
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = (self.RR - 3.0) ** 2 + (self.ZZ / 1.5) ** 2


def test_generate_poincare_plot_traces_fieldline_points() -> None:
    kernel = _DummyKernel()
    fig = DashboardGenerator(kernel).generate_poincare_plot(n_lines=4, n_transits=25)
    try:
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert len(ax.lines) == 4
        for line in ax.lines:
            assert line.get_xdata().size >= 2
            assert line.get_ydata().size >= 2
    finally:
        plt.close(fig)
