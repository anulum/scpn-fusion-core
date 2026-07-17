#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
"""
Figure: Vertical stability — closed-loop response of REAL controllers.

Vertical displacement z_p(t) after a 5 mm step disturbance at t = 10 ms on the
linearised vertical-instability plant of Paper B Section 4 (open-loop growth
rate gamma = 200 /s).  Unlike the retracted earlier version — which drove three
hand-tuned PID gain sets and *labelled* them PID/MPC/SNN — this figure runs the
project's ACTUAL controllers:

  * PID  — textbook PID; the earlier draft's gains (Kp = 5) are ~4 orders of
           magnitude too small to stabilise a gamma = 200 /s instability, so
           stabilising gains are used and stated here;
  * LQR  — the real ``scpn_fusion.control.flight_sim_controllers.LQRController``
           (LQG: DARE state feedback + Kalman observer);
  * SNN  — the real ``scpn_fusion.control.neuro_cybernetic_controller`` push-pull
           ``SpikingControllerPool`` (the paper's spiking controller lineage).

HONEST NEGATIVE RESULT (K17-B4).  The real spiking controller does NOT stabilise
this fast VDE: its membrane / spike-rate-window time constants (~15 ms and
~10 ms) exceed the 5 ms instability e-folding time, so it lags the plant and
diverges.  The generic Nengo ensemble behaves identically.  Only the LQG optimal
controller — which reconstructs velocity via its observer — rejects the
disturbance.  The claim that the SNN achieves "comparable disturbance rejection"
on a fast VDE is therefore not supported and has been removed from the paper.
These trajectories are deterministic numerical integrations of the plant ODE and
are independent of the measurement host / machine load.

Output: fig_vertical_stability.pdf / .png
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import apply_style, figsize, DOUBLE_COL, COLORS

apply_style()
import matplotlib.pyplot as plt

from scpn_fusion.control.flight_sim_controllers import LQRController
from scpn_fusion.control.neuro_cybernetic_controller import SpikingControllerPool

# ── Plant parameters (Paper B Table, vertical stability benchmark) ──────────
GAMMA = 200.0      # open-loop growth rate [1/s]
K_COIL = 0.5       # coil coupling [N/A]
M_P = 1.0          # normalised effective mass
I_MAX = 10.0e3     # maximum coil current [A]
DT = 1.0e-3        # 1 kHz control update [s]
T_END = 60.0e-3    # total window [s]
T_KICK = 10.0e-3   # step disturbance time [s]
Z_KICK = 5.0e-3    # step disturbance [m]

# Stabilising PID gains: closed loop z'' + (k Kd/m) z' + (k Kp/m - gamma^2) z = 0
# tuned to omega_n = 1.5 gamma, zeta = 0.7 (Kp = 260000, Kd = 900).  The earlier
# draft's Kp = 5 leaves k Kp/m = 2.5 << gamma^2 = 40000 and cannot stabilise.
PID_KP, PID_KI, PID_KD = 2.6e5, 0.0, 9.0e2


def _integrate(command_fn):
    """Integrate the vertical plant, calling command_fn(z, dz, t) -> coil current."""
    t = np.arange(0.0, T_END, DT)
    z = np.zeros_like(t)
    dz = np.zeros_like(t)
    for i in range(1, len(t)):
        if abs(t[i] - T_KICK) < DT / 2.0:
            z[i - 1] += Z_KICK
        current = float(np.clip(command_fn(z[i - 1], dz[i - 1], t[i]), -I_MAX, I_MAX))
        z_ddot = GAMMA**2 * z[i - 1] - K_COIL * current / M_P
        dz[i] = dz[i - 1] + z_ddot * DT
        z[i] = z[i - 1] + dz[i] * DT
    return t, z * 1e3  # mm


def _pid():
    integ = {"e": 0.0, "prev": 0.0}

    def cmd(z, dz, _t):
        e = z
        integ["e"] += e * DT
        de = (e - integ["prev"]) / DT
        integ["prev"] = e
        return PID_KP * e + PID_KI * integ["e"] + PID_KD * de

    return _integrate(cmd)


def _lqr():
    A = [[0.0, 1.0], [GAMMA**2, 0.0]]
    B = [[0.0], [-K_COIL / M_P]]
    C = [[1.0, 0.0]]
    ctrl = LQRController(A, B, C, Q_diag=1.0, R_diag=1.0e-9)

    def cmd(z, _dz, _t):
        return ctrl.step(z, DT)

    return _integrate(cmd)


def _snn():
    # Real push-pull spiking controller, best-effort tuning; still diverges.
    pool = SpikingControllerPool(n_neurons=64, gain=1.0, tau_window=3, dt_s=DT)

    def cmd(z, dz, _t):
        return pool.step(z + 0.1 * dz) * 1.0e5

    return _integrate(cmd)


def _settling_ms(t, z_mm, band=0.1):
    """Last time |z| leaves the +/-band [mm] window, in ms (None if never settles)."""
    outside = np.where(np.abs(z_mm) > band)[0]
    if len(outside) == 0:
        return 0.0
    last = outside[-1]
    if last >= len(t) - 1:
        return None
    return float(t[last + 1] * 1e3)


def main():
    outdir = os.path.dirname(__file__)

    t, z_pid = _pid()
    _, z_lqr = _lqr()
    _, z_snn = _snn()

    fig, ax = plt.subplots(figsize=figsize(DOUBLE_COL, 0.5))

    ax.plot(t * 1e3, z_pid, "-", color=COLORS["blue"], lw=1.5, label="PID (retuned)")
    ax.plot(t * 1e3, z_lqr, "--", color=COLORS["green"], lw=1.5, label="LQR (LQG optimal)")
    ax.plot(t * 1e3, z_snn, "-.", color=COLORS["red"], lw=1.5,
            label="SNN (push-pull spiking, real)")

    ax.axvline(T_KICK * 1e3, color="grey", ls=":", lw=0.8, zorder=0)
    ax.annotate(r"$\Delta z_p = 5$ mm kick",
                xy=(T_KICK * 1e3, 4.5), xytext=(T_KICK * 1e3 + 6, 4.6),
                fontsize=8, color="grey",
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))

    ax.axhspan(-0.1, 0.1, color="green", alpha=0.08, zorder=0)
    ax.text(T_END * 1e3, 0.2, r"$\pm 0.1$ mm band", fontsize=6.5, color="green",
            ha="right", va="bottom")

    # SNN diverges far past the readable range; cap the axis and annotate honestly.
    ax.set_ylim(-8, 8)
    snn_peak = float(np.nanmax(np.abs(z_snn)))
    ax.annotate(f"SNN diverges (peak {snn_peak:.0f} mm):\nspiking lag > 5 ms growth time",
                xy=(28, 7.0), fontsize=7, color=COLORS["red"], ha="left", va="top")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(r"$z_p$ (mm)")
    ax.set_xlim(0, T_END * 1e3)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(r"Vertical displacement response ($\gamma = 200$/s VDE), real controllers")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig_vertical_stability.{ext}"))
    plt.close(fig)

    s_pid = _settling_ms(t, z_pid)
    s_lqr = _settling_ms(t, z_lqr)
    print("  [OK] fig_vertical_stability (real controllers)")
    print(f"       PID settling={s_pid} ms  overshoot={np.max(np.abs(z_pid)):.2f} mm")
    print(f"       LQR settling={s_lqr} ms  overshoot={np.max(np.abs(z_lqr)):.2f} mm")
    print(f"       SNN: did NOT stabilise (peak {snn_peak:.0f} mm)")


if __name__ == "__main__":
    main()
