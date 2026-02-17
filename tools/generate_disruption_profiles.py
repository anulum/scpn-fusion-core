# -------------------------------------------------------------------------
# SCPN Fusion Core -- Synthetic DIII-D Disruption Shot Generator
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# -------------------------------------------------------------------------
"""Generate 10 synthetic DIII-D NPZ shot files (5 disruptions + 5 safe).

Each file contains 1000-point time-series at float64 spanning 0-3.0 s
with physically motivated disruption or safe-operation profiles.

Disruption physics references:
    - Locked mode growth via modified Rutherford equation
      (de Vries et al., Nucl. Fusion 51 053018, 2011)
    - Greenwald density limit (Greenwald, PPCF 44, R27, 2002)
    - Vertical displacement events (Strait, Nucl. Fusion 46, S649, 2006)
    - Tearing mode island growth (La Haye, Phys. Plasmas 13, 055501, 2006)
    - Troyon beta limit (Troyon et al., PPCF 26, 209, 1984)
    - Safe H-mode / hybrid / advanced scenarios (ITER Physics Basis, 1999)

Seeding: each shot uses np.random.default_rng(shot_number) for full
reproducibility.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
N_STEPS = 1000
T_END = 3.0  # seconds
TIME = np.linspace(0.0, T_END, N_STEPS, dtype=np.float64)
DT = TIME[1] - TIME[0]  # ~0.003003 s

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "validation" / "reference_data" / "diiid" / "disruption_shots"
)

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _add_noise(
    arr: NDArray[np.float64],
    frac: float,
    *,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Add Gaussian noise at *frac* fraction of abs-mean (1-3 % typical)."""
    scale = max(np.mean(np.abs(arr)), 1e-12) * frac
    return arr + rng.normal(0.0, scale, size=arr.shape)


def _smooth(arr: NDArray[np.float64], window: int = 7) -> NDArray[np.float64]:
    """Simple moving-average smoother."""
    if window < 2:
        return arr
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(arr, kernel, mode="same")


def _time_to_idx(t: float) -> int:
    """Convert a time in seconds to the nearest index in TIME."""
    return int(np.argmin(np.abs(TIME - t)))


def _exp_decay(arr: NDArray[np.float64], start_idx: int, tau: float) -> NDArray[np.float64]:
    """Exponential decay from *start_idx* with time-constant *tau* (seconds)."""
    out = arr.copy()
    val0 = arr[start_idx]
    for i in range(start_idx, N_STEPS):
        out[i] = val0 * np.exp(-(TIME[i] - TIME[start_idx]) / tau)
    return out


def _save_shot(path: Path, **data: Any) -> None:
    """Save a single shot to a compressed NPZ file."""
    np.savez_compressed(
        str(path),
        time_s=data["time_s"].astype(np.float64),
        Ip_MA=data["Ip_MA"].astype(np.float64),
        BT_T=data["BT_T"].astype(np.float64),
        beta_N=data["beta_N"].astype(np.float64),
        q95=data["q95"].astype(np.float64),
        ne_1e19=data["ne_1e19"].astype(np.float64),
        n1_amp=data["n1_amp"].astype(np.float64),
        n2_amp=data["n2_amp"].astype(np.float64),
        locked_mode_amp=data["locked_mode_amp"].astype(np.float64),
        dBdt_gauss_per_s=data["dBdt_gauss_per_s"].astype(np.float64),
        vertical_position_m=data["vertical_position_m"].astype(np.float64),
        is_disruption=np.bool_(data["is_disruption"]),
        disruption_time_idx=np.int64(data["disruption_time_idx"]),
        disruption_type=np.str_(data["disruption_type"]),
    )


# -------------------------------------------------------------------------
# Disruption generators
# -------------------------------------------------------------------------


def generate_locked_mode(rng: np.random.Generator) -> dict:
    """Shot 155916 -- Locked mode disruption (~t=2.0 s).

    n=1 MHD mode grows via Rutherford-like saturation, locks to the
    resistive wall when island width exceeds threshold, leading to
    thermal quench + current quench.

    Refs: de Vries et al., NF 51 (2011) 053018;
          Hender et al., NF 47 (2007) S128.
    """
    # -- Flat-top equilibrium values ----------------------------------------
    Ip = np.full(N_STEPS, 1.2, dtype=np.float64)
    BT = np.full(N_STEPS, 2.1, dtype=np.float64)
    q95 = np.full(N_STEPS, 3.5, dtype=np.float64)
    ne = np.full(N_STEPS, 5.5, dtype=np.float64)
    beta_N = np.full(N_STEPS, 2.2, dtype=np.float64)

    # -- n=1 island growth (logistic, seed at t~1.0 s) ---------------------
    onset_idx = _time_to_idx(1.0)
    lock_target_idx = _time_to_idx(1.8)
    disrupt_target = _time_to_idx(2.0)

    n1 = np.full(N_STEPS, 0.02, dtype=np.float64)
    locked = np.zeros(N_STEPS, dtype=np.float64)
    growth_rate = 4.0  # 1/s

    for i in range(onset_idx, N_STEPS):
        dt_s = TIME[i] - TIME[onset_idx]
        # logistic growth:  0.05 -> 0.45 over ~1 s
        n1[i] = 0.05 + 0.40 / (1.0 + np.exp(-growth_rate * (dt_s - 0.6)))

    # Mode locks around t=1.8 s
    for i in range(lock_target_idx, N_STEPS):
        dt_lock = TIME[i] - TIME[lock_target_idx]
        locked[i] = min(0.8, 2.0 * dt_lock)

    # n=2 sideband (smaller)
    n2 = np.clip(n1 * 0.4, 0.01, 0.3)

    # beta_N drops as island grows, collapse at disruption
    for i in range(onset_idx, N_STEPS):
        frac = min((TIME[i] - TIME[onset_idx]) / (TIME[disrupt_target] - TIME[onset_idx] + 1e-9), 1.0)
        beta_N[i] -= 0.3 * frac
    beta_N = _exp_decay(beta_N, disrupt_target, tau=0.01)

    # Ip current quench (10 ms e-fold)
    Ip = _exp_decay(Ip, disrupt_target, tau=0.010)

    # q95 rises as Ip drops
    for i in range(disrupt_target, N_STEPS):
        if Ip[i] > 0.1:
            q95[i] = 3.5 * 1.2 / Ip[i]
        else:
            q95[i] = 50.0
    q95 = np.clip(q95, 1.0, 50.0)

    # dB/dt: quiet until disruption, then spike to ~2500 G/s
    dBdt = np.full(N_STEPS, 20.0, dtype=np.float64)
    for i in range(disrupt_target, min(disrupt_target + 30, N_STEPS)):
        dt_s = TIME[i] - TIME[disrupt_target]
        dBdt[i] += 2500.0 * np.exp(-dt_s / 0.01)

    # vertical position: small drift during CQ
    vert = np.zeros(N_STEPS, dtype=np.float64)
    for i in range(disrupt_target, N_STEPS):
        vert[i] = 0.04 * (1.0 - np.exp(-(TIME[i] - TIME[disrupt_target]) / 0.02))

    # -- Apply 1-3 % noise to all signals ----------------------------------
    return dict(
        time_s=TIME.copy(),
        Ip_MA=np.clip(_add_noise(Ip, 0.01, rng=rng), 0.0, None),
        BT_T=_add_noise(BT, 0.01, rng=rng),
        beta_N=np.clip(_add_noise(beta_N, 0.02, rng=rng), 0.0, None),
        q95=np.clip(_add_noise(q95, 0.01, rng=rng), 1.0, None),
        ne_1e19=np.clip(_add_noise(ne, 0.02, rng=rng), 0.0, None),
        n1_amp=np.clip(_add_noise(n1, 0.02, rng=rng), 0.01, 0.5),
        n2_amp=np.clip(_add_noise(n2, 0.02, rng=rng), 0.01, 0.3),
        locked_mode_amp=np.clip(_add_noise(locked, 0.01, rng=rng), 0.0, 0.8),
        dBdt_gauss_per_s=np.clip(_add_noise(dBdt, 0.02, rng=rng), 0.0, None),
        vertical_position_m=_add_noise(vert, 0.02, rng=rng),
        is_disruption=True,
        disruption_time_idx=int(disrupt_target),
        disruption_type="locked_mode",
    )


def generate_density_limit(rng: np.random.Generator) -> dict:
    """Shot 160409 -- Greenwald density-limit disruption (~t=2.5 s).

    Density ramps above n_GW = Ip/(pi*a^2), triggering a MARFE and
    radiative collapse.

    Refs: Greenwald, PPCF 44 (2002) R27;
          Greenwald et al., NF 28 (1988) 2199.
    """
    Ip_base = 1.0
    Ip = np.full(N_STEPS, Ip_base, dtype=np.float64)
    BT = np.full(N_STEPS, 2.0, dtype=np.float64)
    q95 = np.full(N_STEPS, 3.8, dtype=np.float64)
    beta_N = np.full(N_STEPS, 2.0, dtype=np.float64)

    # Density ramp: 4 -> ~14 over 3 s
    ne = 4.0 + 3.5 * TIME
    ne = np.clip(ne, 3.0, 15.0)

    # Greenwald limit for DIII-D (a~0.67 m): n_GW ~ 7.1e19
    a = 0.67
    n_GW = Ip_base / (np.pi * a ** 2) * 10.0  # ~7.1

    disrupt_target = _time_to_idx(2.5)
    marfe_onset = _time_to_idx(2.2)

    # n=1 grows during radiation collapse
    n1 = np.full(N_STEPS, 0.02, dtype=np.float64)
    for i in range(marfe_onset, N_STEPS):
        frac = min((TIME[i] - TIME[marfe_onset]) / 0.4, 1.0)
        n1[i] = 0.02 + 0.35 * frac ** 2
    n1 = np.clip(n1, 0.01, 0.5)

    n2 = np.clip(n1 * 0.35, 0.01, 0.3)
    locked = np.zeros(N_STEPS, dtype=np.float64)
    for i in range(disrupt_target, N_STEPS):
        locked[i] = min(0.5, 1.5 * (TIME[i] - TIME[disrupt_target]))

    # beta_N collapse
    beta_N = _exp_decay(beta_N, disrupt_target, tau=0.015)

    # Ip current quench
    Ip = _exp_decay(Ip, disrupt_target, tau=0.015)

    # q95 rises
    for i in range(disrupt_target, N_STEPS):
        if Ip[i] > 0.1:
            q95[i] = 3.8 * Ip_base / Ip[i]
        else:
            q95[i] = 50.0
    q95 = np.clip(q95, 1.0, 50.0)

    # dB/dt spike
    dBdt = np.full(N_STEPS, 15.0, dtype=np.float64)
    for i in range(disrupt_target, min(disrupt_target + 30, N_STEPS)):
        dt_s = TIME[i] - TIME[disrupt_target]
        dBdt[i] += 2000.0 * np.exp(-dt_s / 0.012)

    vert = np.zeros(N_STEPS, dtype=np.float64)
    for i in range(disrupt_target, N_STEPS):
        vert[i] = 0.02 * (1.0 - np.exp(-(TIME[i] - TIME[disrupt_target]) / 0.02))

    return dict(
        time_s=TIME.copy(),
        Ip_MA=np.clip(_add_noise(Ip, 0.01, rng=rng), 0.0, None),
        BT_T=_add_noise(BT, 0.01, rng=rng),
        beta_N=np.clip(_add_noise(beta_N, 0.02, rng=rng), 0.0, None),
        q95=np.clip(_add_noise(q95, 0.01, rng=rng), 1.0, None),
        ne_1e19=np.clip(_add_noise(ne, 0.02, rng=rng), 0.0, None),
        n1_amp=np.clip(_add_noise(n1, 0.02, rng=rng), 0.01, 0.5),
        n2_amp=np.clip(_add_noise(n2, 0.02, rng=rng), 0.01, 0.3),
        locked_mode_amp=np.clip(_add_noise(locked, 0.01, rng=rng), 0.0, 0.8),
        dBdt_gauss_per_s=np.clip(_add_noise(dBdt, 0.02, rng=rng), 0.0, None),
        vertical_position_m=_add_noise(vert, 0.02, rng=rng),
        is_disruption=True,
        disruption_time_idx=int(disrupt_target),
        disruption_type="density_limit",
    )


def generate_vde(rng: np.random.Generator) -> dict:
    """Shot 161598 -- Vertical Displacement Event (~t=1.8 s).

    Loss of vertical position control triggers exponential growth of
    the n=0 vertical instability (gamma ~ 100 /s).

    Refs: Strait, NF 46 (2006) S649;
          Humphreys & Kellman, Phys. Plasmas 6 (1999) 2742.
    """
    Ip_base = 1.5
    Ip = np.full(N_STEPS, Ip_base, dtype=np.float64)
    BT = np.full(N_STEPS, 2.1, dtype=np.float64)
    q95 = np.full(N_STEPS, 3.2, dtype=np.float64)
    ne = np.full(N_STEPS, 6.0, dtype=np.float64)
    beta_N = np.full(N_STEPS, 2.3, dtype=np.float64)

    # VDE onset at t~1.2 s, wall contact at t~1.8 s
    vde_onset = _time_to_idx(1.2)
    disrupt_target = _time_to_idx(1.8)

    # Vertical position: exponential growth
    gamma_vde = 5.0  # effective growth rate over the 0.6 s window
    z0 = 0.002
    vert = np.zeros(N_STEPS, dtype=np.float64)
    for i in range(vde_onset, N_STEPS):
        dt_s = TIME[i] - TIME[vde_onset]
        vert[i] = z0 * np.exp(gamma_vde * dt_s)
        if vert[i] > 0.18:
            vert[i] = 0.18  # wall limit

    # n=1 grows during VDE
    n1 = np.full(N_STEPS, 0.02, dtype=np.float64)
    for i in range(vde_onset, N_STEPS):
        frac = min((TIME[i] - TIME[vde_onset]) / (TIME[disrupt_target] - TIME[vde_onset] + 1e-9), 1.5)
        n1[i] = 0.02 + 0.40 * frac ** 1.5
    n1 = np.clip(n1, 0.01, 0.5)

    n2 = np.clip(n1 * 0.3, 0.01, 0.3)

    # Locked mode at wall contact
    locked = np.zeros(N_STEPS, dtype=np.float64)
    for i in range(disrupt_target, N_STEPS):
        locked[i] = min(0.7, 2.0 * (TIME[i] - TIME[disrupt_target]))

    # Current quench (fast, 5 ms)
    Ip = _exp_decay(Ip, disrupt_target, tau=0.005)

    # beta collapse
    beta_N = _exp_decay(beta_N, disrupt_target, tau=0.003)

    # q95
    for i in range(disrupt_target, N_STEPS):
        if Ip[i] > 0.1:
            q95[i] = 3.2 * Ip_base / Ip[i]
        else:
            q95[i] = 50.0
    q95 = np.clip(q95, 1.0, 50.0)

    # dB/dt: large halo current spike
    dBdt = np.full(N_STEPS, 15.0, dtype=np.float64)
    for i in range(disrupt_target, min(disrupt_target + 40, N_STEPS)):
        dt_s = TIME[i] - TIME[disrupt_target]
        dBdt[i] += 3000.0 * np.exp(-dt_s / 0.008)

    return dict(
        time_s=TIME.copy(),
        Ip_MA=np.clip(_add_noise(Ip, 0.01, rng=rng), 0.0, None),
        BT_T=_add_noise(BT, 0.01, rng=rng),
        beta_N=np.clip(_add_noise(beta_N, 0.02, rng=rng), 0.0, None),
        q95=np.clip(_add_noise(q95, 0.01, rng=rng), 1.0, None),
        ne_1e19=np.clip(_add_noise(ne, 0.02, rng=rng), 0.0, None),
        n1_amp=np.clip(_add_noise(n1, 0.02, rng=rng), 0.01, 0.5),
        n2_amp=np.clip(_add_noise(n2, 0.02, rng=rng), 0.01, 0.3),
        locked_mode_amp=np.clip(_add_noise(locked, 0.01, rng=rng), 0.0, 0.8),
        dBdt_gauss_per_s=np.clip(_add_noise(dBdt, 0.02, rng=rng), 0.0, None),
        vertical_position_m=_add_noise(vert, 0.02, rng=rng),
        is_disruption=True,
        disruption_time_idx=int(disrupt_target),
        disruption_type="vde",
    )


def generate_tearing(rng: np.random.Generator) -> dict:
    """Shot 164965 -- 2/1 neoclassical tearing mode disruption (~t=2.2 s).

    Bootstrap-current-driven 2/1 island grows via modified Rutherford
    equation.  When island width exceeds a critical fraction of the
    minor radius, mode overlap triggers disruption.

    Refs: La Haye, Phys. Plasmas 13 (2006) 055501;
          Sauter et al., Phys. Plasmas 4 (1997) 1654.
    """
    Ip_base = 1.3
    Ip = np.full(N_STEPS, Ip_base, dtype=np.float64)
    BT = np.full(N_STEPS, 2.0, dtype=np.float64)
    q95 = np.full(N_STEPS, 3.4, dtype=np.float64)
    ne = np.full(N_STEPS, 5.0, dtype=np.float64)
    beta_N = np.full(N_STEPS, 2.2, dtype=np.float64)

    onset_idx = _time_to_idx(1.0)
    disrupt_target = _time_to_idx(2.2)

    # n=2 dominant (2/1 NTM):  oscillates then grows
    n2 = np.full(N_STEPS, 0.02, dtype=np.float64)
    for i in range(onset_idx, N_STEPS):
        dt_s = TIME[i] - TIME[onset_idx]
        # oscillation + envelope growth
        osc = 0.03 * np.sin(2.0 * np.pi * 8.0 * dt_s)
        envelope = 0.02 + 0.25 / (1.0 + np.exp(-3.5 * (dt_s - 0.8)))
        n2[i] = envelope + osc
    n2 = np.clip(n2, 0.01, 0.3)

    # n=1 secondary, grows later
    n1 = np.full(N_STEPS, 0.02, dtype=np.float64)
    for i in range(_time_to_idx(1.5), N_STEPS):
        frac = min((TIME[i] - 1.5) / 0.8, 1.0)
        n1[i] = 0.02 + 0.30 * frac ** 2
    n1 = np.clip(n1, 0.01, 0.5)

    # Locked mode near disruption
    locked = np.zeros(N_STEPS, dtype=np.float64)
    lock_onset = _time_to_idx(2.0)
    for i in range(lock_onset, N_STEPS):
        locked[i] = min(0.6, 2.0 * (TIME[i] - TIME[lock_onset]))

    # beta_N gradual drop then collapse
    for i in range(onset_idx, disrupt_target):
        frac = (TIME[i] - TIME[onset_idx]) / (TIME[disrupt_target] - TIME[onset_idx] + 1e-9)
        beta_N[i] -= 0.2 * frac
    beta_N = _exp_decay(beta_N, disrupt_target, tau=0.008)

    Ip = _exp_decay(Ip, disrupt_target, tau=0.012)

    for i in range(disrupt_target, N_STEPS):
        if Ip[i] > 0.1:
            q95[i] = 3.4 * Ip_base / Ip[i]
        else:
            q95[i] = 50.0
    q95 = np.clip(q95, 1.0, 50.0)

    dBdt = np.full(N_STEPS, 15.0, dtype=np.float64)
    for i in range(disrupt_target, min(disrupt_target + 30, N_STEPS)):
        dt_s = TIME[i] - TIME[disrupt_target]
        dBdt[i] += 2200.0 * np.exp(-dt_s / 0.010)

    vert = np.zeros(N_STEPS, dtype=np.float64)
    for i in range(disrupt_target, N_STEPS):
        vert[i] = 0.03 * (1.0 - np.exp(-(TIME[i] - TIME[disrupt_target]) / 0.015))

    return dict(
        time_s=TIME.copy(),
        Ip_MA=np.clip(_add_noise(Ip, 0.01, rng=rng), 0.0, None),
        BT_T=_add_noise(BT, 0.01, rng=rng),
        beta_N=np.clip(_add_noise(beta_N, 0.02, rng=rng), 0.0, None),
        q95=np.clip(_add_noise(q95, 0.01, rng=rng), 1.0, None),
        ne_1e19=np.clip(_add_noise(ne, 0.02, rng=rng), 0.0, None),
        n1_amp=np.clip(_add_noise(n1, 0.02, rng=rng), 0.01, 0.5),
        n2_amp=np.clip(_add_noise(n2, 0.02, rng=rng), 0.01, 0.3),
        locked_mode_amp=np.clip(_add_noise(locked, 0.01, rng=rng), 0.0, 0.8),
        dBdt_gauss_per_s=np.clip(_add_noise(dBdt, 0.02, rng=rng), 0.0, None),
        vertical_position_m=_add_noise(vert, 0.02, rng=rng),
        is_disruption=True,
        disruption_time_idx=int(disrupt_target),
        disruption_type="tearing",
    )


def generate_beta_limit(rng: np.random.Generator) -> dict:
    """Shot 166000 -- Beta-limit (Troyon limit) disruption (~t=2.0 s).

    beta_N is ramped above the Troyon limit (g~3.5), driving ideal MHD
    kink/ballooning instabilities and rapid collapse.

    Refs: Troyon et al., PPCF 26 (1984) 209;
          Strait et al., Phys. Plasmas 4 (1997) 1783.
    """
    Ip_base = 1.1
    Ip = np.full(N_STEPS, Ip_base, dtype=np.float64)
    BT = np.full(N_STEPS, 2.0, dtype=np.float64)
    q95 = np.full(N_STEPS, 3.5, dtype=np.float64)
    ne = np.full(N_STEPS, 5.8, dtype=np.float64)

    # beta_N ramp: 1.8 -> ~3.8 over 2.0 s, then collapse
    beta_N = np.where(TIME <= 2.0, 1.8 + 1.0 * TIME, 1.8 + 2.0)
    beta_N = np.clip(beta_N, 0.0, 4.0)

    disrupt_target = _time_to_idx(2.0)

    # n=1 ideal kink grows near beta limit
    n1 = np.full(N_STEPS, 0.02, dtype=np.float64)
    exceed_idx = _time_to_idx(1.6)  # beta_N ~ 3.4 approaching limit
    for i in range(exceed_idx, N_STEPS):
        excess = max(beta_N[i] - 3.2, 0.0)
        n1[i] = 0.02 + 0.6 * excess ** 2
    n1 = np.clip(n1, 0.01, 0.5)

    n2 = np.clip(n1 * 0.4, 0.01, 0.3)

    locked = np.zeros(N_STEPS, dtype=np.float64)
    for i in range(disrupt_target, N_STEPS):
        locked[i] = min(0.5, 1.5 * (TIME[i] - TIME[disrupt_target]))

    # Beta collapse at disruption
    beta_N = _exp_decay(beta_N, disrupt_target, tau=0.005)

    Ip = _exp_decay(Ip, disrupt_target, tau=0.008)

    for i in range(disrupt_target, N_STEPS):
        if Ip[i] > 0.1:
            q95[i] = 3.5 * Ip_base / Ip[i]
        else:
            q95[i] = 50.0
    q95 = np.clip(q95, 1.0, 50.0)

    dBdt = np.full(N_STEPS, 15.0, dtype=np.float64)
    for i in range(disrupt_target, min(disrupt_target + 30, N_STEPS)):
        dt_s = TIME[i] - TIME[disrupt_target]
        dBdt[i] += 2800.0 * np.exp(-dt_s / 0.008)

    vert = np.zeros(N_STEPS, dtype=np.float64)
    for i in range(disrupt_target, N_STEPS):
        vert[i] = 0.04 * (1.0 - np.exp(-(TIME[i] - TIME[disrupt_target]) / 0.015))

    return dict(
        time_s=TIME.copy(),
        Ip_MA=np.clip(_add_noise(Ip, 0.01, rng=rng), 0.0, None),
        BT_T=_add_noise(BT, 0.01, rng=rng),
        beta_N=np.clip(_add_noise(beta_N, 0.02, rng=rng), 0.0, None),
        q95=np.clip(_add_noise(q95, 0.01, rng=rng), 1.0, None),
        ne_1e19=np.clip(_add_noise(ne, 0.02, rng=rng), 0.0, None),
        n1_amp=np.clip(_add_noise(n1, 0.02, rng=rng), 0.01, 0.5),
        n2_amp=np.clip(_add_noise(n2, 0.02, rng=rng), 0.01, 0.3),
        locked_mode_amp=np.clip(_add_noise(locked, 0.01, rng=rng), 0.0, 0.8),
        dBdt_gauss_per_s=np.clip(_add_noise(dBdt, 0.02, rng=rng), 0.0, None),
        vertical_position_m=_add_noise(vert, 0.02, rng=rng),
        is_disruption=True,
        disruption_time_idx=int(disrupt_target),
        disruption_type="beta_limit",
    )


# -------------------------------------------------------------------------
# Safe-shot generators
# -------------------------------------------------------------------------


def generate_hmode_safe(rng: np.random.Generator) -> dict:
    """Shot 163303 -- Standard ELMy H-mode (safe).

    Stationary H-mode with Type-I ELMs, well below all stability limits.

    Refs: ITER Physics Basis, NF 39 (1999) 2137;
          Luce et al., NF 43 (2003) 321.
    """
    Ip = np.full(N_STEPS, 1.6, dtype=np.float64)
    BT = np.full(N_STEPS, 2.1, dtype=np.float64)
    q95 = np.full(N_STEPS, 4.2, dtype=np.float64)
    beta_N = np.full(N_STEPS, 1.9, dtype=np.float64)
    ne = np.full(N_STEPS, 6.5, dtype=np.float64)

    # Type-I ELM bursts on beta_N and density
    elm_freq = 60.0
    for i in range(N_STEPS):
        phase = (TIME[i] * elm_freq) % 1.0
        if phase < 0.04:
            burst = 0.06 * (1.0 - phase / 0.04)
            beta_N[i] += burst * rng.uniform(0.7, 1.3)
            ne[i] -= 0.15 * rng.uniform(0.8, 1.2)

    n1 = np.full(N_STEPS, 0.04, dtype=np.float64)
    n2 = np.full(N_STEPS, 0.02, dtype=np.float64)
    locked = np.full(N_STEPS, 0.005, dtype=np.float64)
    dBdt = np.full(N_STEPS, 15.0, dtype=np.float64)
    vert = np.zeros(N_STEPS, dtype=np.float64)

    return dict(
        time_s=TIME.copy(),
        Ip_MA=np.clip(_add_noise(Ip, 0.01, rng=rng), 0.0, None),
        BT_T=_add_noise(BT, 0.01, rng=rng),
        beta_N=np.clip(_add_noise(beta_N, 0.02, rng=rng), 0.0, None),
        q95=np.clip(_add_noise(q95, 0.01, rng=rng), 1.0, None),
        ne_1e19=np.clip(_add_noise(ne, 0.02, rng=rng), 0.0, None),
        n1_amp=np.clip(_add_noise(n1, 0.03, rng=rng), 0.01, 0.5),
        n2_amp=np.clip(_add_noise(n2, 0.03, rng=rng), 0.01, 0.3),
        locked_mode_amp=np.clip(_add_noise(locked, 0.03, rng=rng), 0.0, 0.8),
        dBdt_gauss_per_s=np.clip(_add_noise(dBdt, 0.02, rng=rng), 0.0, None),
        vertical_position_m=_add_noise(vert, 0.03, rng=rng),
        is_disruption=False,
        disruption_time_idx=-1,
        disruption_type="safe",
    )


def generate_hybrid_safe(rng: np.random.Generator) -> dict:
    """Shot 154406 -- Hybrid scenario (safe).

    Elevated q_min > 1 avoids sawteeth; broad current profile gives
    excellent confinement at moderate beta.

    Refs: Petty et al., PRL 102 (2009) 045005.
    """
    Ip = np.full(N_STEPS, 1.0, dtype=np.float64)
    BT = np.full(N_STEPS, 2.0, dtype=np.float64)
    q95 = np.full(N_STEPS, 4.5, dtype=np.float64)
    beta_N = np.full(N_STEPS, 2.5, dtype=np.float64)
    ne = np.full(N_STEPS, 5.0, dtype=np.float64)

    # Fishbone oscillations
    fishbone = 0.015 * np.sin(2.0 * np.pi * 12.0 * TIME) * rng.uniform(0.8, 1.2, N_STEPS)
    beta_N += fishbone

    n1 = np.full(N_STEPS, 0.03, dtype=np.float64)
    n2 = np.full(N_STEPS, 0.015, dtype=np.float64)
    locked = np.full(N_STEPS, 0.003, dtype=np.float64)
    dBdt = np.full(N_STEPS, 12.0, dtype=np.float64)
    vert = np.zeros(N_STEPS, dtype=np.float64)

    return dict(
        time_s=TIME.copy(),
        Ip_MA=np.clip(_add_noise(Ip, 0.01, rng=rng), 0.0, None),
        BT_T=_add_noise(BT, 0.01, rng=rng),
        beta_N=np.clip(_add_noise(beta_N, 0.02, rng=rng), 0.0, None),
        q95=np.clip(_add_noise(q95, 0.01, rng=rng), 1.0, None),
        ne_1e19=np.clip(_add_noise(ne, 0.02, rng=rng), 0.0, None),
        n1_amp=np.clip(_add_noise(n1, 0.03, rng=rng), 0.01, 0.5),
        n2_amp=np.clip(_add_noise(n2, 0.03, rng=rng), 0.01, 0.3),
        locked_mode_amp=np.clip(_add_noise(locked, 0.03, rng=rng), 0.0, 0.8),
        dBdt_gauss_per_s=np.clip(_add_noise(dBdt, 0.02, rng=rng), 0.0, None),
        vertical_position_m=_add_noise(vert, 0.03, rng=rng),
        is_disruption=False,
        disruption_time_idx=-1,
        disruption_type="safe",
    )


def generate_negdelta_safe(rng: np.random.Generator) -> dict:
    """Shot 175970 -- Negative triangularity (safe).

    Negative triangularity plasmas have L-mode edge with H-mode-level
    core confinement.  Intrinsically ELM-free and low MHD.

    Refs: Austin et al., PRL 122 (2019) 115001;
          Marinoni et al., NF 61 (2021) 116010.
    """
    Ip = np.full(N_STEPS, 0.9, dtype=np.float64)
    BT = np.full(N_STEPS, 2.0, dtype=np.float64)
    q95 = np.full(N_STEPS, 5.0, dtype=np.float64)
    beta_N = np.full(N_STEPS, 1.6, dtype=np.float64)
    ne = np.full(N_STEPS, 4.5, dtype=np.float64)

    # Very quiet -- no ELMs
    n1 = np.full(N_STEPS, 0.02, dtype=np.float64)
    n2 = np.full(N_STEPS, 0.012, dtype=np.float64)
    locked = np.full(N_STEPS, 0.002, dtype=np.float64)
    dBdt = np.full(N_STEPS, 8.0, dtype=np.float64)
    vert = np.zeros(N_STEPS, dtype=np.float64)

    return dict(
        time_s=TIME.copy(),
        Ip_MA=np.clip(_add_noise(Ip, 0.01, rng=rng), 0.0, None),
        BT_T=_add_noise(BT, 0.01, rng=rng),
        beta_N=np.clip(_add_noise(beta_N, 0.01, rng=rng), 0.0, None),
        q95=np.clip(_add_noise(q95, 0.01, rng=rng), 1.0, None),
        ne_1e19=np.clip(_add_noise(ne, 0.01, rng=rng), 0.0, None),
        n1_amp=np.clip(_add_noise(n1, 0.02, rng=rng), 0.01, 0.5),
        n2_amp=np.clip(_add_noise(n2, 0.02, rng=rng), 0.01, 0.3),
        locked_mode_amp=np.clip(_add_noise(locked, 0.02, rng=rng), 0.0, 0.8),
        dBdt_gauss_per_s=np.clip(_add_noise(dBdt, 0.02, rng=rng), 0.0, None),
        vertical_position_m=_add_noise(vert, 0.02, rng=rng),
        is_disruption=False,
        disruption_time_idx=-1,
        disruption_type="safe",
    )


def generate_snowflake_safe(rng: np.random.Generator) -> dict:
    """Shot 166549 -- Snowflake divertor (safe).

    Advanced divertor geometry with secondary X-point; good power
    handling, standard H-mode core.

    Refs: Petrie et al., NF 53 (2013) 113024;
          Soukhanovskii et al., J. Nucl. Mater. 463 (2015) 1191.
    """
    Ip = np.full(N_STEPS, 1.2, dtype=np.float64)
    BT = np.full(N_STEPS, 2.1, dtype=np.float64)
    q95 = np.full(N_STEPS, 3.8, dtype=np.float64)
    beta_N = np.full(N_STEPS, 2.0, dtype=np.float64)
    ne = np.full(N_STEPS, 6.0, dtype=np.float64)

    # Mild ELMs
    elm_freq = 70.0
    for i in range(N_STEPS):
        phase = (TIME[i] * elm_freq) % 1.0
        if phase < 0.04:
            burst = 0.04 * (1.0 - phase / 0.04)
            beta_N[i] += burst * rng.uniform(0.7, 1.3)
            ne[i] -= 0.10 * rng.uniform(0.8, 1.2)

    n1 = np.full(N_STEPS, 0.035, dtype=np.float64)
    n2 = np.full(N_STEPS, 0.015, dtype=np.float64)
    locked = np.full(N_STEPS, 0.004, dtype=np.float64)
    dBdt = np.full(N_STEPS, 12.0, dtype=np.float64)
    vert = np.zeros(N_STEPS, dtype=np.float64)

    return dict(
        time_s=TIME.copy(),
        Ip_MA=np.clip(_add_noise(Ip, 0.01, rng=rng), 0.0, None),
        BT_T=_add_noise(BT, 0.01, rng=rng),
        beta_N=np.clip(_add_noise(beta_N, 0.02, rng=rng), 0.0, None),
        q95=np.clip(_add_noise(q95, 0.01, rng=rng), 1.0, None),
        ne_1e19=np.clip(_add_noise(ne, 0.02, rng=rng), 0.0, None),
        n1_amp=np.clip(_add_noise(n1, 0.03, rng=rng), 0.01, 0.5),
        n2_amp=np.clip(_add_noise(n2, 0.03, rng=rng), 0.01, 0.3),
        locked_mode_amp=np.clip(_add_noise(locked, 0.03, rng=rng), 0.0, 0.8),
        dBdt_gauss_per_s=np.clip(_add_noise(dBdt, 0.02, rng=rng), 0.0, None),
        vertical_position_m=_add_noise(vert, 0.03, rng=rng),
        is_disruption=False,
        disruption_time_idx=-1,
        disruption_type="safe",
    )


def generate_highbeta_safe(rng: np.random.Generator) -> dict:
    """Shot 176673 -- High-beta stable plasma (safe).

    Sustained beta_N ~ 3.0 with wall stabilisation and active RWM
    control.

    Refs: Garofalo et al., NF 55 (2015) 123025;
          Holcomb et al., Phys. Plasmas 22 (2015) 055904.
    """
    Ip = np.full(N_STEPS, 1.5, dtype=np.float64)
    BT = np.full(N_STEPS, 2.2, dtype=np.float64)
    q95 = np.full(N_STEPS, 3.6, dtype=np.float64)
    beta_N = np.full(N_STEPS, 3.0, dtype=np.float64)
    ne = np.full(N_STEPS, 7.0, dtype=np.float64)

    # RWM oscillations (controlled)
    rwm = 0.012 * np.sin(2.0 * np.pi * 5.0 * TIME) * rng.uniform(0.7, 1.3, N_STEPS)
    beta_N += rwm

    # ELMs
    elm_freq = 50.0
    for i in range(N_STEPS):
        phase = (TIME[i] * elm_freq) % 1.0
        if phase < 0.04:
            burst = 0.05 * (1.0 - phase / 0.04)
            beta_N[i] += burst * rng.uniform(0.7, 1.3)
            ne[i] -= 0.12 * rng.uniform(0.8, 1.2)

    # Slightly elevated MHD (still safe)
    n1 = np.full(N_STEPS, 0.06, dtype=np.float64)
    n2 = np.full(N_STEPS, 0.025, dtype=np.float64)
    locked = np.full(N_STEPS, 0.008, dtype=np.float64)
    dBdt = np.full(N_STEPS, 18.0, dtype=np.float64)
    vert = np.zeros(N_STEPS, dtype=np.float64)

    return dict(
        time_s=TIME.copy(),
        Ip_MA=np.clip(_add_noise(Ip, 0.01, rng=rng), 0.0, None),
        BT_T=_add_noise(BT, 0.01, rng=rng),
        beta_N=np.clip(_add_noise(beta_N, 0.02, rng=rng), 0.0, None),
        q95=np.clip(_add_noise(q95, 0.01, rng=rng), 1.0, None),
        ne_1e19=np.clip(_add_noise(ne, 0.02, rng=rng), 0.0, None),
        n1_amp=np.clip(_add_noise(n1, 0.03, rng=rng), 0.01, 0.5),
        n2_amp=np.clip(_add_noise(n2, 0.03, rng=rng), 0.01, 0.3),
        locked_mode_amp=np.clip(_add_noise(locked, 0.03, rng=rng), 0.0, 0.8),
        dBdt_gauss_per_s=np.clip(_add_noise(dBdt, 0.02, rng=rng), 0.0, None),
        vertical_position_m=_add_noise(vert, 0.03, rng=rng),
        is_disruption=False,
        disruption_time_idx=-1,
        disruption_type="safe",
    )


# -------------------------------------------------------------------------
# Shot manifest -- (filename_stem, shot_number, generator_func)
# -------------------------------------------------------------------------
SHOT_MANIFEST: list[tuple[str, int, callable]] = [
    # Disruption shots
    ("shot_155916_locked_mode", 155916, generate_locked_mode),
    ("shot_160409_density_limit", 160409, generate_density_limit),
    ("shot_161598_vde", 161598, generate_vde),
    ("shot_164965_tearing", 164965, generate_tearing),
    ("shot_166000_beta_limit", 166000, generate_beta_limit),
    # Safe shots
    ("shot_163303_hmode_safe", 163303, generate_hmode_safe),
    ("shot_154406_hybrid_safe", 154406, generate_hybrid_safe),
    ("shot_175970_negdelta_safe", 175970, generate_negdelta_safe),
    ("shot_166549_snowflake_safe", 166549, generate_snowflake_safe),
    ("shot_176673_highbeta_safe", 176673, generate_highbeta_safe),
]


# -------------------------------------------------------------------------
# Generate / verify
# -------------------------------------------------------------------------


def generate_all(
    output_dir: Path | str | None = None,
    *,
    verbose: bool = True,
) -> list[Path]:
    """Generate all 10 NPZ shot files and return their paths.

    Each shot is seeded with ``np.random.default_rng(shot_number)``
    for deterministic reproducibility.
    """
    out_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []

    for name, shot_number, generator in SHOT_MANIFEST:
        if verbose:
            print(f"  Generating {name} (seed={shot_number}) ...", end=" ", flush=True)

        rng = np.random.default_rng(shot_number)
        data = generator(rng)

        # Validate array shapes
        for key in [
            "time_s", "Ip_MA", "BT_T", "beta_N", "q95", "ne_1e19",
            "n1_amp", "n2_amp", "locked_mode_amp", "dBdt_gauss_per_s",
            "vertical_position_m",
        ]:
            arr = data[key]
            assert arr.shape == (N_STEPS,), (
                f"{name}/{key}: expected ({N_STEPS},), got {arr.shape}"
            )
            assert arr.dtype == np.float64, (
                f"{name}/{key}: expected float64, got {arr.dtype}"
            )

        path = out_dir / f"{name}.npz"
        _save_shot(path, **data)
        paths.append(path)

        if verbose:
            dtype_tag = data["disruption_type"]
            idx_tag = data["disruption_time_idx"]
            label = (
                f"disruption ({dtype_tag}) at idx {idx_tag}"
                if data["is_disruption"]
                else "safe"
            )
            print(f"OK  [{label}]")

    if verbose:
        print(f"\n  All {len(paths)} shots written to {out_dir}")
    return paths


def verify_all(output_dir: Path | str | None = None) -> None:
    """Load every generated NPZ file and check keys, shapes, dtypes."""
    out_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR

    required_keys = {
        "time_s", "Ip_MA", "BT_T", "beta_N", "q95", "ne_1e19",
        "n1_amp", "n2_amp", "locked_mode_amp", "dBdt_gauss_per_s",
        "vertical_position_m", "is_disruption", "disruption_time_idx",
        "disruption_type",
    }
    array_keys = required_keys - {"is_disruption", "disruption_time_idx", "disruption_type"}

    n_disruption = 0
    n_safe = 0

    for name, shot_number, _ in SHOT_MANIFEST:
        path = out_dir / f"{name}.npz"
        assert path.exists(), f"Missing: {path}"
        data = np.load(str(path), allow_pickle=True)
        keys = set(data.files)
        missing = required_keys - keys
        assert not missing, f"{name}: missing keys {missing}"

        for k in array_keys:
            arr = data[k]
            assert arr.shape == (N_STEPS,), f"{name}/{k}: shape {arr.shape}"
            assert arr.dtype == np.float64, f"{name}/{k}: dtype {arr.dtype}"
            assert np.all(np.isfinite(arr)), f"{name}/{k}: non-finite values"

        is_disrupt = bool(data["is_disruption"])
        disrupt_idx = int(data["disruption_time_idx"])
        disrupt_type = str(data["disruption_type"])

        if is_disrupt:
            assert 0 <= disrupt_idx < N_STEPS, (
                f"{name}: bad disruption_time_idx={disrupt_idx}"
            )
            assert disrupt_type != "safe", f"{name}: disruption but type='safe'"
            n_disruption += 1
        else:
            assert disrupt_idx == -1, (
                f"{name}: safe but disruption_time_idx={disrupt_idx}"
            )
            assert disrupt_type == "safe", (
                f"{name}: safe but type='{disrupt_type}'"
            )
            n_safe += 1

        print(f"  VERIFIED {name}: type={disrupt_type}, idx={disrupt_idx}")

    assert n_disruption == 5, f"Expected 5 disruptions, got {n_disruption}"
    assert n_safe == 5, f"Expected 5 safe, got {n_safe}"
    print(
        f"\n  All verified: {n_disruption} disruptions + "
        f"{n_safe} safe = {n_disruption + n_safe} total"
    )


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic DIII-D disruption shot NPZ files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing files, do not regenerate",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None

    if args.verify_only:
        print("Verifying existing disruption shot files...")
        verify_all(out_dir)
    else:
        print("Generating DIII-D synthetic disruption shot profiles...")
        print(f"  Steps: {N_STEPS}, time: 0 - {T_END} s, dt: {DT:.6f} s")
        print(f"  Seeding: np.random.default_rng(shot_number)")
        print()
        generate_all(out_dir)
        print()
        print("Verifying generated files...")
        verify_all(out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
