# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Plasma Startup Sequence
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from scpn_fusion.core.impurity_transport import CoolingCurve


class PaschenBreakdown:
    def __init__(self, gas: str = "D2", R0: float = 6.2, a: float = 2.0):
        self.gas = gas
        self.R0 = R0
        self.a = a

        # Townsend coefficients for D₂ — Lieberman & Lichtenberg (2005), Ch. 14
        self.A = 44.7  # effective ionization coefficient [1/(Pa·m)]
        self.C2 = 2.64  # ln(ln(1+1/gamma_SE)) term [dimensionless]
        self.B_V = 155.0  # excitation coefficient [V/(Pa·m)]

    def breakdown_voltage(self, p_Pa: float, connection_length_m: float) -> float:
        """Paschen breakdown voltage [V]."""
        pd = p_Pa * connection_length_m

        if pd <= 0.0:
            return float("inf")

        denom = self.A * math.log(max(pd, 1e-6)) - self.C2
        if denom <= 0.0:
            return float("inf")

        return float(self.B_V * pd / denom)

    def is_breakdown(self, V_loop: float, p_Pa: float, connection_length_m: float = 100.0) -> bool:
        V_req = self.breakdown_voltage(p_Pa, connection_length_m)
        return V_loop > V_req

    def paschen_curve(self, p_range: np.ndarray, connection_length_m: float = 100.0) -> np.ndarray:
        return np.array([self.breakdown_voltage(p, connection_length_m) for p in p_range])

    def optimal_prefill_pressure(
        self, V_loop_max: float, connection_length_m: float = 100.0
    ) -> float:
        # Minimum of V_breakdown occurs at pd = exp(1 + C2/A)
        pd_opt = math.exp(1.0 + self.C2 / self.A)
        return float(pd_opt / connection_length_m)


@dataclass
class AvalancheResult:
    ne_trace: np.ndarray
    Te_trace: np.ndarray
    time_to_full_ionization_ms: float


class TownsendAvalanche:
    def __init__(self, V_loop: float, p_Pa: float, R0: float, a: float):
        self.V_loop = V_loop
        self.p_Pa = p_Pa
        self.R0 = R0
        self.a = a

        self.E_par = V_loop / (2.0 * math.pi * R0)
        self.n_neutral = p_Pa / (1.38e-23 * 300.0)  # roughly ideal gas law at 300K

    def ionization_rate(self, Te_eV: float) -> float:
        # Simplistic ionization rate coefficient
        # ~ exp(-E_iz / Te)
        E_iz = 13.6
        if Te_eV < 0.1:
            return 0.0
        sig_v = 1e-14 * math.exp(-E_iz / max(Te_eV, 0.1))
        return float(self.n_neutral * sig_v)

    def evolve(self, dt: float, n_steps: int) -> AvalancheResult:
        ne = 1e13  # initial seed density
        Te = 1.0  # 1 eV starting electron temp

        ne_trace = np.zeros(n_steps)
        Te_trace = np.zeros(n_steps)

        full_ion_time = -1.0

        for i in range(n_steps):
            t = i * dt

            nu_ion = self.ionization_rate(Te)
            dne = ne * nu_ion * dt
            ne += dne

            # Simple ohmic heating of cold electrons
            # P_ohmic = E^2 / eta
            # eta ~ T_e^-1.5
            eta = 1e-4 / max(Te, 0.1) ** 1.5
            P_ohmic = self.E_par**2 / eta

            # Energy balance: lose 13.6 eV per ionization
            P_loss = nu_ion * 13.6 * 1.6e-19 * ne

            dTe_J = (P_ohmic - P_loss) * dt / max(ne, 1e-6)
            dTe = dTe_J / 1.6e-19

            Te += dTe
            Te = min(max(Te, 0.5), 10.0)  # bound it before burn-through

            ne = min(ne, self.n_neutral)  # can't ionize more than we have

            ne_trace[i] = ne
            Te_trace[i] = Te

            if ne >= 0.99 * self.n_neutral and full_ion_time < 0.0:
                full_ion_time = t * 1000.0

        return AvalancheResult(ne_trace, Te_trace, full_ion_time)


@dataclass
class BurnThroughResult:
    Te_trace: np.ndarray
    success: bool
    time_to_burn_through_ms: float


class BurnThrough:
    def __init__(self, R0: float, a: float, B0: float, V_loop: float):
        self.R0 = R0
        self.a = a
        self.B0 = B0
        self.V_loop = V_loop
        self.E_par = V_loop / (2.0 * math.pi * R0)

    def ohmic_power(self, Te_eV: float, ne_19: float, Ip_kA: float) -> float:
        # P_ohmic = I_p^2 * R_p
        # eta ~ 1.65e-9 * Z_eff * 10 / T_keV^1.5
        T_keV = max(Te_eV / 1000.0, 1e-6)
        eta = 1.65e-9 * 1.5 * 10.0 / (T_keV**1.5)

        R_p = eta * 2.0 * math.pi * self.R0 / (math.pi * self.a**2)
        P_Ohmic = (Ip_kA * 1e3) ** 2 * R_p
        return float(P_Ohmic)

    def radiation_barrier(
        self, Te_eV: float, ne_19: float, f_imp: float, impurity: str = "C"
    ) -> float:
        curve = CoolingCurve(impurity)
        L_z = curve.L_z(np.array([Te_eV]))[0]

        ne = ne_19 * 1e19
        n_imp = ne * f_imp

        V = 2.0 * math.pi**2 * self.R0 * self.a**2
        P_rad = ne * n_imp * L_z * V
        return float(P_rad)

    def burn_through_condition(
        self, Te_eV: float, ne_19: float, Ip_kA: float, f_imp: float, impurity: str = "C"
    ) -> bool:
        P_oh = self.ohmic_power(Te_eV, ne_19, Ip_kA)
        P_rad = self.radiation_barrier(Te_eV, ne_19, f_imp, impurity)
        return P_oh > P_rad

    def critical_impurity_fraction(
        self, Te_eV: float, ne_19: float, Ip_kA: float, impurity: str
    ) -> float:
        # Find f_imp such that P_oh = P_rad
        curve = CoolingCurve(impurity)
        L_z = curve.L_z(np.array([Te_eV]))[0]

        if L_z <= 0.0:
            return 1.0

        P_oh = self.ohmic_power(Te_eV, ne_19, Ip_kA)

        ne = ne_19 * 1e19
        V = 2.0 * math.pi**2 * self.R0 * self.a**2

        n_imp_crit = P_oh / (ne * L_z * V)
        return float(n_imp_crit / ne)

    def evolve(
        self, ne_19: float, f_imp: float, dt: float, n_steps: int, impurity: str = "C"
    ) -> BurnThroughResult:
        Te = 5.0  # start at 5 eV
        Ip = 100.0  # start at 100 kA

        Te_trace = np.zeros(n_steps)
        success = False
        time_to_bt = -1.0

        V = 2.0 * math.pi**2 * self.R0 * self.a**2
        ne = ne_19 * 1e19

        for i in range(n_steps):
            t = i * dt

            P_oh = self.ohmic_power(Te, ne_19, Ip)
            P_rad = self.radiation_barrier(Te, ne_19, f_imp, impurity)

            # simple energy eq: 1.5 n_e V dTe = P_oh - P_rad
            dTe_J = (P_oh - P_rad) * dt / (1.5 * ne * V)
            dTe = dTe_J / 1.6e-19

            Te += dTe

            # Simple Ip ramp if Te rises
            if Te > 20.0:
                Ip += 1000.0 * dt  # 1 MA/s

            Te_trace[i] = Te

            if Te > 100.0 and not success:
                success = True
                time_to_bt = t * 1000.0

        return BurnThroughResult(Te_trace, success, time_to_bt)


@dataclass
class StartupResult:
    breakdown_time_ms: float
    burn_through_time_ms: float
    Ip_at_100ms_kA: float
    Te_at_100ms_eV: float
    success: bool


class StartupSequence:
    def __init__(
        self,
        R0: float,
        a: float,
        B0: float,
        V_loop: float,
        p_prefill_Pa: float,
        f_imp: float = 0.01,
    ):
        self.R0 = R0
        self.a = a
        self.B0 = B0
        self.V_loop = V_loop
        self.p_prefill_Pa = p_prefill_Pa
        self.f_imp = f_imp

    def run(self) -> StartupResult:
        conn = 100.0
        paschen = PaschenBreakdown("D2", self.R0, self.a)

        if not paschen.is_breakdown(self.V_loop, self.p_prefill_Pa, conn):
            return StartupResult(-1.0, -1.0, 0.0, 0.0, False)

        ava = TownsendAvalanche(self.V_loop, self.p_prefill_Pa, self.R0, self.a)
        ava_res = ava.evolve(1e-4, 50)

        bt = BurnThrough(self.R0, self.a, self.B0, self.V_loop)
        # 1 mPa ~ 2e18 m^-3 ~ 0.2 in 1e19 units
        ne_19 = 0.2
        bt_res = bt.evolve(ne_19, self.f_imp, 1e-3, 200, impurity="C")

        return StartupResult(
            breakdown_time_ms=ava_res.time_to_full_ionization_ms,
            burn_through_time_ms=bt_res.time_to_burn_through_ms,
            Ip_at_100ms_kA=100.0 + 1000.0 * 0.1 if bt_res.success else 0.0,
            Te_at_100ms_eV=bt_res.Te_trace[-1],
            success=bt_res.success,
        )


class StartupPhase(Enum):
    GAS_PUFF = auto()
    BREAKDOWN = auto()
    BURN_THROUGH = auto()
    RAMP = auto()


@dataclass
class StartupCommand:
    V_loop: float
    gas_puff_rate: float
    phase: StartupPhase


class StartupController:
    def __init__(self, V_loop_max: float, gas_puff_max: float):
        self.V_loop_max = V_loop_max
        self.gas_puff_max = gas_puff_max
        self.phase = StartupPhase.GAS_PUFF

    def step(self, ne: float, Te: float, Ip: float, t: float, dt: float) -> StartupCommand:
        # Check transitions first
        if self.phase == StartupPhase.GAS_PUFF:
            if t > 0.1:
                self.phase = StartupPhase.BREAKDOWN
        elif self.phase == StartupPhase.BREAKDOWN:
            if ne > 1e18:
                self.phase = StartupPhase.BURN_THROUGH
        elif self.phase == StartupPhase.BURN_THROUGH:
            if Te > 50.0:
                self.phase = StartupPhase.RAMP

        # Now act based on current phase
        if self.phase == StartupPhase.GAS_PUFF:
            return StartupCommand(0.0, self.gas_puff_max, self.phase)
        elif self.phase == StartupPhase.BREAKDOWN or self.phase == StartupPhase.BURN_THROUGH:
            return StartupCommand(self.V_loop_max, 0.0, self.phase)
        else:  # RAMP
            return StartupCommand(self.V_loop_max * 0.5, self.gas_puff_max * 0.1, self.phase)
