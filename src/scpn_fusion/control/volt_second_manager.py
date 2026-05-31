# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Volt-Second Management
"""Volt-second budget, ramp, and flux-consumption utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
TransportModel = Callable[[FloatArray], FloatArray]


@dataclass
class FluxStatus:
    """Instantaneous central-solenoid flux-consumption status."""

    flux_consumed_Vs: float
    flux_remaining_Vs: float
    estimated_remaining_time_s: float
    fraction_consumed: float


@dataclass
class FluxReport:
    """Scenario-level volt-second budget report."""

    ramp_flux: float
    flat_top_flux: float
    ramp_down_flux: float
    total_flux: float
    within_budget: bool
    margin_Vs: float


class FluxBudget:
    """Central-solenoid flux budget with plasma inductive and resistive terms."""

    def __init__(self, Phi_CS_Vs: float, L_plasma_uH: float, R_plasma_uOhm: float):
        self.Phi_CS_Vs = Phi_CS_Vs
        self.L_plasma_H = L_plasma_uH * 1e-6
        self.R_plasma_Ohm = R_plasma_uOhm * 1e-6

    def inductive_flux(self, Ip_MA: float) -> float:
        """Return inductive flux required to reach the target plasma current."""
        return self.L_plasma_H * (Ip_MA * 1e6)

    def resistive_flux_ramp(self, Ip_trace: FloatArray, dt: float) -> float:
        """Integrate resistive volt-second consumption over a current trace."""
        # Integral of R_p * I_p dt
        return float(np.sum(self.R_plasma_Ohm * (Ip_trace * 1e6) * dt))

    def remaining_flux(self, Ip_MA: float, ramp_flux: float) -> float:
        """Return remaining central-solenoid flux after ramp consumption."""
        ind = self.inductive_flux(Ip_MA)
        consumed = ind + ramp_flux
        return max(0.0, self.Phi_CS_Vs - consumed)

    def max_flattop_duration(self, Ip_MA: float, I_bs_MA: float, ramp_flux: float) -> float:
        """Estimate maximum flat-top duration from driven current demand."""
        rem = self.remaining_flux(Ip_MA, ramp_flux)
        I_driven = max((Ip_MA - I_bs_MA) * 1e6, 1e-6)
        return float(rem / (self.R_plasma_Ohm * I_driven))


class VoltSecondOptimizer:
    """Generate current-ramp candidates under a flux budget."""

    def __init__(self, flux_budget: FluxBudget, transport_model: TransportModel | None = None):
        self.budget = flux_budget
        self.transport_model = transport_model

    def optimize_ramp(
        self, Ip_target_MA: float, t_ramp_max: float, n_segments: int = 10
    ) -> FloatArray:
        """Return a deterministic plasma-current ramp to the target current."""
        # A simple optimal ramp: ramp as fast as possible, but we just generate a linear ramp for testing
        # Real optimization would consider CS stress and MHD stability (li limits)
        t_arr = np.linspace(0, t_ramp_max, n_segments)
        Ip_trace = Ip_target_MA * (t_arr / t_ramp_max)
        return Ip_trace


class BootstrapCurrentEstimate:
    """Bootstrap-current proxy from pressure-gradient profiles."""

    @staticmethod
    def from_profiles(
        ne: FloatArray,
        Te: FloatArray,
        Ti: FloatArray,
        q: FloatArray,
        rho: FloatArray,
        R0: float,
        a: float,
    ) -> float:
        """Estimate bootstrap current from density, temperature, q, and geometry profiles."""
        # Simplistic proxy: I_bs scales with pressure gradient and epsilon^0.5
        # We assume a fixed profile and just return a scalable estimate
        epsilon = a / R0
        p = 2.0 * ne * 1e19 * Te * 1e3 * 1.6e-19
        grad_p = np.gradient(p, rho[1] - rho[0] if len(rho) > 1 else 0.1)

        # J_bs ~ -epsilon^0.5 / B_pol * dp/dr
        # Very rough scaling
        J_bs_integral = np.sum(-grad_p * math.sqrt(epsilon)) * 1e-5

        I_bs_MA = max(0.0, J_bs_integral * 0.1)
        return float(I_bs_MA)


class FluxConsumptionMonitor:
    """Online flux-consumption monitor for loop-voltage integration."""

    def __init__(self, flux_budget: FluxBudget):
        self.budget = flux_budget
        self.consumed = 0.0

    def step(self, Ip: float, V_loop: float, dt: float) -> FluxStatus:
        """Advance the flux-consumption estimate by one loop-voltage sample."""
        self.consumed += V_loop * dt
        rem = self.budget.Phi_CS_Vs - self.consumed

        # Current required V_loop roughly
        est_time = rem / max(V_loop, 1e-3) if rem > 0 else 0.0
        frac = self.consumed / self.budget.Phi_CS_Vs

        return FluxStatus(
            flux_consumed_Vs=self.consumed,
            flux_remaining_Vs=max(0.0, rem),
            estimated_remaining_time_s=est_time,
            fraction_consumed=frac,
        )


class ScenarioFluxAnalysis:
    """Compute scenario-level ramp, flat-top, and ramp-down flux use."""

    def __init__(self, flux_budget: FluxBudget):
        self.budget = flux_budget

    def analyze(
        self, ramp_dur: float, flat_dur: float, down_dur: float, Ip_MA: float, I_bs_MA: float
    ) -> FluxReport:
        """Return a deterministic volt-second budget report for a discharge scenario."""
        # Mock scenario analysis
        L_term = self.budget.inductive_flux(Ip_MA)

        # Resistive during ramp
        R_term_ramp = self.budget.R_plasma_Ohm * (Ip_MA * 1e6 * 0.5) * ramp_dur
        ramp_flux = L_term + R_term_ramp

        flat_flux = self.budget.R_plasma_Ohm * max((Ip_MA - I_bs_MA) * 1e6, 0.0) * flat_dur

        # During ramp down, we recover some L, but lose some to resistive. Net is often small.
        down_flux = self.budget.R_plasma_Ohm * (Ip_MA * 1e6 * 0.5) * down_dur - L_term * 0.5

        tot = ramp_flux + flat_flux + down_flux

        return FluxReport(
            ramp_flux=ramp_flux,
            flat_top_flux=flat_flux,
            ramp_down_flux=down_flux,
            total_flux=tot,
            within_budget=tot <= self.budget.Phi_CS_Vs,
            margin_Vs=self.budget.Phi_CS_Vs - tot,
        )
