# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Current Drive Physics Module
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np

E_CHARGE = 1.602176634e-19
M_E = 9.1093837e-31
M_P = 1.6726219e-27
EPS_0 = 8.8541878e-12


class ECCDSource:
    """Electron Cyclotron Current Drive (ECCD)."""

    def __init__(self, P_ec_MW: float, rho_dep: float, sigma_rho: float, eta_cd: float = 0.03):
        self.P_ec_MW = P_ec_MW
        self.rho_dep = rho_dep
        self.sigma_rho = sigma_rho
        self.eta_cd = eta_cd

    def P_absorbed(self, rho: np.ndarray) -> np.ndarray:
        """Absorbed power density profile [W/m^3]."""
        if self.sigma_rho <= 0.0:
            return np.zeros_like(rho)
        P_W = self.P_ec_MW * 1e6
        P_dens = (
            P_W
            / (np.sqrt(2 * np.pi) * self.sigma_rho)
            * np.exp(-((rho - self.rho_dep) ** 2) / (2 * self.sigma_rho**2))
        )
        return np.asarray(P_dens)

    def j_cd(self, rho: np.ndarray, ne_19: np.ndarray, Te_keV: np.ndarray) -> np.ndarray:
        """Driven current density profile [A/m^2]."""
        p_abs = self.P_absorbed(rho)
        denom = np.maximum(ne_19 * Te_keV, 1e-3)
        return np.asarray(self.eta_cd * p_abs / denom)


class NBISource:
    """Neutral Beam Injection (NBI)."""

    def __init__(
        self,
        P_nbi_MW: float,
        E_beam_keV: float,
        rho_tangency: float,
        sigma_rho: float = 0.15,
    ):
        self.P_nbi_MW = P_nbi_MW
        self.E_beam_keV = E_beam_keV
        self.rho_tangency = rho_tangency
        self.sigma_rho = sigma_rho

    def P_heating(self, rho: np.ndarray) -> np.ndarray:
        """Heating power deposition profile [W/m^3]."""
        if self.sigma_rho <= 0.0:
            return np.zeros_like(rho)
        P_W = self.P_nbi_MW * 1e6
        P_dens = (
            P_W
            / (np.sqrt(2 * np.pi) * self.sigma_rho)
            * np.exp(-((rho - self.rho_tangency) ** 2) / (2 * self.sigma_rho**2))
        )
        return np.asarray(P_dens)

    def j_cd(
        self, rho: np.ndarray, ne_19: np.ndarray, Te_keV: np.ndarray, Ti_keV: np.ndarray
    ) -> np.ndarray:
        """Driven current density profile [A/m^2]."""
        p_heat = self.P_heating(rho)

        A_beam = 2.0
        Z_beam = 1.0
        m_beam = A_beam * M_P
        E_beam_J = self.E_beam_keV * 1e3 * E_CHARGE
        v_parallel = np.sqrt(2.0 * E_beam_J / m_beam)

        m_crit = m_beam * (0.75 * np.sqrt(np.pi) * M_E / m_beam) ** (2.0 / 3.0)
        Z_eff = 1.5

        j_prof = np.zeros_like(rho)
        for i in range(len(rho)):
            if p_heat[i] <= 0:
                continue

            Te_J = max(Te_keV[i], 1e-3) * 1e3 * E_CHARGE
            ne = max(ne_19[i], 1e-3) * 1e19
            ln_Lambda = 17.0

            tau_e = (12.0 * np.pi**1.5 * EPS_0**2 * np.sqrt(M_E) * Te_J**1.5) / (
                ne * Z_eff * E_CHARGE**4 * ln_Lambda
            )
            denom = (1.0 + m_beam / (m_crit * Z_eff)) ** 1.5
            tau_s = (0.75 * np.sqrt(np.pi)) * (m_beam / M_E) * tau_e / denom
            n_fast = p_heat[i] * tau_s / E_beam_J
            j_prof[i] = E_CHARGE * n_fast * v_parallel / Z_beam

        return j_prof


class LHCDSource:
    """Lower Hybrid Current Drive (LHCD)."""

    def __init__(self, P_lh_MW: float, rho_dep: float, sigma_rho: float, eta_cd: float = 0.15):
        self.P_lh_MW = P_lh_MW
        self.rho_dep = rho_dep
        self.sigma_rho = sigma_rho
        self.eta_cd = eta_cd

    def P_absorbed(self, rho: np.ndarray) -> np.ndarray:
        if self.sigma_rho <= 0.0:
            return np.zeros_like(rho)
        P_W = self.P_lh_MW * 1e6
        P_dens = (
            P_W
            / (np.sqrt(2 * np.pi) * self.sigma_rho)
            * np.exp(-((rho - self.rho_dep) ** 2) / (2 * self.sigma_rho**2))
        )
        return np.asarray(P_dens)

    def j_cd(self, rho: np.ndarray, ne_19: np.ndarray, Te_keV: np.ndarray) -> np.ndarray:
        p_abs = self.P_absorbed(rho)
        denom = np.maximum(ne_19 * Te_keV, 1e-3)
        return np.asarray(self.eta_cd * p_abs / denom)


class CurrentDriveMix:
    """Combines multiple current drive sources."""

    def __init__(self, a: float = 1.0):
        self.sources: list[ECCDSource | NBISource | LHCDSource] = []
        self.a = a

    def add_source(self, source: ECCDSource | NBISource | LHCDSource) -> None:
        self.sources.append(source)

    def total_j_cd(
        self, rho: np.ndarray, ne: np.ndarray, Te: np.ndarray, Ti: np.ndarray
    ) -> np.ndarray:
        j_tot = np.zeros_like(rho)
        for src in self.sources:
            if isinstance(src, NBISource):
                j_tot += src.j_cd(rho, ne, Te, Ti)
            else:
                j_tot += src.j_cd(rho, ne, Te)
        return j_tot

    def total_heating_power(self, rho: np.ndarray) -> np.ndarray:
        p_tot = np.zeros_like(rho)
        for src in self.sources:
            if isinstance(src, NBISource):
                p_tot += src.P_heating(rho)
            else:
                p_tot += src.P_absorbed(rho)
        return p_tot

    def total_driven_current(
        self, rho: np.ndarray, ne: np.ndarray, Te: np.ndarray, Ti: np.ndarray
    ) -> float:
        """Integrate total driven current [A] assuming circular cross-section."""
        j_tot = self.total_j_cd(rho, ne, Te, Ti)
        drho = rho[1] - rho[0] if len(rho) > 1 else 0.0
        dA = 2.0 * np.pi * rho * self.a**2 * drho
        return float(np.sum(j_tot * dA))
