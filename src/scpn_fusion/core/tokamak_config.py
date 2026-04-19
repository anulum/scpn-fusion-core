# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tokamak Configuration
"""Tokamak machine parameter container with named presets.

Provides a frozen dataclass holding the geometric, magnetic, and kinetic
parameters that define a tokamak operating point.  Class methods create
presets for ITER, SPARC, DIII-D, and JET.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokamakConfig:
    """Tokamak equilibrium operating point.

    Units: lengths [m], field [T], current [MA], density [1e19 m^-3],
    temperature [keV], power [MW].
    """

    name: str
    R0: float  # major radius [m]
    a: float  # minor radius [m]
    B0: float  # toroidal field on axis [T]
    Ip: float  # plasma current [MA]
    kappa: float  # elongation
    delta: float  # triangularity
    n_e: float  # line-averaged electron density [1e19 m^-3]
    T_e: float  # central electron temperature [keV]
    P_aux: float  # auxiliary heating power [MW]

    @property
    def aspect_ratio(self) -> float:
        return self.R0 / self.a

    @property
    def epsilon(self) -> float:
        return self.a / self.R0

    # ── Named presets ─────────────────────────────────────────────────

    @classmethod
    def iter(cls) -> TokamakConfig:
        """ITER baseline H-mode. ITER Research Plan (2018)."""
        return cls(
            name="ITER",
            R0=6.2,
            a=2.0,
            B0=5.3,
            Ip=15.0,
            kappa=1.7,
            delta=0.33,
            n_e=10.1,
            T_e=25.0,
            P_aux=50.0,
        )

    @classmethod
    def sparc(cls) -> TokamakConfig:
        """SPARC V2C. Creely et al., J. Plasma Phys. 86, 865860502 (2020)."""
        return cls(
            name="SPARC",
            R0=1.85,
            a=0.57,
            B0=12.2,
            Ip=8.7,
            kappa=1.97,
            delta=0.54,
            n_e=30.0,
            T_e=21.0,
            P_aux=25.0,
        )

    @classmethod
    def diiid(cls) -> TokamakConfig:
        """DIII-D typical H-mode. Luxon, Nucl. Fusion 42, 614 (2002)."""
        return cls(
            name="DIII-D",
            R0=1.67,
            a=0.67,
            B0=2.1,
            Ip=1.5,
            kappa=1.8,
            delta=0.4,
            n_e=5.0,
            T_e=4.0,
            P_aux=10.0,
        )

    @classmethod
    def jet(cls) -> TokamakConfig:
        """JET with ITER-like wall. Joffrin et al., Nucl. Fusion 59 (2019)."""
        return cls(
            name="JET",
            R0=2.96,
            a=1.25,
            B0=3.45,
            Ip=3.5,
            kappa=1.68,
            delta=0.3,
            n_e=7.0,
            T_e=8.0,
            P_aux=25.0,
        )
