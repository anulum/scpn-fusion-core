# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Integrated Scenario Simulator
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from scpn_fusion.core.current_diffusion import CurrentDiffusionSolver
from scpn_fusion.core.current_drive import CurrentDriveMix, ECCDSource, NBISource
from scpn_fusion.core.integrated_transport_solver import TransportSolver
from scpn_fusion.core.ntm_dynamics import NTMController
from scpn_fusion.core.sawtooth import SawtoothCycler
from scpn_fusion.core.sol_model import TwoPointSOL


@dataclass
class ScenarioConfig:
    # Geometry
    R0: float
    a: float
    B0: float
    kappa: float
    delta: float

    # Actuators
    Ip_MA: float
    P_aux_MW: float

    # CD parameters (simplified)
    P_eccd_MW: float = 0.0
    rho_eccd: float = 0.5
    P_nbi_MW: float = 0.0
    E_nbi_keV: float = 100.0

    # Duration
    t_start: float = 0.0
    t_end: float = 10.0
    dt: float = 0.1

    transport_model: str = "gyro_bohm"

    # Flags
    include_sawteeth: bool = True
    include_ntm: bool = True
    include_sol: bool = True


@dataclass
class ScenarioState:
    time: float
    rho: np.ndarray
    Te: np.ndarray
    Ti: np.ndarray
    ne: np.ndarray
    q: np.ndarray
    psi: np.ndarray
    j_total: np.ndarray
    j_bs: np.ndarray
    j_cd: np.ndarray

    Ip_MA: float
    beta_N: float
    tau_E: float
    P_loss: float
    W_thermal: float
    li: float

    ballooning_stable: bool
    troyon_stable: bool
    ntm_island_widths: dict[str, float]

    T_target: float
    q_peak: float
    detached: bool

    last_crash_time: float
    n_crashes: int


def iter_baseline_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        R0=6.2,
        a=2.0,
        B0=5.3,
        kappa=1.7,
        delta=0.33,
        Ip_MA=15.0,
        P_aux_MW=50.0,
        P_eccd_MW=17.0,
        rho_eccd=0.5,
        P_nbi_MW=33.0,
        E_nbi_keV=1000.0,
        t_end=100.0,
        dt=1.0,
    )


def iter_hybrid_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        R0=6.2,
        a=2.0,
        B0=5.3,
        kappa=1.7,
        delta=0.33,
        Ip_MA=12.0,
        P_aux_MW=50.0,
        t_end=100.0,
        dt=1.0,
    )


def nstx_u_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        R0=0.93,
        a=0.58,
        B0=1.0,
        kappa=2.0,
        delta=0.4,
        Ip_MA=1.0,
        P_aux_MW=10.0,
        t_end=2.0,
        dt=0.01,
    )


class IntegratedScenarioSimulator:
    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.time = config.t_start

        # Initialize internal models
        self.nr = 50
        self.rho = np.linspace(0, 1, self.nr)

        # Current Drive
        self.cd_mix = CurrentDriveMix(a=self.config.a)
        self.eccd: ECCDSource | None = None
        if self.config.P_eccd_MW > 0:
            self.eccd = ECCDSource(self.config.P_eccd_MW, self.config.rho_eccd, 0.05)
            self.cd_mix.add_source(self.eccd)

        if self.config.P_nbi_MW > 0:
            self.nbi = NBISource(self.config.P_nbi_MW, self.config.E_nbi_keV, 0.2)
            self.cd_mix.add_source(self.nbi)

        # Current Diffusion
        self.cd_solver = CurrentDiffusionSolver(
            self.rho, self.config.R0, self.config.a, self.config.B0
        )

        # Sawtooth
        self.sawtooth = SawtoothCycler(self.rho, self.config.R0, self.config.a)
        self.n_crashes = 0
        self.last_crash_time = 0.0

        # SOL
        self.sol = TwoPointSOL(
            self.config.R0, self.config.a, q95=3.0, B_pol=0.5, kappa=self.config.kappa
        )

        # NTMs (Track q=2/1 and q=3/2)
        self.ntm_widths: dict[str, float] = {}  # island name → width [m]
        self.ntm_controller = NTMController()

    def _setup_transport_solver(self) -> TransportSolver:
        # Create a temporary config for TransportSolver
        cfg_dict = {
            "reactor_name": "IntegratedScenario",
            "grid_resolution": [33, 33],
            "tokamak": {
                "R0": self.config.R0,
                "a": self.config.a,
                "B0": self.config.B0,
            },
            "dimensions": {
                "R_min": self.config.R0 - self.config.a - 0.5,
                "R_max": self.config.R0 + self.config.a + 0.5,
                "Z_min": -self.config.a * self.config.kappa - 0.5,
                "Z_max": self.config.a * self.config.kappa + 0.5,
                "nR": 33,
                "nZ": 33,
            },
            "coils": [],
            "grid": {"nr": self.nr},
            "physics": {"transport_model": "gyro_bohm"},
            "control": {"P_aux": self.config.P_aux_MW},
        }
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
            json.dump(cfg_dict, f)
            temp_path = f.name

        solver = TransportSolver(temp_path, nr=self.nr, transport_model=self.config.transport_model)

        # Initialize profiles in solver
        solver.Te = np.ones(self.nr) * 1.0  # keV
        solver.Ti = np.ones(self.nr) * 1.0
        solver.ne = np.ones(self.nr) * 5.0  # 10^19 m^-3

        return solver

    def initialize(self, profiles: dict | None = None) -> ScenarioState:
        self.ts_solver = self._setup_transport_solver()
        if profiles:
            if "Te" in profiles:
                self.ts_solver.Te = profiles["Te"]
            if "Ti" in profiles:
                self.ts_solver.Ti = profiles["Ti"]
            if "ne" in profiles:
                self.ts_solver.ne = profiles["ne"]
            if "psi" in profiles:
                self.cd_solver.psi = profiles["psi"]

        return self._build_state()

    def _build_state(self) -> ScenarioState:
        from scpn_fusion.core.current_diffusion import q_from_psi

        q_prof = q_from_psi(
            self.rho, self.cd_solver.psi, self.config.R0, self.config.a, self.config.B0
        )

        # Compute macroscopic quantities
        vol = 2.0 * np.pi**2 * self.config.R0 * self.config.a**2 * self.config.kappa
        e_charge = 1.602e-19
        energy_dens = (
            1.5
            * (self.ts_solver.ne * 1e19)
            * (self.ts_solver.Te + self.ts_solver.Ti)
            * 1e3
            * e_charge
        )
        W_th = trapezoid(energy_dens * self.rho, self.rho) * vol * 2.0

        P_loss = self.config.P_aux_MW  # steady state assumption
        tau_E = (W_th / 1e6) / max(P_loss, 1.0)
        beta_N = 2.0  # mock
        li = 1.0  # mock

        j_bs = np.zeros(self.nr)  # simplified
        j_cd = self.cd_mix.total_j_cd(
            self.rho, self.ts_solver.ne, self.ts_solver.Te, self.ts_solver.Ti
        )

        T_t, q_peak, detached = 0.0, 0.0, False
        if self.config.include_sol:
            n_u = self.ts_solver.ne[-1]
            sol_res = self.sol.solve(self.config.P_aux_MW, n_u, f_rad=0.3)
            T_t = sol_res.T_target_eV
            q_peak = sol_res.q_parallel_MW_m2  # simplified
            detached = T_t < 5.0

        ntm_widths = dict(self.ntm_widths)

        return ScenarioState(
            time=self.time,
            rho=self.rho.copy(),
            Te=self.ts_solver.Te.copy(),
            Ti=self.ts_solver.Ti.copy(),
            ne=self.ts_solver.ne.copy(),
            q=q_prof.copy(),
            psi=self.cd_solver.psi.copy(),
            j_total=j_bs + j_cd,  # missing ohmic
            j_bs=j_bs,
            j_cd=j_cd,
            Ip_MA=self.config.Ip_MA,
            beta_N=beta_N,
            tau_E=tau_E,
            P_loss=P_loss,
            W_thermal=W_th,
            li=li,
            ballooning_stable=True,
            troyon_stable=True,
            ntm_island_widths=ntm_widths,
            T_target=T_t,
            q_peak=q_peak,
            detached=detached,
            last_crash_time=self.last_crash_time,
            n_crashes=self.n_crashes,
        )

    def step(self) -> ScenarioState:
        dt = self.config.dt

        # 1. Transport advance (mocked by calling step, actually ITS step needs P_aux etc)
        # We just approximate it by letting the solver step if we implemented the full interface
        # For this simulator, we just update the time.
        self.ts_solver.time = self.time
        # In a real run we would call self.ts_solver.step(...)

        # 2. Current drive and bootstrap
        j_cd = self.cd_mix.total_j_cd(
            self.rho, self.ts_solver.ne, self.ts_solver.Te, self.ts_solver.Ti
        )
        # Mock bootstrap current for integration
        j_bs = np.zeros(self.nr)

        # 3. Current diffusion
        self.cd_solver.step(dt, self.ts_solver.Te, self.ts_solver.ne, 1.5, j_bs, j_cd)
        from scpn_fusion.core.current_diffusion import q_from_psi

        q_prof = q_from_psi(
            self.rho, self.cd_solver.psi, self.config.R0, self.config.a, self.config.B0
        )

        # 4. Sawteeth
        if self.config.include_sawteeth:
            shear = np.gradient(q_prof, self.rho) * (self.rho / np.maximum(q_prof, 1e-3))
            event = self.sawtooth.step(dt, q_prof, shear, self.ts_solver.Te, self.ts_solver.ne)
            if event:
                self.n_crashes += 1
                self.last_crash_time = event.crash_time
                # Recalculate q after crash
                q_prof = q_from_psi(
                    self.rho, self.cd_solver.psi, self.config.R0, self.config.a, self.config.B0
                )

        # 5. NTM
        if self.config.include_ntm:
            # simple mock tracking
            pass

        self.time += dt
        return self._build_state()

    def run(self) -> list[ScenarioState]:
        if not hasattr(self, "ts_solver"):
            self.initialize()
        states = []
        n_steps = int((self.config.t_end - self.config.t_start) / self.config.dt)
        for _ in range(n_steps):
            states.append(self.step())
        return states

    def to_json(self, path: Path) -> None:
        # serialization
        pass
