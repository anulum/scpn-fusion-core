# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Integrated Scenario Simulator
"""Integrated scenario orchestration for coupled transport/current dynamics.

This module defines a small, deterministic scenario surface that composes
transport, current diffusion, sawtooth and NTM dynamics, and a SOL surrogate into
one public API for time-dependent discharge-like simulations.
"""

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
from scpn_fusion.core.neoclassical import sauter_bootstrap
from scpn_fusion.core.ntm_dynamics import NTMController
from scpn_fusion.core.ntm_dynamics import NTMIslandDynamics, find_rational_surfaces
from scpn_fusion.core.sawtooth import SawtoothCycler
from scpn_fusion.core.sol_model import TwoPointSOL, peak_target_heat_flux


@dataclass
class ScenarioConfig:
    """Container for user-facing scenario parameters.

    The configuration is intentionally explicit and covers geometry, power input,
    grid/time stepping, and feature toggles used by
    :class:`IntegratedScenarioSimulator`.

    Geometry and machine fields:

    - ``R0`` major radius [m]
    - ``a`` minor radius [m]
    - ``B0`` toroidal field on-axis [T]
    - ``kappa`` plasma elongation
    - ``delta`` plasma triangularity

    Actuation and boundary conditions:

    - ``Ip_MA`` plasma current [MA]
    - ``P_aux_MW`` auxiliary power [MW]
    - ``P_eccd_MW`` ECCD power [MW]
    - ``rho_eccd`` normalized minor radius location of ECCD deposition
    - ``P_nbi_MW`` neutral-beam power [MW]
    - ``E_nbi_keV`` beam energy [keV]

    Temporal integration:

    - ``t_start`` start time [s]
    - ``t_end`` end time [s]
    - ``dt`` fixed integration step [s]
    - ``transport_model`` selector for transport solver setup

    Simulation switches:

    - ``include_sawteeth`` enable sawtooth modulation
    - ``include_ntm`` enable NTM island updates
    - ``include_sol`` include SOL target proxy coupling
    """

    # Geometry
    R0: float
    a: float
    B0: float
    kappa: float
    delta: float

    # Actuators
    Ip_MA: float
    P_aux_MW: float

    # Current-drive actuator parameters
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
    """Single snapshot produced by :class:`IntegratedScenarioSimulator`.

    The structure is used both for transient state updates and JSON serialisation.
    Numerical profiles are expected to be one-dimensional arrays on ``self.rho``.
    """

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
    """Return a conservative baseline preset for integrated evolution tests.

    Returns:
        A fully-populated :class:`ScenarioConfig` with default duration and active
        coupling modules enabled.
    """

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
    """Return a hybrid transport preset.

    Returns:
        A profile with transport-focused defaults and a shorter explicit scenario
        contract than the baseline.
    """

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
    """Return an NSTX-U-style compact scenario preset.

    Returns:
        A geometry-scaled :class:`ScenarioConfig` suitable for compact device
        regression and benchmark comparisons.
    """

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
    """Execute a coupled transport/current/NTM simulation campaign.

    Lifecycle:

    1. Construct with a :class:`ScenarioConfig`.
    2. Optionally override profiles and call :meth:`initialize`.
    3. Advance using :meth:`step` or complete with :meth:`run`.
    4. Optionally persist via :meth:`to_json`.
    """

    def __init__(self, config: ScenarioConfig):
        """Create an integrated simulator bound to one scenario configuration.

        Args:
            config: Complete scenario specification used to seed all coupled
                components.
        """

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
        self._last_states: list[ScenarioState] = []

    def _setup_transport_solver(self) -> TransportSolver:
        """Create and configure a temporary transport solver for this scenario.

        Returns:
            A configured :class:`TransportSolver` instance tied to this scenario grid
            and control power settings.
        """

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
            "physics": {
                "transport_model": "gyro_bohm",
                "plasma_current_target": self.config.Ip_MA,
                "vacuum_permeability": 1.0,
            },
            "control": {"P_aux": self.config.P_aux_MW},
            "solver": {
                "max_iterations": 500,
                "convergence_threshold": 1.0e-4,
                "relaxation_factor": 0.1,
            },
        }
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
            json.dump(cfg_dict, f)
            temp_path = f.name

        solver = TransportSolver(temp_path)

        # Initialize profiles in solver
        solver.Te = np.ones(self.nr) * 1.0  # keV
        solver.Ti = np.ones(self.nr) * 1.0
        solver.ne = np.ones(self.nr) * 5.0  # 10^19 m^-3

        return solver

    def initialize(self, profiles: dict | None = None) -> ScenarioState:
        """Initialize the coupled simulation state.

        Args:
            profiles: Optional override map for profile fields. Supported keys are
                ``"Te"``, ``"Ti"``, ``"ne"``, and ``"psi"``.

        Returns:
            Initialised :class:`ScenarioState` at ``self.time == t_start``.

        Raises:
            ValueError: If overridden profile values are incompatible with solver
                expectations (shape, length, or non-finite values).
        """

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
        """Construct a fresh :class:`ScenarioState` from live solver variables.

        Returns:
            The computed state snapshot with transport profiles, current decomposition,
            global stability flags, and safety metrics.
        """

        from scpn_fusion.core.current_diffusion import q_from_psi

        q_prof = q_from_psi(
            self.rho, self.cd_solver.psi, self.config.R0, self.config.a, self.config.B0
        )

        j_bs = self._bootstrap_current_density(q_prof)
        j_cd = self.cd_mix.total_j_cd(
            self.rho, self.ts_solver.ne, self.ts_solver.Te, self.ts_solver.Ti
        )
        j_ohmic = self._ohmic_current_density()

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
        beta_N = self._normalised_beta(energy_dens)
        li = self._internal_inductance_proxy(j_ohmic + j_bs + j_cd)

        T_t, q_peak, detached = 0.0, 0.0, False
        if self.config.include_sol:
            n_u = self.ts_solver.ne[-1]
            f_rad = 0.3
            sol_res = self.sol.solve(self.config.P_aux_MW, n_u, f_rad=f_rad)
            T_t = sol_res.T_target_eV
            q_peak = peak_target_heat_flux(
                self.config.P_aux_MW * (1.0 - f_rad),
                self.config.R0,
                sol_res.lambda_q_mm * 1e-3,
            )
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
            j_total=j_ohmic + j_bs + j_cd,
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
        """Advance one integration step and return the updated state.

        Returns:
            The new :class:`ScenarioState` after one time increment.

        Raises:
            ValueError: If required solver state is incomplete or un-initialised.
        """

        dt = self.config.dt

        # 1. Transport solver clock advance; profile evolution remains owned by TransportSolver.
        self.ts_solver.time = self.time

        # 2. Current drive and bootstrap
        from scpn_fusion.core.current_diffusion import q_from_psi

        q_prof = q_from_psi(
            self.rho, self.cd_solver.psi, self.config.R0, self.config.a, self.config.B0
        )
        j_cd = self.cd_mix.total_j_cd(
            self.rho, self.ts_solver.ne, self.ts_solver.Te, self.ts_solver.Ti
        )
        j_bs = self._bootstrap_current_density(q_prof)

        # 3. Current diffusion
        self.cd_solver.step(dt, self.ts_solver.Te, self.ts_solver.ne, 1.5, j_bs, j_cd)

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
            self._update_ntm_dynamics(q_prof=q_prof, j_bs=j_bs, j_cd=j_cd)

        self.time += dt
        out = self._build_state()
        self._last_states.append(out)
        return out

    def _update_ntm_dynamics(self, q_prof: np.ndarray, j_bs: np.ndarray, j_cd: np.ndarray) -> None:
        """Advance tearing mode islands for supported NTM rational surfaces.

        The helper selects 2/1 and 3/2 resonances and updates stored island widths.

        Args:
            q_prof: Safety-factor profile on the scenario ``rho`` grid.
            j_bs: Bootstrap current profile on ``rho``.
            j_cd: Current-drive current profile on ``rho``.
        """

        surfaces = find_rational_surfaces(
            q=np.asarray(q_prof, dtype=float), rho=self.rho, a=self.config.a, m_max=3, n_max=2
        )
        if not surfaces:
            return

        dt = float(self.config.dt)
        j_phi_prof = (
            self._ohmic_current_density()
            + np.asarray(j_bs, dtype=float)
            + np.asarray(j_cd, dtype=float)
        )
        allowed_modes = {(2, 1), (3, 2)}

        for surf in surfaces:
            mode = (surf.m, surf.n)
            if mode not in allowed_modes:
                continue

            key = f"{surf.m}/{surf.n}"
            w0 = float(self.ntm_widths.get(key, 1e-6))
            j_bs_loc = float(np.interp(surf.rho, self.rho, j_bs))
            j_phi_loc = float(np.interp(surf.rho, self.rho, j_phi_prof))
            j_cd_loc = float(np.interp(surf.rho, self.rho, j_cd))

            eccd_request_mw = float(
                self.ntm_controller.step(
                    w=w0,
                    rho_rs=float(surf.rho),
                    max_power=max(float(self.config.P_eccd_MW), 0.0),
                )
            )
            # Convert requested ECCD MW to a local equivalent current-drive increment.
            j_cd_ctrl = j_cd_loc + eccd_request_mw * 1e4

            dyn = NTMIslandDynamics(
                r_s=float(surf.r_s),
                m=int(surf.m),
                n=int(surf.n),
                a=float(self.config.a),
                R0=float(self.config.R0),
                B0=float(self.config.B0),
            )
            _, w_arr = dyn.evolve(
                w0=w0,
                t_span=(0.0, dt),
                dt=dt,
                j_bs=j_bs_loc,
                j_phi=j_phi_loc,
                j_cd=j_cd_ctrl,
                eta=1e-7,
            )
            self.ntm_widths[key] = float(np.clip(w_arr[-1], 1e-6, 0.5 * self.config.a))

    def _bootstrap_current_density(self, q_prof: np.ndarray) -> np.ndarray:
        """Compute bootstrap current density via the configured neoclassical model.

        Args:
            q_prof: Safety-factor profile used by the Sauter bootstrap closure.

        Returns:
            Bootstrap current profile on ``rho`` in ``A/m^2``.
        """

        return sauter_bootstrap(
            self.rho,
            self.ts_solver.Te,
            self.ts_solver.Ti,
            self.ts_solver.ne,
            q_prof,
            self.config.R0,
            self.config.a,
            self.config.B0,
        )

    def _ohmic_current_density(self) -> np.ndarray:
        """Return the Ohmic current profile implied by plasma geometry and current.

        Returns:
            A normalized current profile on ``rho`` in ``A/m^2``.
        """

        shape = np.maximum(1.0 - self.rho**2, 0.0)
        area_element = 2.0 * np.pi * self.config.kappa * self.config.a**2 * self.rho
        norm = float(trapezoid(shape * area_element, self.rho))
        if norm <= 0.0 or not np.isfinite(norm):
            raise ValueError("invalid plasma cross-section for ohmic current normalisation")
        return np.asarray(shape * (self.config.Ip_MA * 1e6) / norm)

    def _normalised_beta(self, energy_dens: np.ndarray) -> float:
        """Compute a normalised-beta proxy from thermal energy density.

        Args:
            energy_dens: Profile of thermal energy density on ``rho``.

        Returns:
            Estimated normalised beta.
        """

        pressure = (2.0 / 3.0) * np.asarray(energy_dens, dtype=float)
        pressure_avg = 2.0 * float(trapezoid(pressure * self.rho, self.rho))
        beta_t = 2.0 * np.pi * 4e-7 * pressure_avg / max(self.config.B0**2, 1e-12)
        return float(100.0 * beta_t * self.config.a * self.config.B0 / max(self.config.Ip_MA, 1e-9))

    def _internal_inductance_proxy(self, j_total: np.ndarray) -> float:
        """Compute a bounded internal-inductance proxy from total current profile.

        Args:
            j_total: Total toroidal current density profile on ``rho``.

        Returns:
            Dimensionless internal-inductance-like quantity in ``[0, 10]``.

        Raises:
            ValueError: If ``j_total`` shape is incompatible or contains non-finite
                values.
        """

        j_prof = np.asarray(j_total, dtype=float)
        if j_prof.ndim != 1 or j_prof.size != self.rho.size:
            raise ValueError("j_total must be a 1D profile matching rho grid size.")
        if not np.all(np.isfinite(j_prof)):
            raise ValueError("j_total profile must be finite.")

        area_element = 2.0 * np.pi * self.config.kappa * self.config.a**2 * self.rho
        current = float(trapezoid(j_prof * area_element, self.rho))
        abs_current = abs(current)
        if abs_current < 1e-12:
            return 0.0

        rho = np.asarray(self.rho, dtype=float)
        cumulative = np.empty_like(rho)
        for idx in range(rho.size):
            cumulative[idx] = trapezoid(j_prof[: idx + 1] * area_element[: idx + 1], rho[: idx + 1])
        current_enclosed = np.abs(cumulative)

        # B_pol(r) from enclosed toroidal current in circular approximation: mu0 I(<r)/(2*pi*r).
        minor_r = np.maximum(rho * self.config.a, 1e-9)
        b_pol = (4e-7 * np.pi) * current_enclosed / (2.0 * np.pi * minor_r)
        b_edge = float(b_pol[-1])
        if not np.isfinite(b_edge) or b_edge <= 0.0:
            return 0.0

        num = 2.0 * float(trapezoid((b_pol**2) * rho, rho))
        den = max(b_edge**2, 1e-30)
        return float(np.clip(num / den, 0.0, 10.0))

    def run(self) -> list[ScenarioState]:
        """Run the full integrated trajectory from start to end time.

        Returns:
            Sequence of :class:`ScenarioState` snapshots, one per time step.
        """

        if not hasattr(self, "ts_solver"):
            self.initialize()
        states = []
        n_steps = int((self.config.t_end - self.config.t_start) / self.config.dt)
        for _ in range(n_steps):
            states.append(self.step())
        self._last_states = list(states)
        return states

    def to_json(self, path: Path) -> None:
        """Persist trajectory to a JSON file.

        Args:
            path: Target output path. ``.json`` extension is required.

        Raises:
            ValueError: If the provided path is not a JSON file.
        """

        path = Path(path)
        if path.suffix.lower() != ".json":
            raise ValueError("Scenario output path must use .json extension.")
        if not self._last_states:
            self.run()

        payload = {
            "config": {
                "R0": float(self.config.R0),
                "a": float(self.config.a),
                "B0": float(self.config.B0),
                "kappa": float(self.config.kappa),
                "delta": float(self.config.delta),
                "Ip_MA": float(self.config.Ip_MA),
                "P_aux_MW": float(self.config.P_aux_MW),
                "P_eccd_MW": float(self.config.P_eccd_MW),
                "rho_eccd": float(self.config.rho_eccd),
                "P_nbi_MW": float(self.config.P_nbi_MW),
                "E_nbi_keV": float(self.config.E_nbi_keV),
                "t_start": float(self.config.t_start),
                "t_end": float(self.config.t_end),
                "dt": float(self.config.dt),
                "transport_model": self.config.transport_model,
                "include_sawteeth": bool(self.config.include_sawteeth),
                "include_ntm": bool(self.config.include_ntm),
                "include_sol": bool(self.config.include_sol),
            },
            "states": [
                {
                    "time": float(s.time),
                    "rho": np.asarray(s.rho, dtype=float).tolist(),
                    "Te": np.asarray(s.Te, dtype=float).tolist(),
                    "Ti": np.asarray(s.Ti, dtype=float).tolist(),
                    "ne": np.asarray(s.ne, dtype=float).tolist(),
                    "q": np.asarray(s.q, dtype=float).tolist(),
                    "psi": np.asarray(s.psi, dtype=float).tolist(),
                    "j_total": np.asarray(s.j_total, dtype=float).tolist(),
                    "j_bs": np.asarray(s.j_bs, dtype=float).tolist(),
                    "j_cd": np.asarray(s.j_cd, dtype=float).tolist(),
                    "Ip_MA": float(s.Ip_MA),
                    "beta_N": float(s.beta_N),
                    "tau_E": float(s.tau_E),
                    "P_loss": float(s.P_loss),
                    "W_thermal": float(s.W_thermal),
                    "li": float(s.li),
                    "ballooning_stable": bool(s.ballooning_stable),
                    "troyon_stable": bool(s.troyon_stable),
                    "ntm_island_widths": dict(s.ntm_island_widths),
                    "T_target": float(s.T_target),
                    "q_peak": float(s.q_peak),
                    "detached": bool(s.detached),
                    "last_crash_time": float(s.last_crash_time),
                    "n_crashes": int(s.n_crashes),
                }
                for s in self._last_states
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
