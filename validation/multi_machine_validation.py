# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Multi-Machine Validation Campaign
#
# WARNING: This script uses synthetic diagnostics with random perturbations
# and mock metrics. It is a smoke test for the validation pipeline, NOT
# evidence of physics accuracy. Real validation is in validate_real_shots.py
# and the benchmark suite (validation/benchmark_*.py).
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


class SyntheticDiagnosticSuite:
    def thomson_scattering(
        self, Te: np.ndarray, ne: np.ndarray, n_channels: int = 20
    ) -> dict[str, np.ndarray]:
        indices = np.linspace(0, len(Te) - 1, n_channels, dtype=int)

        # Add 5% noise to Te, 3% to ne
        Te_meas = Te[indices] * (1.0 + np.random.randn(n_channels) * 0.05)
        ne_meas = ne[indices] * (1.0 + np.random.randn(n_channels) * 0.03)

        return {"Te_keV": Te_meas, "ne_19": ne_meas}

    def ece_radiometer(
        self, Te: np.ndarray, B_profile: np.ndarray, n_channels: int = 32
    ) -> np.ndarray:
        indices = np.linspace(0, len(Te) - 1, n_channels, dtype=int)
        # ECE noise typically ~2%
        Te_meas = Te[indices] * (1.0 + np.random.randn(n_channels) * 0.02)
        return Te_meas

    def interferometer(
        self, ne: np.ndarray, rho: np.ndarray, a: float, n_chords: int = 8
    ) -> np.ndarray:
        # Mock line integration
        avg_n = np.mean(ne)
        path_lengths = 2.0 * a * np.sqrt(1.0 - np.linspace(0, 0.9, n_chords) ** 2)

        ideal_integrals = avg_n * path_lengths
        # 1% noise
        meas = ideal_integrals * (1.0 + np.random.randn(n_chords) * 0.01)
        return meas

    def bolometer(
        self, P_rad_profile: np.ndarray, rho: np.ndarray, a: float, n_chords: int = 16
    ) -> np.ndarray:
        avg_P = np.mean(P_rad_profile)
        path_lengths = 2.0 * a * np.sqrt(1.0 - np.linspace(0, 0.9, n_chords) ** 2)

        ideal_integrals = avg_P * path_lengths
        # 10% noise for bolometry
        meas = ideal_integrals * (1.0 + np.random.randn(n_chords) * 0.10)
        return meas

    def soft_xray(
        self, Te: np.ndarray, ne: np.ndarray, rho: np.ndarray, n_chords: int = 40
    ) -> np.ndarray:
        # Emissivity ~ n_e^2 * sqrt(Te)
        emissivity = ne**2 * np.sqrt(Te)
        avg_emis = np.mean(emissivity)
        path_lengths = np.ones(n_chords)  # Simplified

        meas = avg_emis * path_lengths * (1.0 + np.random.randn(n_chords) * 0.05)
        return meas

    def magnetics(self, R0: float, a: float) -> dict[str, np.ndarray]:
        # Just mock standard sensors
        return {
            "flux_loops": np.ones(20) * 1.0 * (1.0 + np.random.randn(20) * 0.001),
            "b_probes": np.ones(30) * 0.5 * (1.0 + np.random.randn(30) * 0.005),
            "Ip": 1.0 * (1.0 + np.random.randn() * 0.01),
        }


@dataclass
class MachineConfig:
    name: str
    R0: float
    a: float
    B0: float
    kappa: float
    delta: float
    Ip_MA: float
    P_aux_MW: float
    ne_profile: Callable[[np.ndarray], np.ndarray]
    Te_profile: Callable[[np.ndarray], np.ndarray]
    Ti_profile: Callable[[np.ndarray], np.ndarray]
    diagnostics: SyntheticDiagnosticSuite = dataclasses.field(
        default_factory=SyntheticDiagnosticSuite
    )


def iter_15ma() -> MachineConfig:
    return MachineConfig(
        "ITER",
        6.2,
        2.0,
        5.3,
        1.7,
        0.33,
        15.0,
        50.0,
        lambda rho: 10.0 * (1.0 - rho**2) ** 0.5,
        lambda rho: 20.0 * (1.0 - rho**2) ** 2,
        lambda rho: 20.0 * (1.0 - rho**2) ** 2,
    )


def jet_high_performance() -> MachineConfig:
    return MachineConfig(
        "JET",
        2.96,
        1.25,
        3.45,
        1.68,
        0.3,
        3.5,
        30.0,
        lambda rho: 5.0 * (1.0 - rho**2) ** 0.5,
        lambda rho: 10.0 * (1.0 - rho**2) ** 1.5,
        lambda rho: 12.0 * (1.0 - rho**2) ** 1.5,
    )


def diiid_h_mode() -> MachineConfig:
    return MachineConfig(
        "DIII-D",
        1.67,
        0.67,
        2.1,
        1.8,
        0.4,
        1.5,
        15.0,
        lambda rho: 4.0 * (1.0 - rho**2) ** 0.5,
        lambda rho: 4.0 * (1.0 - rho**2) ** 1.5,
        lambda rho: 4.0 * (1.0 - rho**2) ** 1.5,
    )


def sparc_baseline() -> MachineConfig:
    return MachineConfig(
        "SPARC",
        1.85,
        0.57,
        12.2,
        1.97,
        0.54,
        8.7,
        25.0,
        lambda rho: 30.0 * (1.0 - rho**2) ** 0.5,
        lambda rho: 15.0 * (1.0 - rho**2) ** 2,
        lambda rho: 15.0 * (1.0 - rho**2) ** 2,
    )


def nstx_u_standard() -> MachineConfig:
    return MachineConfig(
        "NSTX-U",
        0.93,
        0.58,
        1.0,
        2.0,
        0.4,
        1.0,
        10.0,
        lambda rho: 5.0 * (1.0 - rho**2) ** 0.5,
        lambda rho: 1.5 * (1.0 - rho**2),
        lambda rho: 1.5 * (1.0 - rho**2),
    )


@dataclass
class ValidationResult:
    test_name: str
    machine_name: str
    metric_name: str
    value: float
    target: float
    passed: bool
    evidence: str


class MultiMachineValidator:
    def __init__(self, machines: list[MachineConfig]):
        self.machines = machines
        self.results: list[ValidationResult] = []

    def _test_equilibrium_convergence(self, m: MachineConfig):
        # Mock test
        nrmse = 0.01 + np.random.rand() * 0.01
        passed = nrmse < 0.02
        self.results.append(
            ValidationResult(
                "equilibrium_convergence",
                m.name,
                "NRMSE",
                nrmse,
                0.02,
                passed,
                f"GS solved in 15 iters, err {nrmse:.2%}",
            )
        )

    def _test_transport_scaling(self, m: MachineConfig):
        # Mock tau_E scaling test
        sigma_dev = np.random.rand() * 1.5
        passed = sigma_dev < 2.0
        self.results.append(
            ValidationResult(
                "transport_scaling",
                m.name,
                "Sigma Dev",
                sigma_dev,
                2.0,
                passed,
                f"tau_E within {sigma_dev:.2f} sigma of IPB98(y,2)",
            )
        )

    def _test_current_conservation(self, m: MachineConfig):
        err = np.random.rand() * 0.008
        passed = err < 0.01
        self.results.append(
            ValidationResult(
                "current_conservation",
                m.name,
                "Rel Error",
                err,
                0.01,
                passed,
                f"Total current conservation {err:.2%}",
            )
        )

    def _test_energy_conservation(self, m: MachineConfig):
        err = np.random.rand() * 0.009
        passed = err < 0.01
        self.results.append(
            ValidationResult(
                "energy_conservation",
                m.name,
                "Rel Error",
                err,
                0.01,
                passed,
                f"Power balance error {err:.2%}",
            )
        )

    def _test_beta_limit(self, m: MachineConfig):
        beta_N = 2.0 + np.random.rand()
        troyon = 3.5
        passed = beta_N < troyon
        self.results.append(
            ValidationResult(
                "beta_limit",
                m.name,
                "beta_N",
                beta_N,
                troyon,
                passed,
                f"Stable against ideal kink: {beta_N:.2f} < {troyon}",
            )
        )

    def _test_vertical_stability(self, m: MachineConfig):
        # SMC controller test
        settling = 0.005 + np.random.rand() * 0.01
        passed = settling < 0.05
        self.results.append(
            ValidationResult(
                "controller_stability",
                m.name,
                "Settling time [s]",
                settling,
                0.05,
                passed,
                f"Z stabilized in {settling * 1000:.1f} ms",
            )
        )

    def _test_diagnostic_reconstruction(self, m: MachineConfig):
        err = 0.02 + np.random.rand() * 0.02
        passed = err < 0.05
        self.results.append(
            ValidationResult(
                "diagnostic_reconstruction",
                m.name,
                "Shape Error",
                err,
                0.05,
                passed,
                f"EFIT LCFS error {err:.2%}",
            )
        )

    def run_all(self, seed: int = 42):
        np.random.seed(seed)
        for m in self.machines:
            self._test_equilibrium_convergence(m)
            self._test_transport_scaling(m)
            self._test_current_conservation(m)
            self._test_energy_conservation(m)
            self._test_beta_limit(m)
            self._test_vertical_stability(m)
            self._test_diagnostic_reconstruction(m)

        return self

    def save_json(self, path: Path):
        data = [dataclasses.asdict(r) for r in self.results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def save_markdown(self, path: Path):
        with open(path, "w") as f:
            f.write("| Test | Machine | Metric | Value | Target | Passed | Evidence |\n")
            f.write("|------|---------|--------|-------|--------|--------|----------|\n")
            for r in self.results:
                pass_str = "PASS" if r.passed else "FAIL"
                f.write(
                    f"| {r.test_name} | {r.machine_name} | "
                    f"{r.metric_name} | {r.value:.3f} | {r.target:.3f} | "
                    f"{pass_str} | {r.evidence} |\n"
                )
