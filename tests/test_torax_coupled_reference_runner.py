# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — tests of the coupled TORAX reference runner
"""The runner integrates TORAX outputs correctly and keeps the minimum warm wall time.

TORAX lives only in the dedicated ``.venv-torax`` environment.  These tests drive the
runner through ``main`` and ``build_reference`` with a stand-in ``torax`` module that
returns analytic profiles in the layout TORAX 1.4.3 uses (state on ``rho_norm`` with its
two faces, sources on the uniform ``rho_cell_norm`` grid) and with a scripted clock.
Everything the runner computes itself runs for real, and its output must pass the parity
benchmark's own authentication.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from validation import benchmark_torax_coupled_transport_parity as benchmark
from validation import torax_coupled_reference_runner as runner
from validation.reference_data.torax.coupled_transport_model_intersection_deck import (
    MODEL_INTERSECTION,
)

FloatArray = npt.NDArray[np.float64]
CELLS = 16
CENTRES = (np.arange(CELLS, dtype=np.float64) + 0.5) / CELLS
RHO = np.concatenate(([0.0], CENTRES, [1.0]))
TIME = np.array([0.0, 0.01, 0.02])
TEMPERATURE_KEV = 2.0
DENSITY_M3 = 3.0e19
SOURCES = {
    "p_generic_heat_i": 5.0e4,
    "p_generic_heat_e": 7.0e4,
    "s_generic_particle": 2.0e18,
    "j_generic_current": 1.0e5,
    "ei_exchange": -3.0e3,
}
COLD_S = 7.5
WARM_S = (0.12, 0.09, 0.11, 0.10, 0.13)
REFINED_S = 0.4


@dataclass(frozen=True)
class _Coordinate:
    values: FloatArray


@dataclass(frozen=True)
class _Variable:
    values: FloatArray
    dims: tuple[str, str]
    coords: dict[str, _Coordinate]


def _data_tree(temperature_offset_kev: float, centres: FloatArray) -> types.SimpleNamespace:
    def state(values: FloatArray) -> _Variable:
        return _Variable(
            np.tile(values, (TIME.size, 1)), ("time", "rho_norm"), {"rho_norm": _Coordinate(RHO)}
        )

    def source(value: float) -> _Variable:
        return _Variable(
            np.full((TIME.size, centres.size), value),
            ("time", "rho_cell_norm"),
            {"rho_cell_norm": _Coordinate(centres)},
        )

    profiles = {
        "T_i": state(np.full(RHO.size, TEMPERATURE_KEV + temperature_offset_kev)),
        "T_e": state(np.full(RHO.size, TEMPERATURE_KEV)),
        "n_e": state(np.full(RHO.size, DENSITY_M3)),
        "psi": state(RHO**2),
    }
    profiles.update({name: source(value) for name, value in SOURCES.items()})
    return types.SimpleNamespace(profiles=profiles, time=_Coordinate(TIME))


@dataclass
class _FakeTorax:
    """Records every requested fixed step; perturbs the profiles on one chosen call."""

    perturb_call: int | None = None
    source_centres: FloatArray = field(default_factory=lambda: CENTRES.copy())
    requested_dt: list[float] = field(default_factory=list)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = types.ModuleType("torax")
        module.__dict__["__version__"] = "1.4.3"
        module.__dict__["ToraxConfig"] = types.SimpleNamespace(from_dict=lambda config: config)
        module.__dict__["run_simulation"] = self.run_simulation
        monkeypatch.setitem(sys.modules, "torax", module)

    def run_simulation(
        self, config: dict[str, Any], *, progress_bar: bool
    ) -> tuple[types.SimpleNamespace, types.SimpleNamespace]:
        assert progress_bar is False
        self.requested_dt.append(float(config["numerics"]["fixed_dt"]))
        offset = 0.5 if len(self.requested_dt) == self.perturb_call else 0.0
        tree = _data_tree(offset, self.source_centres)
        return tree, types.SimpleNamespace(sim_error="SimError.NO_ERROR")


def _scripted_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks: list[float] = []
    for index, seconds in enumerate((COLD_S, *WARM_S, REFINED_S)):
        ticks.extend((100.0 * index, 100.0 * index + seconds))
    clock: Iterator[float] = iter(ticks)
    monkeypatch.setattr(runner, "time", types.SimpleNamespace(perf_counter=lambda: next(clock)))


def test_main_writes_an_analytic_reference_that_the_benchmark_authenticates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    torax = _FakeTorax()
    torax.install(monkeypatch)
    _scripted_clock(monkeypatch)
    target = tmp_path / "out" / "reference.json"

    assert runner.main(["--output", str(target)]) == 0

    assert "deterministic=True" in capsys.readouterr().out
    reference = json.loads(target.read_text(encoding="utf-8"))
    time_config = MODEL_INTERSECTION["time"]
    assert torax.requested_dt == [time_config["primary_dt_s"]] * 6 + [time_config["refined_dt_s"]]
    assert reference["schema"] == runner.SCHEMA
    assert (
        reference["provenance"]["runner_sha256"]
        == hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest()
    )

    geometry = MODEL_INTERSECTION["geometry"]
    volume = 4.0 * np.pi**2 * geometry["major_radius_m"] * geometry["minor_radius_m"] ** 2
    area = 2.0 * np.pi * geometry["minor_radius_m"] ** 2
    # Both rules integrate rho exactly: the trapezoid on rho_norm, the midpoint sum on cells.
    budget = reference["primary"]["state_budgets"][0]
    kev_j = 1.602176634e-16
    assert budget["thermal_energy_j"] == pytest.approx(
        1.5 * DENSITY_M3 * kev_j * 2.0 * TEMPERATURE_KEV * volume * 0.5, rel=1e-13
    )
    assert budget["particle_inventory"] == pytest.approx(DENSITY_M3 * volume * 0.5, rel=1e-13)
    assert budget["poloidal_flux_l2"] == pytest.approx(float(np.linalg.norm(RHO**2)))
    totals = reference["primary"]["source_totals"]
    assert totals["ion_heat_w"] == pytest.approx([SOURCES["p_generic_heat_i"] * volume * 0.5] * 3)
    assert totals["driven_current_a"] == pytest.approx(
        [SOURCES["j_generic_current"] * area * 0.5] * 3
    )
    comparison = reference["primary"]["comparison_profiles"]["ion_temperature_kev"]
    assert np.allclose(comparison, TEMPERATURE_KEV)
    assert len(comparison[0]) == geometry["comparison_points"]

    samples = reference["runtime_samples_seconds"]["warm_primary"]
    assert samples == pytest.approx(list(WARM_S))
    assert reference["runtime_seconds"]["warm_primary"] == min(samples)
    assert reference["runtime_seconds"]["cold_primary"] == pytest.approx(COLD_S)
    assert reference["runtime_seconds"]["refined"] == pytest.approx(REFINED_S)
    determinism = reference["determinism"]
    assert determinism["byte_identical_scientific_projection"] is True
    assert determinism["warm_projection_sha256s"] == [determinism["primary_projection_sha256"]] * 5

    assert benchmark.load_reference(target)["runtime_seconds"]["warm_primary"] == min(samples)


def test_a_replay_that_differs_from_the_primary_fails_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _FakeTorax(perturb_call=4).install(monkeypatch)
    target = tmp_path / "reference.json"

    assert runner.main(["--output", str(target)]) == 0

    assert "deterministic=False" in capsys.readouterr().out
    determinism = json.loads(target.read_text(encoding="utf-8"))["determinism"]
    digests = determinism["warm_projection_sha256s"]
    assert digests[2] != determinism["primary_projection_sha256"]
    assert digests[:2] + digests[3:] == [determinism["primary_projection_sha256"]] * 4
    with pytest.raises(ValueError, match="lacks deterministic replay"):
        benchmark.load_reference(target)


def test_build_reference_refuses_to_run_without_a_warm_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeTorax().install(monkeypatch)
    monkeypatch.setattr(runner, "WARM_REPEATS", 0)

    with pytest.raises(ValueError, match="at least one warm replay"):
        runner.build_reference()


def test_build_reference_rejects_a_torax_source_grid_that_is_not_uniform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skewed = CENTRES.copy()
    skewed[3] += 0.01
    _FakeTorax(source_centres=skewed).install(monkeypatch)

    with pytest.raises(ValueError, match="uniform cell grid"):
        runner.build_reference()
