# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — 3D Field-Line Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Unit tests for reduced 3D field-line tracing and Poincare diagnostics."""

from __future__ import annotations

import numpy as np

from scpn_fusion.core.equilibrium_3d import FourierMode3D, VMECStyleEquilibrium3D
from scpn_fusion.core.fieldline_3d import FieldLineTracer3D
from scpn_fusion.core.geometry_3d import Reactor3DBuilder


class _DummyKernel:
    def __init__(self) -> None:
        self.NR = 81
        self.NZ = 81
        self.R = np.linspace(1.0, 3.0, self.NR)
        self.Z = np.linspace(-1.0, 1.0, self.NZ)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        radius2 = (self.RR - 2.0) ** 2 + self.ZZ**2
        self.Psi = 1.0 - radius2
        self.cfg = {"coils": []}

    def solve_equilibrium(self) -> None:
        return None

    def find_x_point(self, psi: np.ndarray) -> tuple[tuple[float, float], float]:
        _ = psi
        return (2.894, 0.0), 0.2


def test_fieldline_trace_shape_and_monotonic_phi() -> None:
    eq = VMECStyleEquilibrium3D(r_axis=2.0, z_axis=0.0, a_minor=0.45, kappa=1.4)
    tracer = FieldLineTracer3D(eq, rotational_transform=0.42)
    trace = tracer.trace_line(toroidal_turns=6, steps_per_turn=128)

    assert trace.xyz.shape == (6 * 128 + 1, 3)
    assert trace.rho.shape == (6 * 128 + 1,)
    assert trace.theta.shape == (6 * 128 + 1,)
    assert trace.phi.shape == (6 * 128 + 1,)
    assert np.isfinite(trace.xyz).all()
    assert np.all(np.diff(trace.phi) > 0.0)


def test_poincare_section_crossing_count_matches_turns() -> None:
    eq = VMECStyleEquilibrium3D(r_axis=2.0, z_axis=0.0, a_minor=0.45, kappa=1.4)
    tracer = FieldLineTracer3D(eq, rotational_transform=0.37)
    turns = 10
    trace = tracer.trace_line(toroidal_turns=turns, steps_per_turn=192)
    section = tracer.poincare_section(trace, phi_plane=0.0)

    assert section.xyz.shape[1] == 3
    assert section.rz.shape[1] == 2
    assert len(section.rz) == turns


def test_helical_mode_changes_3d_trace_geometry() -> None:
    eq_axisym = VMECStyleEquilibrium3D(
        r_axis=2.0,
        z_axis=0.0,
        a_minor=0.45,
        kappa=1.4,
    )
    eq_3d = VMECStyleEquilibrium3D(
        r_axis=2.0,
        z_axis=0.0,
        a_minor=0.45,
        kappa=1.4,
        modes=[FourierMode3D(m=1, n=1, r_cos=0.06, z_sin=0.04)],
    )

    tracer_axis = FieldLineTracer3D(eq_axisym, rotational_transform=0.4)
    tracer_3d = FieldLineTracer3D(eq_3d, rotational_transform=0.4)

    trace_axis = tracer_axis.trace_line(toroidal_turns=8, steps_per_turn=160)
    trace_3d = tracer_3d.trace_line(toroidal_turns=8, steps_per_turn=160)

    delta = np.max(np.abs(trace_axis.xyz - trace_3d.xyz))
    assert float(delta) > 1e-3


def test_builder_poincare_map_from_kernel_path() -> None:
    kernel = _DummyKernel()
    builder = Reactor3DBuilder(kernel=kernel, solve_equilibrium=False)
    trace, poincare = builder.generate_poincare_map(
        toroidal_turns=8,
        steps_per_turn=160,
        phi_planes=[0.0, np.pi / 2.0],
        toroidal_modes=[FourierMode3D(m=1, n=1, r_cos=0.04)],
    )

    assert trace.xyz.shape[1] == 3
    assert len(poincare) == 2
    assert 0.0 in poincare
    assert np.pi / 2.0 in poincare
    assert poincare[0.0].shape[1] == 2
    assert poincare[np.pi / 2.0].shape[1] == 2
    assert len(poincare[0.0]) > 0
    assert len(poincare[np.pi / 2.0]) > 0
