# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Shared physical case public regression tests
"""Real direct and dispatched numerical ABI admission and ownership regressions."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
import importlib

import numpy as np
from numpy.typing import NDArray
import pytest

from scpn_fusion.core.multigrid_solve import multigrid_solve
from scpn_fusion.diagnostics.synthetic_sensors import measure_magnetics
from scpn_fusion.core._multi_compat import BackendTier, dispatch_for_tier

FloatArray = NDArray[np.float64]
Sensor = Callable[..., FloatArray]
Multigrid = Callable[..., tuple[FloatArray, float, int, bool]]


@pytest.fixture(params=["numpy", "rust", "dispatch_numpy", "dispatch_rust", "rust_compat"])
def sensor(request: pytest.FixtureRequest) -> Sensor:
    """Exercise actual direct functions and registered providers without fake backends."""
    if request.param == "numpy":
        return measure_magnetics
    if request.param == "rust":
        return cast(Sensor, importlib.import_module("scpn_fusion_rs").measure_magnetics)
    if request.param == "rust_compat":
        return cast(
            Sensor, importlib.import_module("scpn_fusion.core._rust_compat").rust_measure_magnetics
        )
    tier = BackendTier.NUMPY if request.param == "dispatch_numpy" else BackendTier.RUST
    return cast(Sensor, dispatch_for_tier("measure_magnetics", tier))


@pytest.fixture(params=["numpy", "rust", "dispatch_numpy", "dispatch_rust", "rust_compat"])
def multigrid(request: pytest.FixtureRequest) -> Multigrid:
    """Exercise the actual full-solve function and both registered multigrid providers."""
    if request.param == "numpy":
        return multigrid_solve
    if request.param == "rust":
        return cast(Multigrid, importlib.import_module("scpn_fusion_rs").multigrid_vcycle)
    if request.param == "rust_compat":
        return cast(
            Multigrid,
            importlib.import_module("scpn_fusion.core._rust_compat").rust_multigrid_vcycle,
        )
    tier = BackendTier.NUMPY if request.param == "dispatch_numpy" else BackendTier.RUST
    return cast(Multigrid, dispatch_for_tier("multigrid_solve", tier))


def multigrid_arguments() -> dict[str, Any]:
    """Return a tiny valid full-solve problem whose initial flux already converges."""
    return dict(
        source=np.zeros((5, 7)),
        psi_bc=np.zeros((5, 7)),
        r_min=1.0,
        r_max=3.0,
        z_min=-1.0,
        z_max=1.0,
        nr=7,
        nz=5,
        tol=1e-6,
        max_cycles=500,
    )


def sensor_arguments() -> dict[str, Any]:
    """Return affordable, valid interpolation inputs spanning every wall probe."""
    return dict(psi=np.ones((5, 7)), nr=7, nz=5, r_min=-10.0, r_max=20.0, z_min=-10.0, z_max=10.0)


@pytest.mark.parametrize("field", ["nr", "nz"])
@pytest.mark.parametrize("value", [True, np.bool_(False), 7.0, "7", 7j, np.array(7), Decimal(7)])
def test_sensor_integer_kind(sensor: Sensor, field: str, value: object) -> None:
    """Reject each unsupported dimension kind through each real entry point."""
    arguments = sensor_arguments()
    arguments[field] = value
    with pytest.raises(TypeError):
        sensor(**arguments)


@pytest.mark.parametrize("field", ["nr", "nz"])
@pytest.mark.parametrize("value", [-1, 0, 1, 1 << 63, 1 << 200, np.uint64((1 << 64) - 1)])
def test_sensor_integer_domain(sensor: Sensor, field: str, value: object) -> None:
    """Reject each invalid mathematical dimension before unsigned extraction or copies."""
    arguments = sensor_arguments()
    arguments[field] = value
    with pytest.raises(ValueError):
        sensor(**arguments)


@pytest.mark.parametrize("field", ["r_min", "r_max", "z_min", "z_max"])
@pytest.mark.parametrize(
    "value", [True, np.bool_(False), "1", 1j, np.array(1.0), Decimal(1), np.longdouble(1)]
)
def test_sensor_real_kind(sensor: Sensor, field: str, value: object) -> None:
    """Reject unsupported real scalar kinds independently for all four bounds."""
    arguments = sensor_arguments()
    arguments[field] = value
    with pytest.raises(TypeError):
        sensor(**arguments)


@pytest.mark.parametrize("field", ["r_min", "r_max", "z_min", "z_max"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 9007199254740993, 1 << 2000])
def test_sensor_real_domain(sensor: Sensor, field: str, value: object) -> None:
    """Reject nonfinite and mathematically inexact binary64 bounds."""
    arguments = sensor_arguments()
    arguments[field] = value
    with pytest.raises(ValueError):
        sensor(**arguments)


@pytest.mark.parametrize(
    "value",
    [
        None,
        [[1.0]],
        (1.0,),
        np.ones((5, 7), dtype=np.float32),
        np.ones((5, 7), dtype=np.int64),
        np.ones((5, 7), dtype=complex),
        np.ones((5, 7), dtype=object),
        np.ones((5, 7), dtype=">f8"),
    ],
)
def test_sensor_array_kind(sensor: Sensor, value: object) -> None:
    """Refuse array-like, wrong dtype and byte-swapped inputs before shape checks."""
    arguments = sensor_arguments()
    arguments["psi"] = value
    with pytest.raises(TypeError):
        sensor(**arguments)


@pytest.mark.parametrize("value", [np.array(1.0), np.ones(5), np.ones((1, 5, 7)), np.ones((3, 3))])
def test_sensor_array_rank_shape(sensor: Sensor, value: FloatArray) -> None:
    """Give wrong rank and wrong extents the shared ValueError class."""
    arguments = sensor_arguments()
    arguments["psi"] = value
    with pytest.raises(ValueError):
        sensor(**arguments)


@pytest.mark.parametrize("shape", [(2, 2), (2, 3), (3, 2)])
@pytest.mark.parametrize("kind", [int, np.int32, np.int64, np.uint64])
def test_sensor_minimal_dimensions(
    sensor: Sensor, shape: tuple[int, int], kind: Callable[[int], object]
) -> None:
    """Accept signed and unsigned scalar dimensions at every bilinear minimum shape."""
    nz, nr = shape
    arguments = sensor_arguments()
    arguments.update(psi=np.ones(shape), nr=kind(nr), nz=kind(nz))
    np.testing.assert_allclose(sensor(**arguments), np.ones(20), rtol=4e-16, atol=0.0)


@pytest.mark.parametrize(
    "kind", [float, np.float16, np.float32, np.float64, int, np.int64, np.uint64]
)
def test_sensor_real_promotions(sensor: Sensor, kind: Callable[[int], object]) -> None:
    """Accept finite scalar floating promotions and exact integral bounds."""
    arguments = sensor_arguments()
    arguments.update(r_min=kind(1), r_max=kind(20), z_min=kind(1), z_max=kind(10))
    np.testing.assert_allclose(sensor(**arguments), np.ones(20), rtol=4e-16, atol=0.0)


@pytest.mark.parametrize(
    "layout",
    [
        "c",
        "fortran",
        "transpose",
        "reverse",
        "positive_stride",
        "zero_stride",
        "readonly",
        "unaligned",
    ],
)
def test_sensor_logical_layout_and_fresh_output(sensor: Sensor, layout: str) -> None:
    """Preserve real logical data across supported storage layouts and own fresh output."""
    arguments = sensor_arguments()
    base = np.arange(35.0).reshape(5, 7)
    if layout == "fortran":
        array = np.asfortranarray(base)
    elif layout == "transpose":
        array = base.T.copy().T
    elif layout == "reverse":
        array = base[::-1, ::-1]
    elif layout == "positive_stride":
        array = np.arange(140.0).reshape(10, 14)[::2, ::2]
    elif layout == "zero_stride":
        array = np.broadcast_to(base[:1], base.shape)
    elif layout == "readonly":
        array = base.copy()
        array.flags.writeable = False
    elif layout == "unaligned":
        array = np.ndarray(
            base.shape, dtype=np.float64, buffer=bytearray(base.nbytes + 1), offset=1
        )
        array[:] = base
    else:
        array = base
    before = array.copy()
    arguments["psi"] = array
    actual = sensor(**arguments)
    arguments["psi"] = np.array(before, order="C", copy=True)
    np.testing.assert_array_equal(actual, sensor(**arguments))
    np.testing.assert_array_equal(array, before)
    assert actual.dtype == np.float64 and actual.flags.c_contiguous
    assert not np.shares_memory(actual, array)
    actual[:] = -1.0
    np.testing.assert_array_equal(array, before)


def test_sensor_analytic_bilinear_mapping(sensor: Sensor) -> None:
    """Use an independent linear-field oracle over the actual endpoint-inclusive wall."""
    arguments = sensor_arguments()
    rr, zz = np.meshgrid(np.linspace(-10.0, 20.0, 7), np.linspace(-10.0, 10.0, 5))
    arguments["psi"] = 2.0 * rr + 3.0 * zz + 1.0
    theta = np.linspace(0.0, 2.0 * np.pi, 20)
    expected = 2.0 * (6.0 + 3.5 * np.cos(theta)) + 3.0 * (6.3 * np.sin(theta)) + 1.0
    np.testing.assert_allclose(sensor(**arguments), expected, rtol=2e-15, atol=2e-14)


def test_sensor_precedence_and_all_cells(sensor: Sensor) -> None:
    """Prove dtype before rank, scalar kind before domain, and all-cell finiteness."""
    arguments = sensor_arguments()
    arguments.update(psi=np.ones(3, dtype=np.float32), nr=-1)
    with pytest.raises(TypeError):
        sensor(**arguments)
    arguments = sensor_arguments()
    arguments.update(nr=-1, z_max=True)
    with pytest.raises(TypeError):
        sensor(**arguments)
    arguments = sensor_arguments()
    arguments["psi"][0, 0] = np.nan
    with pytest.raises(ValueError):
        sensor(**arguments)
    arguments = sensor_arguments()
    arguments.update(nr=1 << 40, nz=1 << 40)
    with pytest.raises(ValueError):
        sensor(**arguments)


@pytest.mark.parametrize("witness", ["coordinate", "weighted_sum"])
def test_sensor_actual_arithmetic_failure(sensor: Sensor, witness: str) -> None:
    """Distinguish invalid numerical execution from admitted finite interpolation inputs."""
    arguments = sensor_arguments()
    if witness == "coordinate":
        arguments.update(r_min=0.0, r_max=1e-308)
    else:
        arguments["psi"][:] = np.finfo(np.float64).max
        arguments.update(r_min=3.0, r_max=20.0)
    with pytest.raises(RuntimeError):
        sensor(**arguments)


@pytest.mark.parametrize("shape", [(3, 3), (3, 5), (5, 3), (6, 8), (8, 6), (13, 15), (17, 17)])
def test_multigrid_manufactured_solution(multigrid: Multigrid, shape: tuple[int, int]) -> None:
    """Recover a nonzero manufactured field on every required rectangular geometry."""
    nz, nr = shape
    _, zz = np.meshgrid(np.linspace(1.0, 3.0, nr), np.linspace(-1.0, 1.0, nz))
    expected: FloatArray = np.asarray(zz**2 + 1.0, dtype=np.float64)
    initial = expected.copy()
    initial[1:-1, 1:-1] = 0.0
    source = np.full(shape, 2.0)
    result, residual, cycles, converged = multigrid(
        source, initial, 1.0, 3.0, -1.0, 1.0, nr, nz, tol=1e-9, max_cycles=300
    )
    assert converged and residual < 1e-9 and cycles > 0
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-8)
    np.testing.assert_array_equal(result[[0, -1]], initial[[0, -1]])
    np.testing.assert_array_equal(result[:, [0, -1]], initial[:, [0, -1]])
    assert result.flags.c_contiguous and not np.shares_memory(result, initial)


@pytest.mark.parametrize(
    "field", ["omega", "pre_smooth", "post_smooth", "min_grid", "max_cycles", "tol"]
)
def test_numpy_multigrid_initial_convergence_control_validation(field: str) -> None:
    """Validate every tuning kind even when an all-zero initial field already converges."""
    keywords: dict[str, Any] = {field: True}
    with pytest.raises(TypeError):
        multigrid_solve(np.zeros((3, 3)), np.zeros((3, 3)), 1.0, 3.0, -1.0, 1.0, 3, 3, **keywords)


@pytest.mark.parametrize("field", ["source", "psi_bc"])
@pytest.mark.parametrize(
    "value,error",
    [
        (None, TypeError),
        ([[1.0]], TypeError),
        (np.ones((5, 7), dtype=np.float32), TypeError),
        (np.ones((5, 7), dtype=">f8"), TypeError),
        (np.ones((5, 7), dtype=np.int64), TypeError),
        (np.ones((5, 7), dtype=complex), TypeError),
        (np.ones((5, 7), dtype=object), TypeError),
        (np.array(1.0), ValueError),
        (np.ones(7), ValueError),
        (np.ones((1, 5, 7)), ValueError),
        (np.ones((3, 3)), ValueError),
        (np.full((5, 7), np.nan), ValueError),
    ],
)
def test_multigrid_array_admission(
    multigrid: Multigrid, field: str, value: object, error: type[Exception]
) -> None:
    """Exercise exact array kind, rank, shape and all-cell finiteness on each argument."""
    arguments = multigrid_arguments()
    arguments[field] = value
    with pytest.raises(error):
        multigrid(**arguments)


@pytest.mark.parametrize("field", ["nr", "nz", "max_cycles"])
@pytest.mark.parametrize(
    "value,error",
    [
        (True, TypeError),
        (np.bool_(True), TypeError),
        (3.0, TypeError),
        ("3", TypeError),
        (np.array(3), TypeError),
        (-1, ValueError),
        (0, ValueError),
        (1 << 63, ValueError),
        (1 << 200, ValueError),
        (np.uint64((1 << 64) - 1), ValueError),
    ],
)
def test_multigrid_integer_admission(
    multigrid: Multigrid, field: str, value: object, error: type[Exception]
) -> None:
    """Reject delivered integer kind and value failures before narrowing or allocation."""
    arguments = multigrid_arguments()
    arguments[field] = value
    with pytest.raises(error):
        multigrid(**arguments)


@pytest.mark.parametrize("field", ["r_min", "r_max", "z_min", "z_max", "tol"])
@pytest.mark.parametrize(
    "value,error",
    [
        (True, TypeError),
        (np.bool_(True), TypeError),
        ("1", TypeError),
        (1j, TypeError),
        (np.array(1.0), TypeError),
        (Decimal(1), TypeError),
        (np.longdouble(1), TypeError),
        (np.inf, ValueError),
        (np.nan, ValueError),
        (9007199254740993, ValueError),
        (1 << 2000, ValueError),
    ],
)
def test_multigrid_real_admission(
    multigrid: Multigrid, field: str, value: object, error: type[Exception]
) -> None:
    """Reject unsupported, inexact and nonfinite real values for each bound and tolerance."""
    arguments = multigrid_arguments()
    arguments[field] = value
    with pytest.raises(error):
        multigrid(**arguments)


@pytest.mark.parametrize("shape", [(0, 3), (1, 3), (2, 3), (3, 0), (3, 1), (3, 2)])
def test_multigrid_minimum_dimensions(multigrid: Multigrid, shape: tuple[int, int]) -> None:
    """Refuse each undersized axis before any stencil or unsigned subtraction."""
    nz, nr = shape
    arguments = multigrid_arguments()
    arguments.update(source=np.zeros(shape), psi_bc=np.zeros(shape), nr=nr, nz=nz)
    with pytest.raises(ValueError):
        multigrid(**arguments)


def test_multigrid_initial_zero_cycle_and_defaults(multigrid: Multigrid) -> None:
    """Report zero cycles on initial convergence with documented default controls."""
    arguments = multigrid_arguments()
    del arguments["tol"], arguments["max_cycles"]
    result, residual, cycles, converged = multigrid(**arguments)
    assert cycles == 0 and residual == 0.0 and converged
    assert result.flags.c_contiguous and result.dtype == np.float64
    assert not np.shares_memory(result, arguments["psi_bc"])


def test_multigrid_finite_budget_exhaustion(multigrid: Multigrid) -> None:
    """Keep truthful finite nonconvergence distinct from numerical exceptions."""
    arguments = multigrid_arguments()
    arguments.update(
        source=np.ones((17, 17)), psi_bc=np.zeros((17, 17)), nr=17, nz=17, tol=1e-30, max_cycles=1
    )
    result, residual, cycles, converged = multigrid(**arguments)
    assert not converged and cycles == 1 and np.isfinite(residual) and residual >= 1e-30
    assert np.all(np.isfinite(result))


def test_multigrid_actual_numerical_failure(multigrid: Multigrid) -> None:
    """Reject actual stencil overflow before NaN can disappear in a maximum reduction."""
    arguments = multigrid_arguments()
    arguments["psi_bc"][:] = np.finfo(np.float64).max
    with pytest.raises(RuntimeError):
        multigrid(**arguments)


@pytest.mark.parametrize(
    "geometry",
    ["nominal_overflow", "spacing_square", "underflow", "collapsed_axis", "nonpositive_radius"],
)
def test_multigrid_geometry_refusal(multigrid: Multigrid, geometry: str) -> None:
    """Refuse actual nonrepresentable geometry before any solve, even on zero inputs."""
    arguments = multigrid_arguments()
    if geometry == "nominal_overflow":
        arguments.update(z_min=-1e308, z_max=1e308)
    elif geometry == "spacing_square":
        arguments.update(r_min=1e200, r_max=2e200)
    elif geometry == "underflow":
        arguments.update(r_min=1e-200, r_max=2e-200)
    elif geometry == "collapsed_axis":
        arguments.update(r_min=1.0, r_max=np.nextafter(1.0, 2.0))
    else:
        arguments.update(r_min=0.0)
    with pytest.raises(ValueError):
        multigrid(**arguments)


def test_multigrid_tiny_positive_radius_operator(multigrid: Multigrid) -> None:
    """Recover the cubic radial stencil using actual positive radii below the former floor."""
    nr, nz = 9, 7
    radius = np.linspace(1e-12, 3e-12, nr)
    rr, _ = np.meshgrid(radius, np.linspace(-1e-12, 1e-12, nz))
    expected: FloatArray = np.asarray(rr**3, dtype=np.float64)
    initial = expected.copy()
    initial[1:-1, 1:-1] = 0.0
    dr = radius[1] - radius[0]
    source = 3.0 * rr - dr * dr / rr
    result, residual, cycles, converged = multigrid(
        source, initial, 1e-12, 3e-12, -1e-12, 1e-12, nr, nz, tol=1e-22, max_cycles=200
    )
    assert converged and cycles > 0 and residual < 1e-22
    np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-42)


@pytest.mark.parametrize("field", ["source", "psi_bc"])
@pytest.mark.parametrize(
    "layout", ["fortran", "transpose", "reverse", "stride", "zero_stride", "readonly", "unaligned"]
)
def test_multigrid_layout_ownership(multigrid: Multigrid, field: str, layout: str) -> None:
    """Accept each independent logical layout and preserve every original input cell."""
    arguments = multigrid_arguments()
    arguments.update(source=np.ones((5, 7)), max_cycles=1, tol=1e-30)
    base = np.arange(35.0, dtype=np.float64).reshape(5, 7)
    if layout == "fortran":
        array = np.asfortranarray(base)
    elif layout == "transpose":
        array = base.T.copy().T
    elif layout == "reverse":
        array = base[::-1, ::-1]
    elif layout == "stride":
        array = np.arange(140.0).reshape(10, 14)[::2, ::2]
    elif layout == "zero_stride":
        array = np.broadcast_to(base[:1], base.shape)
    elif layout == "unaligned":
        array = np.ndarray(
            base.shape, dtype=np.float64, buffer=bytearray(base.nbytes + 1), offset=1
        )
        array[:] = base
    else:
        array = base.copy()
        array.flags.writeable = False
    arguments[field] = array
    source_before = arguments["source"].copy()
    boundary_before = arguments["psi_bc"].copy()
    result = multigrid(**arguments)
    arguments[field] = np.array(array, order="C", copy=True)
    expected = multigrid(**arguments)
    np.testing.assert_array_equal(result[0], expected[0])
    assert result[1:] == expected[1:]
    np.testing.assert_array_equal(array, source_before if field == "source" else boundary_before)
    assert result[0].flags.c_contiguous
    assert not np.shares_memory(result[0], array)
    result[0][:] = -99.0
    np.testing.assert_array_equal(array, source_before if field == "source" else boundary_before)


@pytest.mark.parametrize("field", ["nr", "nz", "max_cycles"])
@pytest.mark.parametrize("kind", [np.int32, np.int64, np.uint64])
def test_multigrid_numpy_integer_kinds(
    multigrid: Multigrid, field: str, kind: Callable[[int], object]
) -> None:
    """Accept native signed and unsigned NumPy integers for every integer argument."""
    arguments = multigrid_arguments()
    arguments[field] = kind(arguments[field])
    result, residual, cycles, converged = multigrid(**arguments)
    assert converged and residual == 0.0 and cycles == 0
    assert np.all(result == 0.0)


def test_multigrid_combined_invalid_precedence(multigrid: Multigrid) -> None:
    """Keep array order and scalar kind precedence independent of simultaneous failures."""
    arguments = multigrid_arguments()
    arguments.update(source=np.ones(5, dtype=np.float64), psi_bc=np.ones((5, 7), dtype=np.float32))
    with pytest.raises(ValueError):
        multigrid(**arguments)
    arguments = multigrid_arguments()
    arguments.update(nr=-1, tol=True)
    with pytest.raises(TypeError):
        multigrid(**arguments)
    arguments = multigrid_arguments()
    arguments.update(nr=1 << 60, nz=3, psi_bc=np.full((5, 7), np.nan))
    with pytest.raises(ValueError):
        multigrid(**arguments)


@pytest.mark.parametrize("axis", ["r", "z"])
@pytest.mark.parametrize("value", ["equal", "reverse", "overflow", "underflow"])
def test_sensor_derived_geometry(sensor: Sensor, axis: str, value: str) -> None:
    """Check nominal spacing on each axis before evaluating any wall coordinates."""
    arguments = sensor_arguments()
    lower, upper = f"{axis}_min", f"{axis}_max"
    if value == "equal":
        arguments[upper] = arguments[lower]
    elif value == "reverse":
        arguments[upper] = arguments[lower] - 1.0
    elif value == "overflow":
        arguments.update({lower: -1e308, upper: 1e308})
    else:
        arguments.update({lower: 0.0, upper: 5e-324})
    with pytest.raises(ValueError):
        sensor(**arguments)


@pytest.mark.parametrize("extra_memory_mb", [20, 36])
def test_native_allocation_exhaustion_is_memory_error(extra_memory_mb: int) -> None:
    """Constrain only a real child process and recover native allocation failure as MemoryError."""
    import os
    import subprocess
    import sys

    code = (
        "extra_memory_mb = "
        + str(extra_memory_mb)
        + "\n"
        + """import resource
from pathlib import Path
import numpy as np
import scpn_fusion_rs
source = np.zeros((1000,1000), dtype=np.float64)
boundary = np.zeros_like(source)
virtual_kib = int(next(line.split()[1] for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("VmSize:")))
limit = virtual_kib * 1024 + extra_memory_mb * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
try:
    scpn_fusion_rs.multigrid_vcycle(source,boundary,1.,3.,-1.,1.,1000,1000)
except MemoryError as error:
    if extra_memory_mb == 20:
        assert "allocation failure:" in str(error), str(error)
    print("PASS: native recoverable allocation failure")
else:
    raise AssertionError("bounded native allocation unexpectedly succeeded")
"""
    )
    environment = os.environ.copy()
    environment["OPENBLAS_NUM_THREADS"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code], env=environment, text=True, capture_output=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "native recoverable allocation failure" in result.stdout


class StoredInteger(int):
    """Expose real integer storage while refusing user conversion callbacks."""

    def __int__(self) -> int:
        """Fail if an ABI calls a user hook instead of builtin integer storage."""
        raise AssertionError("integer conversion hook executed")

    def __float__(self) -> float:
        """Fail if an ABI coerces a subclass rather than normalizing its integer."""
        raise AssertionError("float conversion hook executed")


class StoredFloat(float):
    """Retain real float storage without allowing an overridden conversion hook."""

    def __float__(self) -> float:
        """Fail if a public ABI invokes a caller-controlled conversion callback."""
        raise AssertionError("float conversion hook executed")


class StoredArray(np.ndarray[Any, np.dtype[np.float64]]):
    """Carry ordinary ndarray storage while refusing subclass arithmetic."""

    def __array_ufunc__(self, *args: object, **kwargs: object) -> object:
        """Fail if a kernel retains subclass arithmetic instead of a base view."""
        raise AssertionError("subclass arithmetic executed")


def test_sensor_subclass_storage_and_output_lifetime(sensor: Sensor) -> None:
    """Use actual subclass data and keep independent results alive after input deletion."""
    import gc

    arguments = sensor_arguments()
    arguments.update(
        psi=np.ones((5, 7)).view(StoredArray), nr=StoredInteger(7), z_min=StoredFloat(-10.0)
    )
    first = sensor(**arguments)
    second = sensor(**arguments)
    assert not np.shares_memory(first, second)
    del arguments
    gc.collect()
    second[:] = -1.0
    np.testing.assert_allclose(first, np.ones(20), rtol=4e-16, atol=0.0)
    assert first.dtype.isnative and first.flags.c_contiguous


def test_multigrid_subclass_storage_and_output_lifetime(multigrid: Multigrid) -> None:
    """Strip both array subclasses and scalar hooks, preserving independent returned storage."""
    import gc

    arguments = multigrid_arguments()
    arguments.update(
        source=np.zeros((5, 7)).view(StoredArray),
        psi_bc=np.ones((5, 7)).view(StoredArray),
        nr=StoredInteger(7),
        r_min=StoredInteger(1),
        z_min=StoredFloat(-1.0),
    )
    first = multigrid(**arguments)[0]
    second = multigrid(**arguments)[0]
    assert not np.shares_memory(first, second)
    del arguments
    gc.collect()
    second[:] = -1.0
    np.testing.assert_allclose(first, np.ones((5, 7)), rtol=0.0, atol=1e-14)
    assert first.dtype.isnative and first.flags.c_contiguous


@pytest.mark.parametrize("axis", ["r", "z"])
def test_multigrid_axis_order(multigrid: Multigrid, axis: str) -> None:
    """Reject equal and reversed endpoints on each axis through every public route."""
    for reverse in (False, True):
        arguments = multigrid_arguments()
        arguments[f"{axis}_max"] = arguments[f"{axis}_min"] - float(reverse)
        with pytest.raises(ValueError):
            multigrid(**arguments)


@pytest.mark.parametrize(
    "witness", ["vertical_square", "center_overflow", "radial_product_overflow"]
)
def test_multigrid_stencil_geometry_extremes(multigrid: Multigrid, witness: str) -> None:
    """Reject separately representable axes whose derived stencil cannot be represented."""
    arguments = multigrid_arguments()
    if witness == "vertical_square":
        arguments.update(z_min=0.0, z_max=1e200)
    elif witness == "center_overflow":
        arguments.update(r_min=1e-152, r_max=1.06e-152)
    else:
        arguments.update(r_min=1e155, r_max=1.06e155)
    with pytest.raises(ValueError):
        multigrid(**arguments)


@pytest.mark.parametrize(
    "field,value",
    [
        ("omega", 0.9),
        ("omega", 2.0),
        ("pre_smooth", -1),
        ("post_smooth", -1),
        ("min_grid", 2),
        ("tol", 0.0),
        ("tol", -1.0),
        ("max_cycles", 0),
    ],
)
def test_numpy_multigrid_tuning_domains_on_initial_solution(field: str, value: object) -> None:
    """Refuse every invalid tuning domain even for an initially converged field."""
    arguments = multigrid_arguments()
    arguments[field] = value
    with pytest.raises(ValueError):
        multigrid_solve(**arguments)


def test_public_residual_helper_empty_and_nonfinite() -> None:
    """Exercise the public helper's empty interior and explicit nonfinite residual refusal."""
    from scpn_fusion.core.multigrid_solve import residual_linf

    field = np.zeros((2, 2))
    assert residual_linf(field, field, np.ones((2, 2)), 1.0, 1.0) == 0.0
    field = np.full((3, 3), np.finfo(np.float64).max)
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(RuntimeError, match="residual arithmetic"),
    ):
        residual_linf(field, np.zeros((3, 3)), np.ones((3, 3)), 1.0, 1.0)


def test_actual_kernel_retains_explicit_radius_policy(tmp_path: Path) -> None:
    """Exercise real configured kernel helper callers below their legacy radius floor."""
    import json

    from scpn_fusion.core.fusion_kernel import FusionKernel
    from scpn_fusion.core.multigrid_solve import mg_residual

    root = Path(__file__).resolve().parents[1]
    configuration = json.loads((root / "validation/iter_config.json").read_text())
    configuration["dimensions"].update(R_min=1e-12, R_max=3e-12, Z_min=-1e-12, Z_max=1e-12)
    configuration["grid_resolution"] = [9, 7]
    path = tmp_path / "kernel.json"
    path.write_text(json.dumps(configuration))
    kernel = FusionKernel(path)
    field = kernel.RR**3
    radius = np.maximum(kernel.RR[1:-1, 1:-1], 1e-10)
    source = np.zeros_like(field)
    source[1:-1, 1:-1] = (
        field[1:-1, 2:] - 2 * field[1:-1, 1:-1] + field[1:-1, :-2]
    ) / kernel.dR**2 - (field[1:-1, 2:] - field[1:-1, :-2]) / (2 * kernel.dR) / radius
    np.testing.assert_allclose(
        kernel._mg_residual(field, source, kernel.RR, kernel.dR, kernel.dZ), 0.0, atol=1e-26
    )
    assert np.max(np.abs(mg_residual(field, source, kernel.RR, kernel.dR, kernel.dZ))) > 1e-12
    smoothed = kernel._mg_smooth(field.copy(), source, kernel.RR, kernel.dR, kernel.dZ, 1.0, 2)
    np.testing.assert_allclose(smoothed, field, rtol=1e-12, atol=1e-46)
    cycled = kernel._multigrid_vcycle(
        field.copy(), source, kernel.RR, kernel.dR, kernel.dZ, omega=1.0
    )
    np.testing.assert_allclose(cycled, field, rtol=1e-10, atol=1e-44)


def test_actual_optional_backend_absence_preserves_availability_result() -> None:
    """Run an actual clean subprocess without the extension and retain optional availability semantics."""
    import os
    import subprocess
    import sys

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    code = """import importlib.util
assert importlib.util.find_spec('scpn_fusion_rs') is None
from scpn_fusion.core._rust_compat import rust_multigrid_vcycle, rust_measure_magnetics
import numpy as np
assert rust_multigrid_vcycle(np.zeros((3,3)),np.zeros((3,3)),1.,3.,-1.,1.,3,3) is None
try:
    rust_measure_magnetics(np.ones((2,2)),2,2,1.,3.,-1.,1.)
except ImportError:
    print('PASS: actual optional backend absence')
else:
    raise AssertionError('missing backend did not signal ImportError')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], env=environment, text=True, capture_output=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "actual optional backend absence" in result.stdout


class StoredNumpyInteger(np.int64):
    """Retain actual NumPy integer storage while refusing caller conversion hooks."""

    def __int__(self) -> int:
        """Fail if an accepted NumPy integer is read through its override."""
        raise AssertionError("NumPy integer conversion override executed")

    def __float__(self) -> float:
        """Fail if exact integer admission delegates to a float override."""
        raise AssertionError("NumPy integer float override executed")


class StoredNumpyFloat(np.float32):
    """Retain actual narrow floating storage without caller metadata or conversion."""

    def __float__(self) -> float:
        """Fail if a public ABI uses an overridden numeric conversion."""
        raise AssertionError("NumPy float conversion override executed")

    @property
    def dtype(self) -> np.dtype[StoredNumpyFloat]:
        """Fail if admission asks a caller attribute instead of NumPy storage."""
        raise AssertionError("NumPy dtype override executed")


@pytest.mark.parametrize("argument", ["nr", "nz"])
def test_sensor_numpy_integer_subclass_storage(sensor: Sensor, argument: str) -> None:
    """Read each public dimension from actual NumPy storage without caller callbacks."""
    arguments = sensor_arguments()
    arguments[argument] = StoredNumpyInteger(arguments[argument])
    np.testing.assert_allclose(sensor(**arguments), np.ones(20), rtol=4e-16, atol=0.0)


@pytest.mark.parametrize("argument", ["r_min", "r_max", "z_min", "z_max"])
@pytest.mark.parametrize("kind", [StoredNumpyInteger, StoredNumpyFloat])
def test_sensor_numpy_real_subclass_storage(
    sensor: Sensor, argument: str, kind: type[np.int64 | np.float32]
) -> None:
    """Use stored admitted integer/float reals in every sensing scalar position."""
    arguments = sensor_arguments()
    arguments[argument] = kind(arguments[argument])
    np.testing.assert_allclose(sensor(**arguments), np.ones(20), rtol=4e-16, atol=0.0)


@pytest.mark.parametrize("argument", ["nr", "nz", "max_cycles"])
def test_multigrid_numpy_integer_subclass_storage(multigrid: Multigrid, argument: str) -> None:
    """Validate each common solve integer from storage even on initial convergence."""
    arguments = multigrid_arguments()
    arguments[argument] = StoredNumpyInteger(arguments[argument])
    result = multigrid(**arguments)
    assert result[2:] == (0, True)


@pytest.mark.parametrize("argument", ["r_min", "r_max", "z_min", "z_max", "tol"])
def test_multigrid_numpy_float_subclass_storage(multigrid: Multigrid, argument: str) -> None:
    """Read every solve real through base NumPy floating storage and metadata."""
    arguments = multigrid_arguments()
    arguments[argument] = StoredNumpyFloat(arguments[argument])
    result = multigrid(**arguments)
    assert result[2:] == (0, True)


@pytest.mark.parametrize("argument", ["pre_smooth", "post_smooth", "min_grid", "omega"])
def test_numpy_only_control_subclass_storage(argument: str) -> None:
    """Admit real stored NumPy subclasses for every NumPy-only tuning control."""
    arguments = multigrid_arguments()
    arguments[argument] = StoredNumpyFloat(1.0) if argument == "omega" else StoredNumpyInteger(5)
    assert multigrid_solve(**arguments)[2:] == (0, True)


class ReportedNarrowFloat(np.longdouble):
    """Expose genuine wide storage with a misleading caller dtype property."""

    @property
    def dtype(self) -> np.dtype[ReportedNarrowFloat]:
        """Report a narrow dtype so admission must inspect actual base storage."""
        return cast(np.dtype[ReportedNarrowFloat], np.dtype(np.float32))


@pytest.mark.parametrize("argument", ["r_min", "r_max", "z_min", "z_max"])
def test_sensor_wide_subclass_metadata(sensor: Sensor, argument: str) -> None:
    """Reject actual wide NumPy storage regardless of a caller dtype override."""
    assert np.dtype(np.longdouble).itemsize > 8
    arguments = sensor_arguments()
    arguments[argument] = ReportedNarrowFloat(arguments[argument])
    with pytest.raises(TypeError):
        sensor(**arguments)


@pytest.mark.parametrize("argument", ["r_min", "r_max", "z_min", "z_max", "tol"])
def test_multigrid_wide_subclass_metadata(multigrid: Multigrid, argument: str) -> None:
    """Refuse each wide solve real before conversion despite misleading metadata."""
    assert np.dtype(np.longdouble).itemsize > 8
    arguments = multigrid_arguments()
    arguments[argument] = ReportedNarrowFloat(arguments[argument])
    with pytest.raises(TypeError):
        multigrid(**arguments)
