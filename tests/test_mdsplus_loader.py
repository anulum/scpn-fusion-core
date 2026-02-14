# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
"""Tests for MDSplusLoader -- all run WITHOUT MDSplus installed."""

from __future__ import annotations

import importlib
import sys
import types
from unittest import mock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers: build a fake MDSplus module so we can import the loader
# ---------------------------------------------------------------------------


def _make_fake_mdsplus() -> types.ModuleType:
    """Return a minimal mock ``MDSplus`` module with a ``Connection`` class."""
    fake = types.ModuleType("MDSplus")

    class _FakeData:
        def __init__(self, arr):
            self._arr = arr

        def data(self):
            return self._arr

    class FakeConnection:
        def __init__(self, server: str):
            self.server = server
            self._node_map: dict[str, np.ndarray] = {}

        def openTree(self, tree: str, shot: int) -> None:  # noqa: N802
            pass

        def closeTree(self, tree: str, shot: int) -> None:  # noqa: N802
            pass

        def get(self, node: str) -> _FakeData:
            if node not in self._node_map:
                raise KeyError(f"Node {node!r} not registered in mock")
            return _FakeData(self._node_map[node])

    fake.Connection = FakeConnection
    return fake


def _import_loader_with_mdsplus(fake_mds: types.ModuleType):
    """(Re-)import ``mdsplus_loader`` with *fake_mds* injected.

    We must:
    1. Insert the fake MDSplus into sys.modules
    2. Evict cached copies of the loader module
    3. Re-import so the module-level try/except re-runs
    """
    sys.modules["MDSplus"] = fake_mds

    # Evict cached loader module so the try/except block runs again
    for key in list(sys.modules):
        if "mdsplus_loader" in key:
            del sys.modules[key]

    mod = importlib.import_module("scpn_fusion.io.mdsplus_loader")
    assert mod._HAS_MDSPLUS is True, (
        "Fake MDSplus injection failed -- _HAS_MDSPLUS is still False"
    )
    return mod


def _import_loader_without_mdsplus():
    """(Re-)import ``mdsplus_loader`` with MDSplus absent."""
    # Remove any real or fake MDSplus from sys.modules
    sys.modules.pop("MDSplus", None)

    # Evict cached loader module
    for key in list(sys.modules):
        if "mdsplus_loader" in key:
            del sys.modules[key]

    # Block the MDSplus import entirely
    _real_import = builtins_import()

    def _no_mdsplus(name, *args, **kwargs):
        if name == "MDSplus":
            raise ImportError("No module named 'MDSplus'")
        return _real_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=_no_mdsplus):
        mod = importlib.import_module("scpn_fusion.io.mdsplus_loader")

    return mod


def builtins_import():
    """Get the real __import__ safely regardless of __builtins__ type."""
    import builtins
    return builtins.__import__


# ===========================================================================
# Tests
# ===========================================================================


class TestImportGuard:
    """Verify graceful behaviour when MDSplus is not installed."""

    def test_import_error_without_mdsplus(self):
        """Instantiating MDSplusLoader without MDSplus raises ImportError."""
        mod = _import_loader_without_mdsplus()
        assert mod._HAS_MDSPLUS is False
        with pytest.raises(ImportError, match="MDSplus not installed"):
            mod.MDSplusLoader()


class TestMDSplusLoaderWithMock:
    """Tests that exercise loader logic using a mocked MDSplus module."""

    @pytest.fixture(autouse=True)
    def _setup_loader(self):
        """Inject fake MDSplus and re-import the loader module."""
        self.fake_mds = _make_fake_mdsplus()
        self.mod = _import_loader_with_mdsplus(self.fake_mds)
        yield
        # Clean up injected module
        sys.modules.pop("MDSplus", None)

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    def _make_loader(self, shot: int = 166439):
        """Create a loader and connect with a pre-populated mock."""
        loader = self.mod.MDSplusLoader(server="mock.server")
        loader.connect(shot)
        return loader

    def _populate_efit_nodes(
        self,
        conn,
        nr: int = 65,
        nz: int = 65,
        nt: int = 10,
        npsi: int = 101,
        nbdry: int = 50,
    ):
        """Register all GEQDSK nodes expected by get_efit_equilibrium."""
        rng = np.random.default_rng(42)
        # Time array in seconds (will be multiplied by 1000 in loader)
        conn._node_map["\\EFIT01::TOP.RESULTS.GEQDSK:GTIME"] = np.linspace(
            1.0, 5.0, nt
        )
        conn._node_map["\\EFIT01::TOP.RESULTS.GEQDSK:R"] = np.linspace(
            1.0, 2.4, nr
        )
        conn._node_map["\\EFIT01::TOP.RESULTS.GEQDSK:Z"] = np.linspace(
            -1.5, 1.5, nz
        )
        conn._node_map["\\EFIT01::TOP.RESULTS.GEQDSK:PSIRZ"] = rng.random(
            (nt, nr, nz)
        )
        conn._node_map["\\EFIT01::TOP.RESULTS.GEQDSK:RMAXIS"] = np.array(1.7)
        conn._node_map["\\EFIT01::TOP.RESULTS.GEQDSK:ZMAXIS"] = np.array(0.01)
        conn._node_map["\\EFIT01::TOP.RESULTS.GEQDSK:SSIMAG"] = np.array(0.5)
        conn._node_map["\\EFIT01::TOP.RESULTS.GEQDSK:SSIBRY"] = np.array(1.2)
        conn._node_map["\\EFIT01::TOP.RESULTS.GEQDSK:RBDRY"] = rng.random(
            (nt, nbdry)
        )
        conn._node_map["\\EFIT01::TOP.RESULTS.GEQDSK:ZBDRY"] = rng.random(
            (nt, nbdry)
        )
        conn._node_map["\\EFIT01::TOP.RESULTS.GEQDSK:QPSI"] = rng.random(
            (nt, npsi)
        )
        conn._node_map["\\EFIT01::TOP.RESULTS.GEQDSK:PRES"] = rng.random(
            (nt, npsi)
        )
        conn._node_map["\\EFIT01::TOP.RESULTS.GEQDSK:FPOL"] = rng.random(
            (nt, npsi)
        )
        conn._node_map["\\EFIT01::TOP.RESULTS.GEQDSK:CPASMA"] = np.array(
            1.2e6
        )
        conn._node_map["\\EFIT01::TOP.RESULTS.GEQDSK:BCENTR"] = np.array(2.1)

    # ---------------------------------------------------------------
    # Actual tests
    # ---------------------------------------------------------------

    def test_connect_stores_shot(self):
        """After connect(), the shot number is stored on the loader."""
        loader = self._make_loader(shot=166439)
        assert loader._shot == 166439
        assert loader._conn is not None

    def test_close_without_connect(self):
        """Calling close() before connect() must not raise."""
        loader = self.mod.MDSplusLoader(server="mock.server")
        assert loader._conn is None
        loader.close()  # should be a safe no-op

    def test_nearest_time_index(self):
        """_nearest_time_index returns the index closest to requested ms."""
        loader = self._make_loader()
        # Populate a simple time vector: [1000, 2000, 3000, 4000, 5000] ms
        loader._conn._node_map[
            "\\EFIT01::TOP.RESULTS.GEQDSK:GTIME"
        ] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert loader._nearest_time_index(3000.0) == 2
        assert loader._nearest_time_index(3400.0) == 2  # closer to 3000
        assert loader._nearest_time_index(3600.0) == 3  # closer to 4000
        assert loader._nearest_time_index(1000.0) == 0
        assert loader._nearest_time_index(5000.0) == 4

    def test_get_efit_equilibrium_keys(self):
        """get_efit_equilibrium returns a dict with all expected keys."""
        loader = self._make_loader()
        self._populate_efit_nodes(loader._conn)

        efit = loader.get_efit_equilibrium(time_ms=3000.0)

        expected_keys = {
            "psi_rz",
            "r_grid",
            "z_grid",
            "r_axis",
            "z_axis",
            "psi_axis",
            "psi_boundary",
            "r_boundary",
            "z_boundary",
            "q_profile",
            "pressure",
            "fpol",
            "ip",
            "bt",
        }
        assert set(efit.keys()) == expected_keys

    def test_get_efit_equilibrium_shapes(self):
        """Returned arrays have the right shapes (time-sliced, not full)."""
        nr, nz, nt, npsi, nbdry = 65, 65, 10, 101, 50
        loader = self._make_loader()
        self._populate_efit_nodes(
            loader._conn, nr=nr, nz=nz, nt=nt, npsi=npsi, nbdry=nbdry
        )

        efit = loader.get_efit_equilibrium(time_ms=3000.0)

        # psi_rz should be a single time-slice: (nr, nz)
        assert efit["psi_rz"].shape == (nr, nz)
        assert efit["r_grid"].shape == (nr,)
        assert efit["z_grid"].shape == (nz,)
        assert efit["r_boundary"].shape == (nbdry,)
        assert efit["z_boundary"].shape == (nbdry,)
        assert efit["q_profile"].shape == (npsi,)
        assert efit["pressure"].shape == (npsi,)
        assert efit["fpol"].shape == (npsi,)

    def test_get_efit_equilibrium_scalar_types(self):
        """Scalar values are plain Python floats, not arrays."""
        loader = self._make_loader()
        self._populate_efit_nodes(loader._conn)

        efit = loader.get_efit_equilibrium(time_ms=3000.0)

        for key in (
            "r_axis",
            "z_axis",
            "psi_axis",
            "psi_boundary",
            "ip",
            "bt",
        ):
            assert isinstance(efit[key], float), f"{key} should be float"

    def test_magnetic_probes_not_implemented(self):
        """get_magnetic_probes raises NotImplementedError."""
        loader = self._make_loader()
        with pytest.raises(NotImplementedError, match="DIII-D-specific"):
            loader.get_magnetic_probes(time_ms=3000.0)

    def test_default_server(self):
        """Without explicit server, defaults to atlas.gat.com."""
        loader = self.mod.MDSplusLoader()
        assert loader.server == "atlas.gat.com"

    def test_custom_server(self):
        """Explicit server is stored correctly."""
        loader = self.mod.MDSplusLoader(server="custom.server.edu")
        assert loader.server == "custom.server.edu"

    def test_close_resets_state(self):
        """After close(), _conn and _shot are reset to None."""
        loader = self._make_loader()
        assert loader._conn is not None
        assert loader._shot is not None
        loader.close()
        assert loader._conn is None
        assert loader._shot is None
