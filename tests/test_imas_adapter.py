# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Tests for IMAS/OMAS equilibrium adapter.

import numpy as np

from scpn_fusion.core.imas_adapter import (
    EquilibriumIDS,
    from_geqdsk,
    from_kernel,
    from_omas,
    to_kernel_arrays,
    to_omas,
)


def _make_ids(**kwargs):
    nr, nz = 5, 7
    defaults = dict(
        r=np.linspace(4.0, 8.0, nr),
        z=np.linspace(-3.0, 3.0, nz),
        psi=np.random.default_rng(0).standard_normal((nz, nr)),
        j_tor=np.random.default_rng(1).standard_normal((nz, nr)),
        ip=15e6,
        b0=5.3,
        r0=6.2,
        time=1.5,
    )
    defaults.update(kwargs)
    return EquilibriumIDS(**defaults)


class TestEquilibriumIDS:
    def test_fields(self):
        ids = _make_ids()
        assert ids.ip == 15e6
        assert ids.b0 == 5.3
        assert ids.r0 == 6.2
        assert ids.time == 1.5
        assert ids.psi.shape == (7, 5)

    def test_default_time(self):
        ids = EquilibriumIDS(
            r=np.array([1.0]),
            z=np.array([0.0]),
            psi=np.array([[1.0]]),
            j_tor=np.array([[0.0]]),
            ip=1e6,
            b0=5.0,
            r0=1.0,
        )
        assert ids.time == 0.0


class TestToKernelArrays:
    def test_round_trip_shapes(self):
        ids = _make_ids()
        arrays = to_kernel_arrays(ids)
        assert arrays["R"].shape == ids.r.shape
        assert arrays["Z"].shape == ids.z.shape
        assert arrays["Psi"].shape == ids.psi.shape
        assert arrays["J_phi"].shape == ids.j_tor.shape

    def test_arrays_are_copies(self):
        ids = _make_ids()
        arrays = to_kernel_arrays(ids)
        arrays["Psi"][0, 0] = 999.0
        assert ids.psi[0, 0] != 999.0


class TestFromKernel:
    def test_extracts_from_mock_kernel(self):
        nr, nz = 4, 6
        rng = np.random.default_rng(42)

        class _MockKernel:
            R = np.linspace(4.0, 8.0, nr)
            Z = np.linspace(-3.0, 3.0, nz)
            Psi = rng.standard_normal((nz, nr))
            J_phi = rng.standard_normal((nz, nr))
            cfg = {
                "physics": {"plasma_current_target": 12e6, "B0": 4.5},
                "dimensions": {"R0": 5.5},
            }

        ids = from_kernel(_MockKernel(), time=2.0)
        assert ids.ip == 12e6
        assert ids.b0 == 4.5
        assert ids.r0 == 5.5
        assert ids.time == 2.0
        assert ids.psi.shape == (nz, nr)

    def test_defaults_for_missing_keys(self):
        nr, nz = 3, 3

        class _BareKernel:
            R = np.linspace(4.0, 8.0, nr)
            Z = np.linspace(-3.0, 3.0, nz)
            Psi = np.zeros((nz, nr))
            J_phi = np.zeros((nz, nr))

        ids = from_kernel(_BareKernel())
        assert ids.ip == 15e6
        assert ids.b0 == 5.3
        assert ids.r0 == 6.2


class TestToOmas:
    def test_returns_none_without_omas(self):
        ids = _make_ids()
        result = to_omas(ids)
        # Either None (omas not installed) or an ODS object
        assert result is None or hasattr(result, "__getitem__")


class TestFromOmas:
    def _make_mock_ods(self):
        """Build a minimal dict-based ODS mimic for from_omas."""
        nr, nz = 5, 7
        rng = np.random.default_rng(0)
        r = np.linspace(4.0, 8.0, nr)
        z = np.linspace(-3.0, 3.0, nz)
        psi = rng.standard_normal((nz, nr))
        j_tor = rng.standard_normal((nz, nr))

        class _DictLike(dict):
            """Dict that also supports .get()."""

            pass

        p2d = _DictLike(
            {
                "grid": {"dim1": r, "dim2": z},
                "psi": psi,
                "j_tor": j_tor,
            }
        )
        gq = _DictLike({"ip": 15e6, "magnetic_axis": {"r": 6.2}})
        ts = _DictLike({"profiles_2d": [p2d], "global_quantities": gq, "time": 1.5})
        ts.get = lambda k, d=None: ts[k] if k in ts else d  # noqa: SIM401
        ods = {"equilibrium": {"time_slice": [ts]}}
        return ods

    def test_from_omas_extracts_fields(self):
        ods = self._make_mock_ods()
        ids = from_omas(ods)
        assert ids.ip == 15e6
        assert ids.r0 == 6.2
        assert ids.psi.shape == (7, 5)
        assert ids.time == 1.5

    def test_from_omas_default_time_index(self):
        ods = self._make_mock_ods()
        ids = from_omas(ods, time_index=0)
        assert ids.r.shape == (5,)


class TestFromGeqdsk:
    def test_from_geqdsk_round_trip(self, tmp_path):
        from scpn_fusion.core.eqdsk import GEqdsk, write_geqdsk

        nr, nz = 8, 10
        g = GEqdsk(
            description="test",
            nw=nr,
            nh=nz,
            rdim=4.0,
            zdim=6.0,
            rcentr=6.2,
            rleft=4.0,
            zmid=0.0,
            rmaxis=6.2,
            zmaxis=0.0,
            simag=-5.0,
            sibry=-1.0,
            bcentr=5.3,
            current=15e6,
            fpol=np.full(nr, 12.0),
            pres=np.zeros(nr),
            ffprime=np.zeros(nr),
            pprime=np.zeros(nr),
            qpsi=np.linspace(1.0, 4.0, nr),
            psirz=np.random.default_rng(0).standard_normal((nz, nr)),
        )
        path = tmp_path / "test.geqdsk"
        write_geqdsk(g, str(path))

        ids = from_geqdsk(str(path))
        assert ids.r.shape == (nr,)
        assert ids.z.shape == (nz,)
        assert ids.psi.shape == (nz, nr)
        assert ids.ip == 15e6
        assert ids.b0 == 5.3
        assert ids.r0 == 6.2
