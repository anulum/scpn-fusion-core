# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
"""Load DIII-D shot data via MDSplus for validation against EFIT."""

from __future__ import annotations

import numpy as np

try:
    import MDSplus

    _HAS_MDSPLUS = True
except ImportError:
    _HAS_MDSPLUS = False


class MDSplusLoader:
    """Load DIII-D equilibrium data from MDSplus.

    Usage::

        loader = MDSplusLoader()
        loader.connect(166439)
        efit = loader.get_efit_equilibrium(time_ms=3000.0)
        probes = loader.get_magnetic_probes(time_ms=3000.0)
    """

    DIII_D_SERVER = "atlas.gat.com"
    EFIT_TREE = "EFIT01"

    def __init__(self, server: str | None = None):
        if not _HAS_MDSPLUS:
            raise ImportError(
                "MDSplus not installed. Run: pip install mdsplus"
            )
        self.server = server or self.DIII_D_SERVER
        self._conn = None
        self._shot = None

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def connect(self, shot: int) -> None:
        """Open connection and EFIT tree for given shot."""
        self._conn = MDSplus.Connection(self.server)
        self._conn.openTree(self.EFIT_TREE, shot)
        self._shot = shot

    def close(self) -> None:
        """Close the EFIT tree (no-op if never connected)."""
        if self._conn is not None and self._shot is not None:
            self._conn.closeTree(self.EFIT_TREE, self._shot)
            self._conn = None
            self._shot = None

    # ------------------------------------------------------------------
    # Low-level fetch
    # ------------------------------------------------------------------

    def _get(self, node: str) -> np.ndarray:
        """Fetch *node* from the open tree and return as ndarray."""
        assert self._conn is not None, "Call connect() first"
        return np.array(self._conn.get(node).data())

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------

    def get_time_array(self) -> np.ndarray:
        """Return EFIT time base in **milliseconds**."""
        return self._get("\\EFIT01::TOP.RESULTS.GEQDSK:GTIME") * 1000.0

    def _nearest_time_index(self, time_ms: float) -> int:
        """Index into the EFIT time array closest to *time_ms*."""
        times = self.get_time_array()
        return int(np.argmin(np.abs(times - time_ms)))

    # ------------------------------------------------------------------
    # Full EFIT equilibrium
    # ------------------------------------------------------------------

    def get_efit_equilibrium(self, time_ms: float) -> dict:
        """Load full EFIT reconstruction at given time.

        Returns
        -------
        dict
            Keys: ``psi_rz``, ``r_grid``, ``z_grid``, ``r_axis``,
            ``z_axis``, ``psi_axis``, ``psi_boundary``, ``r_boundary``,
            ``z_boundary``, ``q_profile``, ``pressure``, ``fpol``,
            ``ip``, ``bt``.
        """
        idx = self._nearest_time_index(time_ms)

        # Spatial grids (1-D)
        r_grid = self._get("\\EFIT01::TOP.RESULTS.GEQDSK:R")
        z_grid = self._get("\\EFIT01::TOP.RESULTS.GEQDSK:Z")

        # Poloidal-flux map (may be time x R x Z)
        psirz_full = self._get("\\EFIT01::TOP.RESULTS.GEQDSK:PSIRZ")
        psi_rz = psirz_full[idx] if psirz_full.ndim == 3 else psirz_full

        # Scalar axis quantities
        r_axis = float(self._get("\\EFIT01::TOP.RESULTS.GEQDSK:RMAXIS"))
        z_axis = float(self._get("\\EFIT01::TOP.RESULTS.GEQDSK:ZMAXIS"))
        psi_axis = float(self._get("\\EFIT01::TOP.RESULTS.GEQDSK:SSIMAG"))
        psi_bdry = float(self._get("\\EFIT01::TOP.RESULTS.GEQDSK:SSIBRY"))

        # Boundary shape (may be time x N)
        r_bdry = self._get("\\EFIT01::TOP.RESULTS.GEQDSK:RBDRY")
        z_bdry = self._get("\\EFIT01::TOP.RESULTS.GEQDSK:ZBDRY")
        if r_bdry.ndim == 2:
            r_bdry = r_bdry[idx]
            z_bdry = z_bdry[idx]

        # 1-D profiles (may be time x N)
        q_profile = self._get("\\EFIT01::TOP.RESULTS.GEQDSK:QPSI")
        if q_profile.ndim == 2:
            q_profile = q_profile[idx]

        pressure = self._get("\\EFIT01::TOP.RESULTS.GEQDSK:PRES")
        if pressure.ndim == 2:
            pressure = pressure[idx]

        fpol = self._get("\\EFIT01::TOP.RESULTS.GEQDSK:FPOL")
        if fpol.ndim == 2:
            fpol = fpol[idx]

        # Global scalars
        ip = float(self._get("\\EFIT01::TOP.RESULTS.GEQDSK:CPASMA"))
        bt = float(self._get("\\EFIT01::TOP.RESULTS.GEQDSK:BCENTR"))

        return {
            "psi_rz": psi_rz,
            "r_grid": r_grid,
            "z_grid": z_grid,
            "r_axis": r_axis,
            "z_axis": z_axis,
            "psi_axis": psi_axis,
            "psi_boundary": psi_bdry,
            "r_boundary": r_bdry,
            "z_boundary": z_bdry,
            "q_profile": q_profile,
            "pressure": pressure,
            "fpol": fpol,
            "ip": ip,
            "bt": bt,
        }

    # ------------------------------------------------------------------
    # Magnetic probes (stub)
    # ------------------------------------------------------------------

    def get_magnetic_probes(self, time_ms: float) -> dict:
        """Load magnetic probe measurements for inverse reconstruction.

        .. note::

           Probe geometry loading requires DIII-D-specific node paths
           that vary across campaigns.  This method is intentionally left
           as a stub until exact tree paths are confirmed.
        """
        raise NotImplementedError(
            "Probe geometry loading requires DIII-D-specific node paths. "
            "See DIII-D MDSplus documentation for exact tree structure."
        )
