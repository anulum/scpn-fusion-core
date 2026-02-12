# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — G-EQDSK Equilibrium File Parser
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Reader and writer for the G-EQDSK (EFIT) equilibrium file format.

The G-EQDSK format is the *de facto* standard for tokamak equilibrium
data exchange.  It stores the poloidal flux map ψ(R,Z) on a uniform
(R,Z) grid together with 1-D profile arrays and scalar quantities.

References
----------
- Lao et al., Nucl. Fusion 25 (1985) 1611
- FreeQDSK documentation: https://freeqdsk.readthedocs.io/
- ITER IMAS Data Dictionary (equilibrium IDS)

Format specification
--------------------
- Fortran fixed-width: 5 values per line, ``(5e16.9)``
- Header line: 48-char description, 3 ints (idum, nw, nh)
- Scalars block (20 values): rdim, zdim, rcentr, rleft, zmid,
  rmaxis, zmaxis, simag, sibry, bcentr, current, …
- 1-D arrays (each nw values): fpol, pres, ffprime, pprime, qpsi
- 2-D array: psirz (nh × nw values, row-major)
- Boundary & limiter point counts + (R,Z) pairs
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray


# ── Data container ────────────────────────────────────────────────────

@dataclass
class GEqdsk:
    """Container for all data in a G-EQDSK file."""

    # Header
    description: str = ""
    nw: int = 0              # number of horizontal (R) grid points
    nh: int = 0              # number of vertical (Z) grid points

    # Scalars
    rdim: float = 0.0        # horizontal dimension (m)
    zdim: float = 0.0        # vertical dimension (m)
    rcentr: float = 0.0      # reference R for vacuum toroidal field (m)
    rleft: float = 0.0       # R at left of computational domain (m)
    zmid: float = 0.0        # Z at centre of computational domain (m)
    rmaxis: float = 0.0      # R of magnetic axis (m)
    zmaxis: float = 0.0      # Z of magnetic axis (m)
    simag: float = 0.0       # ψ at magnetic axis (Wb/rad)
    sibry: float = 0.0       # ψ at plasma boundary (Wb/rad)
    bcentr: float = 0.0      # vacuum toroidal field at rcentr (T)
    current: float = 0.0     # plasma current (A)

    # 1-D profile arrays (length nw)
    fpol: NDArray = field(default_factory=lambda: np.array([]))
    pres: NDArray = field(default_factory=lambda: np.array([]))
    ffprime: NDArray = field(default_factory=lambda: np.array([]))
    pprime: NDArray = field(default_factory=lambda: np.array([]))
    qpsi: NDArray = field(default_factory=lambda: np.array([]))

    # 2-D flux map (nh × nw)
    psirz: NDArray = field(default_factory=lambda: np.array([]))

    # Boundary and limiter contours
    rbdry: NDArray = field(default_factory=lambda: np.array([]))
    zbdry: NDArray = field(default_factory=lambda: np.array([]))
    rlim: NDArray = field(default_factory=lambda: np.array([]))
    zlim: NDArray = field(default_factory=lambda: np.array([]))

    # ── Derived grids ─────────────────────────────────────────────────

    @property
    def r(self) -> NDArray:
        """1-D array of R grid values."""
        return np.linspace(self.rleft, self.rleft + self.rdim, self.nw)

    @property
    def z(self) -> NDArray:
        """1-D array of Z grid values."""
        return np.linspace(
            self.zmid - self.zdim / 2, self.zmid + self.zdim / 2, self.nh
        )

    @property
    def psi_norm(self) -> NDArray:
        """Normalised poloidal flux ψ_N ∈ [0, 1] (axis=0, boundary=1)."""
        return np.linspace(0.0, 1.0, self.nw)

    def psi_to_norm(self, psi: NDArray) -> NDArray:
        """Map raw ψ to normalised ψ_N."""
        return (psi - self.simag) / (self.sibry - self.simag)

    def to_config(self, name: str = "eqdsk") -> dict:
        """Convert to FusionKernel JSON config dict (approximate)."""
        r = self.r
        z = self.z
        return {
            "reactor_name": name,
            "grid_resolution": [self.nw, self.nh],
            "dimensions": {
                "R_min": float(r[0]),
                "R_max": float(r[-1]),
                "Z_min": float(z[0]),
                "Z_max": float(z[-1]),
            },
            "physics": {
                "plasma_current_target": float(self.current / 1e6),
                "vacuum_permeability": 1.0,
            },
            "coils": [],  # not stored in GEQDSK
            "solver": {
                "max_iterations": 1000,
                "convergence_threshold": 1e-4,
                "relaxation_factor": 0.1,
            },
        }


# ── Reader ────────────────────────────────────────────────────────────

# Regex that matches Fortran-style floats: optional sign, digits,
# optional decimal, optional exponent.  Handles the common case where
# two values like "2.385E+00-1.216E+01" run together (no whitespace).
_FORTRAN_FLOAT_RE = re.compile(
    r"[+-]?\d*\.?\d+(?:[eEdD][+-]?\d+)?"
)


def _split_fortran(line: str) -> list[str]:
    """Extract all Fortran-format floats from *line*."""
    return _FORTRAN_FLOAT_RE.findall(line)


def _parse_header(line: str):
    """Parse the first line: 48-char description + idum nw nh."""
    parts = line.split()
    nh = int(parts[-1])
    nw = int(parts[-2])
    desc = " ".join(parts[:-3]) if len(parts) > 3 else ""
    return desc, nw, nh


def read_geqdsk(path: Union[str, Path]) -> GEqdsk:
    """
    Read a G-EQDSK file and return a :class:`GEqdsk` container.

    Handles both whitespace-separated and fixed-width Fortran formats,
    including the common case where values run together without spaces
    (e.g. ``2.385E+00-1.216E+01``).

    Parameters
    ----------
    path : str or Path
        Path to the G-EQDSK file.

    Returns
    -------
    GEqdsk
        Parsed equilibrium data.
    """
    path = Path(path)
    with open(path, "r") as f:
        lines = f.readlines()

    # Parse header
    desc, nw, nh = _parse_header(lines[0])

    # Gather all numeric tokens using Fortran-aware splitting
    tokens: list[str] = []
    for line in lines[1:]:
        tokens.extend(_split_fortran(line))

    idx = 0

    def _next() -> float:
        nonlocal idx
        val = float(tokens[idx].replace("D", "E").replace("d", "e"))
        idx += 1
        return val

    def _read_array(n: int) -> NDArray:
        nonlocal idx
        arr = np.array(
            [float(tokens[idx + i].replace("D", "E").replace("d", "e"))
             for i in range(n)]
        )
        idx += n
        return arr

    # 20 scalar values
    rdim = _next()
    zdim = _next()
    rcentr = _next()
    rleft = _next()
    zmid = _next()
    rmaxis = _next()
    zmaxis = _next()
    simag = _next()
    sibry = _next()
    bcentr = _next()
    current = _next()
    _next()  # simag (duplicate)
    _next()  # padding
    _next()  # rmaxis duplicate
    _next()  # padding
    _next()  # zmaxis duplicate
    _next()  # padding
    _next()  # sibry duplicate
    _next()  # padding
    _next()  # padding

    # 1-D arrays (each nw values)
    fpol = _read_array(nw)
    pres = _read_array(nw)
    ffprime = _read_array(nw)
    pprime = _read_array(nw)

    # 2-D flux map: nh rows × nw columns (row-major)
    psirz = _read_array(nh * nw).reshape(nh, nw)

    # q profile
    qpsi = _read_array(nw)

    # Boundary and limiter
    nbdry = int(_next())
    nlim = int(_next())

    rbdry = np.zeros(nbdry)
    zbdry = np.zeros(nbdry)
    for i in range(nbdry):
        rbdry[i] = _next()
        zbdry[i] = _next()

    rlim = np.zeros(nlim)
    zlim = np.zeros(nlim)
    for i in range(nlim):
        rlim[i] = _next()
        zlim[i] = _next()

    return GEqdsk(
        description=desc,
        nw=nw,
        nh=nh,
        rdim=rdim,
        zdim=zdim,
        rcentr=rcentr,
        rleft=rleft,
        zmid=zmid,
        rmaxis=rmaxis,
        zmaxis=zmaxis,
        simag=simag,
        sibry=sibry,
        bcentr=bcentr,
        current=current,
        fpol=fpol,
        pres=pres,
        ffprime=ffprime,
        pprime=pprime,
        qpsi=qpsi,
        psirz=psirz,
        rbdry=rbdry,
        zbdry=zbdry,
        rlim=rlim,
        zlim=zlim,
    )


# ── Writer ────────────────────────────────────────────────────────────

def write_geqdsk(eq: GEqdsk, path: Union[str, Path]) -> None:
    """
    Write a :class:`GEqdsk` to a G-EQDSK file.

    Parameters
    ----------
    eq : GEqdsk
        Equilibrium data.
    path : str or Path
        Output file path.
    """
    path = Path(path)

    def _fmt(val: float) -> str:
        return f"{val:16.9e}"

    def _write_array(f, arr: NDArray) -> None:
        for i, v in enumerate(arr.ravel()):
            f.write(_fmt(v))
            if (i + 1) % 5 == 0:
                f.write("\n")
        if len(arr.ravel()) % 5 != 0:
            f.write("\n")

    with open(path, "w") as f:
        # Header
        desc = eq.description[:48].ljust(48)
        f.write(f"{desc}   0 {eq.nw:4d} {eq.nh:4d}\n")

        # Scalars (20 values, 5 per line)
        scalars = [
            eq.rdim, eq.zdim, eq.rcentr, eq.rleft, eq.zmid,
            eq.rmaxis, eq.zmaxis, eq.simag, eq.sibry, eq.bcentr,
            eq.current, eq.simag, 0.0, eq.rmaxis, 0.0,
            eq.zmaxis, 0.0, eq.sibry, 0.0, 0.0,
        ]
        for i, v in enumerate(scalars):
            f.write(_fmt(v))
            if (i + 1) % 5 == 0:
                f.write("\n")

        # 1-D arrays
        _write_array(f, eq.fpol)
        _write_array(f, eq.pres)
        _write_array(f, eq.ffprime)
        _write_array(f, eq.pprime)

        # 2-D flux map
        _write_array(f, eq.psirz)

        # q profile
        _write_array(f, eq.qpsi)

        # Boundary / limiter counts
        nbdry = len(eq.rbdry)
        nlim = len(eq.rlim)
        f.write(f"{nbdry:5d}{nlim:5d}\n")

        # Boundary pairs
        for i in range(nbdry):
            f.write(_fmt(eq.rbdry[i]))
            f.write(_fmt(eq.zbdry[i]))
            if (i + 1) % 2 == 0 or i == nbdry - 1:
                # not quite standard but widely accepted
                pass
            if ((i + 1) * 2) % 5 == 0 or i == nbdry - 1:
                f.write("\n")
        if nbdry > 0 and (nbdry * 2) % 5 != 0:
            f.write("\n")

        # Limiter pairs
        for i in range(nlim):
            f.write(_fmt(eq.rlim[i]))
            f.write(_fmt(eq.zlim[i]))
            if ((i + 1) * 2) % 5 == 0 or i == nlim - 1:
                f.write("\n")
        if nlim > 0 and (nlim * 2) % 5 != 0:
            f.write("\n")
