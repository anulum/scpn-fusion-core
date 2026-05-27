# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — G-EQDSK Equilibrium File Parser
"""
Reader and writer for the G-EQDSK (EFIT) equilibrium file format.

The G-EQDSK format is the *de facto* standard for tokamak equilibrium
data exchange.  It stores the poloidal flux map ψ(R,Z) on a uniform
(R,Z) grid together with 1-D profile arrays and scalar quantities.

References include Lao et al., Nucl. Fusion 25 (1985) 1611, FreeQDSK
documentation, and the ITER IMAS equilibrium IDS.

The parser expects the standard Fortran fixed-width layout: five values per
line using ``(5e16.9)``, a 48-character header description followed by idum,
``nw`` and ``nh``, the 20-scalar equilibrium block, five 1-D profile arrays,
the row-major ``psirz`` flux map, and boundary/limiter ``(R, Z)`` point lists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Union, cast

import numpy as np
from numpy.typing import NDArray

MAX_GEQDSK_BYTES = 10 * 1024 * 1024
MAX_GEQDSK_GRID_POINTS = 1_000_000
MAX_GEQDSK_CONTOUR_POINTS = 100_000

GEQDSK_SOURCE_CONVENTION_MODES = {
    "raw_canonical": "raw_canonical",
    "public_sparc_named_adapter": "public_sparc_named_adapter",
}

GEQDSK_PUBLIC_SPARC_SOURCE_ADAPTERS = {
    "sparc_1305.eqdsk": "scaled_by_2pi",
    "sparc_1310.eqdsk": "scaled_by_2pi",
    "sparc_1315.eqdsk": "scaled_by_2pi",
    "sparc_1349.eqdsk": "scaled_by_2pi",
}

GEQDSK_SOURCE_CONVENTION_ADAPTERS = {
    "scaled_by_2pi": 2.0 * np.pi,
}


# ── Data container ────────────────────────────────────────────────────


@dataclass
class GEqdsk:
    """Container for all data in a G-EQDSK file."""

    # Header
    description: str = ""
    nw: int = 0  # number of horizontal (R) grid points
    nh: int = 0  # number of vertical (Z) grid points

    # Scalars
    rdim: float = 0.0  # horizontal dimension (m)
    zdim: float = 0.0  # vertical dimension (m)
    rcentr: float = 0.0  # reference R for vacuum toroidal field (m)
    rleft: float = 0.0  # R at left of computational domain (m)
    zmid: float = 0.0  # Z at centre of computational domain (m)
    rmaxis: float = 0.0  # R of magnetic axis (m)
    zmaxis: float = 0.0  # Z of magnetic axis (m)
    simag: float = 0.0  # ψ at magnetic axis (Wb/rad)
    sibry: float = 0.0  # ψ at plasma boundary (Wb/rad)
    bcentr: float = 0.0  # vacuum toroidal field at rcentr (T)
    current: float = 0.0  # plasma current (A)

    # 1-D profile arrays (length nw)
    fpol: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    pres: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    ffprime: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    pprime: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    qpsi: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))

    # 2-D flux map (nh × nw)
    psirz: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))

    # Boundary and limiter contours
    rbdry: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    zbdry: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    rlim: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    zlim: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))

    # Source convention metadata
    source_convention: str = "raw_canonical"
    source_convention_adapter: str = "not_applied"
    source_convention_adapter_pass: bool = False
    source_convention_metadata: dict[str, object] = field(default_factory=dict)

    # ── Derived grids ─────────────────────────────────────────────────

    @property
    def r(self) -> NDArray[np.float64]:
        """1-D array of R grid values."""
        return cast(NDArray[np.float64], np.linspace(self.rleft, self.rleft + self.rdim, self.nw))

    @property
    def z(self) -> NDArray[np.float64]:
        """1-D array of Z grid values."""
        return cast(
            NDArray[np.float64],
            np.linspace(self.zmid - self.zdim / 2, self.zmid + self.zdim / 2, self.nh),
        )

    @property
    def psi_norm(self) -> NDArray[np.float64]:
        """Normalised poloidal flux ψ_N ∈ [0, 1] (axis=0, boundary=1)."""
        return cast(NDArray[np.float64], np.linspace(0.0, 1.0, self.nw))

    def psi_to_norm(self, psi: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map raw ψ to normalised ψ_N."""
        return (psi - self.simag) / (self.sibry - self.simag)

    def to_config(self, name: str = "eqdsk") -> dict[str, object]:
        """Convert to a FusionKernel JSON config with GEQDSK shape metadata.

        GEQDSK files do not contain an external-coil set, so the returned
        ``coils`` list remains empty.  They do contain plasma-boundary and
        limiter contours; those are exported into ``free_boundary`` so native
        free-boundary workflows can use the EFIT boundary as an isoflux shape
        contract instead of silently discarding it.
        """
        r = self.r
        z = self.z
        free_boundary: dict[str, object] = {
            "magnetic_axis": [float(self.rmaxis), float(self.zmaxis)],
            "psi_axis": float(self.simag),
            "psi_boundary": float(self.sibry),
            "boundary_source": "geqdsk_rbdry_zbdry",
        }
        if self.rbdry.size and self.zbdry.size:
            if self.rbdry.shape != self.zbdry.shape:
                raise ValueError("GEQDSK boundary R/Z arrays must have matching lengths.")
            if not np.all(np.isfinite(self.rbdry)) or not np.all(np.isfinite(self.zbdry)):
                raise ValueError("GEQDSK boundary contour must contain finite values only.")
            boundary_points = np.column_stack([self.rbdry, self.zbdry]).astype(np.float64)
            free_boundary["target_flux_points"] = boundary_points.tolist()
            free_boundary["target_flux_values"] = np.full(
                boundary_points.shape[0],
                float(self.sibry),
                dtype=np.float64,
            ).tolist()
        if self.rlim.size or self.zlim.size:
            if self.rlim.shape != self.zlim.shape:
                raise ValueError("GEQDSK limiter R/Z arrays must have matching lengths.")
            if not np.all(np.isfinite(self.rlim)) or not np.all(np.isfinite(self.zlim)):
                raise ValueError("GEQDSK limiter contour must contain finite values only.")
            free_boundary["limiter_points"] = (
                np.column_stack([self.rlim, self.zlim]).astype(np.float64).tolist()
            )

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
            "free_boundary": free_boundary,
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
_FORTRAN_FLOAT_RE = re.compile(r"[+-]?\d*\.?\d+(?:[eEdD][+-]?\d+)?")


def _split_fortran(line: str) -> list[str]:
    """Extract all Fortran-format floats from *line*."""
    return _FORTRAN_FLOAT_RE.findall(line)


def _parse_header(line: str) -> tuple[str, int, int]:
    """Parse the first line: 48-char description + idum nw nh."""
    parts = line.split()
    if len(parts) < 3:
        raise ValueError("GEQDSK header must contain idum, nw, and nh")
    nh = int(parts[-1])
    nw = int(parts[-2])
    desc = " ".join(parts[:-3]) if len(parts) > 3 else ""
    return desc, nw, nh


def _validate_dimensions(nw: int, nh: int) -> None:
    if nw < 2 or nh < 2:
        raise ValueError(f"GEQDSK grid dimensions must be >= 2x2, got {(nw, nh)}")
    if nw * nh > MAX_GEQDSK_GRID_POINTS:
        raise ValueError(
            f"GEQDSK grid dimensions exceed safety limit {MAX_GEQDSK_GRID_POINTS}: got {nw}x{nh}"
        )


def _validate_contour_count(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"GEQDSK {name} count must be non-negative")
    if value > MAX_GEQDSK_CONTOUR_POINTS:
        raise ValueError(f"GEQDSK {name} count exceeds safety limit {MAX_GEQDSK_CONTOUR_POINTS}")


def _detect_named_source_adapter(path: Path) -> str | None:
    """Return the explicit source adapter for known public SPARC filenames."""
    return GEQDSK_PUBLIC_SPARC_SOURCE_ADAPTERS.get(path.name.lower())


def _apply_source_adapter(
    eq: GEqdsk,
    *,
    convention: str,
    source_file: Path,
    mode: str,
) -> GEqdsk:
    """Apply a documented source-convention adapter and record provenance."""
    if convention == "canonical":
        eq.source_convention = "canonical"
        eq.source_convention_adapter = "not_needed"
        eq.source_convention_adapter_pass = True
        eq.source_convention_metadata = {
            "requested_mode": mode,
            "source_file": source_file.name,
            "adapter": "not_applied",
            "applied_scale": 1.0,
            "provenance": "none",
        }
        return eq

    multiplier = GEQDSK_SOURCE_CONVENTION_ADAPTERS.get(convention)
    if multiplier is None:
        eq.source_convention = "unsupported"
        eq.source_convention_adapter = convention
        eq.source_convention_adapter_pass = False
        eq.source_convention_metadata = {
            "requested_mode": mode,
            "source_file": source_file.name,
            "adapter": convention,
            "applied_scale": float("nan"),
            "provenance": "unsupported_named_adapter",
            "error": "unsupported source convention adapter",
        }
        return eq

    eq.ffprime = multiplier * eq.ffprime
    eq.pprime = multiplier * eq.pprime
    eq.source_convention = "canonical"
    eq.source_convention_adapter = convention
    eq.source_convention_adapter_pass = True
    eq.source_convention_metadata = {
        "requested_mode": mode,
        "source_file": source_file.name,
        "adapter": convention,
        "applied_scale": float(multiplier),
        "provenance": "public_sparc_named_adapter",
        "public_case": source_file.name,
    }
    return eq


def read_geqdsk(
    path: Union[str, Path],
    *,
    source_convention_mode: str = "raw_canonical",
) -> GEqdsk:
    """
    Read a G-EQDSK file and return a :class:`GEqdsk` container.

    Handles both whitespace-separated and fixed-width Fortran formats,
    including the common case where values run together without spaces
    (e.g. ``2.385E+00-1.216E+01``).

    Parameters
    ----------
    path : str or Path
        Path to the G-EQDSK file.
    source_convention_mode : {"raw_canonical", "public_sparc_named_adapter"}
        ``raw_canonical`` keeps parsed ``ffprime`` and ``pprime`` in raw file
        units (default, strict mode).

        ``public_sparc_named_adapter`` applies a documented, source-aware
        adapter only for explicitly recognized public SPARC files.

    Returns
    -------
    GEqdsk
        Parsed equilibrium data.
    """
    path = Path(path)
    if source_convention_mode not in GEQDSK_SOURCE_CONVENTION_MODES:
        raise ValueError(
            "Unsupported source_convention_mode: "
            f"{source_convention_mode}. Expected one of: "
            + ", ".join(sorted(GEQDSK_SOURCE_CONVENTION_MODES.keys()))
        )
    size = path.stat().st_size
    if size > MAX_GEQDSK_BYTES:
        raise ValueError(f"GEQDSK file too large: {size} bytes exceeds {MAX_GEQDSK_BYTES}")
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        raise ValueError("GEQDSK file is empty")

    # Parse header
    desc, nw, nh = _parse_header(lines[0])
    _validate_dimensions(nw, nh)

    # Gather all numeric tokens using Fortran-aware splitting
    tokens: list[str] = []
    for line in lines[1:]:
        tokens.extend(_split_fortran(line))

    idx = 0

    def _next() -> float:
        nonlocal idx
        if idx >= len(tokens):
            raise ValueError("GEQDSK file ended before all required values were present")
        val = float(tokens[idx].replace("D", "E").replace("d", "e"))
        idx += 1
        return val

    def _read_array(n: int) -> NDArray[np.float64]:
        nonlocal idx
        if n < 0:
            raise ValueError("GEQDSK array length must be non-negative")
        if idx + n > len(tokens):
            raise ValueError("GEQDSK file ended before all required array values were present")
        arr = np.array(
            [float(tokens[idx + i].replace("D", "E").replace("d", "e")) for i in range(n)]
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
    _validate_contour_count("boundary", nbdry)
    _validate_contour_count("limiter", nlim)

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

    eq = GEqdsk(
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

    if source_convention_mode == "public_sparc_named_adapter":
        adapter = _detect_named_source_adapter(path)
        if adapter is None:
            eq.source_convention_adapter = "no_named_adapter"
            eq.source_convention_adapter_pass = False
            eq.source_convention = "raw_canonical"
            eq.source_convention_metadata = {
                "requested_mode": source_convention_mode,
                "source_file": path.name,
                "adapter": "no_named_adapter",
                "applied_scale": 1.0,
                "provenance": "public_sparc_named_adapter_no_match",
                "error": "no recognized public SPARC convention mapping",
            }
        else:
            eq = _apply_source_adapter(
                eq,
                convention=adapter,
                source_file=path,
                mode=source_convention_mode,
            )

    return eq


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

    def _write_array(f: "IO[str]", arr: NDArray[np.float64]) -> None:
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
            eq.rdim,
            eq.zdim,
            eq.rcentr,
            eq.rleft,
            eq.zmid,
            eq.rmaxis,
            eq.zmaxis,
            eq.simag,
            eq.sibry,
            eq.bcentr,
            eq.current,
            eq.simag,
            0.0,
            eq.rmaxis,
            0.0,
            eq.zmaxis,
            0.0,
            eq.sibry,
            0.0,
            0.0,
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
