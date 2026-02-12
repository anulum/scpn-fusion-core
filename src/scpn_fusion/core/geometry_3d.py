# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Geometry 3d
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""3D flux-surface mesh generation utilities.

This module extracts an approximate last-closed flux surface (LCFS) from the
2D Grad-Shafranov equilibrium and revolves it toroidally into a triangular mesh.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel


class Reactor3DBuilder:
    """Build a 3D toroidal mesh from a solved 2D equilibrium."""

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        *,
        kernel: Optional[FusionKernel] = None,
        solve_equilibrium: bool = True,
    ) -> None:
        if kernel is not None:
            self.kernel = kernel
        elif config_path is not None:
            self.kernel = FusionKernel(str(config_path))
        else:
            raise ValueError("Provide either `config_path` or `kernel`.")

        if solve_equilibrium:
            self.kernel.solve_equilibrium()

    def _inside_domain(self, r_val: float, z_val: float) -> bool:
        return (
            self.kernel.R[0] <= r_val <= self.kernel.R[-1]
            and self.kernel.Z[0] <= z_val <= self.kernel.Z[-1]
        )

    def _sample_psi_bilinear(self, r_val: float, z_val: float) -> float:
        if not self._inside_domain(r_val, z_val):
            raise ValueError("Requested sample outside kernel domain.")

        ir = int(np.searchsorted(self.kernel.R, r_val) - 1)
        iz = int(np.searchsorted(self.kernel.Z, z_val) - 1)
        ir = int(np.clip(ir, 0, self.kernel.NR - 2))
        iz = int(np.clip(iz, 0, self.kernel.NZ - 2))

        r0, r1 = self.kernel.R[ir], self.kernel.R[ir + 1]
        z0, z1 = self.kernel.Z[iz], self.kernel.Z[iz + 1]
        fr = float((r_val - r0) / (r1 - r0 + 1e-12))
        fz = float((z_val - z0) / (z1 - z0 + 1e-12))

        p00 = self.kernel.Psi[iz, ir]
        p01 = self.kernel.Psi[iz, ir + 1]
        p10 = self.kernel.Psi[iz + 1, ir]
        p11 = self.kernel.Psi[iz + 1, ir + 1]

        p0 = (1.0 - fr) * p00 + fr * p01
        p1 = (1.0 - fr) * p10 + fr * p11
        return float((1.0 - fz) * p0 + fz * p1)

    def _axis_and_boundary_flux(self) -> tuple[float, float, float, float]:
        idx_max = int(np.argmax(self.kernel.Psi))
        iz_ax, ir_ax = np.unravel_index(idx_max, self.kernel.Psi.shape)
        r_ax = float(self.kernel.R[ir_ax])
        z_ax = float(self.kernel.Z[iz_ax])
        psi_axis = float(self.kernel.Psi[iz_ax, ir_ax])

        _, psi_x = self.kernel.find_x_point(self.kernel.Psi)
        psi_boundary = float(psi_x)

        if not np.isfinite(psi_boundary) or abs(psi_boundary - psi_axis) < 1e-10:
            psi_boundary = float(np.nanmin(self.kernel.Psi))

        if abs(psi_boundary - psi_axis) < 1e-10:
            raise RuntimeError("Failed to infer a valid boundary flux.")

        return r_ax, z_ax, psi_axis, psi_boundary

    def _fallback_ellipse(
        self,
        resolution_poloidal: int,
        r_ax: float,
        z_ax: float,
    ) -> np.ndarray:
        """Build a conservative elliptical boundary fallback inside the domain."""
        r_margin = min(r_ax - float(self.kernel.R[0]), float(self.kernel.R[-1]) - r_ax)
        z_margin = min(z_ax - float(self.kernel.Z[0]), float(self.kernel.Z[-1]) - z_ax)
        semi_r = max(0.85 * r_margin, 1e-3)
        semi_z = max(0.85 * z_margin, 1e-3)

        thetas = np.linspace(0.0, 2.0 * np.pi, resolution_poloidal, endpoint=False)
        points = np.zeros((resolution_poloidal, 2), dtype=float)
        points[:, 0] = r_ax + semi_r * np.cos(thetas)
        points[:, 1] = z_ax + semi_z * np.sin(thetas)
        return points

    def _trace_lcfs(self, resolution_poloidal: int, radial_steps: int) -> np.ndarray:
        if resolution_poloidal < 8:
            raise ValueError("resolution_poloidal must be >= 8.")
        if radial_steps < 16:
            raise ValueError("radial_steps must be >= 16.")

        r_ax, z_ax, psi_axis, psi_boundary = self._axis_and_boundary_flux()

        max_radius = 1.05 * max(
            abs(self.kernel.R[-1] - r_ax),
            abs(self.kernel.R[0] - r_ax),
            abs(self.kernel.Z[-1] - z_ax),
            abs(self.kernel.Z[0] - z_ax),
        )

        points: list[list[float]] = []
        thetas = np.linspace(0.0, 2.0 * np.pi, resolution_poloidal, endpoint=False)
        ray_samples = np.linspace(0.0, max_radius, radial_steps + 1)

        for theta in thetas:
            cos_t = float(np.cos(theta))
            sin_t = float(np.sin(theta))

            prev_radius = 0.0
            prev_diff = psi_axis - psi_boundary
            found = False
            sampled_radii: list[float] = []
            sampled_diffs: list[float] = []

            for radius in ray_samples[1:]:
                r_test = r_ax + radius * cos_t
                z_test = z_ax + radius * sin_t

                if not self._inside_domain(r_test, z_test):
                    break

                psi_test = self._sample_psi_bilinear(r_test, z_test)
                diff = psi_test - psi_boundary
                sampled_radii.append(float(radius))
                sampled_diffs.append(float(diff))

                crossed = diff == 0.0 or (prev_diff > 0.0 and diff < 0.0) or (
                    prev_diff < 0.0 and diff > 0.0
                )
                if crossed:
                    delta = diff - prev_diff
                    frac = 0.0 if abs(delta) < 1e-12 else float(-prev_diff / delta)
                    frac = float(np.clip(frac, 0.0, 1.0))
                    hit_radius = prev_radius + frac * (radius - prev_radius)
                    points.append(
                        [
                            r_ax + hit_radius * cos_t,
                            z_ax + hit_radius * sin_t,
                        ]
                    )
                    found = True
                    break

                prev_radius = radius
                prev_diff = diff

            if not found and sampled_diffs:
                # If no strict crossing is found on this ray (common on coarse/sparse
                # equilibria), use the closest sampled point to the boundary level.
                min_idx = int(np.argmin(np.abs(np.asarray(sampled_diffs, dtype=float))))
                hit_radius = sampled_radii[min_idx]
                points.append(
                    [
                        r_ax + hit_radius * cos_t,
                        z_ax + hit_radius * sin_t,
                    ]
                )

        if len(points) < max(8, resolution_poloidal // 3):
            return self._fallback_ellipse(resolution_poloidal, r_ax, z_ax)

        return np.asarray(points, dtype=float)

    def generate_plasma_surface(
        self,
        resolution_toroidal: int = 60,
        resolution_poloidal: int = 60,
        radial_steps: int = 512,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a triangulated toroidal surface mesh.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(vertices, faces)``, where vertices are shaped ``(N, 3)`` and
            faces are shaped ``(M, 3)`` with zero-based indices.
        """

        if resolution_toroidal < 3:
            raise ValueError("resolution_toroidal must be >= 3.")

        poloidal_points = self._trace_lcfs(resolution_poloidal, radial_steps)
        n_pol = int(poloidal_points.shape[0])
        n_tor = int(resolution_toroidal)

        vertices: list[list[float]] = []
        phi_values = np.linspace(0.0, 2.0 * np.pi, n_tor, endpoint=False)
        for phi in phi_values:
            cos_phi = float(np.cos(phi))
            sin_phi = float(np.sin(phi))
            for r_val, z_val in poloidal_points:
                vertices.append([r_val * cos_phi, r_val * sin_phi, z_val])

        faces: list[list[int]] = []
        for i in range(n_tor):
            i_next = (i + 1) % n_tor
            for j in range(n_pol):
                j_next = (j + 1) % n_pol

                a = i * n_pol + j
                b = i * n_pol + j_next
                c = i_next * n_pol + j
                d = i_next * n_pol + j_next

                faces.append([a, b, c])
                faces.append([b, d, c])

        return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)

    def generate_coil_meshes(self) -> list[dict[str, float | str]]:
        """Return compact metadata for external coil geometry placeholders."""
        meshes: list[dict[str, float | str]] = []
        for coil in self.kernel.cfg["coils"]:
            meshes.append(
                {
                    "name": coil["name"],
                    "R": float(coil["r"]),
                    "Z": float(coil["z"]),
                }
            )
        return meshes

    def export_obj(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        filename: str | Path = "plasma.obj",
    ) -> Path:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as handle:
            handle.write("# SCPN 3D Plasma Export\n")
            for v in vertices:
                handle.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

        return output_path

    def export_preview_png(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        filename: str | Path = "plasma_preview.png",
        *,
        dpi: int = 140,
    ) -> Path:
        """Render mesh preview to a PNG using matplotlib."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(8.0, 8.0), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            triangles=faces,
            cmap="viridis",
            linewidth=0.05,
            antialiased=True,
            alpha=0.95,
        )
        ax.set_title("SCPN LCFS 3D Mesh Preview")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.view_init(elev=18, azim=45)

        # Keep axes visually proportional.
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        center = 0.5 * (mins + maxs)
        radius = 0.55 * float(np.max(maxs - mins))
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)
        return output_path


def _default_config_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "validation" / "iter_validated_config.json"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate SCPN 3D plasma OBJ mesh.")
    parser.add_argument(
        "--config",
        default=str(_default_config_path()),
        help="Path to reactor JSON config.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/SCPN_Plasma_3D.obj",
        help="Output OBJ path.",
    )
    parser.add_argument(
        "--toroidal",
        type=int,
        default=60,
        help="Toroidal mesh resolution.",
    )
    parser.add_argument(
        "--poloidal",
        type=int,
        default=60,
        help="Poloidal contour resolution.",
    )
    parser.add_argument(
        "--radial-steps",
        type=int,
        default=512,
        help="Ray marching samples per poloidal angle.",
    )
    parser.add_argument(
        "--preview-png",
        default="",
        help="Optional PNG preview output path.",
    )
    args = parser.parse_args(argv)

    builder = Reactor3DBuilder(args.config)
    vertices, faces = builder.generate_plasma_surface(
        resolution_toroidal=args.toroidal,
        resolution_poloidal=args.poloidal,
        radial_steps=args.radial_steps,
    )
    path = builder.export_obj(vertices, faces, args.output)
    if args.preview_png:
        png_path = builder.export_preview_png(vertices, faces, args.preview_png)
        print(f"Preview PNG: {png_path}")
    print(
        f"Exported {path} with {len(vertices)} vertices and {len(faces)} faces "
        f"(toroidal={args.toroidal}, poloidal={args.poloidal})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
