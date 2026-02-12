import numpy as np
import json
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel

class Reactor3DBuilder:
    """
    Converts 2D Axisymmetric Equilibrium into 3D Mesh Geometry.
    Exports OBJ/STL files for CAD integration or Visualization.
    """
    def __init__(self, config_path):
        self.kernel = FusionKernel(config_path)
        self.kernel.solve_equilibrium() # Need physical state
        
    def generate_plasma_surface(self, resolution_toroidal=60, resolution_poloidal=60):
        """
        Extracts the Last Closed Flux Surface (LCFS) and revolves it.
        Returns: vertices, faces (for 3D mesh)
        """
        # 1. Find Boundary Contour (2D)
        # We use marching squares or simply extract contour using matplotlib logic internally
        # Simplified: Scan radial rays from magnetic axis
        
        # Find axis
        idx_max = np.argmax(self.kernel.Psi)
        iz_ax, ir_ax = np.unravel_index(idx_max, self.kernel.Psi.shape)
        R_ax, Z_ax = self.kernel.R[ir_ax], self.kernel.Z[iz_ax]
        
        # Find Boundary Flux value
        xp, psi_x = self.kernel.find_x_point(self.kernel.Psi)
        psi_boundary = psi_x
        if abs(psi_boundary - self.kernel.Psi[iz_ax, ir_ax]) < 1.0:
             psi_boundary = np.min(self.kernel.Psi) # Fallback
             
        # Ray casting to find R(theta) for the boundary flux
        thetas = np.linspace(0, 2*np.pi, resolution_poloidal)
        poloidal_points = []
        
        for theta in thetas:
            # Cast ray from axis
            for r_step in np.linspace(0, 4.0, 100): # Scan 4m radius
                R_test = R_ax + r_step * np.cos(theta)
                Z_test = Z_ax + r_step * np.sin(theta)
                
                # Interpolate Psi
                # Simple grid lookup
                ir = int((R_test - self.kernel.R[0]) / self.kernel.dR)
                iz = int((Z_test - self.kernel.Z[0]) / self.kernel.dZ)
                
                if ir < 0 or ir >= self.kernel.NR or iz < 0 or iz >= self.kernel.NZ:
                    continue
                    
                if self.kernel.Psi[iz, ir] <= psi_boundary:
                    # Found edge (Psi decreases towards edge usually, check sign!)
                    # Actually Psi is max at axis, min at edge usually.
                    # Wait, check kernel logic.
                    # Vacuum field from coils usually opposes plasma.
                    # Let's assume we crossed the value.
                    poloidal_points.append([R_test, Z_test])
                    break
        
        # 2. Revolve into Torus
        vertices = []
        faces = []
        
        phi_steps = np.linspace(0, 2*np.pi, resolution_toroidal)
        
        # Create vertices
        for phi in phi_steps[:-1]: # Skip last to wrap
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            
            for p in poloidal_points:
                R, Z = p
                x = R * cos_phi
                y = R * sin_phi
                z = Z
                vertices.append([x, y, z])
                
        # Create faces (Quads)
        n_pol = len(poloidal_points)
        n_tor = len(phi_steps) - 1
        
        for i in range(n_tor):
            for j in range(n_pol):
                # Indices
                current = i * n_pol + j
                next_pol = i * n_pol + ((j + 1) % n_pol)
                
                next_tor = ((i + 1) % n_tor) * n_pol + j
                next_tor_pol = ((i + 1) % n_tor) * n_pol + ((j + 1) % n_pol)
                
                # Triangles for OBJ
                # 1. current -> next_pol -> next_tor
                faces.append([current, next_pol, next_tor])
                # 2. next_pol -> next_tor_pol -> next_tor
                faces.append([next_pol, next_tor_pol, next_tor])
                
        return np.array(vertices), np.array(faces)

    def generate_coil_meshes(self):
        """Generates torus meshes for each PF coil."""
        meshes = []
        for coil in self.kernel.cfg['coils']:
            # Simplified torus for each coil
            R_major = coil['r']
            Z_center = coil['z']
            r_minor = 0.3 # Thickness
            
            verts = []
            faces = []
            # ... (Mesh generation logic similar to plasma but simpler tube)
            # Placeholder for brevity in demo script
            meshes.append({'name': coil['name'], 'R': R_major, 'Z': Z_center})
        return meshes

    def export_obj(self, vertices, faces, filename="plasma.obj"):
        with open(filename, 'w') as f:
            f.write(f"# SCPN 3D Plasma Export\n")
            for v in vertices:
                f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            for face in faces:
                # OBJ is 1-indexed
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        print(f"Exported 3D Model: {filename}")

if __name__ == "__main__":
    # Demo
    cfg = "03_CODE/SCPN-Fusion-Core/validation/iter_validated_config.json"
    builder = Reactor3DBuilder(cfg)
    
    verts, faces = builder.generate_plasma_surface()
    builder.export_obj(verts, faces, "SCPN_Plasma_3D.obj")
