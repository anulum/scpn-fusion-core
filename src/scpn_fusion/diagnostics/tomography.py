import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear

class PlasmaTomography:
    """
    Reconstructs 2D Emissivity Profile from 1D Chord Measurements.
    Uses Tikhonov Regularization (Pixel-based inversion).
    """
    def __init__(self, sensors, grid_res=20):
        self.sensors = sensors
        self.res = grid_res
        
        # Reconstruction Grid (Coarser than physics grid)
        self.R_rec = np.linspace(sensors.kernel.R[0], sensors.kernel.R[-1], grid_res)
        self.Z_rec = np.linspace(sensors.kernel.Z[0], sensors.kernel.Z[-1], grid_res)
        self.n_pixels = grid_res * grid_res
        
        # Build Geometry Matrix (A)
        # Signal_i = Sum_j (A_ij * Pixel_j)
        self.A = self._build_geometry_matrix()
        
    def _build_geometry_matrix(self):
        print("[Tomography] Building Geometry Matrix A...")
        n_chords = len(self.sensors.bolo_chords)
        A = np.zeros((n_chords, self.n_pixels))
        
        # For each chord
        for i, (start, end) in enumerate(self.sensors.bolo_chords):
            # Ray trace through reconstruction grid
            # Simplified: Check which pixels the line intersects
            # Here using sample points
            num_samples = 100
            r_samples = np.linspace(start[0], end[0], num_samples)
            z_samples = np.linspace(start[1], end[1], num_samples)
            dl = np.linalg.norm(end - start) / num_samples
            
            for k in range(num_samples):
                r, z = r_samples[k], z_samples[k]
                
                # Find pixel index
                ir = int((r - self.R_rec[0]) / (self.R_rec[1]-self.R_rec[0]))
                iz = int((z - self.Z_rec[0]) / (self.Z_rec[1]-self.Z_rec[0]))
                
                if 0 <= ir < self.res and 0 <= iz < self.res:
                    pixel_idx = iz * self.res + ir
                    A[i, pixel_idx] += dl
                    
        return A

    def reconstruct(self, signals):
        """
        Solves: min ||Ax - b||^2 + lambda * ||Lx||^2
        """
        # Regularization (Smoothness)
        lambda_reg = 0.1
        
        # Laplacian Matrix L (Finite difference of pixels)
        # Hard to construct for 1D flattened vector, so we use Identity (Ridge Regression)
        # min ||Ax - b||^2 + lambda * ||x||^2
        
        # Augmented System
        # [ A          ] [ x ]   [ b ]
        # [ sqrt(lam)I ]         [ 0 ]
        
        A_aug = np.vstack([self.A, np.sqrt(lambda_reg) * np.eye(self.n_pixels)])
        b_aug = np.concatenate([signals, np.zeros(self.n_pixels)])
        
        # Solve Least Squares
        # Use lsq_linear to enforce non-negativity (Emissivity > 0)
        res = lsq_linear(A_aug, b_aug, bounds=(0, np.inf), tol=1e-4)
        
        return res.x.reshape((self.res, self.res))

    def plot_reconstruction(self, ground_truth, reconstruction):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.set_title("Ground Truth (Phantom)")
        ax1.imshow(ground_truth, origin='lower', cmap='hot')
        
        ax2.set_title("Tomographic Reconstruction")
        ax2.imshow(reconstruction, origin='lower', cmap='hot')
        
        return fig
