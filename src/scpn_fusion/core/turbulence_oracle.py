import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import rand
from scipy.linalg import solve
import sys
import os

# --- HASEGAWA-WAKATANI PARAMETERS ---
GRID = 64
L = 10.0
ALPHA = 0.1  # Adiabaticity parameter
KAPPA = 0.5  # Density gradient drive
NU = 0.01    # Viscosity
DT = 0.05

class DriftWavePhysics:
    """
    Solves the 2D Hasegawa-Wakatani equations for Plasma Edge Turbulence.
    Variables:
      phi: Electrostatic potential (Stream function)
      n:   Density fluctuation
    """
    def __init__(self, N=GRID):
        self.N = N
        self.k = np.fft.fftfreq(N, d=L/(2*np.pi*N))
        self.kx, self.ky = np.meshgrid(self.k, self.k)
        self.k2 = self.kx**2 + self.ky**2
        self.k2[0,0] = 1.0 # Avoid division by zero
        
        # De-aliasing mask (2/3 rule)
        # Filters out high-k modes that cause spectral blocking explosion
        k_max = np.max(np.abs(self.k))
        self.mask = np.where(self.k2 < (2./3. * k_max)**2, 1.0, 0.0)
        
        # Init State (Random Noise)
        self.phi_k = np.fft.fft2(np.random.randn(N,N) * 0.01) * self.mask
        self.n_k   = np.fft.fft2(np.random.randn(N,N) * 0.01) * self.mask

    def bracket(self, A_k, B_k):
        """Calculates Poisson Bracket [A, B] with de-aliasing"""
        # Derivatives in spectral space
        dxA = np.fft.ifft2(1j * self.kx * A_k)
        dyA = np.fft.ifft2(1j * self.ky * A_k)
        dxB = np.fft.ifft2(1j * self.kx * B_k)
        dyB = np.fft.ifft2(1j * self.ky * B_k)
        
        # Nonlinear product in real space
        nonlin = dxA * dyB - dyA * dxB
        
        # Back to spectral + De-aliasing
        return np.fft.fft2(nonlin) * self.mask

    def step(self):
        """Runge-Kutta 4th Order Step with Stability Clamp"""
        p = self.phi_k
        n = self.n_k
        
        # Reduced time step for stability
        local_dt = 0.01
        
        def rhs(p_in, n_in):
            # Enforce mask
            p_in *= self.mask
            n_in *= self.mask
            
            # Vorticity w = -k^2 phi
            w_in = -self.k2 * p_in 
            
            # Non-linear terms
            brack_phi_w = self.bracket(p_in, w_in)
            brack_phi_n = self.bracket(p_in, n_in)
            
            # Linear terms (Hasegawa-Wakatani)
            # dw/dt = -[phi,w] + alpha*(phi-n) - nu*k^4*w
            # dn/dt = -[phi,n] + alpha*(phi-n) - kappa*dy_phi - nu*k^4*n
            
            coupling = ALPHA * (p_in - n_in)
            
            # Viscosity (Hyper-viscosity k^4 often used, here k^2 for simplicity)
            dissip_w = NU * self.k2 * w_in 
            dissip_n = NU * self.k2 * n_in
            
            dw_dt = -brack_phi_w + coupling - dissip_w
            
            # Invert to get d(phi)/dt: dphi = -dw / k^2
            dp_dt = -dw_dt / self.k2
            dp_dt[0,0] = 0.0 # Zero mean
            
            dn_dt = -brack_phi_n + coupling - KAPPA * (1j * self.ky * p_in) - dissip_n
            
            return dp_dt, dn_dt

        # RK4 Integration
        k1_p, k1_n = rhs(p, n)
        k2_p, k2_n = rhs(p + 0.5*local_dt*k1_p, n + 0.5*local_dt*k1_n)
        k3_p, k3_n = rhs(p + 0.5*local_dt*k2_p, n + 0.5*local_dt*k2_n)
        k4_p, k4_n = rhs(p + local_dt*k3_p, n + local_dt*k3_n)
        
        self.phi_k += (local_dt/6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
        self.n_k   += (local_dt/6.0) * (k1_n + 2*k2_n + 2*k3_n + k4_n)
        
        # Stability Clamp (Prevent blow-up)
        max_amp = np.max(np.abs(self.phi_k))
        if max_amp > 100.0:
            # Rescale to prevent overflow
            scale = 100.0 / max_amp
            self.phi_k *= scale
            self.n_k *= scale
            
        return np.real(np.fft.ifft2(self.phi_k)), np.real(np.fft.ifft2(self.n_k))

class OracleESN:
    """
    Echo State Network (Reservoir Computing).
    Specialized for predicting chaotic time series.
    """
    def __init__(self, input_dim, reservoir_size=500, spectral_radius=0.95):
        self.W_in = np.random.uniform(-1, 1, (reservoir_size, input_dim))
        
        # Sparse Reservoir
        self.W_res = rand(reservoir_size, reservoir_size, density=0.1).toarray()
        
        # Scale spectral radius
        eigenvalues = np.linalg.eigvals(self.W_res)
        self.W_res *= spectral_radius / np.max(np.abs(eigenvalues))
        
        self.state = np.zeros(reservoir_size)
        self.W_out = None # Trained later
        
    def train(self, inputs, targets):
        """Ridge Regression training"""
        print(f"[Oracle] Training on {len(inputs)} chaotic states...")
        states = []
        
        # Harvest states
        for u in inputs:
            self.state = np.tanh(np.dot(self.W_in, u) + np.dot(self.W_res, self.state))
            states.append(self.state)
            
        S = np.array(states) # [Time, Reservoir]
        
        # Solve W_out * S.T = Targets.T
        # W_out = Targets.T * S * inv(S.T * S + beta*I)
        reg = 1e-4
        self.W_out = np.dot(np.dot(targets.T, S), np.linalg.inv(np.dot(S.T, S) + reg * np.eye(S.shape[1])))
        
        print("[Oracle] Mental Model Formed.")

    def predict(self, u_current, steps=50):
        predictions = []
        curr = u_current
        
        for _ in range(steps):
            # Update reservoir with current feedback (Closed Loop prediction)
            self.state = np.tanh(np.dot(self.W_in, curr) + np.dot(self.W_res, self.state))
            
            # Readout
            pred = np.dot(self.W_out, self.state)
            predictions.append(pred)
            curr = pred # Feed prediction back
            
        return np.array(predictions)

def run_turbulence_oracle():
    print("--- SCPN TURBULENCE ORACLE: PREDICTING CHAOS ---")
    
    # 1. Generate Chaos (Ground Truth)
    hw = DriftWavePhysics()
    
    # Warmup physics
    for _ in range(100): hw.step()
    
    print("Generating Training Data (Hasegawa-Wakatani)...")
    data_phi = []
    
    # We sample the field at 16 probe locations (Sparse Sensing)
    # Reconstructing full field is Tomography job, here we predict probes
    probe_idx = np.linspace(0, GRID*GRID-1, 16, dtype=int)
    
    for _ in range(1000):
        phi, _ = hw.step()
        probes = phi.flatten()[probe_idx]
        data_phi.append(probes)
        
    data = np.array(data_phi)
    
    # Split Train/Test
    train_len = 800
    X_train = data[:train_len]
    Y_train = data[1:train_len+1] # Next step target
    
    # 2. Train Oracle
    oracle = OracleESN(input_dim=16)
    oracle.train(X_train, Y_train)
    
    # 3. Test Prediction Horizon
    print("Testing Prediction Horizon...")
    start_state = data[train_len]
    horizon = 150
    
    # Physics Future (Ground Truth)
    truth = data[train_len:train_len+horizon]
    
    # AI Future (Hallucination)
    prediction = oracle.predict(start_state, steps=horizon)
    
    # 4. Analysis
    # Calculate Divergence (Lyapunov)
    mse = np.mean((truth - prediction)**2, axis=1)
    
    # Find "Trust Horizon" (where error exceeds threshold)
    threshold = 0.5 * np.var(truth)
    try:
        trust_steps = np.where(mse > threshold)[0][0]
    except IndexError:
        trust_steps = horizon
        
    print(f"Prediction Horizon: {trust_steps} steps")
    print(f"Physics Time: {trust_steps * DT:.2f} normalized units")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot one probe trace
    ax1.plot(truth[:,0], 'k-', label='Reality (H-W Physics)')
    ax1.plot(prediction[:,0], 'r--', label='Oracle Prediction (ESN)')
    ax1.axvline(trust_steps, color='b', linestyle=':', label='Lyapunov Horizon')
    ax1.set_title("Turbulence Probe Signal Prediction")
    ax1.legend()
    
    # Plot Divergence
    ax2.plot(mse, 'g-')
    ax2.axhline(threshold, color='k', linestyle='--')
    ax2.set_title("Forecast Error Divergence (Chaos)")
    ax2.set_xlabel("Steps into Future")
    ax2.set_ylabel("MSE")
    
    plt.tight_layout()
    plt.savefig("Turbulence_Oracle.png")
    print("Saved: Turbulence_Oracle.png")

if __name__ == "__main__":
    run_turbulence_oracle()
