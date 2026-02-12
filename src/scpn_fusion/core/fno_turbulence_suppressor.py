import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# --- FNO PARAMETERS ---
MODES = 12  # Fourier Modes to keep
WIDTH = 32  # Feature channels
GRID_SIZE = 64
TIME_STEPS = 200

class SpectralTurbulenceGenerator:
    """
    Generates synthetic ITG (Ion Temperature Gradient) turbulence.
    Uses Kolmogorov spectrum dynamics.
    """
    def __init__(self, size=GRID_SIZE):
        self.size = size
        self.x = np.linspace(0, 2*np.pi, size)
        self.y = np.linspace(0, 2*np.pi, size)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Initialize Field (Density Fluctuations)
        self.field = np.random.randn(size, size) * 0.1
        self.field_k = np.fft.fft2(self.field)
        
    def step(self, dt=0.01, damping=0.0):
        """
        Evolves turbulence in Fourier Space.
        dk/dt = -Linear_Dispersion + NonLinear_Coupling - Damping
        """
        # 1. Wavenumbers
        kx = np.fft.fftfreq(self.size) * self.size
        ky = np.fft.fftfreq(self.size) * self.size
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2
        K2[0,0] = 1.0 # Avoid div by zero
        
        # 2. Linear Drift Wave Physics (simplified dispersion)
        # omega ~ ky / (1 + k^2)
        omega = KY / (1.0 + K2)
        
        # 3. Evolution (Phase Shift)
        # exp(-i * omega * dt)
        phase_shift = np.exp(-1j * omega * dt)
        
        # 4. Non-Linear Forcing (Kolmogorov Cascade)
        # Energy moves from source (low k) to dissipation (high k)
        forcing = np.random.randn(self.size, self.size) + 1j*np.random.randn(self.size, self.size)
        forcing_k = np.fft.fft2(forcing)
        
        # Filter forcing to be low-k driven
        mask = K2 < 25.0
        forcing_k *= mask * 5.0 # Stronger forcing to fight damping
        
        # 5. Apply
        self.field_k = (self.field_k * phase_shift) + (forcing_k * dt)
        
        # Damping (Viscosity) + Active Suppression
        self.field_k *= np.exp(-0.001 * K2 * dt) # Reduced viscosity
        self.field_k *= (1.0 - damping)         # Control Action
        
        # Inverse FFT
        self.field = np.real(np.fft.ifft2(self.field_k))
        return self.field

class FNO_Block:
    """
    Fourier Neural Operator Layer (Simplified numpy implementation).
    Performs convolution in spectral domain.
    """
    def __init__(self, modes):
        self.modes = modes
        self.scale = 1.0 / (modes**2)
        self.weights = np.random.randn(modes, modes, 2) * 1.0 # Larger weights for stronger initial signal
        
    def forward(self, x_field):
        # 1. FFT
        x_k = np.fft.fft2(x_field)
        
        # 2. Filter (Keep only low modes)
        x_k_trunc = x_k[:self.modes, :self.modes]
        
        # 3. Spectral Convolution (Element-wise multiplication with weights)
        # Z = R + i*I
        # W = Wr + i*Wi
        # Z*W = (R*Wr - I*Wi) + i*(R*Wi + I*Wr)
        
        R = np.real(x_k_trunc)
        I = np.imag(x_k_trunc)
        Wr = self.weights[:,:,0]
        Wi = self.weights[:,:,1]
        
        out_r = R*Wr - I*Wi
        out_i = R*Wi + I*Wr
        
        out_k_trunc = out_r + 1j*out_i
        
        # 4. Pad back to full size
        out_k = np.zeros_like(x_k)
        out_k[:self.modes, :self.modes] = out_k_trunc
        
        # 5. IFFT
        return np.real(np.fft.ifft2(out_k))

class FNO_Controller:
    """
    Predicts turbulence evolution and generates counter-waves.
    """
    def __init__(self):
        self.fno = FNO_Block(MODES)
        
    def predict_and_suppress(self, field):
        # FNO predicts the 'structure' of the next turbulent eddy
        prediction = self.fno.forward(field)
        
        # Control Strategy:
        # If we know where the eddy will be, we apply local shear flow.
        # Here we calculate a global 'suppression metric' based on prediction magnitude.
        
        energy = np.mean(prediction**2)
        
        # Simple policy: Higher predicted energy -> Stronger damping
        suppression = np.tanh(energy * 10.0) 
        return suppression, prediction

def run_fno_simulation():
    print("--- SCPN FNO: Spectral Turbulence Suppression ---")
    
    sim = SpectralTurbulenceGenerator()
    ai = FNO_Controller()
    
    # Visualization setup
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    history_energy = []
    
    print(f"Running {TIME_STEPS} steps of Gyro-Fluid Dynamics...")
    
    for t in range(TIME_STEPS):
        # 1. Physics Step (with active control)
        # We start with no control, then turn on AI at t=50
        control = 0.0
        prediction = np.zeros_like(sim.field)
        
        if t > 50:
            control, prediction = ai.predict_and_suppress(sim.field)
            
        field = sim.step(damping=control)
        
        # Metrics
        turb_energy = np.mean(field**2)
        pred_energy = np.mean(prediction**2)
        history_energy.append(turb_energy)
        
        if t % 20 == 0:
            print(f"Step {t}: Energy={turb_energy:.4f} | PredE={pred_energy:.4f} | Suppression={control:.2f}")
            
    # Final Plot
    ax1.imshow(sim.field, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax1.set_title(f"Turbulence Density (t={TIME_STEPS})")
    
    ax2.plot(history_energy, 'k-', label='Turbulence Energy')
    ax2.axvline(50, color='r', linestyle='--', label='AI ON')
    ax2.set_title("Suppression Performance")
    ax2.set_xlabel("Time Step")
    ax2.legend()
    
    plt.savefig("FNO_Turbulence_Result.png")
    print("Analysis saved: FNO_Turbulence_Result.png")

if __name__ == "__main__":
    run_fno_simulation()
