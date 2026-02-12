import numpy as np
import matplotlib.pyplot as plt
try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel
import sys
import os
import time

# --- SOTA PARAMETERS ---
PREDICTION_HORIZON = 10  # Look-ahead steps (MPC)
SHOT_LENGTH = 100
LEARNING_SAMPLES = 500   # Data points to train the Surrogate

class NeuralSurrogate:
    """
    SOTA Concept: Differentiable Surrogate Model.
    Approximates the heavy Grad-Shafranov physics with a fast Neural Net.
    Maps: [Current_State, Coil_Action] -> [Next_State_Change]
    """
    def __init__(self, n_coils, n_state):
        # Weights for a simple linear predictor (A*x + B*u)
        # In a full SOTA implementation, this would be a deep MLP or LSTM
        # We initialize with identifying the linearized dynamics (System ID)
        self.A = np.eye(n_state) # State transition
        self.B = np.zeros((n_state, n_coils)) # Control impact
        
    def train_on_kernel(self, kernel):
        """
        'Distills' the physics kernel into the matrix model.
        Performs System Identification.
        """
        print("[SOTA] Training Neural Surrogate on Physics Kernel...")
        
        # 1. Base State
        kernel.solve_equilibrium()
        base_state = self.get_state(kernel)
        
        # 2. Perturbation Analysis
        perturbation = 1.0 # Amps
        
        for i in range(len(kernel.cfg['coils'])):
            # Record old
            old_I = kernel.cfg['coils'][i]['current']
            
            # Step
            kernel.cfg['coils'][i]['current'] += perturbation
            kernel.solve_equilibrium()
            new_state = self.get_state(kernel)
            
            # Derivative dState/dCurrent
            delta = (new_state - base_state) / perturbation
            self.B[:, i] = delta
            
            # Restore
            kernel.cfg['coils'][i]['current'] = old_I
            
        # Reset physics
        kernel.solve_equilibrium()
        print("[SOTA] Surrogate Training Complete.")
        
    def get_state(self, kernel):
        """Extracts critical state vector: [R_axis, Z_axis, X_point_R, X_point_Z]"""
        # Axis
        idx_max = np.argmax(kernel.Psi)
        iz, ir = np.unravel_index(idx_max, kernel.Psi.shape)
        r_ax = kernel.R[ir]
        z_ax = kernel.Z[iz]
        
        # X-point
        xp_pos, _ = kernel.find_x_point(kernel.Psi)
        
        return np.array([r_ax, z_ax, xp_pos[0], xp_pos[1]])

    def predict(self, current_state, action_delta):
        """
        Fast forward pass.
        Next_State = Current_State + B * Delta_Currents
        """
        return current_state + np.dot(self.B, action_delta)

class ModelPredictiveController:
    """
    SOTA Concept: MPC (Model Predictive Control).
    Optimizes a sequence of future actions to minimize cost.
    """
    def __init__(self, surrogate, target_state):
        self.model = surrogate
        self.target = target_state
        self.horizon = PREDICTION_HORIZON
        
    def plan_trajectory(self, current_state):
        """
        Solves the Optimal Control Problem over the horizon.
        Cost J = Sum( (State - Target)^2 + lambda * Action^2 )
        
        Since our surrogate is Linear (State_{k+1} = State_k + B*u), 
        we can solve this analytically (Least Squares) or via Gradient Descent.
        Here we use a simplified Gradient Descent planning.
        """
        n_coils = self.model.B.shape[1]
        
        # Initialize planned actions (Delta currents) with zeros
        # Plan is matrix [Horizon, Coils]
        planned_actions = np.zeros((self.horizon, n_coils))
        
        # Gradient Descent Optimization of the Plan
        # We iterate to refine the plan
        learning_rate = 0.5
        iterations = 20
        
        for _ in range(iterations):
            temp_state = current_state.copy()
            grads = np.zeros_like(planned_actions)
            
            # Forward Rollout
            trajectory = []
            for t in range(self.horizon):
                # Physics Step (Surrogate)
                next_state = self.model.predict(temp_state, planned_actions[t])
                trajectory.append(next_state)
                
                # Calculate Error Gradient
                # Loss = 0.5 * (next - target)^2
                # dLoss/dState = (next - target)
                # dState/dAction = B
                # dLoss/dAction = B.T * (next - target)
                
                error = next_state - self.target
                grad_step = np.dot(self.model.B.T, error)
                
                # Regularization (Penalize large currents)
                grad_step += 0.1 * planned_actions[t] 
                
                grads[t] = grad_step
                temp_state = next_state # Propagate state
            
            # Backward Update (Update Plan)
            planned_actions -= learning_rate * grads
            
            # Clip actions (Physical limits of power supplies)
            planned_actions = np.clip(planned_actions, -2.0, 2.0)
            
        # Return the first action of the optimal sequence
        return planned_actions[0]

def run_sota_simulation():
    print("\n--- SCPN FUSION SOTA: Neural-MPC Hybrid Control ---")
    
    # 1. Init Real Physics (Ground Truth)
    config_path = "03_CODE/SCPN-Fusion-Core/iter_config.json"
    kernel = FusionKernel(config_path)
    
    # 2. Train Surrogate (The "Digital Twin" used for planning)
    # State Vector: [R_axis, Z_axis, X_R, X_Z]
    surrogate = NeuralSurrogate(n_coils=7, n_state=4)
    surrogate.train_on_kernel(kernel)
    
    # 3. Define Mission (Targets)
    # We want to move plasma to R=6.0, Z=0.0 and maintain X-point geometry
    TARGET_VECTOR = np.array([6.0, 0.0, 5.0, -3.5]) # Target Axis & X-point
    
    mpc = ModelPredictiveController(surrogate, TARGET_VECTOR)
    
    # History
    h_r, h_z, h_xr, h_xz = [], [], [], []
    
    print(f"Starting {SHOT_LENGTH} step simulation with MPC Horizon={PREDICTION_HORIZON}...")
    
    start_time = time.time()
    
    for t in range(SHOT_LENGTH):
        # A. Measure State (from Real Kernel)
        curr_state = surrogate.get_state(kernel)
        
        # B. MPC Planning (Brain)
        # The AI "thinks" 10 steps ahead using the Surrogate
        best_action = mpc.plan_trajectory(curr_state)
        
        # C. Actuate (Apply to Real Kernel)
        for i, delta in enumerate(best_action):
            kernel.cfg['coils'][i]['current'] += delta
            
        # D. Physics Step (Reality)
        # Apply external disturbance (Shafranov Shift scenario)
        # This tests if MPC can adapt to unmodeled drift
        if t > 20: 
            kernel.cfg['physics']['plasma_current_target'] += 0.1 # Drift
            
        kernel.solve_equilibrium()
        
        # Log
        h_r.append(curr_state[0])
        h_z.append(curr_state[1])
        h_xr.append(curr_state[2])
        h_xz.append(curr_state[3])
        
        # Calc Real Error
        err = np.linalg.norm(curr_state - TARGET_VECTOR)
        if t % 10 == 0:
            print(f"Step {t}: R={curr_state[0]:.2f}, Z={curr_state[1]:.2f} | X-Point=({curr_state[2]:.2f},{curr_state[3]:.2f}) | Err={err:.3f}")

    print(f"Simulation finished in {time.time() - start_time:.2f}s")
    
    # --- VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Axis Control
    ax1.set_title("MPC Axis Tracking")
    ax1.plot(h_r, label='R Axis')
    ax1.plot(h_z, label='Z Axis')
    ax1.axhline(TARGET_VECTOR[0], color='blue', linestyle='--', alpha=0.5, label='Target R')
    ax1.axhline(TARGET_VECTOR[1], color='orange', linestyle='--', alpha=0.5, label='Target Z')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Divertor Control (X-Point)
    ax2.set_title("MPC Divertor (X-Point) Stabilization")
    ax2.plot(h_xr, label='X-Point R')
    ax2.plot(h_xz, label='X-Point Z')
    ax2.axhline(TARGET_VECTOR[2], color='blue', linestyle='--', alpha=0.5, label='Target XR')
    ax2.axhline(TARGET_VECTOR[3], color='orange', linestyle='--', alpha=0.5, label='Target XZ')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("SOTA_MPC_Results.png")
    print("SOTA Analysis saved: SOTA_MPC_Results.png")

if __name__ == "__main__":
    run_sota_simulation()
