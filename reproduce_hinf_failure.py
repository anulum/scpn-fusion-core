import numpy as np
from scpn_fusion.control.h_infinity_controller import get_radial_robust_controller

def reproduce_failure():
    print("Reproducing H-infinity Spectral Feasibility Failure...")
    # Using default gamma_growth=100.0
    try:
        ctrl = get_radial_robust_controller(gamma_growth=100.0, enforce_robust_feasibility=True)
        print(f"Success! rho(XY) = {ctrl.spectral_radius_xy}")
    except ValueError as e:
        print(f"Caught Expected Failure: {e}")

    print("\nTesting with lower growth rate (50.0)...")
    try:
        ctrl = get_radial_robust_controller(gamma_growth=50.0, enforce_robust_feasibility=True)
        print(f"Success! rho(XY) = {ctrl.spectral_radius_xy} gamma^2 = {ctrl.gamma**2}")
    except ValueError as e:
        print(f"Failure with 50.0: {e}")

if __name__ == "__main__":
    reproduce_failure()
