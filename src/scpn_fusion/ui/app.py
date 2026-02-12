# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — App
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import json
try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel
from scpn_fusion.core.fusion_ignition_sim import FusionBurnPhysics
from scpn_fusion.nuclear.nuclear_wall_interaction import NuclearEngineeringLab
from scpn_fusion.engineering.balance_of_plant import PowerPlantModel

# --- CONFIG ---
st.set_page_config(page_title="SCPN Fusion Reactor", layout="wide", page_icon="⚛️")

# Helper to find config
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "iter_config.json")
if not os.path.exists(CONFIG_PATH):
    # Fallback for dev mode
    CONFIG_PATH = "iter_config.json"

st.title("⚛️ SCPN Fusion Reactor Control Room")
st.markdown("### Digital Twin & Engineering Suite v1.0")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Reactor Parameters")
reactor_size = st.sidebar.slider("Major Radius (m)", 3.0, 9.0, 6.2)
plasma_current = st.sidebar.slider("Target Current (MA)", 1.0, 20.0, 15.0)
aux_heating = st.sidebar.slider("Auxiliary Heating (MW)", 0.0, 100.0, 50.0)

# Modify Config in memory (Simulated)
# In real app we would write back to JSON or create object directly
# Here we hack the config file path because our classes read from file
# Ideally classes should accept dicts. Refactor opportunity for v1.1.

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["Plasma Physics", "Ignition & Q", "Nuclear Engineering", "Power Plant"])

with tab1:
    st.header("Grad-Shafranov Equilibrium")
    
    if st.button("Solve Equilibrium"):
        with st.spinner("Solving Non-Linear MHD Equations..."):
            kernel = FusionKernel(CONFIG_PATH)
            # Inject parameters
            kernel.cfg['physics']['plasma_current_target'] = float(plasma_current)
            
            kernel.solve_equilibrium()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots()
                ax.contour(kernel.RR, kernel.ZZ, kernel.Psi, levels=20, colors='black')
                im = ax.imshow(kernel.J_phi, extent=[kernel.R[0], kernel.R[-1], kernel.Z[0], kernel.Z[-1]], origin='lower', cmap='hot', alpha=0.6)
                plt.colorbar(im, label='Current Density')
                ax.set_title("Magnetic Flux & Current")
                st.pyplot(fig)
                
            with col2:
                st.metric("Magnetic Axis Flux", f"{np.max(kernel.Psi):.2f} Wb")
                xp, psi_x = kernel.find_x_point(kernel.Psi)
                st.metric("X-Point Location", f"R={xp[0]:.2f}, Z={xp[1]:.2f} m")

with tab2:
    st.header("Thermonuclear Performance")
    
    if st.button("Run Burn Simulation"):
        physics = FusionBurnPhysics(CONFIG_PATH)
        physics.solve_equilibrium()
        metrics = physics.calculate_thermodynamics(aux_heating)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fusion Power", f"{metrics['P_fusion_MW']:.1f} MW")
        c2.metric("Q-Factor", f"{metrics['Q']:.2f}")
        c3.metric("Alpha Heating", f"{metrics['P_alpha_MW']:.1f} MW")
        c4.metric("Status", "IGNITION" if metrics['Q'] > 10 else "Driven")
        
        # Popcon Plot
        fig, ax = plt.subplots()
        ax.bar(['Aux Heat', 'Alpha Heat'], [metrics['P_aux_MW'], metrics['P_alpha_MW']], color=['orange', 'red'])
        ax.bar(['Losses'], [metrics['P_loss_MW']], color='blue')
        ax.set_ylabel("Power (MW)")
        ax.set_title("Power Balance")
        st.pyplot(fig)

with tab3:
    st.header("Nuclear Wall Loading")
    
    if st.button("Calculate Neutron Flux"):
        with st.spinner("Ray-Tracing Neutron Paths..."):
            lab = NuclearEngineeringLab(CONFIG_PATH)
            lab.solve_equilibrium()
            Rw, Zw, flux = lab.calculate_neutron_wall_loading()
            
            # Analyze
            lifespans, load_mw = lab.analyze_materials(flux)
            
            st.warning(f"Peak Neutron Load: {np.max(load_mw):.2f} MW/m2")
            
            fig, ax = plt.subplots()
            sc = ax.scatter(Rw, Zw, c=load_mw, cmap='inferno')
            plt.colorbar(sc, label='MW/m2')
            ax.set_aspect('equal')
            st.pyplot(fig)
            
            st.subheader("Component Lifespan")
            st.json(lifespans)

with tab4:
    st.header("Balance of Plant (Electricity Generation)")
    
    if st.button("Calculate Grid Output"):
        # Chain simulations: Physics -> Plant
        physics = FusionBurnPhysics(CONFIG_PATH)
        physics.solve_equilibrium()
        plasma_metrics = physics.calculate_thermodynamics(aux_heating)
        
        plant = PowerPlantModel()
        plant_metrics = plant.calculate_plant_performance(plasma_metrics['P_fusion_MW'], aux_heating)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Gross Electric", f"{plant_metrics['P_gross']:.1f} MWe")
        c2.metric("House Load", f"{plant_metrics['P_recirc']:.1f} MWe")
        c3.metric("NET TO GRID", f"{plant_metrics['P_net']:.1f} MWe", delta_color="normal")
        
        if plant_metrics['P_net'] > 0:
            st.success("✅ SYSTEM IS PRODUCING POWER!")
        else:
            st.error("❌ SYSTEM IS CONSUMING POWER!")
            
        st.pyplot(plant.plot_sankey_diagram(plant_metrics))
        st.subheader("Load Breakdown")
        st.json(plant_metrics['breakdown'])

st.sidebar.markdown("---")
st.sidebar.info("Developed by SCPN AI Agent (2026)")
