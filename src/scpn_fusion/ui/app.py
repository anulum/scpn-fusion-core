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
from pathlib import Path
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
st.markdown("### Digital Twin & Engineering Suite v2.1.0")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Reactor Parameters")
reactor_size = st.sidebar.slider("Major Radius (m)", 3.0, 9.0, 6.2)
plasma_current = st.sidebar.slider("Target Current (MA)", 1.0, 20.0, 15.0)
aux_heating = st.sidebar.slider("Auxiliary Heating (MW)", 0.0, 100.0, 50.0)

# Modify Config in memory (simulated — classes read from file)

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Plasma Physics", "Ignition & Q", "Nuclear Engineering",
    "Power Plant", "Shot Replay"
])

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

with tab5:
    st.header("DIII-D Shot Replay & Disruption Analysis")

    # Locate disruption shots directory relative to app.py
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _DISRUPTION_DIR = _PROJECT_ROOT / "validation" / "reference_data" / "diiid" / "disruption_shots"

    npz_files = sorted(_DISRUPTION_DIR.glob("*.npz")) if _DISRUPTION_DIR.is_dir() else []

    if not npz_files:
        st.info(
            "No disruption shot NPZ files found. Expected location: "
            f"`{_DISRUPTION_DIR}`"
        )
    else:
        shot_names = [p.stem for p in npz_files]
        selected_shot = st.selectbox("Select Shot", shot_names, index=0)

        if st.button("Load & Replay"):
            # --- Import IO loader and disruption predictor inside this block ---
            from scpn_fusion.io.tokamak_archive import load_disruption_shot
            from scpn_fusion.control.disruption_predictor import predict_disruption_risk

            with st.spinner("Loading shot data..."):
                try:
                    shot_data = load_disruption_shot(selected_shot, disruption_dir=_DISRUPTION_DIR)
                except Exception as e:
                    st.error(f"Failed to load shot: {e}")
                    st.stop()

                time_s = shot_data["time_s"]
                is_disruption = shot_data["is_disruption"]
                disruption_idx = shot_data["disruption_time_idx"]
                disruption_type = shot_data["disruption_type"]

                # --- Metadata ---
                st.subheader("Shot Metadata")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Shot Name", selected_shot)
                mc2.metric("Disruption Type", disruption_type if is_disruption else "Safe (no disruption)")
                mc3.metric("Is Disruption", "Yes" if is_disruption else "No")

                # Disruption time marker (in seconds)
                disruption_time_s = float(time_s[disruption_idx]) if is_disruption and 0 <= disruption_idx < len(time_s) else None

                # --- 2x2 time-series subplot grid ---
                st.subheader("Time Series Overview")
                fig1, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

                signals_2x2 = [
                    ("dBdt_gauss_per_s", "dB/dt (Gauss/s)", "tab:blue"),
                    ("beta_N", r"$\beta_N$", "tab:orange"),
                    ("Ip_MA", "Plasma Current $I_p$ (MA)", "tab:green"),
                    ("q95", "$q_{95}$", "tab:red"),
                ]

                for ax, (key, label, color) in zip(axes.flat, signals_2x2):
                    ax.plot(time_s, shot_data[key], color=color, linewidth=0.8)
                    ax.set_ylabel(label)
                    ax.grid(True, alpha=0.3)
                    if disruption_time_s is not None:
                        ax.axvline(disruption_time_s, color="red", linestyle="--", linewidth=1.2, alpha=0.8, label="Disruption")
                        ax.legend(loc="upper right", fontsize=8)

                axes[1, 0].set_xlabel("Time (s)")
                axes[1, 1].set_xlabel("Time (s)")
                fig1.suptitle(f"Shot: {selected_shot}", fontsize=13)
                fig1.tight_layout()
                st.pyplot(fig1)
                plt.close(fig1)

                # --- Toroidal mode amplitudes ---
                st.subheader("Toroidal Mode Amplitudes")
                fig2, ax2 = plt.subplots(figsize=(12, 4))
                ax2.plot(time_s, shot_data["n1_amp"], label="n=1 amplitude", linewidth=0.9)
                ax2.plot(time_s, shot_data["n2_amp"], label="n=2 amplitude", linewidth=0.9)
                ax2.plot(time_s, shot_data["locked_mode_amp"], label="Locked mode amplitude", linewidth=0.9)
                if disruption_time_s is not None:
                    ax2.axvline(disruption_time_s, color="red", linestyle="--", linewidth=1.2, alpha=0.8, label="Disruption")
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Amplitude (a.u.)")
                ax2.set_title("Toroidal Mode Structure")
                ax2.legend(fontsize=9)
                ax2.grid(True, alpha=0.3)
                fig2.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)

                # --- Disruption risk score via sliding window ---
                st.subheader("Disruption Risk Score (Sliding Window)")
                window_size = 50  # samples per window
                n_samples = len(time_s)
                risk_scores = np.full(n_samples, np.nan)

                for i in range(window_size, n_samples):
                    window_signal = shot_data["dBdt_gauss_per_s"][i - window_size : i]
                    toroidal_obs = {
                        "toroidal_n1_amp": float(shot_data["n1_amp"][i]),
                        "toroidal_n2_amp": float(shot_data["n2_amp"][i]),
                        "toroidal_n3_amp": float(shot_data["locked_mode_amp"][i]),
                    }
                    risk_scores[i] = predict_disruption_risk(window_signal, toroidal_obs)

                fig3, ax3 = plt.subplots(figsize=(12, 4))
                valid = ~np.isnan(risk_scores)
                ax3.plot(time_s[valid], risk_scores[valid], color="darkred", linewidth=1.0)
                ax3.axhline(0.5, color="orange", linestyle=":", linewidth=1.0, label="Threshold (0.5)")
                ax3.fill_between(
                    time_s[valid], 0, risk_scores[valid],
                    where=risk_scores[valid] >= 0.5,
                    color="red", alpha=0.2, label="High Risk Region"
                )
                if disruption_time_s is not None:
                    ax3.axvline(disruption_time_s, color="red", linestyle="--", linewidth=1.2, alpha=0.8, label="Disruption")
                ax3.set_xlabel("Time (s)")
                ax3.set_ylabel("Risk Score")
                ax3.set_ylim(-0.05, 1.05)
                ax3.set_title("Disruption Predictor Risk Score vs Time")
                ax3.legend(fontsize=9)
                ax3.grid(True, alpha=0.3)
                fig3.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)

                # --- Summary statistics table ---
                st.subheader("Signal Summary Statistics")
                stat_keys = [
                    ("dBdt_gauss_per_s", "dB/dt (Gauss/s)"),
                    ("beta_N", "beta_N"),
                    ("Ip_MA", "Ip (MA)"),
                    ("q95", "q95"),
                    ("n1_amp", "n=1 Amplitude"),
                    ("n2_amp", "n=2 Amplitude"),
                    ("locked_mode_amp", "Locked Mode Amp"),
                    ("ne_1e19", "ne (1e19 m^-3)"),
                    ("vertical_position_m", "Vertical Position (m)"),
                ]
                rows = []
                for key, label in stat_keys:
                    arr = shot_data[key]
                    rows.append({
                        "Signal": label,
                        "Mean": f"{np.mean(arr):.4f}",
                        "Max": f"{np.max(arr):.4f}",
                        "Min": f"{np.min(arr):.4f}",
                        "Std": f"{np.std(arr):.4f}",
                    })
                st.table(rows)

st.sidebar.markdown("---")
st.sidebar.info("Developed by SCPN AI Agent (2026)")
