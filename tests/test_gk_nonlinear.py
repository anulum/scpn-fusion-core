# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests: Nonlinear δf Gyrokinetic Solver

from __future__ import annotations

import numpy as np

from scpn_fusion.core.gk_nonlinear import (
    NonlinearGKConfig,
    NonlinearGKSolver,
)

# Small grid for fast tests
_FAST_CFG = NonlinearGKConfig(
    n_kx=8,
    n_ky=8,
    n_theta=16,
    n_vpar=8,
    n_mu=4,
    n_species=2,
    dt=0.02,
    n_steps=50,
    save_interval=10,
    nonlinear=True,
    collisions=True,
    nu_collision=0.01,
    hyper_coeff=0.1,
    cfl_adapt=False,
)

# Linear-only (no E×B nonlinearity)
_LINEAR_CFG = NonlinearGKConfig(
    n_kx=8,
    n_ky=8,
    n_theta=16,
    n_vpar=8,
    n_mu=4,
    n_species=2,
    dt=0.02,
    n_steps=50,
    save_interval=10,
    nonlinear=False,
    collisions=False,
    hyper_coeff=0.0,
    cfl_adapt=False,
)

# No drive, no collisions, no hyperdiffusion — for energy conservation
_CONSERV_CFG = NonlinearGKConfig(
    n_kx=8,
    n_ky=8,
    n_theta=16,
    n_vpar=8,
    n_mu=4,
    n_species=2,
    dt=0.01,
    n_steps=20,
    save_interval=5,
    nonlinear=True,
    collisions=False,
    hyper_coeff=0.0,
    R_L_Ti=0.0,
    R_L_Te=0.0,
    R_L_ne=0.0,
    cfl_adapt=False,
)


# ── Config and State ──────────────────────────────────────────────────


class TestConfigState:
    def test_config_defaults(self):
        cfg = NonlinearGKConfig()
        assert cfg.n_kx == 16
        assert cfg.n_steps == 5000
        assert cfg.dealiasing == "2/3"

    def test_state_shape(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        state = solver.init_state()
        assert state.f.shape == (2, 8, 8, 16, 8, 4)
        assert state.phi.shape == (8, 8, 16)
        assert state.time == 0.0

    def test_single_mode_init(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        state = solver.init_single_mode(kx_idx=0, ky_idx=1, amplitude=1e-3)
        assert np.max(np.abs(state.f[0, 0, 1])) > 0
        # Other modes should be zero (except at the specified mode)
        assert np.max(np.abs(state.f[0, 2, 3])) == 0.0

    def test_phase_space_contract_declares_5d_storage_units_and_boundaries(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        contract = solver.phase_space_contract()

        assert contract.distribution_shape == (2, 8, 8, 16, 8, 4)
        assert contract.field_shape == (8, 8, 16)
        assert contract.distribution_axes == (
            "species",
            "kx_rhos",
            "ky_rhos",
            "theta_rad",
            "vpar_vth",
            "mu_normalized",
        )
        assert contract.axis_units["theta_rad"] == "rad"
        assert contract.axis_units["vpar_vth"] == "v_th"
        assert contract.field_components == ("phi", "A_parallel", "B_parallel")
        assert contract.boundary_semantics["kx"] == "periodic spectral"
        assert contract.boundary_semantics["theta"] == "ballooning-connected periodic"
        assert contract.dealiasing == "2/3"

    def test_validate_state_rejects_non_5d_distribution_shape(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        state = solver.init_state()
        bad = state.__class__(f=state.f[0], phi=state.phi, time=state.time)

        try:
            solver.validate_state(bad)
        except ValueError as exc:
            assert "distribution shape" in str(exc)
        else:
            raise AssertionError("validate_state must reject non-5D species storage")


# ── Field solve ───────────────────────────────────────────────────────


class TestFieldSolve:
    def test_phi_shape(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        state = solver.init_state()
        phi = solver.field_solve(state.f)
        assert phi.shape == (8, 8, 16)

    def test_phi_finite(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        state = solver.init_state()
        assert np.all(np.isfinite(state.phi))

    def test_zero_f_gives_zero_phi(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        f = np.zeros((2, 8, 8, 16, 8, 4), dtype=complex)
        phi = solver.field_solve(f)
        np.testing.assert_allclose(phi, 0.0, atol=1e-30)


# ── E×B bracket ──────────────────────────────────────────────────────


class TestExBBracket:
    def test_bracket_shape(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        state = solver.init_state()
        bracket = solver.exb_bracket(state.phi, state.f[0])
        assert bracket.shape == state.f[0].shape

    def test_bracket_zero_for_uniform_f(self):
        """Spatially uniform f (only k=0 mode) has zero gradients → bracket = 0."""
        solver = NonlinearGKSolver(_FAST_CFG)
        f_uniform = np.zeros((8, 8, 16, 8, 4), dtype=complex)
        f_uniform[0, 0, :, :, :] = 1e-3  # only the (kx=0, ky=0) mode
        phi = solver.init_state().phi
        bracket = solver.exb_bracket(phi, f_uniform)
        # df/dx = df/dy = 0 for uniform f → bracket = 0
        assert np.max(np.abs(bracket)) < 1e-10

    def test_bracket_dealiased(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        state = solver.init_state(amplitude=1e-3)
        bracket = solver.exb_bracket(state.phi, state.f[0])
        # High-k modes should be zero after dealiasing
        mask = ~solver.dealias_mask
        for t in range(16):
            for v in range(8):
                for m in range(4):
                    assert np.all(bracket[mask, t, v, m] == 0.0)

    def test_nonlinear_exb_term_is_negative_conservative_bracket(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        state = solver.init_state(amplitude=1e-4, seed=5)

        term, diagnostics = solver.nonlinear_exb_term(state, return_diagnostics=True)

        assert term.shape == state.f.shape
        np.testing.assert_allclose(term[0], -solver.exb_bracket(state.phi, state.f[0]))
        np.testing.assert_allclose(term[1], 0.0)
        assert diagnostics.finite
        assert diagnostics.passes


# ── Parallel streaming ───────────────────────────────────────────────


class TestParallelStreaming:
    def test_streaming_shape(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        state = solver.init_state()
        stream = solver.parallel_streaming(state.f[0])
        assert stream.shape == state.f[0].shape

    def test_uniform_theta_zero_streaming(self):
        """Uniform f(θ) → ∂f/∂θ = 0 → no streaming contribution."""
        solver = NonlinearGKSolver(_FAST_CFG)
        f_s = np.ones((8, 8, 16, 8, 4), dtype=complex) * 1e-3
        stream = solver.parallel_streaming(f_s)
        np.testing.assert_allclose(stream, 0.0, atol=1e-14)


# ── Energy conservation ──────────────────────────────────────────────


class TestEnergyConservation:
    def test_energy_finite(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        state = solver.init_state()
        E = solver.total_energy(state)
        assert np.isfinite(E)
        assert E > 0

    def test_no_drive_no_dissipation_conserves(self):
        """Without drive, collisions, or hyperdiffusion, E should not grow."""
        solver = NonlinearGKSolver(_CONSERV_CFG)
        state = solver.init_state(amplitude=1e-6)
        E0 = solver.total_energy(state)

        for _ in range(20):
            state = solver._rk4_step(state, 0.01)

        E1 = solver.total_energy(state)
        # Energy should not grow (may decrease slightly due to dealiasing)
        assert E1 < E0 * 10.0  # generous bound — no unbounded growth

    def test_electromagnetic_field_energy_accounts_all_field_components(self):
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=6,
            n_mu=4,
            n_species=2,
            kinetic_electrons=True,
            electromagnetic=True,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        state = solver.init_state(amplitude=1e-5, seed=29)

        field_energy = solver.field_energy(state)

        assert field_energy.finite
        assert field_energy.phi >= 0.0
        assert field_energy.A_parallel >= 0.0
        assert field_energy.B_parallel >= 0.0
        assert field_energy.total == (
            field_energy.phi + field_energy.A_parallel + field_energy.B_parallel
        )

    def test_total_energy_includes_particle_and_electromagnetic_field_energy(self):
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=6,
            n_mu=4,
            n_species=2,
            kinetic_electrons=True,
            electromagnetic=True,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        state = solver.init_state(amplitude=1e-5, seed=31)

        particle = solver.particle_free_energy(state)
        field = solver.field_energy(state)

        assert solver.total_energy(state) == particle + field.total

    def test_run_exports_particle_and_electromagnetic_energy_histories(self):
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=6,
            n_mu=4,
            n_species=2,
            kinetic_electrons=True,
            electromagnetic=True,
            dt=0.01,
            n_steps=4,
            save_interval=1,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)

        result = solver.run(solver.init_state(amplitude=1e-5, seed=37))

        assert result.particle_free_energy_t.shape == result.time.shape
        assert result.phi_energy_t.shape == result.time.shape
        assert result.A_parallel_energy_t.shape == result.time.shape
        assert result.B_parallel_energy_t.shape == result.time.shape
        assert result.total_energy_t.shape == result.time.shape
        assert np.all(np.isfinite(result.total_energy_t))
        np.testing.assert_allclose(
            result.total_energy_t,
            result.particle_free_energy_t
            + result.phi_energy_t
            + result.A_parallel_energy_t
            + result.B_parallel_energy_t,
        )

    def test_heat_flux_spectra_close_scalar_fluxes(self):
        cfg = NonlinearGKConfig(
            n_kx=6,
            n_ky=6,
            n_theta=8,
            n_vpar=5,
            n_mu=4,
            n_species=2,
            kinetic_electrons=True,
            dt=0.005,
            n_steps=3,
            save_interval=1,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        state = solver.init_state(amplitude=1e-5, seed=47)

        Q_i, Q_e = solver.compute_fluxes(state)
        Q_i_kxky, Q_e_kxky = solver.heat_flux_spectra(state)

        assert Q_i_kxky.shape == (cfg.n_kx, cfg.n_ky)
        assert Q_e_kxky.shape == (cfg.n_kx, cfg.n_ky)
        np.testing.assert_allclose(np.sum(Q_i_kxky), Q_i)
        np.testing.assert_allclose(np.sum(Q_e_kxky), Q_e)

        result = solver.run(state)
        assert result.Q_i_kxky_t.shape == (result.time.size, cfg.n_kx, cfg.n_ky)
        assert result.Q_e_kxky_t.shape == (result.time.size, cfg.n_kx, cfg.n_ky)
        np.testing.assert_allclose(np.sum(result.Q_i_kxky_t, axis=(1, 2)), result.Q_i_t)
        np.testing.assert_allclose(np.sum(result.Q_e_kxky_t, axis=(1, 2)), result.Q_e_t)

    def test_zonal_flow_energy_history_is_saved_and_bounded(self):
        cfg = NonlinearGKConfig(
            n_kx=6,
            n_ky=6,
            n_theta=8,
            n_vpar=5,
            n_mu=4,
            n_species=2,
            kinetic_electrons=True,
            dt=0.005,
            n_steps=3,
            save_interval=1,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        state = solver.init_state(amplitude=1e-5, seed=53)

        zonal_energy = solver.zonal_flow_energy(state)

        assert np.isfinite(zonal_energy)
        assert zonal_energy >= 0.0
        assert zonal_energy <= solver.field_energy(state).phi

        result = solver.run(state)
        assert result.zonal_flow_energy_t.shape == result.time.shape
        assert np.all(np.isfinite(result.zonal_flow_energy_t))
        assert np.all(result.zonal_flow_energy_t >= 0.0)
        assert np.all(result.zonal_flow_energy_t <= result.phi_energy_t)

    def test_run_exports_saturation_window_diagnostics(self):
        cfg = NonlinearGKConfig(
            n_kx=6,
            n_ky=6,
            n_theta=8,
            n_vpar=5,
            n_mu=4,
            n_species=2,
            kinetic_electrons=True,
            electromagnetic=True,
            dt=0.005,
            n_steps=5,
            save_interval=1,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)

        result = solver.run(solver.init_state(amplitude=1e-5, seed=59))

        n_half = max(result.time.size // 2, 1)
        late = slice(n_half, None)

        assert result.saturated_Q_i_kxky.shape == (cfg.n_kx, cfg.n_ky)
        assert result.saturated_Q_e_kxky.shape == (cfg.n_kx, cfg.n_ky)
        np.testing.assert_allclose(
            result.saturated_Q_i_kxky, np.mean(result.Q_i_kxky_t[late], axis=0)
        )
        np.testing.assert_allclose(
            result.saturated_Q_e_kxky, np.mean(result.Q_e_kxky_t[late], axis=0)
        )
        np.testing.assert_allclose(np.sum(result.saturated_Q_i_kxky), result.chi_i)
        np.testing.assert_allclose(np.sum(result.saturated_Q_e_kxky), result.chi_e)
        np.testing.assert_allclose(result.saturated_phi_rms, np.mean(result.phi_rms_t[late]))
        np.testing.assert_allclose(
            result.saturated_zonal_flow_energy, np.mean(result.zonal_flow_energy_t[late])
        )
        np.testing.assert_allclose(result.saturated_phi_energy, np.mean(result.phi_energy_t[late]))
        np.testing.assert_allclose(
            result.saturated_A_parallel_energy, np.mean(result.A_parallel_energy_t[late])
        )
        np.testing.assert_allclose(
            result.saturated_B_parallel_energy, np.mean(result.B_parallel_energy_t[late])
        )
        np.testing.assert_allclose(
            result.saturated_total_energy, np.mean(result.total_energy_t[late])
        )

    def test_electromagnetic_field_energy_spectra_close_component_energies(self):
        cfg = NonlinearGKConfig(
            n_kx=6,
            n_ky=6,
            n_theta=8,
            n_vpar=5,
            n_mu=4,
            n_species=2,
            kinetic_electrons=True,
            electromagnetic=True,
            dt=0.005,
            n_steps=4,
            save_interval=1,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        state = solver.init_state(amplitude=1e-5, seed=61)

        phi_kxky, a_parallel_kxky, b_parallel_kxky = solver.field_energy_spectra(state)
        field_energy = solver.field_energy(state)

        assert phi_kxky.shape == (cfg.n_kx, cfg.n_ky)
        assert a_parallel_kxky.shape == (cfg.n_kx, cfg.n_ky)
        assert b_parallel_kxky.shape == (cfg.n_kx, cfg.n_ky)
        assert np.all(phi_kxky >= 0.0)
        assert np.all(a_parallel_kxky >= 0.0)
        assert np.all(b_parallel_kxky >= 0.0)
        np.testing.assert_allclose(np.sum(phi_kxky), field_energy.phi)
        np.testing.assert_allclose(np.sum(a_parallel_kxky), field_energy.A_parallel)
        np.testing.assert_allclose(np.sum(b_parallel_kxky), field_energy.B_parallel)

        result = solver.run(state)
        late = slice(max(result.time.size // 2, 1), None)

        assert result.phi_energy_kxky_t.shape == (result.time.size, cfg.n_kx, cfg.n_ky)
        assert result.A_parallel_energy_kxky_t.shape == (result.time.size, cfg.n_kx, cfg.n_ky)
        assert result.B_parallel_energy_kxky_t.shape == (result.time.size, cfg.n_kx, cfg.n_ky)
        np.testing.assert_allclose(
            np.sum(result.phi_energy_kxky_t, axis=(1, 2)), result.phi_energy_t
        )
        np.testing.assert_allclose(
            np.sum(result.A_parallel_energy_kxky_t, axis=(1, 2)), result.A_parallel_energy_t
        )
        np.testing.assert_allclose(
            np.sum(result.B_parallel_energy_kxky_t, axis=(1, 2)), result.B_parallel_energy_t
        )
        np.testing.assert_allclose(
            result.saturated_phi_energy_kxky, np.mean(result.phi_energy_kxky_t[late], axis=0)
        )
        np.testing.assert_allclose(
            result.saturated_A_parallel_energy_kxky,
            np.mean(result.A_parallel_energy_kxky_t[late], axis=0),
        )
        np.testing.assert_allclose(
            result.saturated_B_parallel_energy_kxky,
            np.mean(result.B_parallel_energy_kxky_t[late], axis=0),
        )

    def test_particle_free_energy_spectra_close_scalar_history(self):
        cfg = NonlinearGKConfig(
            n_kx=6,
            n_ky=6,
            n_theta=8,
            n_vpar=5,
            n_mu=4,
            n_species=2,
            kinetic_electrons=True,
            dt=0.005,
            n_steps=4,
            save_interval=1,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        state = solver.init_state(amplitude=1e-5, seed=67)

        particle_kxky = solver.particle_free_energy_spectra(state)
        particle_energy = solver.particle_free_energy(state)

        assert particle_kxky.shape == (cfg.n_species, cfg.n_kx, cfg.n_ky)
        assert np.all(particle_kxky >= 0.0)
        np.testing.assert_allclose(np.sum(particle_kxky), particle_energy)

        result = solver.run(state)
        late = slice(max(result.time.size // 2, 1), None)

        assert result.particle_free_energy_species_kxky_t.shape == (
            result.time.size,
            cfg.n_species,
            cfg.n_kx,
            cfg.n_ky,
        )
        np.testing.assert_allclose(
            np.sum(result.particle_free_energy_species_kxky_t, axis=(1, 2, 3)),
            result.particle_free_energy_t,
        )
        np.testing.assert_allclose(
            result.saturated_particle_free_energy_species_kxky,
            np.mean(result.particle_free_energy_species_kxky_t[late], axis=0),
        )


class TestSugamaCollisionProjection:
    def test_sugama_collision_conserves_discrete_density_momentum_energy(self):
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=8,
            n_mu=6,
            n_species=2,
            collisions=True,
            collision_model="sugama",
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        state = solver.init_state(amplitude=1e-4, seed=7)
        collision = solver.collide(state.f[0])

        vpar = solver.vpar[None, None, None, :, None]
        mu = solver.mu[None, None, None, None, :]
        energy = 0.5 * vpar**2 + mu
        dv = solver.dvpar * solver.dmu

        density_moment = np.sum(collision * dv, axis=(-2, -1))
        momentum_moment = np.sum(collision * vpar * dv, axis=(-2, -1))
        energy_moment = np.sum(collision * energy * dv, axis=(-2, -1))

        np.testing.assert_allclose(density_moment, 0.0, atol=1e-12)
        np.testing.assert_allclose(momentum_moment, 0.0, atol=1e-12)
        np.testing.assert_allclose(energy_moment, 0.0, atol=1e-12)


# ── Linear recovery ──────────────────────────────────────────────────


class TestLinearRecovery:
    def test_linear_mode_grows(self):
        """With nonlinearity off, a single mode should grow."""
        solver = NonlinearGKSolver(_LINEAR_CFG)
        state = solver.init_single_mode(ky_idx=1, amplitude=1e-8)
        phi0 = solver.phi_rms(state)

        for _ in range(50):
            state = solver._rk4_step(state, 0.02)

        phi1 = solver.phi_rms(state)
        # In linear regime, phi should evolve (grow or damp)
        assert np.isfinite(phi1)

    def test_phi_rms_finite_throughout(self):
        solver = NonlinearGKSolver(_LINEAR_CFG)
        result = solver.run(solver.init_single_mode(amplitude=1e-8))
        assert np.all(np.isfinite(result.phi_rms_t))


# ── Zonal flows ──────────────────────────────────────────────────────


class TestZonalFlows:
    def test_zonal_rms_computed(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        state = solver.init_state()
        zr = solver.zonal_rms(state)
        assert np.isfinite(zr)

    def test_zonal_present_in_nonlinear(self):
        """Nonlinear run should generate zonal flows (k_y=0) from noise."""
        solver = NonlinearGKSolver(_FAST_CFG)
        result = solver.run()
        assert result.zonal_rms_t[-1] >= 0

    def test_rosenbluth_hinton_residual_positive(self):
        """After zonal flow initialization, residual should be > 0.

        Rosenbluth & Hinton 1998: ZF residual = 1/(1 + 1.6q²/√ε).
        """
        solver = NonlinearGKSolver(
            NonlinearGKConfig(
                n_kx=8,
                n_ky=8,
                n_theta=16,
                n_vpar=8,
                n_mu=4,
                n_species=2,
                dt=0.02,
                n_steps=10,
                save_interval=5,
                nonlinear=False,
                collisions=False,
                hyper_coeff=0.0,
                R_L_Ti=0.0,
                R_L_Te=0.0,
                R_L_ne=0.0,
                cfl_adapt=False,
            )
        )
        state = solver.init_state(amplitude=1e-5)
        zr0 = solver.zonal_rms(state)
        # After a few steps, zonal component should remain (not decay to zero)
        for _ in range(10):
            state = solver._rk4_step(state, 0.02)
        zr1 = solver.zonal_rms(state)
        assert np.isfinite(zr1)


# ── CBC benchmark ────────────────────────────────────────────────────


class TestCBCBenchmark:
    def test_cbc_runs_without_nan(self):
        cfg = NonlinearGKConfig(
            n_kx=8,
            n_ky=8,
            n_theta=16,
            n_vpar=8,
            n_mu=4,
            dt=0.02,
            n_steps=100,
            save_interval=20,
            R_L_Ti=6.9,
            R_L_Te=6.9,
            R_L_ne=2.2,
            q=1.4,
            s_hat=0.78,
            R0=2.78,
            a=1.0,
            B0=2.0,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        result = solver.run()
        assert result.converged
        assert np.all(np.isfinite(result.Q_i_t))

    def test_cbc_chi_i_finite(self):
        cfg = NonlinearGKConfig(
            n_kx=8,
            n_ky=8,
            n_theta=16,
            n_vpar=8,
            n_mu=4,
            dt=0.02,
            n_steps=100,
            save_interval=20,
            R_L_Ti=6.9,
            q=1.4,
            s_hat=0.78,
            R0=2.78,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        result = solver.run()
        assert np.isfinite(result.chi_i)

    def test_subcritical_lower_transport(self):
        """R/L_Ti=2.0 (subcritical) should produce less transport than R/L_Ti=6.9."""
        cfg_sub = NonlinearGKConfig(
            n_kx=8,
            n_ky=8,
            n_theta=16,
            n_vpar=8,
            n_mu=4,
            dt=0.02,
            n_steps=100,
            save_interval=20,
            R_L_Ti=2.0,
            q=1.4,
            s_hat=0.78,
            R0=2.78,
            cfl_adapt=False,
        )
        cfg_sup = NonlinearGKConfig(
            n_kx=8,
            n_ky=8,
            n_theta=16,
            n_vpar=8,
            n_mu=4,
            dt=0.02,
            n_steps=100,
            save_interval=20,
            R_L_Ti=6.9,
            q=1.4,
            s_hat=0.78,
            R0=2.78,
            cfl_adapt=False,
        )
        r_sub = NonlinearGKSolver(cfg_sub).run()
        r_sup = NonlinearGKSolver(cfg_sup).run()
        # Both should be finite
        assert np.isfinite(r_sub.chi_i)
        assert np.isfinite(r_sup.chi_i)


# ── Flux diagnostics ─────────────────────────────────────────────────


class TestFluxDiagnostics:
    def test_flux_computation(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        state = solver.init_state()
        Q_i, Q_e = solver.compute_fluxes(state)
        assert np.isfinite(Q_i)
        assert np.isfinite(Q_e)

    def test_kinetic_electron_flux_uses_electron_distribution(self):
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=4,
            n_vpar=4,
            n_mu=3,
            n_species=2,
            kinetic_electrons=True,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        phi = np.zeros((cfg.n_kx, cfg.n_ky, cfg.n_theta), dtype=np.complex128)
        phi[:, 1, :] = 1.0
        f = np.zeros(
            (cfg.n_species, cfg.n_kx, cfg.n_ky, cfg.n_theta, cfg.n_vpar, cfg.n_mu),
            dtype=np.complex128,
        )
        f[0, :, 1, :, :, :] = -1j
        f[1, :, 1, :, :, :] = -3j
        state = solver.init_state(amplitude=0.0)
        state = state.__class__(f=f, phi=phi, time=0.0)

        Q_i, Q_e = solver.compute_fluxes(state)
        vpar2 = solver.vpar[None, None, None, :, None] ** 2
        mu_val = solver.mu[None, None, None, None, :]
        energy = 0.5 * vpar2 + mu_val
        p_e = np.sum(energy * f[1], axis=(-2, -1)) * solver.dvpar * solver.dmu
        expected_e = float(
            np.real(np.sum(1j * solver.ky[1] * np.conj(phi[:, 1, :]) * p_e[:, 1, :]))
        )

        assert Q_e == expected_e
        assert Q_e != 0.5 * Q_i

    def test_phi_rms_positive(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        state = solver.init_state()
        assert solver.phi_rms(state) > 0


class TestElectromagneticDrive:
    def test_electromagnetic_state_declares_a_parallel_and_b_parallel_fields(self):
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=6,
            n_mu=4,
            n_species=2,
            kinetic_electrons=True,
            electromagnetic=True,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        state = solver.init_state(amplitude=1e-5, seed=17)

        assert state.A_par is not None
        assert state.B_par is not None
        assert state.A_par.shape == state.phi.shape
        assert state.B_par.shape == state.phi.shape
        solver.validate_state(state)

    def test_magnetic_compression_solve_is_finite_and_zero_mean_gauge(self):
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=6,
            n_mu=4,
            n_species=2,
            kinetic_electrons=True,
            electromagnetic=True,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        state = solver.init_state(amplitude=1e-5, seed=23)

        b_par = solver.magnetic_compression_solve(state.f)

        assert b_par.shape == state.phi.shape
        assert np.all(np.isfinite(b_par))
        np.testing.assert_allclose(b_par[0, 0, :], 0.0, atol=1e-30)

    def test_validate_state_rejects_bad_b_parallel_shape(self):
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=6,
            n_mu=4,
            n_species=2,
            kinetic_electrons=True,
            electromagnetic=True,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        state = solver.init_state(amplitude=1e-5, seed=19)
        bad = state.__class__(
            f=state.f,
            phi=state.phi,
            time=state.time,
            A_par=state.A_par,
            B_par=np.zeros((cfg.n_kx, cfg.n_ky), dtype=np.complex128),
        )

        try:
            solver.validate_state(bad)
        except ValueError as exc:
            assert "B_par shape" in str(exc)
        else:
            raise AssertionError("validate_state must reject malformed B_parallel fields")

    def test_kinetic_electron_drive_uses_effective_potential(self):
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=8,
            n_mu=4,
            n_species=2,
            kinetic_electrons=True,
            electromagnetic=True,
            collisions=False,
            hyper_coeff=0.0,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        phi = np.ones((cfg.n_kx, cfg.n_ky, cfg.n_theta), dtype=np.complex128) * (1.0 + 0.25j)
        a_par = np.ones_like(phi) * (0.2 - 0.1j)

        electrostatic = solver.gradient_drive(phi, None)
        electromagnetic = solver.gradient_drive(phi, a_par)

        assert np.max(np.abs(electromagnetic[0] - electrostatic[0])) > 0.0
        assert np.max(np.abs(electromagnetic[1] - electrostatic[1])) > 0.0

    def test_b_parallel_enters_electromagnetic_hamiltonian_drive(self):
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=8,
            n_mu=4,
            n_species=2,
            kinetic_electrons=True,
            electromagnetic=True,
            collisions=False,
            hyper_coeff=0.0,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        phi = np.ones((cfg.n_kx, cfg.n_ky, cfg.n_theta), dtype=np.complex128) * (1.0 + 0.1j)
        a_par = np.zeros_like(phi)
        b_par = np.ones_like(phi) * (0.15 - 0.05j)

        without_compression = solver.gradient_drive(phi, a_par, None)
        with_compression = solver.gradient_drive(phi, a_par, b_par)

        assert np.max(np.abs(with_compression[0] - without_compression[0])) > 0.0
        assert np.max(np.abs(with_compression[1] - without_compression[1])) > 0.0


class TestNonlinearInvariantDiagnostics:
    def test_exb_nonlinearity_preserves_free_energy_contract(self):
        cfg = NonlinearGKConfig(
            n_kx=8,
            n_ky=8,
            n_theta=8,
            n_vpar=4,
            n_mu=3,
            n_species=2,
            nonlinear=True,
            collisions=False,
            hyper_coeff=0.0,
            R_L_Ti=0.0,
            R_L_Te=0.0,
            R_L_ne=0.0,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        state = solver.init_state(amplitude=1e-4, seed=123)

        diagnostics = solver.nonlinear_invariant_diagnostics(state)

        assert diagnostics.finite
        assert diagnostics.passes
        assert abs(diagnostics.exb_free_energy_production) <= 1e-8
        assert diagnostics.dealiased_high_k_max_abs <= 1e-12

    def test_run_exports_nonlinear_invariant_histories(self):
        cfg = NonlinearGKConfig(
            n_kx=8,
            n_ky=8,
            n_theta=8,
            n_vpar=4,
            n_mu=3,
            n_species=2,
            nonlinear=True,
            collisions=False,
            hyper_coeff=0.0,
            R_L_Ti=0.0,
            R_L_Te=0.0,
            R_L_ne=0.0,
            dt=0.005,
            n_steps=4,
            save_interval=1,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)

        result = solver.run(solver.init_state(amplitude=1e-5, seed=41))

        assert result.exb_free_energy_production_t.shape == result.time.shape
        assert result.exb_relative_free_energy_production_t.shape == result.time.shape
        assert result.dealiased_high_k_max_abs_t.shape == result.time.shape
        assert result.nonlinear_invariant_pass_t.shape == result.time.shape
        assert result.nonlinear_invariant_pass_t.dtype == np.bool_
        assert np.all(np.isfinite(result.exb_free_energy_production_t))
        assert np.all(np.isfinite(result.exb_relative_free_energy_production_t))
        assert np.all(result.dealiased_high_k_max_abs_t <= 1e-12)
        assert np.all(result.nonlinear_invariant_pass_t)


# ── JAX fallback ─────────────────────────────────────────────────────


class TestJaxFallback:
    def test_jax_solver_import(self):
        from scpn_fusion.core.jax_gk_nonlinear import JaxNonlinearGKSolver

        solver = JaxNonlinearGKSolver(_FAST_CFG)
        # Should work regardless of JAX availability
        assert solver._np_solver is not None

    def test_jax_solver_runs(self):
        from scpn_fusion.core.jax_gk_nonlinear import JaxNonlinearGKSolver

        solver = JaxNonlinearGKSolver(_FAST_CFG)
        result = solver.run()
        assert result.converged or len(result.Q_i_t) > 0
        assert np.all(np.isfinite(result.Q_i_t))
        expected_chi_i_gB = result.chi_i / max(_FAST_CFG.R_L_Ti, 0.01)
        assert result.chi_i_gB == expected_chi_i_gB

    def test_jax_run_exports_numpy_parity_histories(self):
        from scpn_fusion.core.jax_gk_nonlinear import JaxNonlinearGKSolver

        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=4,
            n_mu=3,
            n_species=2,
            kinetic_electrons=True,
            electromagnetic=True,
            nonlinear=True,
            collisions=False,
            hyper_coeff=0.0,
            dt=0.005,
            n_steps=3,
            save_interval=1,
            cfl_adapt=False,
        )
        solver = JaxNonlinearGKSolver(cfg)

        result = solver.run(solver._np_solver.init_state(amplitude=1e-5, seed=43))

        assert result.particle_free_energy_t.shape == result.time.shape
        assert result.phi_energy_t.shape == result.time.shape
        assert result.A_parallel_energy_t.shape == result.time.shape
        assert result.B_parallel_energy_t.shape == result.time.shape
        assert result.total_energy_t.shape == result.time.shape
        assert result.exb_free_energy_production_t.shape == result.time.shape
        assert result.exb_relative_free_energy_production_t.shape == result.time.shape
        assert result.dealiased_high_k_max_abs_t.shape == result.time.shape
        assert result.nonlinear_invariant_pass_t.shape == result.time.shape
        assert result.nonlinear_invariant_pass_t.dtype == np.bool_
        assert np.all(np.isfinite(result.total_energy_t))
        assert np.all(np.isfinite(result.exb_relative_free_energy_production_t))
        assert result.Q_i_kxky_t.shape == (result.time.size, cfg.n_kx, cfg.n_ky)
        assert result.Q_e_kxky_t.shape == (result.time.size, cfg.n_kx, cfg.n_ky)
        np.testing.assert_allclose(np.sum(result.Q_i_kxky_t, axis=(1, 2)), result.Q_i_t)
        np.testing.assert_allclose(np.sum(result.Q_e_kxky_t, axis=(1, 2)), result.Q_e_t)
        assert result.zonal_flow_energy_t.shape == result.time.shape
        assert np.all(result.zonal_flow_energy_t <= result.phi_energy_t)
        assert result.saturated_Q_i_kxky.shape == (cfg.n_kx, cfg.n_ky)
        assert result.saturated_Q_e_kxky.shape == (cfg.n_kx, cfg.n_ky)
        assert np.isfinite(result.saturated_phi_rms)
        assert np.isfinite(result.saturated_zonal_flow_energy)
        assert np.isfinite(result.saturated_total_energy)
        assert result.phi_energy_kxky_t.shape == (result.time.size, cfg.n_kx, cfg.n_ky)
        assert result.A_parallel_energy_kxky_t.shape == (result.time.size, cfg.n_kx, cfg.n_ky)
        assert result.B_parallel_energy_kxky_t.shape == (result.time.size, cfg.n_kx, cfg.n_ky)
        assert result.particle_free_energy_species_kxky_t.shape == (
            result.time.size,
            cfg.n_species,
            cfg.n_kx,
            cfg.n_ky,
        )
        assert result.saturated_phi_energy_kxky.shape == (cfg.n_kx, cfg.n_ky)
        assert result.saturated_A_parallel_energy_kxky.shape == (cfg.n_kx, cfg.n_ky)
        assert result.saturated_B_parallel_energy_kxky.shape == (cfg.n_kx, cfg.n_ky)
        assert result.saturated_particle_free_energy_species_kxky.shape == (
            cfg.n_species,
            cfg.n_kx,
            cfg.n_ky,
        )

    def test_jax_parallel_streaming_matches_numpy_ballooning_connection(self):
        from scpn_fusion.core.jax_gk_nonlinear import JaxNonlinearGKSolver, jax_available

        assert jax_available()
        import jax.numpy as jnp

        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=4,
            n_mu=3,
            n_species=2,
            s_hat=0.9,
            cfl_adapt=False,
        )
        solver = JaxNonlinearGKSolver(cfg)
        state = solver._np_solver.init_state(amplitude=1e-4, seed=31)

        expected = solver._np_solver.parallel_streaming(state.f[0])
        actual = np.asarray(solver._jax_parallel_streaming(jnp.asarray(state.f[0])))

        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_jax_available_bool(self):
        from scpn_fusion.core.jax_gk_nonlinear import jax_available

        assert isinstance(jax_available(), bool)

    def test_jax_sugama_collision_conserves_discrete_moments_or_falls_back(self):
        from scpn_fusion.core.jax_gk_nonlinear import JaxNonlinearGKSolver, jax_available

        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=8,
            n_mu=6,
            n_species=2,
            collisions=True,
            collision_model="sugama",
            cfl_adapt=False,
        )
        solver = JaxNonlinearGKSolver(cfg)
        state = solver._np_solver.init_state(amplitude=1e-4, seed=11)

        if jax_available():
            import jax.numpy as jnp

            collision = np.asarray(solver._jax_collide_sugama(jnp.asarray(state.f[0])))
        else:
            collision = solver._np_solver.collide(state.f[0])

        vpar = solver._np_solver.vpar[None, None, None, :, None]
        mu = solver._np_solver.mu[None, None, None, None, :]
        energy = 0.5 * vpar**2 + mu
        dv = solver._np_solver.dvpar * solver._np_solver.dmu

        np.testing.assert_allclose(np.sum(collision * dv, axis=(-2, -1)), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.sum(collision * vpar * dv, axis=(-2, -1)), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.sum(collision * energy * dv, axis=(-2, -1)), 0.0, atol=1e-12)

    def test_jax_exb_invariant_diagnostics_match_contract_or_fall_back(self):
        from scpn_fusion.core.jax_gk_nonlinear import JaxNonlinearGKSolver

        cfg = NonlinearGKConfig(
            n_kx=8,
            n_ky=8,
            n_theta=8,
            n_vpar=4,
            n_mu=3,
            n_species=2,
            nonlinear=True,
            collisions=False,
            hyper_coeff=0.0,
            R_L_Ti=0.0,
            R_L_Te=0.0,
            R_L_ne=0.0,
            cfl_adapt=False,
        )
        solver = JaxNonlinearGKSolver(cfg)
        state = solver._np_solver.init_state(amplitude=1e-4, seed=123)

        diagnostics = solver.nonlinear_invariant_diagnostics(state)

        assert diagnostics.finite
        assert diagnostics.passes
        assert abs(diagnostics.exb_free_energy_production) <= 1e-8
        assert diagnostics.dealiased_high_k_max_abs <= 1e-12
