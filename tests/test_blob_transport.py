# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Blob Transport Tests
from __future__ import annotations

import numpy as np

from scpn_fusion.core.blob_transport import (
    BlobDetector,
    BlobDynamics,
    BlobEnsemble,
    SOLBlobProfile,
)


def test_blob_dynamics_critical_size():
    dyn = BlobDynamics(R0=6.2, B0=5.3, Te_eV=20.0, Ti_eV=20.0, mi_amu=2.0)

    # Check critical size
    delta_star = dyn.critical_size(L_parallel=10.0)
    assert 0.001 < delta_star < 0.1  # Typically ~cm scale

    # Regimes
    v_sheath, reg_sheath = dyn.blob_velocity(delta_star * 0.5, 1e19, 10.0)
    assert reg_sheath == "sheath"

    v_inertial, reg_inertial = dyn.blob_velocity(delta_star * 2.0, 1e19, 10.0)
    assert reg_inertial == "inertial"

    # Velocity at delta_star should be the crossing point
    v_star = dyn.max_velocity(10.0)
    assert v_star > 0.0


def test_blob_ensemble_generation():
    dyn = BlobDynamics(R0=6.2, B0=5.3, Te_eV=20.0, Ti_eV=20.0, mi_amu=2.0)
    ens = BlobEnsemble(dyn, n_blobs=100)
    rng = np.random.default_rng(42)

    pop = ens.generate(0.01, 0.002, 1.0, 1e-4, rng)

    assert len(pop.sizes) == 100
    assert len(pop.birth_times) == 100
    assert np.all(pop.sizes > 0)
    assert np.all(pop.amplitudes > 0)

    # Verify fluxes
    gamma = ens.radial_flux(pop)
    q = ens.heat_flux(pop, 20.0)

    assert gamma > 0.0
    assert q > 0.0


def test_sol_blob_profile() -> None:
    # Without blobs
    r_arr = np.array([0.05])
    prof_clean = SOLBlobProfile.radial_density(r=r_arr, Gamma_blob=0.0, D_perp=1.0, lambda_n=0.02)

    # With blobs
    prof_blobs = SOLBlobProfile.radial_density(r=r_arr, Gamma_blob=1e20, D_perp=1.0, lambda_n=0.02)

    # Profile should be broader (higher density far out)
    assert prof_blobs[0] > prof_clean[0]
    # Wall flux
    wall_clean = SOLBlobProfile.wall_flux(r_wall=0.1, Gamma_blob=1e18, lambda_n=0.02)
    wall_dirty = SOLBlobProfile.wall_flux(r_wall=0.1, Gamma_blob=1e20, lambda_n=0.02)

    assert wall_dirty > wall_clean


def test_blob_detector():
    det = BlobDetector()

    # Create a synthetic signal with a spike
    sig = np.random.randn(1000) * 0.1
    # Spike at index 500
    sig[500:510] += 5.0

    events = det.detect_blobs(sig, dt=1e-6, threshold=2.5)

    assert len(events) >= 1
    # Spike is around index 500
    assert 495 < events[0].start_idx < 505

    avg = det.conditional_average(sig, events, window=20)
    assert len(avg) == 41
    assert avg[20] > 1.0  # Center of average is the spike
