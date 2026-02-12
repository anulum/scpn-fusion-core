// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Types
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
#ifndef SCPN_FUSION_TYPES_H
#define SCPN_FUSION_TYPES_H

#include <vector>
#include <cstdint>

// --- MEMORY LAYOUT FOR HPC ---
// Align to 64 bytes for AVX-512 vectorization
#define ALIGN_64 __attribute__((aligned(64)))

namespace SCPN {

    // Configuration Parameter Pack (Passed from Python)
    struct ALIGN_64 PlasmaConfig {
        int nr;             // Grid Resolution R
        int nz;             // Grid Resolution Z
        double r_min;
        double r_max;
        double z_min;
        double z_max;
        double target_current; // MA
        double vacuum_perm;
        
        // Iteration Control
        int max_iter;
        double tol;
        double alpha; // Relaxation
    };

    // Coil Definition (Struct of Arrays for SIMD efficiency)
    struct ALIGN_64 CoilSet {
        int n_coils;
        // Pointers to arrays allocated in shared memory
        double* r_pos;
        double* z_pos;
        double* current;
    };

    // The Field State (The heavy data)
    struct ALIGN_64 PlasmaState {
        // Flat arrays (Row-major order for C++)
        double* psi;        // Magnetic Flux [nr * nz]
        double* j_phi;      // Current Density [nr * nz]
        double* pressure;   // Pressure Profile [nr * nz]
        
        // Derived metrics
        double axis_r;
        double axis_z;
        double psi_axis;
        double psi_boundary;
    };

}

#endif // SCPN_FUSION_TYPES_H
