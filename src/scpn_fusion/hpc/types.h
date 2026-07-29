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

/** Native-memory types shared by the bundled HPC solver adapters. */
namespace SCPN {

    /** Configuration pack passed across the native bridge. */
    struct ALIGN_64 PlasmaConfig {
        /** Radial grid-point count. */
        int nr;
        /** Vertical grid-point count. */
        int nz;
        /** Minimum major radius in metres. */
        double r_min;
        /** Maximum major radius in metres. */
        double r_max;
        /** Minimum vertical coordinate in metres. */
        double z_min;
        /** Maximum vertical coordinate in metres. */
        double z_max;
        /** Target toroidal plasma current in megaamperes. */
        double target_current;
        /** Vacuum permeability used by the solver. */
        double vacuum_perm;

        /** Maximum solver iteration count. */
        int max_iter;
        /** Absolute convergence tolerance. */
        double tol;
        /** Iteration relaxation factor. */
        double alpha;
    };

    /** Struct-of-arrays coil definition for SIMD-compatible shared memory. */
    struct ALIGN_64 CoilSet {
        /** Number of entries in each coil array. */
        int n_coils;
        /** Coil major-radius coordinates. */
        double* r_pos;
        /** Coil vertical coordinates. */
        double* z_pos;
        /** Coil currents in amperes. */
        double* current;
    };

    /** Row-major plasma field state and derived magnetic-axis metrics. */
    struct ALIGN_64 PlasmaState {
        /** Poloidal flux array with `nr * nz` entries. */
        double* psi;
        /** Toroidal current-density array with `nr * nz` entries. */
        double* j_phi;
        /** Pressure-profile array with `nr * nz` entries. */
        double* pressure;

        /** Derived magnetic-axis major radius. */
        double axis_r;
        /** Derived magnetic-axis vertical coordinate. */
        double axis_z;
        /** Poloidal flux at the magnetic axis. */
        double psi_axis;
        /** Poloidal flux on the fixed boundary. */
        double psi_boundary;
    };

}

#endif // SCPN_FUSION_TYPES_H
