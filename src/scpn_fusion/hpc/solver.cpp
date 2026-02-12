// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Solver
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include "types.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// --- HIGH PERFORMANCE GRAD-SHAFRANOV SOLVER ---

namespace SCPN {

    class FastSolver {
    public:
        FastSolver(PlasmaConfig config) : cfg(config) {
            // Allocate memory
            size_t size = cfg.nr * cfg.nz;
            psi.resize(size, 0.0);
            j_phi.resize(size, 0.0);
            
            // Precompute grid
            dr = (cfg.r_max - cfg.r_min) / (cfg.nr - 1);
            dz = (cfg.z_max - cfg.z_min) / (cfg.nz - 1);
            dr_sq = dr * dr;
            dz_sq = dz * dz;
            
            // Precompute R array for 1/R terms
            r_grid.resize(size);
            
            // Phase 2 Preparation: Domain Decomposition hooks
            // For now, we use OpenMP for shared memory parallelism
            #pragma omp parallel for collapse(2)
            for(int z=0; z<cfg.nz; z++) {
                for(int r=0; r<cfg.nr; r++) {
                    r_grid[z*cfg.nr + r] = cfg.r_min + r*dr;
                }
            }
        }

        // The Hot Loop (This is what runs on Supercomputer Nodes)
        void solve_step_sor(double omega = 1.8) {
            // Successive Over-Relaxation using Red-Black ordering for Parallelism
            // To make it parallel-friendly, we split into Red and Black checkerboards
            
            // Red Pass (i+j even)
            #pragma omp parallel for collapse(2)
            for(int z = 1; z < cfg.nz - 1; z++) {
                for(int r = 1; r < cfg.nr - 1; r++) {
                    if ((z + r) % 2 == 0) update_point(z, r, omega);
                }
            }
            
            // Black Pass (i+j odd)
            #pragma omp parallel for collapse(2)
            for(int z = 1; z < cfg.nz - 1; z++) {
                for(int r = 1; r < cfg.nr - 1; r++) {
                    if ((z + r) % 2 != 0) update_point(z, r, omega);
                }
            }
        }
        
        inline void update_point(int z, int r, double omega) {
            int idx = z * cfg.nr + r;
            
            double R = r_grid[idx];
            double source = -cfg.vacuum_perm * R * j_phi[idx];
            
            // Elliptic Operator Stencil (5-point)
            double c_r_plus = (1.0 / dr_sq) - (1.0 / (2.0 * R * dr));
            double c_r_minus = (1.0 / dr_sq) + (1.0 / (2.0 * R * dr));
            double c_z = 1.0 / dz_sq;
            double center = 2.0/dr_sq + 2.0/dz_sq;
            
            // Neighbors
            double p_up = psi[(z+1)*cfg.nr + r];
            double p_down = psi[(z-1)*cfg.nr + r];
            double p_right = psi[z*cfg.nr + (r+1)];
            double p_left = psi[z*cfg.nr + (r-1)];
            
            // Gauss-Seidel Prediction
            double p_star = (source + c_z*(p_up + p_down) + c_r_plus*p_right + c_r_minus*p_left) / center;
            
            // SOR Update
            psi[idx] = (1.0 - omega) * psi[idx] + omega * p_star;
        }

        // Public Interface for Python
        void set_current_profile(const std::vector<double>& j_in) {
            if(j_in.size() == j_phi.size()) {
                 j_phi = j_in;
            }
        }
        
        std::vector<double> get_psi() {
            return psi;
        }

    private:
        PlasmaConfig cfg;
        std::vector<double> psi;
        std::vector<double> j_phi;
        std::vector<double> r_grid;
        double dr, dz, dr_sq, dz_sq;
    };
}

// C-Style Export for Python CTypes
extern "C" {
    void* create_solver(int nr, int nz, double rmin, double rmax, double zmin, double zmax) {
        SCPN::PlasmaConfig cfg;
        cfg.nr = nr; cfg.nz = nz;
        cfg.r_min = rmin; cfg.r_max = rmax;
        cfg.z_min = zmin; cfg.z_max = zmax;
        cfg.vacuum_perm = 1.0;
        
        return new SCPN::FastSolver(cfg);
    }

    void run_step(void* solver_ptr, double* j_array, double* psi_array, int size, int iterations) {
        SCPN::FastSolver* solver = (SCPN::FastSolver*)solver_ptr;
        
        // Zero-copy usually, but here we copy for safety in demo
        std::vector<double> j_vec(j_array, j_array + size);
        solver->set_current_profile(j_vec);
        
        for(int i=0; i<iterations; i++) {
            solver->solve_step_sor(1.9); // High relaxation for speed
        }
        
        std::vector<double> res = solver->get_psi();
        std::copy(res.begin(), res.end(), psi_array);
    }
}
