// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — C++ Grad-Shafranov Elliptic Solver
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//
// Red-Black SOR solver for the 2-D Poisson-like equation arising from
// the Grad-Shafranov equilibrium:
//
//   Δ*Ψ = -μ₀ R J_φ
//
// with 5-point finite-difference stencil including the toroidal 1/R
// correction terms:
//
//   c_r_plus  = 1/dr² - 1/(2R dr)
//   c_r_minus = 1/dr² + 1/(2R dr)
//   c_z       = 1/dz²
//   center    = 2/dr² + 2/dz²
//
// Build:
//   Linux:   g++ -shared -fPIC -o libscpn_solver.so solver.cpp -O3 -march=native
//   Windows: g++ -shared -o scpn_solver.dll solver.cpp -O3 -mavx2
//
// The shared library exposes a C-linkage API consumed by the Python
// HPCBridge (src/scpn_fusion/hpc/hpc_bridge.py) via ctypes.
// ─────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

// ── Configuration ───────────────────────────────────────────────────

struct PlasmaConfig {
    int    nr;
    int    nz;
    double r_min;
    double r_max;
    double z_min;
    double z_max;
    double vacuum_perm;  // μ₀ (normally 1.0 in normalised units)
};

// ── Solver ──────────────────────────────────────────────────────────

class FastSolver {
public:
    explicit FastSolver(PlasmaConfig config)
        : cfg(config)
    {
        const size_t size = static_cast<size_t>(cfg.nr) * cfg.nz;
        psi.resize(size, 0.0);
        j_phi.resize(size, 0.0);

        dr = (cfg.r_max - cfg.r_min) / (cfg.nr - 1);
        dz = (cfg.z_max - cfg.z_min) / (cfg.nz - 1);
        dr_sq = dr * dr;
        dz_sq = dz * dz;

        r_grid.resize(size);

        #ifdef _OPENMP
        #pragma omp parallel for collapse(2)
        #endif
        for (int z = 0; z < cfg.nz; ++z) {
            for (int r = 0; r < cfg.nr; ++r) {
                r_grid[z * cfg.nr + r] = cfg.r_min + r * dr;
            }
        }
    }

    /// Run one Red-Black SOR sweep.
    void solve_step_sor(double omega = 1.8) {
        // Red pass (iz + ir even)
        #ifdef _OPENMP
        #pragma omp parallel for collapse(2)
        #endif
        for (int z = 1; z < cfg.nz - 1; ++z) {
            for (int r = 1; r < cfg.nr - 1; ++r) {
                if ((z + r) % 2 == 0) update_point(z, r, omega);
            }
        }

        // Black pass (iz + ir odd)
        #ifdef _OPENMP
        #pragma omp parallel for collapse(2)
        #endif
        for (int z = 1; z < cfg.nz - 1; ++z) {
            for (int r = 1; r < cfg.nr - 1; ++r) {
                if ((z + r) % 2 != 0) update_point(z, r, omega);
            }
        }
    }

    void set_current_profile(const double *j_in, size_t size) {
        if (size == j_phi.size()) {
            std::copy(j_in, j_in + size, j_phi.begin());
        }
    }

    const double *get_psi_ptr() const { return psi.data(); }
    size_t get_size() const { return psi.size(); }

private:
    inline void update_point(int z, int r, double omega) {
        const int idx = z * cfg.nr + r;
        const double R = r_grid[idx];
        const double source = -cfg.vacuum_perm * R * j_phi[idx];

        // 5-point stencil with toroidal 1/R correction
        const double c_r_plus  = 1.0 / dr_sq - 1.0 / (2.0 * R * dr);
        const double c_r_minus = 1.0 / dr_sq + 1.0 / (2.0 * R * dr);
        const double c_z       = 1.0 / dz_sq;
        const double center    = 2.0 / dr_sq + 2.0 / dz_sq;

        const double p_up    = psi[(z + 1) * cfg.nr + r];
        const double p_down  = psi[(z - 1) * cfg.nr + r];
        const double p_right = psi[z * cfg.nr + (r + 1)];
        const double p_left  = psi[z * cfg.nr + (r - 1)];

        const double p_gs = (source + c_z * (p_up + p_down)
                             + c_r_plus * p_right
                             + c_r_minus * p_left) / center;

        psi[idx] = (1.0 - omega) * psi[idx] + omega * p_gs;
    }

    PlasmaConfig cfg;
    std::vector<double> psi;
    std::vector<double> j_phi;
    std::vector<double> r_grid;
    double dr, dz, dr_sq, dz_sq;
};

// ── C-linkage API (consumed by Python ctypes) ───────────────────────

extern "C" {

/// Create a new solver instance for an (NR x NZ) grid.
void *create_solver(int nr, int nz,
                    double rmin, double rmax,
                    double zmin, double zmax)
{
    PlasmaConfig cfg;
    cfg.nr = nr;
    cfg.nz = nz;
    cfg.r_min = rmin;
    cfg.r_max = rmax;
    cfg.z_min = zmin;
    cfg.z_max = zmax;
    cfg.vacuum_perm = 1.0;

    return static_cast<void *>(new FastSolver(cfg));
}

/// Run `iterations` Red-Black SOR sweeps.
///
/// @param solver_ptr  opaque handle from create_solver()
/// @param j_array     input toroidal current density (row-major, nz*nr)
/// @param psi_array   output poloidal flux           (row-major, nz*nr)
/// @param size        total grid points (nz * nr)
/// @param iterations  number of SOR sweeps
void run_step(void *solver_ptr,
              const double *j_array,
              double       *psi_array,
              int           size,
              int           iterations)
{
    auto *solver = static_cast<FastSolver *>(solver_ptr);

    solver->set_current_profile(j_array, static_cast<size_t>(size));

    for (int i = 0; i < iterations; ++i) {
        solver->solve_step_sor(1.8);
    }

    const double *result = solver->get_psi_ptr();
    std::copy(result, result + size, psi_array);
}

/// Free a solver instance.
void destroy_solver(void *solver_ptr) {
    delete static_cast<FastSolver *>(solver_ptr);
}

/// Backward-compatible alias for bindings expecting delete_solver().
void delete_solver(void *solver_ptr) {
    destroy_solver(solver_ptr);
}

}  // extern "C"
