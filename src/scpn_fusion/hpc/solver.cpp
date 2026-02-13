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

        apply_dirichlet_boundaries();
    }

    /// Run one Red-Black SOR sweep and return max |delta psi|.
    double solve_step_sor(double omega = 1.8) {
        apply_dirichlet_boundaries();

        double red_max_delta = 0.0;
        // Red pass (iz + ir even)
        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) reduction(max:red_max_delta)
        #endif
        for (int z = 1; z < cfg.nz - 1; ++z) {
            for (int r = 1; r < cfg.nr - 1; ++r) {
                if ((z + r) % 2 == 0) {
                    const double delta = update_point(z, r, omega);
                    red_max_delta = std::max(red_max_delta, delta);
                }
            }
        }

        double black_max_delta = 0.0;
        // Black pass (iz + ir odd)
        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) reduction(max:black_max_delta)
        #endif
        for (int z = 1; z < cfg.nz - 1; ++z) {
            for (int r = 1; r < cfg.nr - 1; ++r) {
                if ((z + r) % 2 != 0) {
                    const double delta = update_point(z, r, omega);
                    black_max_delta = std::max(black_max_delta, delta);
                }
            }
        }

        return std::max(red_max_delta, black_max_delta);
    }

    void set_current_profile(const double *j_in, size_t size) {
        if (size == j_phi.size()) {
            std::copy(j_in, j_in + size, j_phi.begin());
        }
    }

    void set_boundary_value(double value) {
        boundary_value = value;
        apply_dirichlet_boundaries();
    }

    const double *get_psi_ptr() const { return psi.data(); }
    size_t get_size() const { return psi.size(); }

private:
    inline void apply_dirichlet_boundaries() {
        if (cfg.nr <= 0 || cfg.nz <= 0) {
            return;
        }
        const int nr = cfg.nr;
        const int nz = cfg.nz;
        for (int r = 0; r < nr; ++r) {
            psi[r] = boundary_value;
            psi[(nz - 1) * nr + r] = boundary_value;
        }
        for (int z = 0; z < nz; ++z) {
            psi[z * nr] = boundary_value;
            psi[z * nr + (nr - 1)] = boundary_value;
        }
    }

    inline double update_point(int z, int r, double omega) {
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

        const double old_psi = psi[idx];
        const double new_psi = (1.0 - omega) * old_psi + omega * p_gs;
        psi[idx] = new_psi;
        return std::abs(new_psi - old_psi);
    }

    PlasmaConfig cfg;
    std::vector<double> psi;
    std::vector<double> j_phi;
    std::vector<double> r_grid;
    double dr, dz, dr_sq, dz_sq;
    double boundary_value = 0.0;
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

/// Set fixed Dirichlet boundary value for the psi edges.
void set_boundary_dirichlet(void *solver_ptr, double boundary_value)
{
    if (solver_ptr == nullptr) {
        return;
    }
    auto *solver = static_cast<FastSolver *>(solver_ptr);
    solver->set_boundary_value(boundary_value);
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
    if (solver_ptr == nullptr || j_array == nullptr || psi_array == nullptr || size <= 0) {
        return;
    }

    auto *solver = static_cast<FastSolver *>(solver_ptr);

    solver->set_current_profile(j_array, static_cast<size_t>(size));

    const int n_iter = std::max(iterations, 1);
    for (int i = 0; i < n_iter; ++i) {
        solver->solve_step_sor(1.8);
    }

    const double *result = solver->get_psi_ptr();
    std::copy(result, result + size, psi_array);
}

/// Run SOR with early convergence stop based on max |delta psi|.
///
/// @return iterations executed (>=1)
int run_step_converged(void *solver_ptr,
                       const double *j_array,
                       double       *psi_array,
                       int           size,
                       int           max_iterations,
                       double        omega,
                       double        tolerance,
                       double       *final_delta_out)
{
    if (solver_ptr == nullptr || j_array == nullptr || psi_array == nullptr || size <= 0) {
        if (final_delta_out != nullptr) {
            *final_delta_out = 0.0;
        }
        return 0;
    }

    auto *solver = static_cast<FastSolver *>(solver_ptr);
    solver->set_current_profile(j_array, static_cast<size_t>(size));

    const int n_iter = std::max(max_iterations, 1);
    const double omega_clamped = std::clamp(omega, 0.1, 1.99);
    const double tol = std::max(tolerance, 0.0);

    double last_delta = 0.0;
    int performed = 0;
    for (int i = 0; i < n_iter; ++i) {
        last_delta = solver->solve_step_sor(omega_clamped);
        performed = i + 1;
        if (last_delta <= tol) {
            break;
        }
    }

    const double *result = solver->get_psi_ptr();
    std::copy(result, result + size, psi_array);
    if (final_delta_out != nullptr) {
        *final_delta_out = last_delta;
    }
    return performed;
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
