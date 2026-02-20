// ────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Red-Black SOR Compute Shader for GS Solver
// ────────────────────────────────────────────────────────────────────
//
// Solves the Grad-Shafranov equation on a uniform (R,Z) grid:
//   Δ*ψ = ∂²ψ/∂R² - (1/R)∂ψ/∂R + ∂²ψ/∂Z² = S(R,Z)
//
// using Red-Black Successive Over-Relaxation (SOR).
//
// Each dispatch handles one color (red or black) of the checkerboard
// pattern. The host alternates: red sweep → black sweep.

struct Params {
    nr: u32,       // number of R grid points
    nz: u32,       // number of Z grid points
    dr: f32,       // R spacing [m]
    dz: f32,       // Z spacing [m]
    r_left: f32,   // R coordinate of left boundary [m]
    omega: f32,    // SOR relaxation factor (1 < ω < 2)
    color: u32,    // 0 = red, 1 = black
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> psi: array<f32>;
@group(0) @binding(2) var<storage, read> source: array<f32>;

fn idx(iz: u32, ir: u32) -> u32 {
    return iz * params.nr + ir;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ir = gid.x + 1u;  // skip boundary
    let iz = gid.y + 1u;

    // Bounds check (interior only)
    if (ir >= params.nr - 1u || iz >= params.nz - 1u) {
        return;
    }

    // Red-Black pattern: (ir + iz) % 2 == color
    if ((ir + iz) % 2u != params.color) {
        return;
    }

    let dr = params.dr;
    let dz = params.dz;
    let R = params.r_left + f32(ir) * dr;

    // Avoid R = 0 singularity
    let R_safe = max(R, 1e-6);
    let inv_R = 1.0 / R_safe;

    let dr2 = dr * dr;
    let dz2 = dz * dz;

    // Finite-difference stencil for Δ*ψ = S
    // ∂²ψ/∂R² ≈ (ψ[i,j+1] - 2ψ[i,j] + ψ[i,j-1]) / dR²
    // (1/R)∂ψ/∂R ≈ (ψ[i,j+1] - ψ[i,j-1]) / (2R dR)
    // ∂²ψ/∂Z² ≈ (ψ[i+1,j] - 2ψ[i,j] + ψ[i-1,j]) / dZ²

    let c_center = -2.0 / dr2 - 2.0 / dz2;
    let c_right = 1.0 / dr2 + inv_R / (2.0 * dr);
    let c_left = 1.0 / dr2 - inv_R / (2.0 * dr);
    let c_up = 1.0 / dz2;
    let c_down = 1.0 / dz2;

    let psi_right = psi[idx(iz, ir + 1u)];
    let psi_left = psi[idx(iz, ir - 1u)];
    let psi_up = psi[idx(iz + 1u, ir)];
    let psi_down = psi[idx(iz - 1u, ir)];
    let psi_center = psi[idx(iz, ir)];
    let rhs = source[idx(iz, ir)];

    // Gauss-Seidel update: ψ_new = (S - Σ c_k ψ_k) / c_center
    let gs_update = (rhs - c_right * psi_right - c_left * psi_left
                     - c_up * psi_up - c_down * psi_down) / c_center;

    // SOR: ψ = (1 - ω) ψ_old + ω ψ_GS
    let psi_new = (1.0 - params.omega) * psi_center + params.omega * gs_update;

    psi[idx(iz, ir)] = psi_new;
}

// ── Multigrid Kernels ──────────────────────────────────────────────

@group(0) @binding(3) var<storage, read_write> residual_buf: array<f32>;

// 1. Residual calculation: r = S - Delta* psi
@compute @workgroup_size(16, 16)
fn calculate_residual(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ir = gid.x + 1u;
    let iz = gid.y + 1u;

    if (ir >= params.nr - 1u || iz >= params.nz - 1u) {
        return;
    }

    let dr = params.dr;
    let dz = params.dz;
    let R = params.r_left + f32(ir) * dr;
    let inv_R = 1.0 / max(R, 1e-6);

    let c_center = -2.0 / (dr * dr) - 2.0 / (dz * dz);
    let c_right = 1.0 / (dr * dr) + inv_R / (2.0 * dr);
    let c_left = 1.0 / (dr * dr) - inv_R / (2.0 * dr);
    let c_up = 1.0 / (dz * dz);
    let c_down = 1.0 / (dz * dz);

    let p_c = psi[idx(iz, ir)];
    let p_r = psi[idx(iz, ir + 1u)];
    let p_l = psi[idx(iz, ir - 1u)];
    let p_u = psi[idx(iz + 1u, ir)];
    let p_d = psi[idx(iz - 1u, ir)];
    let rhs = source[idx(iz, ir)];

    // r = S - (c_c*p_c + c_r*p_r + c_l*p_l + c_u*p_u + c_d*p_d)
    residual_buf[idx(iz, ir)] = rhs - (c_center * p_c + c_right * p_r + c_left * p_l + c_up * p_u + c_down * p_d);
}

// 2. Restriction: Fine -> Coarse (Half-weight injection)
// Expects fine grid in source, coarse grid in residual_buf
@group(0) @binding(4) var<storage, read> coarse_source: array<f32>;
@group(0) @binding(5) var<storage, read_write> coarse_dest: array<f32>;

@compute @workgroup_size(16, 16)
fn restrict_to_coarse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ir_c = gid.x;
    let iz_c = gid.y;
    
    // Coarse grid size is (nr-1)/2 + 1
    let nr_c = (params.nr - 1u) / 2u + 1u;
    let nz_c = (params.nz - 1u) / 2u + 1u;

    if (ir_c >= nr_c || iz_c >= nz_c) {
        return;
    }

    let ir_f = ir_c * 2u;
    let iz_f = iz_c * 2u;
    
    // Simple injection for now
    coarse_dest[iz_c * nr_c + ir_c] = residual_buf[iz_f * params.nr + ir_f];
}

// 3. Prolongation: Coarse -> Fine (Bilinear interpolation)
// Adds coarse correction to fine solution
@compute @workgroup_size(16, 16)
fn prolong_and_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ir_f = gid.x;
    let iz_f = gid.y;

    if (ir_f >= params.nr || iz_f >= params.nz) {
        return;
    }

    let nr_c = (params.nr - 1u) / 2u + 1u;
    
    // Coarse indices (floor)
    let ir_c = ir_f / 2u;
    let iz_c = iz_f / 2u;
    
    // Fractions
    let fr = f32(ir_f % 2u) * 0.5;
    let fz = f32(iz_f % 2u) * 0.5;

    // Bilinear from coarse_dest
    let c00 = coarse_dest[iz_c * nr_c + ir_c];
    let c10 = coarse_dest[iz_c * nr_c + min(ir_c + 1u, nr_c - 1u)];
    let c01 = coarse_dest[min(iz_c + 1u, nz_c - 1u) * nr_c + ir_c];
    let c11 = coarse_dest[min(iz_c + 1u, nz_c - 1u) * nr_c + min(ir_c + 1u, nr_c - 1u)];

    let correction = (1.0 - fr) * (1.0 - fz) * c00 + 
                     fr * (1.0 - fz) * c10 + 
                     (1.0 - fr) * fz * c01 + 
                     fr * fz * c11;

    psi[idx(iz_f, ir_f)] += correction;
}
