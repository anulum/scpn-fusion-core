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
