# Solver Tuning Guide

Practical guidance for configuring the Grad-Shafranov equilibrium solver,
the Levenberg-Marquardt inverse reconstruction, and the neural transport
surrogate. All parameters live in the reactor JSON config or in
`InverseConfig` (Rust) / function arguments (Python).

---

## 1. Picard Relaxation Factor (`relaxation_factor`)

The equilibrium solver uses under-relaxation to stabilise the nonlinear
Picard iteration: `Psi = (1 - alpha) * Psi_old + alpha * Psi_new`.

| Value | Behaviour | When to use |
|-------|-----------|-------------|
| **0.02–0.05** | Very conservative, slow but stable | High-beta plasmas, strongly shaped equilibria, or when the solver diverges at higher values |
| **0.08–0.12** | Default range (0.1 shipped in all configs) | Most ITER-class and medium-beta scenarios |
| **0.15–0.25** | Aggressive, converges faster but may oscillate | Low-beta L-mode, small grids (33×33), or when speed matters more than robustness |
| **> 0.3** | Likely to diverge | Not recommended without adaptive damping |

### How to tune

1. Start with the default `0.1`.
2. If the solver log shows oscillating residuals (residual goes up/down),
   reduce to `0.05`.
3. If convergence is too slow (hundreds of iterations for a 65×65 grid),
   try `0.15`–`0.2`.
4. For compact high-field designs (SPARC-class, B > 10 T), `0.08` is
   usually more stable than `0.1`.

### Config example

```json
{
    "solver": {
        "max_iterations": 500,
        "convergence_threshold": 1e-6,
        "relaxation_factor": 0.08
    }
}
```

The Rust solver reads this value directly. The Python solver reads it via
`self.cfg["solver"].get("relaxation_factor", 0.1)`. If the field is
missing or zero, both solvers fall back to 0.1.

---

## 2. Tikhonov Regularisation (`tikhonov`)

Tikhonov regularisation adds a penalty `alpha * ||x - x0||^2` to the
inverse solver cost function and `alpha * I` to the normal matrix. This
pulls reconstructed profile parameters towards the initial guess when
measurement data is sparse or noisy.

| Value | Effect | When to use |
|-------|--------|-------------|
| **0** | Disabled (pure least-squares) | Clean synthetic data, many well-placed probes (> 16) |
| **0.001–0.01** | Light regularisation | Moderate noise (SNR > 20 dB), standard probe layouts |
| **0.01–0.1** | Moderate regularisation | High noise, few probes (8–12), or ill-conditioned geometries |
| **0.1–1.0** | Strong regularisation | Very few probes (< 8), heavily contaminated data, or when the reconstruction oscillates wildly |
| **> 1.0** | Over-regularised | Result will be biased towards the initial guess; only useful as a diagnostic |

### How to tune

1. Run a reconstruction with `tikhonov = 0` on clean synthetic data.
   If chi-squared converges to near zero, regularisation is unnecessary.
2. Add noise to the synthetic data (`noise_std = 0.01`). If the
   reconstruction diverges or parameters hit their bounds, increase
   `tikhonov` until parameters stay physical.
3. The L-curve method works well: plot `||x - x0||` vs `||r||` for a
   range of alpha values and pick the "elbow".
4. For real experimental data, start with `0.01` and adjust based on
   the residual pattern.

### Code example (Rust)

```rust
let config = InverseConfig {
    tikhonov: 0.01,
    ..InverseConfig::default()
};
```

---

## 3. Huber Robust Loss (`loss`)

The Huber loss function reduces the influence of outlier measurements.
For residuals with `|r| <= delta` it behaves like least-squares; for
`|r| > delta` it switches to linear growth, limiting the pull of any
single bad probe.

| Delta | Effect | When to use |
|-------|--------|-------------|
| **Large (> 1.0)** | Effectively standard least-squares | Clean data, no outliers expected |
| **0.05–0.5** | Standard robust range | Real experimental data with occasional probe failures or calibration drift |
| **0.01–0.05** | Very aggressive outlier rejection | Known bad probes that can't be excluded, very noisy data |
| **< 0.01** | Nearly L1 loss | Extreme outliers; risk of slow convergence due to non-smooth gradient |

### How to tune

1. Run with standard least-squares first. Inspect the residual vector:
   if 1–2 probes have residuals 10× larger than the rest, those are
   outliers.
2. Set `delta` to roughly 2–3× the median residual magnitude. This
   lets "normal" probes contribute quadratically while capping outliers.
3. Combine with `sigma` weights for best results: sigma handles known
   calibration uncertainty, Huber handles unexpected outliers.

### Code example (Rust)

```rust
let config = InverseConfig {
    loss: LossFunction::Huber(0.1),
    sigma: Some(vec![0.01; n_probes]),  // 1% measurement uncertainty
    ..InverseConfig::default()
};
```

---

## 4. Measurement Weights (`sigma`)

Per-probe inverse-variance weighting: residuals are divided by sigma_i,
so noisier probes contribute less to the fit. This is standard
weighted least-squares.

| Approach | Description |
|----------|-------------|
| **Uniform (None)** | All probes weighted equally. Use when probe uncertainties are unknown or similar. |
| **Calibrated sigma** | Set sigma_i to the known measurement uncertainty of each probe (e.g. 0.5% for flux loops, 2% for B_pol pickups). |
| **SNR-based** | Set sigma_i = noise_floor / signal_amplitude for each probe. |

### Guidelines

- Flux loops are typically more accurate than B_pol pickups; give them
  smaller sigma (higher weight).
- If one probe is known to be malfunctioning, set its sigma to a large
  value (e.g. 1e6) rather than removing it — this preserves the
  Jacobian structure.
- Sigma length must exactly match the number of probes; a mismatch
  returns a `ConfigError`.

---

## 5. Grid Resolution vs Speed

| Grid | Python (NumPy) | Rust (release) | Rust (GPU projected) | Use case |
|------|---------------|----------------|---------------------|----------|
| 33×33 | ~0.8 s | ~2 ms | <1 ms | Quick scoping, parameter sweeps |
| 65×65 | ~5 s | ~15 ms | ~2 ms | Standard production, EFIT-comparable |
| 128×128 | ~30 s | ~95 ms | ~10 ms | High-fidelity, publication figures |
| 256×256 | ~5 min | ~10 s | ~50 ms | Research, pedestal/X-point resolution |

### Recommendations

- **Design scans** (hundreds of configurations): use 33×33 with Rust.
- **Single-shot analysis**: use 65×65, which balances speed and accuracy.
- **Validation against GEQDSK**: use 65×65 or 128×128 to match the
  reference grid.
- **Pedestal studies**: 128×128 minimum; the mtanh pedestal gradient
  requires adequate resolution near psi_norm ~ 0.9.

---

## 6. Inverse Solver: Jacobian Parallelism

As of the latest release, the Rust inverse solver computes the 8 Jacobian
finite-difference columns in parallel using `rayon`. Each column runs an
independent forward solve on a cloned kernel, so the wall time for one LM
iteration drops from ~0.8 s to ~0.15 s on an 8-core machine.

No configuration is needed — parallelism is automatic. To control the
thread count, set the `RAYON_NUM_THREADS` environment variable:

```bash
# Limit to 4 threads (useful on shared machines)
RAYON_NUM_THREADS=4 cargo run --release
```

### Expected speedup

| Cores | Speedup (8 columns) | Notes |
|-------|---------------------|-------|
| 1 | 1× (serial fallback) | Same as before |
| 4 | ~3.5× | Some overhead from cloning |
| 8 | ~5–6× | Matches column count |
| 16 | ~5–6× | No benefit beyond 8 columns |

---

## 7. Neural Transport: Weight Selection

The `NeuralTransportModel` accepts a `weights_path` pointing to an `.npz`
file. When no path is given, it falls back to the analytic
critical-gradient model.

### Weight file requirements

- Must contain arrays: `w1`, `b1`, `w2`, `b2`, `w3`, `b3`,
  `input_mean`, `input_std`, `output_scale`, `version`.
- `version` must equal 1 (checked on load).
- A SHA-256 checksum is computed and logged for reproducibility.

### Training guidance

See [`docs/NEURAL_TRANSPORT_TRAINING.md`](NEURAL_TRANSPORT_TRAINING.md)
for the full training pipeline. Key choices:

| Parameter | Recommended | Range |
|-----------|-------------|-------|
| Hidden layer 1 | 64 | 32–128 |
| Hidden layer 2 | 32 | 16–64 |
| Activation | tanh | tanh or ReLU |
| Training data | QLKNN-10D dataset | 10⁵–10⁶ samples |
| Learning rate | 1e-3 (Adam) | 1e-4 to 1e-2 |
| Epochs | 100–500 | Until validation loss plateaus |

Larger networks (H=128/64) give marginal accuracy gains but double
inference time from ~5 µs to ~10 µs per point.

---

## Quick Reference Card

| Parameter | Location | Default | Safe range |
|-----------|----------|---------|------------|
| `relaxation_factor` | `solver` JSON | 0.1 | 0.02–0.25 |
| `tikhonov` | `InverseConfig` | 0.0 | 0–1.0 |
| `loss` (Huber delta) | `InverseConfig` | LeastSquares | 0.01–1.0 |
| `sigma` | `InverseConfig` | None (uniform) | Per-probe float |
| `fd_step` | `InverseConfig` | 1e-4 | 1e-5–1e-3 |
| `max_iterations` | `solver` JSON | 1000 | 100–5000 |
| `convergence_threshold` | `solver` JSON | 1e-4 | 1e-8–1e-3 |
| Grid resolution | `grid_resolution` JSON | [65, 65] | [33, 33]–[256, 256] |
| MLP hidden dims | Weight file | 64/32 | 32–128 / 16–64 |
