# Benchmark Figures — Static Export

Static tables and figure descriptions from
[`examples/06_inverse_and_transport_benchmarks.ipynb`](../examples/06_inverse_and_transport_benchmarks.ipynb)
for inclusion in PDF/LaTeX/arXiv manuscripts. All values are representative;
run the notebook for live measurements on your hardware.

---

## Figure 1: Forward Solve Scaling (Bar Chart)

```
Forward Solve Time (ms)           Est. LM Iteration (ms)
                                  (8 forward solves)
 ┌──────────────────────┐         ┌──────────────────────┐
 │                      │         │                      │
 │                 ████ │         │                █████ │
 │                 ████ │         │                █████ │
 │          ████   ████ │         │         ████   █████ │
 │          ████   ████ │         │         ████   █████ │
 │   ████   ████   ████ │         │   ████   ████   █████ │
 │   ████   ████   ████ │         │   ████   ████   █████ │
 └──────────────────────┘         └──────────────────────┘
   33×33   49×49   65×65            33×33   49×49   65×65
```

**Data (Python NumPy backend):**

| Grid | Forward Solve | 1 LM Iteration (8×) |
|------|--------------|---------------------|
| 33×33 | ~0.8 s | ~6.4 s |
| 49×49 | ~2.5 s | ~20 s |
| 65×65 | ~5 s | ~40 s |

**Data (Rust release backend):**

| Grid | Forward Solve | 1 LM Iteration (8×) |
|------|--------------|---------------------|
| 33×33 | 2 ms | 16 ms |
| 65×65 | 100 ms | 800 ms |
| 128×128 | 950 ms | 7.6 s |

---

## Table 1: Inverse Reconstruction Config Variants

| Configuration | Overhead per LM iter | Notes |
|---------------|---------------------|-------|
| Default (LS) | 8 forward solves + Cholesky | baseline |
| + Tikhonov (α=0.1) | same + N_PARAMS additions | negligible overhead |
| + Huber (δ=0.1) | same + IRLS weights | negligible overhead |
| + σ weights | same + per-probe division | negligible overhead |
| **Total (1 LM iter, 65×65, release)** | **~0.8 s** | dominated by forward solve |
| **Full reconstruction (5 iters)** | **~4 s** | competitive with EFIT |

---

## Table 2: EFIT Comparison

| Metric | SCPN Fusion Core (Rust) | EFIT (Fortran) |
|--------|------------------------|----------------|
| Forward solve (65×65) | ~100 ms | ~50 ms |
| 1 LM iteration | ~0.8 s | ~0.4 s (Picard) |
| Full reconstruction | ~4 s (5 LM iters) | ~2 s (converged) |
| Regularisation | Tikhonov + Huber + σ | Von-Hagenow smoothing |
| Profile model | mtanh (7 params) | Spline knots (~20 params) |

*Lao, L.L. et al. (1985). "Reconstruction of current profile parameters
and plasma shapes in tokamaks." Nucl. Fusion 25, 1611.*

---

## Table 3: Neural Transport Surrogate Performance

| Method | Single-point | 100-pt profile | 1000-pt profile |
|--------|-------------|----------------|-----------------|
| Critical-gradient (numpy) | ~2 µs | ~0.2 ms | ~2 ms |
| MLP surrogate (numpy, H=64/32) | ~5 µs | ~0.05 ms | ~0.3 ms |
| QuaLiKiz (gyrokinetic) | ~1 s | ~100 s | ~1000 s |
| QLKNN (TensorFlow) | ~10 µs | ~0.1 ms | ~1 ms |

---

## Figure 2: Ion Thermal Diffusivity — Fallback vs MLP (Line Plot)

```
chi_i [m²/s]
  ^
  │                                  ╱ Fallback (analytic)
  │                                ╱
  │                              ╱
  │                            ╱    --- MLP (trained)
  │                          ╱   ---
  │                        ╱  ---
  │                      ╱---
  │                   ╱--
  │                ╱--
  │              ╱-
  │           ╱-
  │·········╱·······   ITG threshold
  │        |
  │        |
  │--------+----------------------------> R/L_Ti
  0    2   4   6   8  10  12  14  16  18  20
```

Both models show zero transport below the ITG critical gradient
(R/L_Ti < 4). Above threshold, the critical-gradient model follows a
power-law (stiffness exponent = 2), while the trained MLP reproduces
the gyrokinetic-level response including saturation effects.

---

## Table 4: Vectorised vs Point-by-Point Speedup

| Evaluation Strategy | 1000-pt profile | Relative |
|--------------------|-----------------|----------|
| MLP vectorised (`predict_profile`) | ~0.3 ms | 1× (baseline) |
| MLP point-by-point loop | ~30 ms | ~100× slower |
| Fallback vectorised | ~2 ms | ~7× slower |
| Fallback point-by-point loop | ~200 ms | ~670× slower |

---

## LaTeX Snippet

For including Table 3 in a manuscript:

```latex
\begin{table}[htbp]
\centering
\caption{Neural transport surrogate inference latency comparison.}
\label{tab:transport-benchmark}
\begin{tabular}{lccc}
\toprule
Method & Single-point & 100-pt profile & 1000-pt profile \\
\midrule
Critical-gradient (NumPy) & $\sim$2\,\textmu s & $\sim$0.2\,ms & $\sim$2\,ms \\
MLP surrogate (NumPy, H=64) & $\sim$5\,\textmu s & $\sim$0.05\,ms & $\sim$0.3\,ms \\
QuaLiKiz (gyrokinetic)~\cite{Citrin2015} & $\sim$1\,s & $\sim$100\,s & $\sim$1000\,s \\
QLKNN (TensorFlow)~\cite{vanDePlassche2020} & $\sim$10\,\textmu s & $\sim$0.1\,ms & $\sim$1\,ms \\
\bottomrule
\end{tabular}
\end{table}
```

For including Table 2 in a manuscript:

```latex
\begin{table}[htbp]
\centering
\caption{Inverse equilibrium reconstruction: SCPN Fusion Core vs EFIT.}
\label{tab:inverse-benchmark}
\begin{tabular}{lcc}
\toprule
Metric & SCPN Fusion Core (Rust) & EFIT~\cite{Lao1985} \\
\midrule
Forward solve (65$\times$65) & $\sim$100\,ms & $\sim$50\,ms \\
1 LM iteration & $\sim$0.8\,s & $\sim$0.4\,s \\
Full reconstruction & $\sim$4\,s & $\sim$2\,s \\
Regularisation & Tikhonov + Huber + $\sigma$ & Von-Hagenow \\
Profile model & mtanh (7 params) & Spline ($\sim$20 params) \\
\bottomrule
\end{tabular}
\end{table}
```

---

*Generated from notebook 06. See [`docs/BENCHMARKS.md`](BENCHMARKS.md) for
the full benchmark comparison tables.*
