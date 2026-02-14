# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
"""Verify SC-NeuroCore integration for Petri net -> SNN compilation.

Tests both the Rust-accelerated path (scpn_fusion_rs) and the
NumPy fallback path used when the native extension is unavailable.

Two import families are tested:

1. **scpn_fusion_rs** (Rust/PyO3 backend for the controller):
   - ``scpn_dense_activations(W_in, marking) -> activations``
   - ``scpn_marking_update(marking, W_in, W_out, firing) -> new_marking``
   - ``scpn_sample_firing(p_fire, n_passes, seed, antithetic) -> mean_firing``

2. **sc_neurocore** (Python package for the compiler):
   - ``StochasticLIFNeuron`` -- threshold comparator for Petri transitions
   - ``generate_bernoulli_bitstream`` + ``pack_bitstream`` / ``vec_and`` /
     ``vec_popcount`` -- stochastic bitstream arithmetic

When neither is available the script falls back to pure NumPy equivalents
to prove the math stays correct without native extensions.

Usage::

    python validation/test_sc_neurocore_integration.py
    # or
    pytest validation/test_sc_neurocore_integration.py -v
"""

from __future__ import annotations

import math
import sys
import time
from typing import Optional

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# ── Probe availability of each native backend ────────────────────────────────

# 1. Rust backend (scpn_fusion_rs) — controller path
HAS_RUST_BACKEND = False
_rust_dense_activations = None
_rust_marking_update = None
_rust_sample_firing = None

try:
    from scpn_fusion_rs import (  # type: ignore[import-not-found]
        scpn_dense_activations as _rust_dense_activations,
        scpn_marking_update as _rust_marking_update,
        scpn_sample_firing as _rust_sample_firing,
    )
    HAS_RUST_BACKEND = True
    print("scpn_fusion_rs (Rust backend) available")
except ImportError:
    print("scpn_fusion_rs (Rust backend) NOT available")

# 2. sc_neurocore Python package — compiler path
HAS_SC_NEUROCORE = False
try:
    from sc_neurocore import StochasticLIFNeuron  # type: ignore[import-not-found]
    from sc_neurocore import generate_bernoulli_bitstream  # type: ignore[import-not-found]
    from sc_neurocore import RNG as _SC_RNG  # type: ignore[import-not-found]
    from sc_neurocore.accel.vector_ops import (  # type: ignore[import-not-found]
        pack_bitstream,
        vec_and,
        vec_popcount,
    )
    HAS_SC_NEUROCORE = True
    print("sc_neurocore (Python package) available")
except ImportError:
    print("sc_neurocore (Python package) NOT available — testing NumPy fallback only")


# ═════════════════════════════════════════════════════════════════════════════
#  Helper: build a reference vertical-control Petri net (8 places, 7 transitions)
# ═════════════════════════════════════════════════════════════════════════════

def _build_control_matrices(
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Build W_in (nT, nP), W_out (nP, nT), marking, thresholds.

    Dimensions: 8 places, 7 transitions — matching a small vertical-control
    Petri net topology.
    """
    nP, nT = 8, 7
    W_in = (rng.random((nT, nP)) * 0.5).astype(np.float64)
    W_out = (rng.random((nP, nT)) * 0.5).astype(np.float64)
    marking = np.zeros(nP, dtype=np.float64)
    marking[0] = 1.0  # single token at first place
    thresholds = np.full(nT, 0.3, dtype=np.float64)
    return W_in, W_out, marking, thresholds


# ═════════════════════════════════════════════════════════════════════════════
#  NumPy reference implementations (matching controller.py fallback)
# ═════════════════════════════════════════════════════════════════════════════

def _numpy_dense_activations(W_in: FloatArray, marking: FloatArray) -> FloatArray:
    """NumPy reference: activations = W_in @ marking."""
    return np.asarray(W_in @ marking, dtype=np.float64)


def _numpy_marking_update(
    marking: FloatArray,
    W_in: FloatArray,
    W_out: FloatArray,
    firing: FloatArray,
) -> FloatArray:
    """NumPy reference: m' = clip(m - W_in^T @ f + W_out @ f, 0, 1)."""
    consumption = W_in.T @ firing
    production = W_out @ firing
    return np.clip(marking - consumption + production, 0.0, 1.0).astype(np.float64)


def _numpy_sample_firing(
    p_fire: FloatArray,
    n_passes: int,
    seed: int,
    antithetic: bool,
) -> FloatArray:
    """NumPy reference for stochastic firing (antithetic Bernoulli sampling)."""
    rng = np.random.default_rng(seed)
    nT = len(p_fire)
    counts = np.zeros(nT, dtype=np.int64)

    if antithetic and n_passes >= 2:
        n_pairs = (n_passes + 1) // 2
        base = rng.random((n_pairs, nT))
        low_hits = np.sum(base < p_fire[np.newaxis, :], axis=0, dtype=np.int64)
        if n_passes % 2 == 0:
            high_hits = np.sum(
                base > (1.0 - p_fire)[np.newaxis, :], axis=0, dtype=np.int64
            )
        else:
            high_hits = np.sum(
                base[:-1, :] > (1.0 - p_fire)[np.newaxis, :], axis=0, dtype=np.int64
            )
        counts[:] = low_hits + high_hits
    else:
        counts[:] = rng.binomial(n_passes, p_fire, size=nT).astype(np.int64)

    return counts.astype(np.float64) / float(max(n_passes, 1))


# ═════════════════════════════════════════════════════════════════════════════
#  Test functions
# ═════════════════════════════════════════════════════════════════════════════

def test_rust_dense_activations():
    """Test scpn_dense_activations matches NumPy reference."""
    if not HAS_RUST_BACKEND:
        print("SKIP: scpn_fusion_rs not available")
        return

    rng = np.random.default_rng(42)
    W_in, _, marking, _ = _build_control_matrices(rng)

    rust_out = np.asarray(_rust_dense_activations(W_in, marking), dtype=np.float64)
    numpy_out = _numpy_dense_activations(W_in, marking)

    assert rust_out.shape == numpy_out.shape, (
        f"Shape mismatch: rust={rust_out.shape} vs numpy={numpy_out.shape}"
    )
    np.testing.assert_allclose(rust_out, numpy_out, atol=1e-12, rtol=1e-12)
    print(f"  activations (nT=7): {rust_out}")
    print("PASS: Rust dense_activations matches NumPy")


def test_rust_marking_update():
    """Test scpn_marking_update matches NumPy reference."""
    if not HAS_RUST_BACKEND:
        print("SKIP: scpn_fusion_rs not available")
        return

    rng = np.random.default_rng(42)
    W_in, W_out, marking, thresholds = _build_control_matrices(rng)

    # Compute activations and binary firing
    activations = _numpy_dense_activations(W_in, marking)
    firing = (activations >= thresholds).astype(np.float64)

    rust_out = np.asarray(
        _rust_marking_update(marking, W_in, W_out, firing), dtype=np.float64
    )
    numpy_out = _numpy_marking_update(marking, W_in, W_out, firing)

    assert rust_out.shape == numpy_out.shape
    np.testing.assert_allclose(rust_out, numpy_out, atol=1e-12, rtol=1e-12)
    assert np.all(rust_out >= 0.0) and np.all(rust_out <= 1.0), (
        "Marking out of [0, 1] range"
    )
    print(f"  marking update: {rust_out}")
    print("PASS: Rust marking_update matches NumPy")


def test_rust_sample_firing():
    """Test scpn_sample_firing produces values in [0, 1]."""
    if not HAS_RUST_BACKEND:
        print("SKIP: scpn_fusion_rs not available")
        return

    p_fire = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], dtype=np.float64)
    n_passes = 1024
    seed = 42

    # Non-antithetic
    out_plain = np.asarray(
        _rust_sample_firing(p_fire, n_passes, seed, False), dtype=np.float64
    )
    assert out_plain.shape == (7,), f"Expected (7,), got {out_plain.shape}"
    assert np.all(out_plain >= 0.0) and np.all(out_plain <= 1.0)
    # With enough passes, output should be close to input probabilities
    np.testing.assert_allclose(out_plain, p_fire, atol=0.08)

    # Antithetic
    out_anti = np.asarray(
        _rust_sample_firing(p_fire, n_passes, seed, True), dtype=np.float64
    )
    assert np.all(out_anti >= 0.0) and np.all(out_anti <= 1.0)
    np.testing.assert_allclose(out_anti, p_fire, atol=0.08)

    print(f"  p_fire:     {p_fire}")
    print(f"  plain est:  {out_plain}")
    print(f"  antithetic: {out_anti}")
    print("PASS: Rust sample_firing")


def test_numpy_fallback_dense_forward():
    """Test NumPy-based dense activation (fallback when Rust unavailable)."""
    rng = np.random.default_rng(42)
    W_in, _, marking, _ = _build_control_matrices(rng)

    output = _numpy_dense_activations(W_in, marking)
    assert output.shape == (7,), f"Expected (7,), got {output.shape}"
    # With marking=[1,0,...,0] and W_in random, output = W_in[:,0]
    np.testing.assert_allclose(output, W_in[:, 0], atol=1e-14)
    print(f"  NumPy dense: input={marking}, output={output}")
    print("PASS: NumPy fallback dense forward")


def test_numpy_fallback_marking_update():
    """Test NumPy-based marking update (fallback when Rust unavailable)."""
    rng = np.random.default_rng(42)
    W_in, W_out, marking, thresholds = _build_control_matrices(rng)

    activations = _numpy_dense_activations(W_in, marking)
    firing = (activations >= thresholds).astype(np.float64)
    new_marking = _numpy_marking_update(marking, W_in, W_out, firing)

    assert new_marking.shape == (8,), f"Expected (8,), got {new_marking.shape}"
    assert np.all(new_marking >= 0.0) and np.all(new_marking <= 1.0), (
        "Marking out of [0, 1] range"
    )
    print(f"  marking before: {marking}")
    print(f"  marking after:  {new_marking}")
    print("PASS: NumPy fallback marking update")


def test_numpy_fallback_sample_firing():
    """Test NumPy-based stochastic firing estimation."""
    p_fire = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.1, 0.9], dtype=np.float64)

    for antithetic in (False, True):
        est = _numpy_sample_firing(p_fire, n_passes=2048, seed=42, antithetic=antithetic)
        assert est.shape == (7,)
        assert np.all(est >= 0.0) and np.all(est <= 1.0)
        np.testing.assert_allclose(est, p_fire, atol=0.06)

    print("PASS: NumPy fallback sample_firing")


def test_marking_bounds_200_steps():
    """Marking stays in [0, 1] over 200 steps with varying input."""
    rng = np.random.default_rng(42)
    W_in, W_out, marking, thresholds = _build_control_matrices(rng)

    for k in range(200):
        # Inject sinusoidal input into place 0
        marking[0] = np.clip(0.5 + 0.5 * math.sin(0.1 * k), 0.0, 1.0)
        activations = _numpy_dense_activations(W_in, marking)
        firing = (activations >= thresholds).astype(np.float64)
        marking = _numpy_marking_update(marking, W_in, W_out, firing)
        assert np.all(marking >= 0.0) and np.all(marking <= 1.0), (
            f"Marking out of [0,1] at step k={k}: {marking}"
        )

    print(f"  Final marking after 200 steps: {marking}")
    print("PASS: marking_bounds_200_steps")


def test_sc_neurocore_lif_neuron():
    """Test StochasticLIFNeuron from sc_neurocore (compiler path)."""
    if not HAS_SC_NEUROCORE:
        print("SKIP: sc_neurocore not available")
        return

    # Create a neuron with threshold matching Petri net transition
    neuron = StochasticLIFNeuron(
        v_rest=0.0,
        v_reset=0.0,
        v_threshold=0.5,
        tau_mem=1e6,  # near-instant pass-through
        dt=1.0,
        noise_std=0.0,
        resistance=1.0,
        refractory_period=0,
        seed=42,
    )

    # Sub-threshold input -> no fire
    neuron.reset_state()
    fired = neuron.step(0.3)
    assert fired == 0.0, f"Expected no fire for sub-threshold input, got {fired}"

    # Supra-threshold input -> fire
    neuron.reset_state()
    fired = neuron.step(0.8)
    assert fired == 1.0, f"Expected fire for supra-threshold input, got {fired}"

    print("PASS: StochasticLIFNeuron threshold behaviour")


def test_sc_neurocore_bitstream_encode():
    """Test Bernoulli bitstream encoding from sc_neurocore."""
    if not HAS_SC_NEUROCORE:
        print("SKIP: sc_neurocore not available")
        return

    L = 4096
    for p_target in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        rng = _SC_RNG(42)
        bits = generate_bernoulli_bitstream(p_target, L, rng=rng)
        packed = pack_bitstream(bits)
        p_est = int(vec_popcount(packed)) / L
        tol = 3.0 / math.sqrt(L)
        assert abs(p_est - p_target) < tol, (
            f"p={p_target}, est={p_est}, tol={tol:.4f}"
        )
        print(f"  p={p_target:.2f}  est={p_est:.4f}  tol={tol:.4f}")

    print("PASS: Bernoulli bitstream encode accuracy")


def test_sc_neurocore_and_product():
    """Test AND+popcount product: E[AND(w,p)] ~ w*p."""
    if not HAS_SC_NEUROCORE:
        print("SKIP: sc_neurocore not available")
        return

    L = 4096
    for w, p in [(0.5, 0.5), (0.8, 0.3), (1.0, 0.7), (0.0, 0.5)]:
        rng_w = _SC_RNG(100)
        rng_p = _SC_RNG(200)
        bits_w = generate_bernoulli_bitstream(w, L, rng=rng_w)
        bits_p = generate_bernoulli_bitstream(p, L, rng=rng_p)
        anded = vec_and(pack_bitstream(bits_w), pack_bitstream(bits_p))
        est = int(vec_popcount(anded)) / L
        tol = 3.0 / math.sqrt(L)
        assert abs(est - w * p) < tol, (
            f"w={w}, p={p}, est={est}, expected={w*p:.4f}, tol={tol:.4f}"
        )
        print(f"  w={w:.1f} * p={p:.1f} = {w*p:.4f}  est={est:.4f}")

    print("PASS: AND+popcount product accuracy")


def test_full_petri_step_rust_vs_numpy():
    """Full Petri net step: Rust path must match NumPy path exactly."""
    if not HAS_RUST_BACKEND:
        print("SKIP: scpn_fusion_rs not available")
        return

    rng = np.random.default_rng(42)
    W_in, W_out, marking, thresholds = _build_control_matrices(rng)

    for k in range(50):
        marking[0] = np.clip(0.5 + 0.4 * math.sin(0.08 * k), 0.0, 1.0)

        # Rust path
        rust_act = np.asarray(_rust_dense_activations(W_in, marking), dtype=np.float64)
        firing = (rust_act >= thresholds).astype(np.float64)
        rust_m = np.asarray(
            _rust_marking_update(marking, W_in, W_out, firing), dtype=np.float64
        )

        # NumPy path
        numpy_act = _numpy_dense_activations(W_in, marking)
        numpy_m = _numpy_marking_update(marking, W_in, W_out, firing)

        np.testing.assert_allclose(rust_act, numpy_act, atol=1e-12)
        np.testing.assert_allclose(rust_m, numpy_m, atol=1e-12)
        marking = rust_m

    print(f"  50-step Petri trace: final marking={marking}")
    print("PASS: Rust vs NumPy full Petri step equivalence")


def benchmark_latency() -> Optional[float]:
    """Measure per-forward-pass latency for the Rust backend."""
    if not HAS_RUST_BACKEND:
        print("SKIP: scpn_fusion_rs not available")
        return None

    rng = np.random.default_rng(42)
    W_in, W_out, marking, thresholds = _build_control_matrices(rng)

    # Warmup
    for _ in range(200):
        act = np.asarray(_rust_dense_activations(W_in, marking), dtype=np.float64)
        firing = (act >= thresholds).astype(np.float64)
        _ = _rust_marking_update(marking, W_in, W_out, firing)

    # Timed: full step = dense_activations + marking_update
    n_iter = 10_000
    t0 = time.perf_counter()
    for _ in range(n_iter):
        act = np.asarray(_rust_dense_activations(W_in, marking), dtype=np.float64)
        firing = (act >= thresholds).astype(np.float64)
        marking = np.asarray(
            _rust_marking_update(marking, W_in, W_out, firing), dtype=np.float64
        )
    t1 = time.perf_counter()

    us_per_step = (t1 - t0) / n_iter * 1e6
    print(f"  Rust full-step latency: {us_per_step:.1f} us/step ({n_iter} iterations)")
    return us_per_step


def benchmark_numpy_latency() -> float:
    """Measure per-forward-pass latency for the NumPy fallback."""
    rng = np.random.default_rng(42)
    W_in, W_out, marking, thresholds = _build_control_matrices(rng)

    # Warmup
    for _ in range(200):
        act = _numpy_dense_activations(W_in, marking)
        firing = (act >= thresholds).astype(np.float64)
        marking = _numpy_marking_update(marking, W_in, W_out, firing)

    # Timed
    rng2 = np.random.default_rng(42)
    _, _, marking, _ = _build_control_matrices(rng2)

    n_iter = 10_000
    t0 = time.perf_counter()
    for _ in range(n_iter):
        act = _numpy_dense_activations(W_in, marking)
        firing = (act >= thresholds).astype(np.float64)
        marking = _numpy_marking_update(marking, W_in, W_out, firing)
    t1 = time.perf_counter()

    us_per_step = (t1 - t0) / n_iter * 1e6
    print(f"  NumPy full-step latency: {us_per_step:.1f} us/step ({n_iter} iterations)")
    return us_per_step


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("SC-NeuroCore Integration Verification")
    print("=" * 70)

    # ── Rust backend tests (controller path) ──
    print("\n--- Rust backend (scpn_fusion_rs) ---")
    test_rust_dense_activations()
    test_rust_marking_update()
    test_rust_sample_firing()
    test_full_petri_step_rust_vs_numpy()

    # ── sc_neurocore tests (compiler path) ──
    print("\n--- sc_neurocore (compiler path) ---")
    test_sc_neurocore_lif_neuron()
    test_sc_neurocore_bitstream_encode()
    test_sc_neurocore_and_product()

    # ── NumPy fallback tests ──
    print("\n--- NumPy fallback ---")
    test_numpy_fallback_dense_forward()
    test_numpy_fallback_marking_update()
    test_numpy_fallback_sample_firing()
    test_marking_bounds_200_steps()

    # ── Benchmarks ──
    print("\n--- Benchmarks ---")
    rust_latency = benchmark_latency()
    numpy_latency = benchmark_numpy_latency()

    # ── Summary ──
    print("\n" + "=" * 70)
    parts = []
    if HAS_RUST_BACKEND:
        parts.append(f"Rust backend ({rust_latency:.1f} us/step)")
    if HAS_SC_NEUROCORE:
        parts.append("sc_neurocore compiler")
    parts.append(f"NumPy fallback ({numpy_latency:.1f} us/step)")

    if HAS_RUST_BACKEND and rust_latency is not None:
        speedup = numpy_latency / max(rust_latency, 0.001)
        print(f"Rust speedup: {speedup:.1f}x over NumPy")

    print(f"ALL TESTS PASSED: {', '.join(parts)}")
