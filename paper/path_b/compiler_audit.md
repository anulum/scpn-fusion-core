# SCPN Fusion Core -- Petri Net Compiler Pipeline Audit

**Date:** 2026-02-14
**Auditor:** Claude Opus 4.6 (automated code audit)
**Scope:** `src/scpn_fusion/scpn/` package (Packets A, B, C) + test files
**Files reviewed:**
- `structure.py` (Packet A -- Stochastic Petri Net definition)
- `compiler.py` (Packet B -- FusionCompiler + CompiledNet)
- `controller.py` (Packet C -- NeuroSymbolicController)
- `contracts.py` (Packet C -- data contracts, feature extraction, action decoding)
- `artifact.py` (Packet C -- `.scpnctl.json` artifact schema)
- `__init__.py` (public API surface)
- `tests/test_scpn_compiler.py` (Packets A+B unit tests)
- `tests/test_controller.py` (Packet C unit + integration tests)
- `tests/test_hypothesis_properties.py` (property-based tests)

---

## 1. Petri Net Types Supported

### Currently Supported: Stochastic Petri Net (SPN) with Continuous Tokens

The `StochasticPetriNet` class implements a **continuous-valued stochastic Petri net** --
not a classical Place/Transition (P/T) net with discrete integer tokens. Key
characteristics:

| Feature | Status | Details |
|---------|--------|---------|
| **Continuous tokens** | Supported | Token densities in [0.0, 1.0] (float64), not discrete integers |
| **Stochastic execution** | Supported | Via packed uint64 Bernoulli bitstream AND+popcount forward pass (requires `sc_neurocore`) |
| **Timed transitions** | Supported | `delay_ticks` parameter per transition (integer tick delays, circular buffer implementation) |
| **Inhibitor arcs** | Supported | Opt-in via `inhibitor=True` on `add_arc()` + `allow_inhibitor=True` on `compile()` |
| **Fractional firing** | Supported | `firing_mode="fractional"` yields graded [0,1] output via `(activation - threshold) / margin` |
| **Binary firing** | Supported | Default mode: `activation >= threshold` yields {0, 1} |
| **Colored tokens** | NOT supported | No token type/colour distinction |
| **Priority transitions** | NOT supported | No priority ordering among enabled transitions |
| **Hierarchical nets** | NOT supported | No subnet/module abstraction |
| **Guard functions** | NOT supported | No arbitrary boolean guards on transitions |

### Token Semantics

Tokens are **fractional densities** clamped to [0, 1], not discrete counts. This is
a deliberate design choice for stochastic computing compatibility: each token density
maps directly to a Bernoulli probability for bitstream encoding.

---

## 2. Compilation: Places/Transitions to Matrices

### Matrix Layout

The `FusionCompiler.compile()` method transforms a `StochasticPetriNet` graph into
two dense weight matrices:

```
W_in  : shape (nT, nP) -- input arc weights
         W_in[t, p] = weight of arc from place p to transition t
         Positive values = normal input arcs
         Negative values = inhibitor arcs (when allow_inhibitor=True)

W_out : shape (nP, nT) -- output arc weights
         W_out[p, t] = weight of arc from transition t to place p
         Always non-negative
```

### Compilation Pipeline

1. **Sparse COO assembly** (`structure.py:compile()`):
   - Iterates arc list `(source, target, weight, inhibitor_flag)`
   - Place->Transition arcs populate `W_in[transition_idx, place_idx]`
   - Transition->Place arcs populate `W_out[place_idx, transition_idx]`
   - Assembled as `scipy.sparse.csr_matrix`

2. **Dense extraction** (`compiler.py:FusionCompiler.compile()`):
   - Converts sparse W_in, W_out to dense float64 numpy arrays
   - Extracts threshold, delay_ticks, and initial_marking vectors

3. **LIF neuron creation** (when `sc_neurocore` is available):
   - One `StochasticLIFNeuron` per transition
   - `v_threshold` = transition threshold
   - `v_rest` = `v_reset` = 0.0 (pure comparator mode)
   - Configurable `tau_mem`, `noise_std`, `dt`, `resistance`, `refractory_period`

4. **Bitstream packing** (when `sc_neurocore` is available):
   - Each W_in and W_out element encoded as a packed uint64 Bernoulli bitstream
   - Shape: `(rows, cols, n_words)` where `n_words = ceil(bitstream_length / 64)`

---

## 3. W_in and W_out Matrix Encoding

### Float Path (always available)

```
W_in  (nT x nP) float64:
  - Each entry w in [0, 1] for normal arcs
  - Entry w in [-1, 0) for inhibitor arcs
  - Row i = input weights for transition i across all places

W_out (nP x nT) float64:
  - Each entry w in [0, 1]
  - Row p = output weights into place p from all transitions
```

### Packed Bitstream Path (requires sc_neurocore)

```
W_in_packed  (nT, nP, n_words) uint64:
W_out_packed (nP, nT, n_words) uint64:
  - n_words = ceil(bitstream_length / 64)
  - Each (i, j, :) slice is a packed Bernoulli bitstream where
    P(bit=1) = |W[i,j]|
  - Generated deterministically via numpy default_rng(seed)
  - Bit packing: bits shifted into uint64 words using np.left_shift
```

### Forward Pass: Stochastic Matrix-Vector Product

```python
# For each output row i:
#   1. Encode input_probs[j] as packed bitstream
#   2. AND with W_packed[i, j, :]
#   3. popcount the AND result
#   4. output[i] = total_ones / bitstream_length
```

This computes an unbiased estimator of `W @ x` via stochastic computing,
where the AND gate implements multiplication and popcount implements summation.

### Artifact Serialization

Weight matrices serialize to `.scpnctl.json` as:
- `weights.w_in.shape`: `[nT, nP]`
- `weights.w_in.data`: row-major float list
- `weights.packed.w_in_packed.data_u64`: flat uint64 list or zlib-compressed base64 (`u64-le-zlib-base64` encoding)

---

## 4. Stochastic Firing in the Controller

### Yes -- the controller supports stochastic firing via two mechanisms:

#### Mechanism 1: Multi-pass Bernoulli Sampling (`sc_n_passes`)

In `_sc_step()`:
- Firing probabilities `p_fire` are computed from activations
- If `sc_n_passes > 1` and firing is not deterministic binary:
  - Each transition is sampled `sc_n_passes` times
  - Final firing = count / sc_n_passes (fractional average)
- Supports **antithetic variates** (`sc_antithetic=True`):
  - Generates `n_pairs = (sc_n_passes + 1) // 2` base samples
  - Uses both `U < p` and `U > 1-p` (variance reduction)
  - Chunked processing via `sc_antithetic_chunk_size`

#### Mechanism 2: Bit-flip Fault Injection (`sc_bitflip_rate`)

In `_apply_bit_flip_faults()`:
- With probability `sc_bitflip_rate`, each float64 value has a random mantissa bit flipped
- Bits 0-51 (mantissa bits only) are targeted
- Result is NaN-cleaned and clamped to [0, 1]
- Applied to both firing vectors and marking vectors

#### Mechanism 3: Binary Probabilistic Margin (`sc_binary_margin`)

For binary firing mode with `sc_binary_margin > 0`:
- Instead of hard threshold: `p_fire = clip(0.5 + 0.5 * (activation - threshold) / margin, 0, 1)`
- This creates a soft probabilistic zone around the threshold
- Default in "adaptive" profile: `sc_binary_margin = 0.05`
- Default in "traceable" profile: `sc_binary_margin = 0.0` (exact threshold)

#### Determinism Guarantees

All stochastic paths are **deterministic** given the same `seed_base` and step index `k`:
- Seeds derived via `sha256(seed_base:sc_step:{k})` truncated to 64 bits
- Verified by test: same controller replayed twice produces identical outputs

---

## 5. Formal Analyses in `structure.py`

### `validate_topology()` provides four diagnostic checks:

| Analysis | Method | Formal Property |
|----------|--------|-----------------|
| **Dead place detection** | Degree counting | Places with zero in-degree AND zero out-degree (isolated nodes) |
| **Dead transition detection** | Degree counting | Transitions with zero in-degree AND zero out-degree |
| **Unseeded place cycle detection** | Tarjan SCC algorithm | Strongly connected components in the place-reachability graph where all places have zero initial tokens (will never fire) |
| **Input weight overflow detection** | Sum of positive input arc weights per transition > 1.0 | Indicates potential marking overflow since tokens are bounded to [0, 1] |

### What is NOT supported:

| Formal Property | Status | Notes |
|-----------------|--------|-------|
| **Reachability analysis** | NOT implemented | No coverability tree / state space enumeration |
| **Liveness** | NOT implemented (partially approximated) | Dead-node detection is a weak proxy; full liveness requires reachability |
| **Boundedness** | NOT implemented (implicit) | Tokens are hard-clamped to [0, 1] -- the net is trivially 1-bounded by construction, but this is enforcement, not verification |
| **S-invariants / T-invariants** | NOT implemented | No incidence matrix analysis for place/transition invariants |
| **Fairness** | NOT implemented | No analysis of transition firing fairness |
| **Mutual exclusion** | NOT implemented | No structural analysis of mutex place pairs |
| **Siphons / Traps** | NOT implemented | No structural deadlock analysis |

### Topology Validation Modes

- `compile(validate_topology=True)`: runs diagnostics, stores report, but does not block compilation
- `compile(strict_validation=True)`: runs diagnostics AND raises `ValueError` if any issues found

---

## 6. What `contracts.py` Verifies

### Feature Extraction (`extract_features()`)

- **Observation key existence**: raises `KeyError` if required observation keys are missing
- **Finiteness**: raises `ValueError` if observation values, targets, or scales are non-finite (NaN/Inf)
- **Scale safety**: replaces near-zero scales with 1e-12 to prevent division by zero
- **Output clamping**: all output features clamped to [0, 1] via `_clip01()`
- **Error normalization**: signed error = `(target - obs) / scale`, clamped to [-1, 1]
- **Unipolar decomposition**: positive error -> `pos_key`, negative error -> `neg_key`

### Action Decoding (`decode_actions()`)

- **Vector length consistency**: all input lists must have equal length
- **`dt` validity**: must be finite and > 0
- **Place index bounds**: must be >= 0 and < len(marking)
- **Slew-rate limiting**: `|action[k] - action[k-1]| <= slew_per_s * dt`
- **Absolute saturation**: `|action| <= abs_max`

### Artifact Validation (`artifact.py:_validate()`)

Comprehensive field-level validation on artifact load:

- `firing_mode` must be `"binary"` or `"fractional"`
- `fixed_point.data_width` must be integer >= 1
- `fixed_point.fraction_bits` must be integer >= 0 and < data_width
- `fixed_point.signed` must be boolean
- `stream_length` must be integer >= 1
- `dt_control_s` must be finite float > 0
- All `w_in` weights in [-1, 1] (allows negative for inhibitor arcs)
- All `w_out` weights in [0, 1]
- All thresholds in [0, 1], finite, not boolean-typed
- All margins >= 0, finite (if present)
- All `delay_ticks` integer >= 0
- `marking` length == nP, all values in [0, 1]
- Place injection `place_id` in bounds, `source` non-empty string, `scale`/`offset` finite
- Readout actions: `id` >= 0, `name` non-empty, `pos_place`/`neg_place` in bounds
- Readout vectors (`gains`, `abs_max`, `slew_per_s`) length == n_actions, all finite

---

## 7. SC-NeuroCore Integration Point

### Import Location

`compiler.py`, lines 40-51:

```python
try:
    from sc_neurocore import StochasticLIFNeuron
    from sc_neurocore import generate_bernoulli_bitstream
    from sc_neurocore import RNG as _SC_RNG
    from sc_neurocore.accel.vector_ops import pack_bitstream, vec_and, vec_popcount
    _HAS_SC_NEUROCORE = True
except ImportError:
    _HAS_SC_NEUROCORE = False
```

### Graceful Degradation

The entire pipeline works without `sc_neurocore`:
- **Float path only**: `CompiledNet.dense_forward_float()` uses plain `W @ inputs`
- **No LIF neurons**: binary threshold comparison via `(currents >= thresholds).astype(float64)`
- **No packed bitstreams**: `W_in_packed` and `W_out_packed` are `None`

When `sc_neurocore` IS available:
- One `StochasticLIFNeuron` per transition with configurable membrane dynamics
- Packed uint64 bitstream weight tensors for stochastic forward passes
- `generate_bernoulli_bitstream()` + `pack_bitstream()` for runtime input encoding
- `vec_and()` + `vec_popcount()` for stochastic matrix-vector products

### Rust Runtime Integration

`controller.py` also optionally imports from `scpn_fusion_rs` (Rust PyO3 extension):

```python
from scpn_fusion_rs import (
    scpn_dense_activations,
    scpn_marking_update,
    scpn_sample_firing,
)
```

Three Rust-accelerated kernels:
- `scpn_dense_activations(W_in, marking)` -- matrix-vector product
- `scpn_marking_update(marking, W_in, W_out, firing)` -- Petri net marking step
- `scpn_sample_firing(p_fire, n_passes, seed, antithetic)` -- stochastic sampling

Backend selection: `runtime_backend` = `"auto"` | `"numpy"` | `"rust"`

---

## 8. Bitstream Encoding Format

### Bernoulli Bitstream Packing

Each matrix weight `w` in [0, 1] is encoded as a packed uint64 bitstream:

```
Bitstream length: L (default 1024 bits)
Words per stream: n_words = ceil(L / 64)
Encoding:
  1. Generate L random floats U ~ Uniform(0, 1) via numpy default_rng(seed)
  2. Set bit[i] = 1 if U[i] < w, else 0
  3. Pad to n_words * 64 bits with zeros
  4. Pack into uint64 words: word[k] = sum(bit[k*64 + j] << j for j in range(64))
```

### Artifact Packed Serialization

Two formats supported in `.scpnctl.json`:

1. **Raw uint64 list**: `"data_u64": [0, 1234567890, ...]`
2. **Compact compressed**: `"encoding": "u64-le-zlib-base64"`, `"data_u64_b64_zlib": "<base64>"`
   - Little-endian uint64 bytes -> zlib compress (level 9) -> base64 encode
   - Bounded decompression: `MAX_DECOMPRESSED_BYTES = 80 MB`, `MAX_COMPRESSED_BYTES = 50 MB`

### Fixed-Point Metadata

The artifact carries fixed-point configuration (for potential hardware targets):
- `data_width`: 16 bits (default)
- `fraction_bits`: 10 (default)
- `signed`: False (default)

This metadata is declarative -- the current Python runtime uses float64 throughout.

### Seed Policy

```json
{
  "id": "default",
  "hash_fn": "splitmix64",
  "rng_family": "xoshiro256++"
}
```

This documents the intended RNG family for hardware implementations; the Python
runtime uses `numpy.random.default_rng()` (PCG64 by default).

---

## 9. Inhibitor Arcs and Priority Transitions

### Inhibitor Arcs: SUPPORTED

Inhibitor arcs are fully implemented:

1. **Declaration**: `net.add_arc("Place", "Transition", weight=w, inhibitor=True)`
   - Only valid for Place->Transition direction
   - Weight must be > 0 (magnitude); stored as `-abs(weight)` internally

2. **Compilation**: requires explicit opt-in
   - `net.compile(allow_inhibitor=True)`
   - `compiler.compile(net, allow_inhibitor=True)`
   - Without opt-in, raises `ValueError: "Negative input arc weights detected; re-run compile with allow_inhibitor=True."`

3. **Matrix encoding**: negative values in W_in
   - Normal arc: `W_in[t, p] = +weight`
   - Inhibitor arc: `W_in[t, p] = -weight`
   - Artifact validation allows W_in values in [-1, 1]

4. **Semantic effect**: the negative weight in the matrix-vector product `W_in @ marking` causes
   the activation to DECREASE when the inhibitor place has tokens. The transition fires
   only when activation (after the negative contribution) exceeds the threshold.

### Priority Transitions: NOT SUPPORTED

There is no priority ordering mechanism for transitions:
- No `priority` field on transitions
- No conflict resolution policy when multiple transitions are simultaneously enabled
- All enabled transitions fire in the same step (parallel firing semantics)
- The marking update `m' = clip(m - W_in^T @ f + W_out @ f, 0, 1)` applies all
  firings simultaneously, with post-hoc clamping to [0, 1]

---

## 10. Test Scenarios

### `test_scpn_compiler.py` -- Packets A+B (46 tests)

**Packet A -- StochasticPetriNet:**
- Place/transition counts, names, initial marking, thresholds
- W_in / W_out shape, sparsity, and exact values for a 3-place traffic-light net
- Compilation flag toggling (add_place invalidates)
- Validation: duplicate nodes, same-kind arcs, unknown nodes, empty net, token range, negative weights
- Transition delay_ticks persistence
- `validate_topology()`: dead node detection, unseeded place cycle detection, input weight overflow
- Strict validation mode rejecting problematic topologies
- Inhibitor arc compilation (opt-in, direction validation, negative weight encoding)

**Packet B -- FusionCompiler:**
- Strict topology rejection via compiler (dead nodes, weight overflow)
- Topology validation report population through compiler
- Inhibitor arc opt-in via compiler
- CompiledNet shapes, thresholds, initial marking
- LIF neuron count, thresholds, membrane parameters (when sc_neurocore available)
- Custom LIF runtime parameters (tau_mem, noise_std, dt, resistance, refractory_period)
- Reactor LIF factory defaults
- Traceable runtime kwargs
- Packed weight shapes and dtype (when sc_neurocore available)
- Float forward pass with known marking
- LIF fire above/below threshold
- Stochastic forward pass high-activation verification (when sc_neurocore available)
- Stochastic vs float path accuracy (atol=0.15)
- Git SHA resolution from environment

**Integration -- 10-step cyclic flow:**
- Traffic-light Red->Green->Yellow->Red cycle over 10 steps (exact marking verification)
- Token conservation (sum = 1.0) over 30 steps
- Bitstream length minimum validation
- Invalid LIF parameter rejection

### `test_controller.py` -- Packet C (55+ tests)

**Level 0 -- Static validation (18 tests):**
- Artifact load/save roundtrip
- Compact packed u64 codec roundtrip, determinism, and rejection of invalid payloads
- Weight, threshold, marking range validation
- Schema version, compiler version verification
- Invalid field rejection: delay_ticks, stream_length, dt_control_s, fixed_point metadata,
  transition thresholds/margins, readout vectors, action specs, place injections

**Level 1 -- Determinism (10 tests):**
- Same seed_base + k sequence -> identical actions (binary and fractional modes)
- Determinism after reset
- Binary threshold mode matching oracle when sc_binary_margin=0
- Deterministic profile matching oracle
- Adaptive profile introducing probabilistic margin (oracle != stochastic)
- Probabilistic binary margin determinism and non-oracle behavior
- Antithetic sampling determinism and variance reduction verification

**Level 2 -- Primitive correctness (2 tests, require sc_neurocore):**
- Encode mean accuracy: `E[popcount(Encode(p))/L] ~= p`
- AND product accuracy: `E[AND(w, p)] ~= w*p`

**Level 3 -- Petri semantics (5 tests):**
- Marking bounds [0, 1] over 200 steps with sinusoidal observations
- Fractional firing range [0, 1]
- Fractional CompiledNet.lif_fire values
- Binary firing unchanged ({0, 1} only)
- Timed transition deferral (delay_ticks=2 defers action by 2 ticks)
- Bit-flip path stays bounded

**Integration (17+ tests):**
- Sinusoidal disturbance bounded actions
- Step disturbance nonzero response
- Slew-rate limiting verification
- JSONL logging with expected keys
- Passthrough injection sources
- Custom feature axes
- Oracle diagnostics disable
- Runtime backend selection (rust fallback, auto threshold, rust preference)
- Marking property copy semantics
- Marking setter validation
- Traceable step matching mapping step
- Antithetic chunked sampling determinism
- Rust kernel execution verification (via monkeypatch)
- Rust sampling path verification (via monkeypatch)

**Contract helpers (10 tests):**
- Feature extraction on-target, positive error, negative error, clamped
- Custom axes, passthrough keys
- Non-finite value rejection
- Action decoding basic, vector length mismatch, invalid dt, bounds checking

### `test_hypothesis_properties.py` -- Property-Based Tests

- Random Petri net generation strategy (1-8 places, 1-8 transitions, random arcs)
- Topology invariants for randomly generated nets
- FusionCompiler matrix property verification

---

## 11. GAPS: Requirements for Vertical Position Control

For a complete vertical (Z-axis) position control system using the Petri net
controller pipeline, the following gaps must be addressed:

### 11.1 Net Topology for Vertical Control

**Current state:** The existing test fixture (`_build_controller_net()`) already includes
Z-axis places (`x_Z_pos`, `x_Z_neg`, `a_Z_pos`, `a_Z_neg`) and a corresponding
`dI_PF_topbot_A` readout action. The controller contracts define `Z_axis_m` observation,
`Z_target_m` setpoint, and `Z_scale_m` normalization.

**Gap:** The 8-place test net is a simple pass-through (one transition per input place).
A production vertical stability controller would need:
- **Cross-coupling transitions**: Z-axis errors should potentially influence R-axis actuators
  and vice versa (the current net has zero cross-talk)
- **Multi-input transitions**: transitions that combine multiple error signals
  (e.g., Z position error + Z velocity error + plasma current error)
- **Priority transitions for safety**: rapid VDE (Vertical Displacement Event) detection
  should preempt normal control -- requires **priority transitions** (not currently supported)

### 11.2 Missing Formal Analyses

- **Reachability analysis**: needed to prove the controller can reach the target marking
  from any initial condition within the operating envelope
- **Liveness**: needed to prove the controller never deadlocks (all transitions remain
  eventually firable)
- **S-invariants**: needed to identify conserved quantities (e.g., total coil current
  budget) that must be maintained as algebraic invariants of the net structure
- **T-invariants**: needed to identify cyclic behaviors that bring the marking back
  to a known state (relevant for periodic disturbance rejection)

### 11.3 Missing Controller Features

| Feature | Why Needed | Current Status |
|---------|-----------|----------------|
| **Priority transitions** | VDE avoidance must preempt normal position control | Not supported |
| **Guard functions** | Conditional firing based on plasma state (e.g., q95 < 2 triggers disruption avoidance) | Not supported |
| **PID-like integral action** | Steady-state error elimination for vertical position | Must be implemented via net topology (accumulator places); no built-in integrator |
| **Anti-windup** | Prevent integrator saturation when actuators are saturated | Must be modeled via net topology |
| **Multiple timescale transitions** | Fast VDE response (sub-ms) vs slow position trim (100ms+) | Partially supported via `delay_ticks`, but no continuous-time transition rates |
| **State estimation coupling** | Real-time magnetic equilibrium reconstruction feeding the controller | Supported via `FeatureAxisSpec` and passthrough injection, but no built-in observer |
| **Gain scheduling** | Different controller gains for different plasma configurations | Would require runtime modification of W_in/W_out weights; currently fixed after compilation |
| **Multi-actuator coordination** | Coordinating PF1-PF6 + CS coils, not just PF3 + PF_topbot | Requires larger net topology (more places/transitions/readouts) |

### 11.4 Missing Plant Model Integration

- **Vertical instability growth rate**: the controller must react faster than the
  characteristic growth time (~10 ms for typical tokamaks). The current `dt_control_s`
  must be chosen accordingly, but there is no formal stability margin analysis.
- **Actuator dynamics**: coil current response time, voltage limits, and mutual
  inductance between PF coils are not modeled in the Petri net. The `slew_per_s`
  limiting is a proxy but does not capture RL circuit dynamics.
- **Sensor noise and latency**: the controller assumes clean observations; real
  magnetic probes have noise, calibration drift, and ~100 us latency. The `sc_bitflip_rate`
  provides fault injection but not systematic noise modeling.

### 11.5 Missing Verification and Validation Infrastructure

- **Closed-loop simulation harness**: no built-in plant simulator for testing the
  controller in a feedback loop (only open-loop step/sinusoidal disturbance tests exist)
- **Stability margin computation**: no Bode/Nyquist analysis capability for the
  linearized closed-loop system
- **Monte Carlo robustness testing**: `test_hypothesis_properties.py` provides random
  net generation but no systematic parameter sweep or worst-case analysis
- **Hardware-in-the-loop (HIL) interface**: no mechanism to deploy the compiled
  artifact to FPGA/real-time hardware for HIL testing

### 11.6 Summary Priority List for Vertical Control

1. **HIGH** -- Add priority transitions for VDE safety response
2. **HIGH** -- Implement closed-loop simulation harness with a vertical stability plant model
3. **HIGH** -- Add reachability and liveness analysis to `structure.py`
4. **MEDIUM** -- Add S-invariant / T-invariant computation
5. **MEDIUM** -- Implement gain scheduling (runtime weight modification or net switching)
6. **MEDIUM** -- Add multi-coil readout actions (PF1-PF6 + CS)
7. **LOW** -- Add guard functions for conditional transition enabling
8. **LOW** -- Add Bode/Nyquist stability margin computation

---

## Appendix A: Public API Surface (`__init__.py`)

```python
__all__ = [
    "StochasticPetriNet",       # Packet A
    "FusionCompiler",           # Packet B
    "CompiledNet",              # Packet B
    "ControlObservation",       # Packet C
    "ControlAction",            # Packet C
    "ControlTargets",           # Packet C
    "ControlScales",            # Packet C
    "extract_features",         # Packet C
    "decode_actions",           # Packet C
    "Artifact",                 # Packet C
    "load_artifact",            # Packet C
    "save_artifact",            # Packet C
    "NeuroSymbolicController",  # Packet C
]
```

## Appendix B: Data Flow Summary

```
Observation dict {R_axis_m, Z_axis_m, ...}
    |
    v
extract_features() --> {x_R_pos, x_R_neg, x_Z_pos, x_Z_neg}
    |
    v
_inject_places() --> marking[injection_places] = features * scale + offset
    |
    v
_dense_activations(): a = W_in @ marking     (numpy or Rust)
    |
    v
_oracle_step() (float path)  |  _sc_step() (stochastic path)
    |                             |
    |  f = threshold(a)           |  f = sample(p_fire, n_passes)
    |                             |
    v                             v
_apply_transition_timing(): delay circular buffer
    |
    v
_marking_update(): m' = clip(m - W_in^T @ f + W_out @ f, 0, 1)
    |
    v
_decode_actions(): action = gain * (marking[pos] - marking[neg])
    |                   + slew limiting + abs saturation
    v
ControlAction {dI_PF3_A, dI_PF_topbot_A}
```

## Appendix C: Key Constants

| Constant | Value | Location |
|----------|-------|----------|
| `ARTIFACT_SCHEMA_VERSION` | `"1.0.0"` | `artifact.py` |
| `MAX_PACKED_WORDS` | 10,000,000 | `artifact.py` |
| `MAX_DECOMPRESSED_BYTES` | 80,000,000 | `artifact.py` |
| `MAX_COMPRESSED_BYTES` | 50,000,000 | `artifact.py` |
| Default `bitstream_length` | 1024 | `compiler.py` |
| Minimum `bitstream_length` | 64 | `compiler.py` |
| Default `lif_tau_mem` | 1e6 (no leak) | `compiler.py` |
| Reactor `lif_tau_mem` | 10.0 | `compiler.py` |
| Default `firing_margin` | 0.05 | `compiler.py` |
| Default `R_target_m` | 6.2 | `contracts.py` |
| Default `Z_target_m` | 0.0 | `contracts.py` |
| Default `R_scale_m` / `Z_scale_m` | 0.5 | `contracts.py` |
| FixedPoint defaults | 16-bit, 10 frac, unsigned | `artifact.py` |

---

*End of audit.*
