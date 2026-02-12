# r/Compilers Post Draft

**Title:** Petri net -> stochastic neuron compiler for real-time plasma control

---

**Body:**

Part of our open-source tokamak simulator (SCPN Fusion Core) includes a neuro-symbolic compiler that might interest this community. It takes plasma control logic expressed as Petri nets and compiles it to stochastic spiking neural networks for sub-millisecond real-time execution.

**Pipeline:**

1. **Petri Net Definition** -- Control logic expressed as place/transition nets with formal contracts (preconditions, postconditions, invariants). Data structures in `scpn/structure.py`.

2. **Compilation** (`scpn/compiler.py`) -- Each Petri net transition maps to a stochastic leaky-integrate-and-fire (LIF) neuron. Places become input/output synapses. Token counts encode as Bernoulli bitstreams (when SC-NeuroCore hardware backend is available) or as float activations (NumPy fallback).

3. **Contract Verification** (`scpn/contracts.py`) -- Formal checks on compiled artifacts: reachability, boundedness, liveness properties of the original net are verified to be preserved in the neural representation.

4. **Execution** -- The compiled SNN runs in the tokamak flight simulator's control loop at sub-millisecond latency. The neural controller handles disruption prediction, heating power allocation, and position control simultaneously.

**What makes this different from typical neural compilation:**
- Source IR is Petri nets, not a compute graph -- captures concurrency and resource contention natively
- Target is stochastic neurons (not deterministic ReLU units) -- natural uncertainty quantification
- Formal contracts survive compilation -- you can verify safety properties on the artifact
- Optional hardware target: SC-NeuroCore (Verilog HDL neuromorphic core with Bernoulli bitstream encoding)

**Benchmark context:**
- The full simulator does equilibrium in 15 ms (Rust multigrid), transport inference in 5 us/point (MLP surrogate), and reconstruction in ~4 s -- the SNN controller adds negligible overhead to the loop
- Tutorial notebook: https://anulum.github.io/scpn-fusion-core/notebooks/ (notebook 02)

**Links:**
- Repository: https://github.com/anulum/scpn-fusion-core
- Compiler source: `src/scpn_fusion/scpn/compiler.py`
- SC-NeuroCore (HDL backend): https://github.com/anulum/sc-neurocore
- Full benchmark tables: https://github.com/anulum/scpn-fusion-core/blob/main/docs/BENCHMARKS.md

Licensed AGPL-3.0. Questions and feedback welcome.
