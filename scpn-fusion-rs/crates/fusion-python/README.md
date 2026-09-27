<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# scpn-fusion-rs

`scpn-fusion-rs` is the independently installable native extension for
[SCPN Fusion Core](https://github.com/anulum/scpn-fusion-core). It exposes
selected equilibrium, control, transport, phase, diagnostic, nuclear, and
machine-learning kernels through the `scpn_fusion_rs` Python module.

Install the Python package and native extension together:

```bash
python -m pip install scpn-fusion scpn-fusion-rs
```

Confirm the extension and its release identity:

```bash
python -c "import scpn_fusion_rs; print(scpn_fusion_rs.__version__)"
```

The extension is optional. `scpn-fusion` retains NumPy/SciPy fallback paths
when a compatible native wheel is unavailable. Backend availability is not a
blanket performance claim; use the repository's checksummed benchmark reports
for equivalent workload and hardware comparisons.

## Numerical boundaries

`measure_magnetics` and the full-solve `multigrid_vcycle` require actual native
float64 ndarrays, accept valid strided/readonly layouts and return independent
C-contiguous output. Invalid kinds, domains and numerical arithmetic propagate
as TypeError, ValueError and RuntimeError; ordinary allocation failure is
MemoryError. Initial multigrid convergence returns zero cycles. See the full
[numerical contracts](../../../docs/NUMERICAL_CONTRACTS.md) for scalar kinds,
dimensions, geometry and explicit dtype migration. Backend availability does
not turn input or numerical errors into fallback.

## Source build

From the repository root:

```bash
python -m pip install 'maturin>=1.14.1,<2.0'
cd scpn-fusion-rs/crates/fusion-python
maturin develop --release
```

Python 3.10–3.12 are supported by the 4.0.0 distribution line.

## Licensing

The open-source path is `AGPL-3.0-or-later`. Separate commercial licences are
available for proprietary use; contact `protoscience@anulum.li`.
