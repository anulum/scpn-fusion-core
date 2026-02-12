Getting Started
===============

Installation
------------

From PyPI::

    pip install scpn-fusion

From source::

    git clone https://github.com/anulum/scpn-fusion-core.git
    cd scpn-fusion-core
    pip install -e ".[dev]"

Quick Example
-------------

Run a Grad-Shafranov equilibrium solve::

    python run_fusion_suite.py kernel

Read a SPARC GEQDSK equilibrium::

    from scpn_fusion.core.eqdsk import read_geqdsk

    eq = read_geqdsk("validation/reference_data/sparc/lmode_vv.geqdsk")
    print(f"B_T = {eq.bcentr:.1f} T, I_p = {eq.current/1e6:.1f} MA")

Rust Acceleration
-----------------

Optional Rust backend for numerical solver speedups::

    cd scpn-fusion-rs
    cargo build --release

    # Build Python bindings
    pip install maturin
    cd crates/fusion-python
    maturin develop --release

The Python package auto-detects the Rust extension and falls back to NumPy.
