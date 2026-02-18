============
Installation
============

SCPN-Fusion-Core supports three installation paths: a pure-Python install
(fastest to get started), a Rust-accelerated install (10--50x faster
numerics), and a Docker-based install (zero local dependencies).

Requirements
------------

- Python 3.9 or later
- NumPy >= 1.22
- SciPy >= 1.8

Optional dependencies are pulled in automatically by the ``[dev]`` extra:
``pytest``, ``hypothesis``, ``matplotlib``, ``streamlit``.

From PyPI (Recommended)
-----------------------

Pre-built wheels include the Rust extension for common platforms::

    pip install scpn-fusion

This is the simplest path and provides both the Python package and the
compiled Rust kernels.

From Source (Pure Python)
-------------------------

Clone the repository and install in editable mode::

    git clone https://github.com/anulum/scpn-fusion-core.git
    cd scpn-fusion-core
    pip install -e .

This installs the pure-Python package.  Every module auto-detects the
Rust extension at import time and falls back to NumPy/SciPy if it is not
available.  You will see a one-time info message at import::

    INFO: scpn_fusion_rs not found -- using NumPy fallback.

Development Install
-------------------

For running the test suite, linters, and building documentation::

    pip install -e ".[dev]"
    pytest tests/ -v

The ``[dev]`` extra installs ``pytest``, ``hypothesis``, ``matplotlib``,
and ``streamlit``.

Rust Kernel Build (Optional)
-----------------------------

The ``scpn-fusion-rs/`` directory contains a 10-crate Rust workspace
that mirrors the Python package structure.  Building it provides 10--50x
speedups for equilibrium solves, inverse reconstruction, and transport
stepping.

Prerequisites:

- Rust stable toolchain (``rustup`` recommended)
- ``maturin`` (``pip install maturin``)

Build steps::

    cd scpn-fusion-rs

    # Build the native library
    cargo build --release

    # Run Rust tests
    cargo test --all-features

    # Build Python bindings (produces scpn_fusion_rs.pyd / .so)
    cd crates/fusion-python
    maturin develop --release

After building, restart Python and the package will auto-detect the Rust
extension::

    >>> from scpn_fusion.core import RUST_BACKEND
    >>> print(RUST_BACKEND)
    True

Rust Benchmarks
^^^^^^^^^^^^^^^

Criterion micro-benchmarks are included for the SOR stencil, inverse
solver, and neural transport MLP::

    cd scpn-fusion-rs
    cargo bench

Docker
------

A Docker image is provided for zero-dependency deployment::

    # Run the Streamlit dashboard
    docker compose up

    # Or build and run manually
    docker build -t scpn-fusion-core .
    docker run -p 8501:8501 scpn-fusion-core

    # With dev dependencies (for running tests inside the container)
    docker build --build-arg INSTALL_DEV=1 -t scpn-fusion-core:dev .
    docker run scpn-fusion-core:dev pytest tests/ -v

The Docker image includes both the Python package and the pre-compiled
Rust extension.

Verifying the Installation
--------------------------

After installation, verify that the package loads correctly::

    python -c "import scpn_fusion; print(scpn_fusion.__version__)"

Run a quick equilibrium solve to confirm numerical correctness::

    scpn-fusion kernel

Expected output: a converged Grad-Shafranov equilibrium with magnetic
axis position, safety factor profile, and plasma current reported to
stdout.
