# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Docker Image
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────

# ── Stage 1: Build Rust workspace + PyO3 bindings ────────────────────
FROM python:3.12-slim@sha256:ccc7089399c8bb65dd1fb3ed6d55efa538a3f5e7fca3f5988ac3b5b87e593bf0 AS rust-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain (stable, minimal profile)
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH="/usr/local/cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh \
    && echo "6c30b75a75b28a96fd913a037c8581b580080b6ee9b8169a3c0feb1af7fe8caf  /tmp/rustup-init.sh" | sha256sum -c - \
    && sh /tmp/rustup-init.sh -y --default-toolchain stable --profile minimal \
    && rm /tmp/rustup-init.sh

WORKDIR /build

# Copy only the Rust workspace first (cache-friendly layer)
COPY scpn-fusion-rs/ scpn-fusion-rs/

# Build the full workspace in release mode
RUN cd scpn-fusion-rs && cargo build --release

# Install maturin and build the PyO3 wheel
RUN pip install --no-cache-dir "maturin==1.12.6"
RUN cd scpn-fusion-rs/crates/fusion-python \
    && maturin build --release --out /build/wheels

# ── Stage 2: Runtime image ───────────────────────────────────────────
FROM python:3.12-slim@sha256:ccc7089399c8bb65dd1fb3ed6d55efa538a3f5e7fca3f5988ac3b5b87e593bf0 AS runtime

WORKDIR /app
ARG INSTALL_DEV=0

# Install the built Rust wheel first (before Python package, for layer caching)
COPY --from=rust-builder /build/wheels/ /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels

# Copy Python project files
COPY pyproject.toml README.md ./
COPY src/ src/
COPY validation/ validation/
COPY requirements/ requirements/

# Install runtime dependencies from hash-pinned lock file, then local package
RUN if [ "$INSTALL_DEV" = "1" ]; then \
      pip install --no-cache-dir --require-hashes -r requirements/ci-py312.txt \
      && pip install --no-cache-dir --no-deps ".[dev]" ; \
    else \
      pip install --no-cache-dir --require-hashes -r requirements/minimal.txt \
      && pip install --no-cache-dir --no-deps . ; \
    fi

# Copy remaining project assets needed at runtime
COPY calibration/ calibration/
COPY schemas/ schemas/
COPY examples/ examples/
COPY tests/ tests/
COPY run_fusion_suite.py conftest.py iter_config.json ./

RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

# Streamlit configuration: disable telemetry, use port 8501
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501

EXPOSE 8501

CMD ["streamlit", "run", "src/scpn_fusion/ui/app.py", \
     "--server.address=0.0.0.0"]
