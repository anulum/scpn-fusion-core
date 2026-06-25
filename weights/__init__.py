# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Bundled Pretrained-Surrogate Weights
"""Bundled pretrained-surrogate weight artifacts.

This package exists so that the small runtime surrogate weights the library loads
by default — the QLKNN neural-transport model, the SPARC and ITER neural-equilibrium
surrogates, and the ITPA/EUROfusion-JET pretrained heads — ship inside the installed
wheel and resolve through :func:`scpn_fusion._data_paths.data_root` alongside the
``validation`` data tree.

Large training-only artifacts (e.g. the 302 MB legacy ``fno_turbulence_jax.npz``
loaded only under ``SCPN_ENABLE_LEGACY_FNO=1``, and the variant/augmented checkpoints)
are intentionally **not** packaged: they live in the source checkout for training and
benchmarking but would bloat the wheel beyond any reasonable distribution size.
"""
