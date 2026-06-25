# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Studio verbs (schema A)
"""The SCPN-FUSION-CORE studio's verbs, on the locked platform contract.

SCPN-FUSION-CORE is the magnetically-confined-fusion laboratory of the SCPN
ecosystem: it reconstructs and evolves tokamak/FRC equilibria, transports the
plasma profiles, analyses MHD and gyrokinetic stability, drives closed-loop
control in a flight simulator, and forecasts disruptions. Six verbs are drawn
from the shared :data:`scpn_studio_platform.verbs.CORE_VERBS` spine
(``reconstruct``, ``simulate``, ``analyse``, ``validate``, ``benchmark``,
``replay``); two are domain-distinctive to a fusion studio (``control``,
``predict``).

Every verb is a :class:`scpn_studio_platform.verbs.Verb` carrying the attribute
contract the Hub federates and gates against. The studio never touches a live
machine — all execution is simulated or read-only, so no verb is
``live-hardware`` and every safety tier is ``research``. Verb attributes use the
platform enums verbatim; the ``studio.*.v1`` names below are the evidence claim
families the verbs produce, mapped from the repository's evidence boundary.
"""

from __future__ import annotations

from scpn_studio_platform.verbs import (
    Fidelity,
    SafetyTier,
    SideEffect,
    Timing,
    TimingClass,
    Verb,
)

STUDIO_ID = "scpn-fusion-core"
"""The studio identifier this vertical implements (also the federation name)."""

# ── evidence schema names this studio emits (studio.*.v1) ──────────────
EQUILIBRIUM_RECONSTRUCTION_SCHEMA = "studio.equilibrium-reconstruction.v1"
TRANSPORT_SOLUTION_SCHEMA = "studio.transport-solution.v1"
GYROKINETIC_PARITY_SCHEMA = "studio.gyrokinetic-parity.v1"
MHD_STABILITY_SCHEMA = "studio.mhd-stability.v1"
PHYSICS_VALIDATION_SCHEMA = "studio.physics-validation.v1"
SOLVER_BENCHMARK_SCHEMA = "studio.solver-benchmark.v1"
EVIDENCE_REPLAY_SCHEMA = "studio.evidence-replay.v1"
CONTROL_REPLAY_SCHEMA = "studio.control-replay.v1"
DISRUPTION_FORECAST_SCHEMA = "studio.disruption-forecast.v1"


# ── core-spine verbs ───────────────────────────────────────────────────
RECONSTRUCT = Verb(
    name="reconstruct",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.READ_ONLY,
    timing=Timing(TimingClass.BATCH),
    fidelity=Fidelity.FIRST_PRINCIPLES,
    produces=(EQUILIBRIUM_RECONSTRUCTION_SCHEMA,),
    backends=("rust", "jax", "python"),
)
"""Reconstruct a free-boundary equilibrium (EFIT / kinetic-EFIT / neural) from diagnostics."""

SIMULATE = Verb(
    name="simulate",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.SIMULATED,
    timing=Timing(TimingClass.BATCH),
    fidelity=Fidelity.FIRST_PRINCIPLES,
    produces=(TRANSPORT_SOLUTION_SCHEMA,),
    backends=("rust", "jax", "python"),
)
"""Evolve the Grad-Shafranov equilibrium and integrated transport profiles forward in time."""

ANALYSE = Verb(
    name="analyse",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.READ_ONLY,
    timing=Timing(TimingClass.BATCH),
    fidelity=Fidelity.ANALYTIC,
    produces=(GYROKINETIC_PARITY_SCHEMA, MHD_STABILITY_SCHEMA),
    backends=("rust", "python"),
)
"""Extract gyrokinetic transport and MHD-stability probes (growth rates, parity, margins)."""

VALIDATE = Verb(
    name="validate",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.READ_ONLY,
    timing=Timing(TimingClass.BATCH),
    fidelity=Fidelity.ANALYTIC,
    produces=(PHYSICS_VALIDATION_SCHEMA,),
    backends=("python",),
)
"""Check a bounded physics claim against an experimental reference (ITER/SPARC/DIII-D/JET/MAST)."""

BENCHMARK = Verb(
    name="benchmark",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.SIMULATED,
    timing=Timing(TimingClass.BATCH),
    fidelity=Fidelity.ANALYTIC,
    produces=(SOLVER_BENCHMARK_SCHEMA,),
    backends=("rust", "julia", "go", "python"),
)
"""Measure the multi-language solver speedup as a reproducible regression guard."""

REPLAY = Verb(
    name="replay",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.READ_ONLY,
    timing=Timing(TimingClass.BATCH),
    fidelity=Fidelity.ANALYTIC,
    produces=(EVIDENCE_REPLAY_SCHEMA,),
    backends=("python",),
)
"""Re-verify a committed evidence pack from its raw artefacts and provenance."""


# ── domain-distinctive verbs ───────────────────────────────────────────
CONTROL = Verb(
    name="control",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.SIMULATED,
    timing=Timing(TimingClass.REALTIME, deadline_us=100.0),
    fidelity=Fidelity.REDUCED_ORDER,
    produces=(CONTROL_REPLAY_SCHEMA,),
    backends=("rust", "python"),
)
"""Drive closed-loop shape / vertical / burn control against the plasma flight simulator.

The real-time deadline is 100 microseconds — the 10 kHz control-loop period the
Rust controller path is benchmarked against (``tests/test_verify_10khz_rust.py``).
"""

PREDICT = Verb(
    name="predict",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.READ_ONLY,
    timing=Timing(TimingClass.BATCH),
    fidelity=Fidelity.ML_SURROGATE,
    produces=(DISRUPTION_FORECAST_SCHEMA,),
    backends=("python",),
)
"""Forecast disruptions from plasma features with a calibrated machine-learning surrogate."""


FUSION_VERBS: tuple[Verb, ...] = (
    RECONSTRUCT,
    SIMULATE,
    ANALYSE,
    VALIDATE,
    BENCHMARK,
    REPLAY,
    CONTROL,
    PREDICT,
)
"""All verbs the FUSION studio advertises on the federation contract."""


def evidence_schemas() -> tuple[str, ...]:
    """Return the ``studio.*.v1`` evidence-schema names this studio emits.

    The order is stable so the content digest over the declared surface is
    reproducible across checkouts.

    Returns
    -------
    tuple[str, ...]
        The evidence-schema identifiers produced by :data:`FUSION_VERBS`.
    """
    return (
        EQUILIBRIUM_RECONSTRUCTION_SCHEMA,
        TRANSPORT_SOLUTION_SCHEMA,
        GYROKINETIC_PARITY_SCHEMA,
        MHD_STABILITY_SCHEMA,
        PHYSICS_VALIDATION_SCHEMA,
        SOLVER_BENCHMARK_SCHEMA,
        EVIDENCE_REPLAY_SCHEMA,
        CONTROL_REPLAY_SCHEMA,
        DISRUPTION_FORECAST_SCHEMA,
    )
