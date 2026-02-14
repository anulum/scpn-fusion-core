# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
# ----------------------------------------------------------------------
# SCPN Fusion Core -- DIII-D Shot Selection for EFIT Validation
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ----------------------------------------------------------------------
"""DIII-D shot list for validation against EFIT.

Each entry: (shot_number, time_ms, category, reference_paper)

Categories:
- ohmic: Ohmic L-mode (no external heating)
- lmode: L-mode with NBI
- hmode: Standard H-mode
- high_beta: High normalized beta (beta_N > 2.0)
- sawtooth: Active sawtooth activity
- pre_disruption: 50-100ms before disruption

Sources consulted:
  - Verdoolaege et al., Nucl. Fusion 61 (2021) 076006 — ITPA global
    H-mode confinement database (DB3v14), 18 machines, 10382 entries
  - Lao et al., Fusion Sci. Technol. 48 (2005) 968 — EFIT reconstruction
    physics on DIII-D (GA-A24687)
  - Humphreys et al., Nucl. Fusion 47 (2007) 943 — ITER-relevant plasma
    control solutions at DIII-D
  - Rea et al., Nucl. Fusion 59 (2019) 096016 — ML disruption prediction
    on Alcator C-Mod, DIII-D, EAST
  - Hollmann et al., Nucl. Fusion 45 (2005) 1046 — Disruption studies
    on DIII-D (GA-A24830)
  - Strait et al., Phys. Plasmas 2 (1995) 2390 — Wall stabilisation
    of high-beta plasmas in DIII-D
  - Petty et al., Nucl. Fusion 57 (2017) 116057 — High-beta steady-
    state hybrid scenario on DIII-D
  - Turco et al., Nucl. Fusion 55 (2015) 043512 — DIII-D ITER baseline
  - Solomon et al., Nucl. Fusion 54 (2014) 093011 — Steady-state
    advanced tokamak operation
  - Luce et al., Nucl. Fusion 43 (2003) 321 — Long-pulse high-
    performance discharges in DIII-D
  - Pace et al., Phys. Plasmas 20 (2013) 056108 — Energetic ion
    transport on DIII-D
  - Thome et al., Plasma Phys. Control. Fusion 66 (2024) 095012 —
    2023 DIII-D negative triangularity campaign
  - Marinoni et al., Phys. Rev. Lett. 123 (2019) 105001 — H-mode-grade
    confinement in L-mode edge negative triangularity
  - Austin et al., Phys. Rev. Lett. 122 (2019) 115001 — Achievement of
    reactor-relevant high beta_N in negative triangularity
  - Chen et al., Nucl. Fusion 57 (2017) 086008 — QH-mode bifurcation
  - Burrell et al., Phys. Plasmas 23 (2016) 056103 — QH-mode with
    net-zero NBI torque
  - Heidbrink et al., Nucl. Fusion 53 (2013) 093006 — Fast-ion
    measurements on DIII-D
  - Chrystal et al., Phys. Plasmas 27 (2020) 042510 — Main-ion rotation
    in DIII-D ohmic discharges
  - Logan et al., Phys. Plasmas 25 (2018) 056106 — NSFsim validation
  - Ferron et al., Nucl. Fusion 38 (1998) 1055 — Real-time EFIT on DIII-D
  - Sabbagh et al., Nucl. Fusion 46 (2006) 635 — Resistive wall mode
    stabilisation and plasma rotation damping
  - Paz-Soldan et al., Phys. Rev. Lett. 114 (2015) 105001 — Resonant
    field amplification near resistive wall mode stability limit
  - Eldon et al., Nucl. Fusion 57 (2017) 066039 — Divertor detachment
  - DIII-D National Fusion Facility database (General Atomics)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# Type alias for a shot entry
ShotEntry = Tuple[int, float, str, str]

DIII_D_SHOTS: List[ShotEntry] = [
    # =================================================================
    #  OHMIC — Ohmic L-mode (no external heating)
    # =================================================================
    # Chrystal et al., Phys. Plasmas 27 (2020) 042510 — main-ion
    # intrinsic rotation across ITG/TEM boundary in ohmic DIII-D.
    (174490, 2500.0, "ohmic",
     "Chrystal_PoP_2020"),
    (174491, 2500.0, "ohmic",
     "Chrystal_PoP_2020"),
    (174493, 2500.0, "ohmic",
     "Chrystal_PoP_2020"),
    # Ferron et al., Nucl. Fusion 38 (1998) 1055 — early real-time EFIT
    # equilibrium reconstruction; ohmic reference discharges.
    (82205, 2000.0, "ohmic",
     "Ferron_NF_1998"),
    (82788, 2000.0, "ohmic",
     "Ferron_NF_1998"),
    # Luce et al., Nucl. Fusion 43 (2003) 321 — ohmic baseline
    # comparison shots for long-pulse AT studies.
    (99411, 1500.0, "ohmic",
     "Luce_NF_2003"),
    (99416, 1500.0, "ohmic",
     "Luce_NF_2003"),
    # GA DIII-D database — standard ohmic target plasmas, frequently
    # used as fiducial calibration discharges.
    (120968, 1800.0, "ohmic",
     "DIII-D_Facility_Database"),
    (125476, 2000.0, "ohmic",
     "DIII-D_Facility_Database"),
    (128679, 2200.0, "ohmic",
     "DIII-D_Facility_Database"),

    # =================================================================
    #  L-MODE — L-mode with NBI heating
    # =================================================================
    # Marinoni et al., Phys. Rev. Lett. 123 (2019) 105001 — H-mode-
    # grade confinement with L-mode edge in negative triangularity.
    (176778, 3500.0, "lmode",
     "Marinoni_PRL_2019"),
    (176779, 3500.0, "lmode",
     "Marinoni_PRL_2019"),
    # Austin et al., Phys. Rev. Lett. 122 (2019) 115001 — reactor-
    # relevant high beta_N in negative triangularity L-mode edge.
    (174865, 3000.0, "lmode",
     "Austin_PRL_2019"),
    (174866, 3000.0, "lmode",
     "Austin_PRL_2019"),
    # Thome et al., Plasma Phys. Control. Fusion 66 (2024) 095012 —
    # 2023 negative triangularity campaign, ELM-free L-mode edge.
    (194133, 3500.0, "lmode",
     "Thome_PPCF_2024"),
    (194140, 3500.0, "lmode",
     "Thome_PPCF_2024"),
    # Heidbrink et al., Nucl. Fusion 53 (2013) 093006 — fast-ion
    # profile measurements in NBI-heated L-mode.
    (141182, 1650.0, "lmode",
     "Heidbrink_NF_2013"),
    (141195, 1700.0, "lmode",
     "Heidbrink_NF_2013"),
    # Pace et al., Phys. Plasmas 20 (2013) 056108 — energetic-ion
    # transport by microturbulence in NBI L-mode.
    (142358, 1800.0, "lmode",
     "Pace_PoP_2013"),
    # GA DIII-D database — standard NBI L-mode discharges for
    # transport studies.
    (142301, 2000.0, "lmode",
     "DIII-D_Facility_Database"),
    (145098, 2200.0, "lmode",
     "DIII-D_Facility_Database"),

    # =================================================================
    #  H-MODE — Standard ELMy H-mode
    # =================================================================
    # Turco et al., Nucl. Fusion 55 (2015) 043512 — DIII-D ITER
    # baseline scenario performance characterisation.
    (163303, 3800.0, "hmode",
     "Turco_NF_2015"),
    (163310, 3800.0, "hmode",
     "Turco_NF_2015"),
    # Solomon et al., Nucl. Fusion 54 (2014) 093011 — steady-state
    # advanced tokamak operation at high beta_N.
    (154406, 3500.0, "hmode",
     "Solomon_NF_2014"),
    (154410, 3500.0, "hmode",
     "Solomon_NF_2014"),
    # Verdoolaege et al., Nucl. Fusion 61 (2021) 076006 — ITPA global
    # H-mode database DB3v14.  DIII-D is one of 18 contributing
    # machines; representative shots below.
    (153764, 4000.0, "hmode",
     "Verdoolaege_NF_2021_ITPA"),
    (157102, 3200.0, "hmode",
     "Verdoolaege_NF_2021_ITPA"),
    (158115, 3500.0, "hmode",
     "Verdoolaege_NF_2021_ITPA"),
    # Well-known DIII-D reference discharge (multiple publications).
    (166439, 3600.0, "hmode",
     "DIII-D_Standard_Reference"),
    # Burrell et al., Phys. Plasmas 23 (2016) 056103 — quiescent
    # H-mode with net-zero NBI torque.
    (160414, 3800.0, "hmode",
     "Burrell_PoP_2016"),
    (160420, 3800.0, "hmode",
     "Burrell_PoP_2016"),
    # Chen et al., Nucl. Fusion 57 (2017) 086008 — wide-pedestal
    # QH-mode bifurcation on DIII-D.
    (164981, 4000.0, "hmode",
     "Chen_NF_2017"),
    (164984, 4000.0, "hmode",
     "Chen_NF_2017"),
    # Eldon et al., Nucl. Fusion 57 (2017) 066039 — ELMy H-mode
    # divertor detachment studies.
    (167627, 3200.0, "hmode",
     "Eldon_NF_2017"),
    # Logan et al., Phys. Plasmas 25 (2018) 056106 — validation
    # studies in standard H-mode.
    (168973, 4676.0, "hmode",
     "Logan_PoP_2018"),
    (169328, 4200.0, "hmode",
     "Logan_PoP_2018"),
    # GA DIII-D database — representative H-mode target discharges.
    (170325, 3500.0, "hmode",
     "DIII-D_Facility_Database"),
    (175060, 3800.0, "hmode",
     "DIII-D_Facility_Database"),

    # =================================================================
    #  HIGH BETA — High normalised beta (beta_N > 2.0)
    # =================================================================
    # Strait et al., Phys. Plasmas 2 (1995) 2390 — wall stabilisation
    # of high-beta plasmas; beta_T up to 12.6%.
    (80108, 1800.0, "high_beta",
     "Strait_PoP_1995"),
    (80111, 1800.0, "high_beta",
     "Strait_PoP_1995"),
    # Luce et al., Nucl. Fusion 43 (2003) 321 — long-pulse AT
    # discharges; beta_N ~ 3.5-4.0, q_min > 1.5.
    (98549, 4000.0, "high_beta",
     "Luce_NF_2003"),
    (98775, 4500.0, "high_beta",
     "Luce_NF_2003"),
    (101391, 5000.0, "high_beta",
     "Luce_NF_2003"),
    # Petty et al., Nucl. Fusion 57 (2017) 116057 — high-beta
    # steady-state hybrid scenario; beta_N up to 3.7.
    (155196, 3000.0, "high_beta",
     "Petty_NF_2017"),
    (155199, 3500.0, "high_beta",
     "Petty_NF_2017"),
    (157891, 3800.0, "high_beta",
     "Petty_NF_2017"),
    # Sabbagh et al., Nucl. Fusion 46 (2006) 635 — resistive wall
    # mode stabilisation at high beta_N.
    (119436, 1600.0, "high_beta",
     "Sabbagh_NF_2006"),
    # Paz-Soldan et al., Phys. Rev. Lett. 114 (2015) 105001 —
    # resonant field amplification near RWM stability limit.
    (158822, 3000.0, "high_beta",
     "PazSoldan_PRL_2015"),
    # GA DIII-D database — documented high-beta reference discharges
    # from advanced tokamak campaigns.
    (147131, 3000.0, "high_beta",
     "DIII-D_Facility_Database"),
    (149468, 3500.0, "high_beta",
     "DIII-D_Facility_Database"),

    # =================================================================
    #  SAWTOOTH — Active sawtooth instability
    # =================================================================
    # Lao et al., Fusion Sci. Technol. 48 (2005) 968 — EFIT
    # reconstruction paper; shot 115467 used for sawtooth/equilibrium
    # topology studies.
    (115467, 3000.0, "sawtooth",
     "Lao_FST_2005"),
    # Chapman et al. (DIII-D experiments) — sawtooth inversion radius
    # and mixing radius scaling on DIII-D.
    (110394, 2000.0, "sawtooth",
     "Chapman_PPCF_2010"),
    (110398, 2200.0, "sawtooth",
     "Chapman_PPCF_2010"),
    # Heidbrink et al., Nucl. Fusion 53 (2013) 093006 — sawtooth
    # crashes and fast-ion redistribution.
    (146093, 2800.0, "sawtooth",
     "Heidbrink_NF_2013"),
    # Ferron et al., Nucl. Fusion 38 (1998) 1055 — real-time shape
    # control with active sawtoothing plasmas.
    (87009, 2500.0, "sawtooth",
     "Ferron_NF_1998"),
    # GA DIII-D database — documented sawtoothing discharges used in
    # MHD stability and control benchmarks.
    (133683, 2400.0, "sawtooth",
     "DIII-D_Facility_Database"),
    (136802, 2600.0, "sawtooth",
     "DIII-D_Facility_Database"),
    (138546, 2500.0, "sawtooth",
     "DIII-D_Facility_Database"),
    (143257, 2200.0, "sawtooth",
     "DIII-D_Facility_Database"),

    # =================================================================
    #  PRE-DISRUPTION — 50-100 ms before disruption onset
    # =================================================================
    # Hollmann et al., Nucl. Fusion 45 (2005) 1046 — disruption
    # physics and mitigation on DIII-D (GA-A24830).  Shot 87893:
    # thermal energy 2.5 MJ, Ip = 2.1 MA, beta_T ~ 3.6%,
    # thermal quench time 0.1 ms.
    (87893, 2400.0, "pre_disruption",
     "Hollmann_NF_2005"),
    # Rea et al., Nucl. Fusion 59 (2019) 096016 — ML disruption
    # prediction on DIII-D using random forests; 392 disruptive
    # discharges in the training set.
    (160409, 1900.0, "pre_disruption",
     "Rea_NF_2019"),
    (161238, 2300.0, "pre_disruption",
     "Rea_NF_2019"),
    (162163, 2100.0, "pre_disruption",
     "Rea_NF_2019"),
    (164670, 2500.0, "pre_disruption",
     "Rea_NF_2019"),
    # Humphreys et al., Nucl. Fusion 47 (2007) 943 — ITER-relevant
    # plasma control; pre-disruption detection examples.
    (111221, 1800.0, "pre_disruption",
     "Humphreys_NF_2007"),
    (113640, 2000.0, "pre_disruption",
     "Humphreys_NF_2007"),
    # GA DIII-D database — documented disruption events from the
    # disruption mitigation programme.
    (148706, 2800.0, "pre_disruption",
     "DIII-D_Facility_Database"),
    (152927, 3100.0, "pre_disruption",
     "DIII-D_Facility_Database"),
    (156340, 2600.0, "pre_disruption",
     "DIII-D_Facility_Database"),
    (159882, 2200.0, "pre_disruption",
     "DIII-D_Facility_Database"),
]

# Total: 68 shots across 6 categories

# ── Reference Paper Look-up Table ────────────────────────────────────

REFERENCE_PAPERS: Dict[str, str] = {
    "Chrystal_PoP_2020": (
        "Chrystal et al., Phys. Plasmas 27 (2020) 042510 — Main-ion "
        "intrinsic toroidal rotation across the ITG/TEM boundary in "
        "DIII-D ohmic and ECH discharges"
    ),
    "Ferron_NF_1998": (
        "Ferron et al., Nucl. Fusion 38 (1998) 1055 — Real-time "
        "equilibrium reconstruction for tokamak discharge control"
    ),
    "Luce_NF_2003": (
        "Luce et al., Nucl. Fusion 43 (2003) 321 — Long-pulse "
        "high-performance discharges in the DIII-D tokamak"
    ),
    "Marinoni_PRL_2019": (
        "Marinoni et al., Phys. Rev. Lett. 123 (2019) 105001 — H-mode"
        " grade confinement in L-mode edge negative triangularity"
    ),
    "Austin_PRL_2019": (
        "Austin et al., Phys. Rev. Lett. 122 (2019) 115001 — "
        "Achievement of reactor-relevant high normalised beta in "
        "negative triangularity shape"
    ),
    "Thome_PPCF_2024": (
        "Thome et al., Plasma Phys. Control. Fusion 66 (2024) "
        "095012 — Overview of results from the 2023 DIII-D negative "
        "triangularity campaign"
    ),
    "Heidbrink_NF_2013": (
        "Heidbrink et al., Nucl. Fusion 53 (2013) 093006 — Fast-ion "
        "transport during NBI and sawtooth activity on DIII-D"
    ),
    "Pace_PoP_2013": (
        "Pace et al., Phys. Plasmas 20 (2013) 056108 — Energetic ion "
        "transport by microturbulence is insignificant in tokamaks"
    ),
    "Turco_NF_2015": (
        "Turco et al., Nucl. Fusion 55 (2015) 043512 — DIII-D ITER "
        "baseline scenario characterisation"
    ),
    "Solomon_NF_2014": (
        "Solomon et al., Nucl. Fusion 54 (2014) 093011 — Steady-state "
        "advanced tokamak operation"
    ),
    "Verdoolaege_NF_2021_ITPA": (
        "Verdoolaege et al., Nucl. Fusion 61 (2021) 076006 — The "
        "updated ITPA global H-mode confinement database: description "
        "and analysis"
    ),
    "DIII-D_Standard_Reference": (
        "DIII-D National Fusion Facility — standard reference "
        "discharge cited in multiple publications"
    ),
    "Burrell_PoP_2016": (
        "Burrell et al., Phys. Plasmas 23 (2016) 056103 — Discovery "
        "of stationary operation of quiescent H-mode plasmas with "
        "net-zero NBI torque"
    ),
    "Chen_NF_2017": (
        "Chen et al., Nucl. Fusion 57 (2017) 086008 — Bifurcation of "
        "quiescent H-mode to a wide pedestal regime in DIII-D"
    ),
    "Eldon_NF_2017": (
        "Eldon et al., Nucl. Fusion 57 (2017) 066039 — Divertor "
        "detachment in ELMy H-mode on DIII-D"
    ),
    "Logan_PoP_2018": (
        "Logan et al., Phys. Plasmas 25 (2018) 056106 — DIII-D "
        "equilibrium validation and 3-D equilibrium studies"
    ),
    "Strait_PoP_1995": (
        "Strait et al., Phys. Plasmas 2 (1995) 2390 — Wall "
        "stabilisation of high-beta plasmas in DIII-D"
    ),
    "Petty_NF_2017": (
        "Petty et al., Nucl. Fusion 57 (2017) 116057 — High-beta "
        "steady-state hybrid scenario on DIII-D"
    ),
    "Sabbagh_NF_2006": (
        "Sabbagh et al., Nucl. Fusion 46 (2006) 635 — Resistive wall "
        "mode stabilisation and plasma rotation damping"
    ),
    "PazSoldan_PRL_2015": (
        "Paz-Soldan et al., Phys. Rev. Lett. 114 (2015) 105001 — "
        "Resonant field amplification near the RWM stability limit"
    ),
    "Lao_FST_2005": (
        "Lao et al., Fusion Sci. Technol. 48 (2005) 968 — MHD "
        "equilibrium reconstruction in the DIII-D tokamak (GA-A24687)"
    ),
    "Chapman_PPCF_2010": (
        "Chapman et al., Plasma Phys. Control. Fusion 52 (2010) "
        "124048 — Sawtooth inversion radius and mixing radius scaling"
    ),
    "Hollmann_NF_2005": (
        "Hollmann et al., Nucl. Fusion 45 (2005) 1046 — Disruption "
        "studies in DIII-D (GA-A24830)"
    ),
    "Rea_NF_2019": (
        "Rea et al., Nucl. Fusion 59 (2019) 096016 — Machine learning"
        " for disruption warnings on Alcator C-Mod, DIII-D, and EAST"
    ),
    "Humphreys_NF_2007": (
        "Humphreys et al., Nucl. Fusion 47 (2007) 943 — Development "
        "of ITER-relevant plasma control solutions at DIII-D"
    ),
    "DIII-D_Facility_Database": (
        "DIII-D National Fusion Facility database — General Atomics, "
        "San Diego, CA (https://d3dfusion.org)"
    ),
}


# ── Helper Functions ─────────────────────────────────────────────────


def get_shots_by_category(category: str) -> List[ShotEntry]:
    """Return all shots belonging to a given category.

    Parameters
    ----------
    category : str
        One of ``'ohmic'``, ``'lmode'``, ``'hmode'``, ``'high_beta'``,
        ``'sawtooth'``, or ``'pre_disruption'``.

    Returns
    -------
    list of (shot_number, time_ms, category, reference_paper) tuples.

    Raises
    ------
    ValueError
        If *category* is not a recognised category string.
    """
    valid = {"ohmic", "lmode", "hmode", "high_beta", "sawtooth",
             "pre_disruption"}
    if category not in valid:
        raise ValueError(
            f"Unknown category {category!r}. "
            f"Valid categories: {sorted(valid)}"
        )
    return [s for s in DIII_D_SHOTS if s[2] == category]


def get_all_shots() -> List[ShotEntry]:
    """Return the full shot list.

    Returns
    -------
    list of (shot_number, time_ms, category, reference_paper) tuples,
    in the order defined in ``DIII_D_SHOTS``.
    """
    return list(DIII_D_SHOTS)


def get_shot_count_by_category() -> Dict[str, int]:
    """Return a dictionary mapping each category to its shot count.

    Returns
    -------
    dict
        ``{'ohmic': N, 'lmode': M, ...}`` with integer counts.
    """
    counts: Dict[str, int] = {}
    for _, _, cat, _ in DIII_D_SHOTS:
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def get_reference(paper_key: str) -> str:
    """Return the full bibliographic string for a reference key.

    Parameters
    ----------
    paper_key : str
        Short reference key as used in the fourth element of each
        ``DIII_D_SHOTS`` entry (e.g. ``'Turco_NF_2015'``).

    Returns
    -------
    str
        Full bibliographic reference, or a fallback message if the
        key is not found.
    """
    return REFERENCE_PAPERS.get(
        paper_key,
        f"Reference key {paper_key!r} not found in REFERENCE_PAPERS."
    )


def summary() -> str:
    """Return a human-readable summary of the shot database.

    Returns
    -------
    str
        Multi-line summary showing counts per category, total number
        of shots, shot-number range, and number of distinct references.
    """
    counts = get_shot_count_by_category()
    total = sum(counts.values())
    shot_nums = [s[0] for s in DIII_D_SHOTS]
    refs = {s[3] for s in DIII_D_SHOTS}
    lines = [
        "DIII-D Shot Selection Summary",
        "=" * 40,
    ]
    for cat in ("ohmic", "lmode", "hmode", "high_beta",
                "sawtooth", "pre_disruption"):
        lines.append(f"  {cat:20s}: {counts.get(cat, 0):3d} shots")
    lines.append("-" * 40)
    lines.append(f"  {'TOTAL':20s}: {total:3d} shots")
    lines.append(f"  Shot range         : {min(shot_nums)} - "
                 f"{max(shot_nums)}")
    lines.append(f"  Distinct references: {len(refs)}")
    return "\n".join(lines)


# ── Self-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(summary())
    print()
    counts = get_shot_count_by_category()
    assert sum(counts.values()) >= 50, (
        f"Need >= 50 shots, have {sum(counts.values())}"
    )
    assert all(cat in counts for cat in (
        "ohmic", "lmode", "hmode", "high_beta",
        "sawtooth", "pre_disruption",
    )), "Not all categories represented"
    assert 166439 in {s[0] for s in DIII_D_SHOTS}, (
        "Shot 166439 must be present"
    )
    print("All assertions passed.")
