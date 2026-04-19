# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Verification Report Tests
from __future__ import annotations

import json

from scpn_fusion.core.gk_corrector import CorrectionRecord
from scpn_fusion.core.gk_verification_report import VerificationReport


def _make_record(surr_i: float, gk_i: float) -> CorrectionRecord:
    return CorrectionRecord(
        rho_idx=0,
        rho=0.5,
        chi_i_surrogate=surr_i,
        chi_i_gk=gk_i,
        chi_e_surrogate=1.0,
        chi_e_gk=1.0,
        D_e_surrogate=0.1,
        D_e_gk=0.1,
    )


def test_empty_report():
    r = VerificationReport()
    assert r.verification_fraction == 0.0
    assert r.max_rel_error == 0.0
    assert r.mean_rel_error == 0.0


def test_add_steps():
    r = VerificationReport()
    r.add_step(verified=True, n_spot_checks=3, n_ood=1)
    r.add_step(verified=False)
    r.add_step(verified=True, n_spot_checks=2)
    assert r.total_steps == 3
    assert r.steps_verified == 2
    assert r.total_spot_checks == 5
    assert r.ood_triggers == 1
    assert r.verification_fraction == 2.0 / 3.0


def test_add_records():
    r = VerificationReport()
    records = [
        _make_record(1.0, 1.5),  # rel_error = 0.5/1.5 = 0.333
        _make_record(2.0, 1.0),  # rel_error = 1.0/1.0 = 1.0
    ]
    r.add_records(records)
    assert len(r.records) == 2
    assert r.max_rel_error > 0.9
    assert r.mean_rel_error > 0


def test_to_dict():
    r = VerificationReport()
    r.add_step(verified=True, n_spot_checks=2, n_ood=1)
    r.add_records([_make_record(1.0, 1.5)])
    r.add_correction_factor(0.5)
    d = r.to_dict()
    assert d["total_steps"] == 1
    assert d["steps_verified"] == 1
    assert "max_rel_error_chi_i" in d
    assert "mean_correction_factor" in d


def test_to_json():
    r = VerificationReport()
    r.add_step(verified=True, n_spot_checks=1)
    r.add_records([_make_record(1.0, 2.0)])
    text = r.to_json()
    parsed = json.loads(text)
    assert parsed["total_steps"] == 1


def test_to_json_file(tmp_path):
    r = VerificationReport()
    r.add_step(verified=True)
    path = tmp_path / "report.json"
    r.to_json(path)
    assert path.exists()
    parsed = json.loads(path.read_text())
    assert parsed["total_steps"] == 1


def test_correction_factor_tracking():
    r = VerificationReport()
    r.add_correction_factor(0.1)
    r.add_correction_factor(0.3)
    r.add_correction_factor(0.2)
    d = r.to_dict()
    assert d["mean_correction_factor"] == round(0.2, 4)
