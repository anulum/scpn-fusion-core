# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — DREAM Full-Kinetic Execution Custody Tests
"""Real HDF5 and filesystem tests for DREAM execution custody."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from validation.dream_full_kinetic_execution_custody import (
    DreamOutputContract,
    atomic_write_json,
    frozen_output_contract,
    inspect_dream_output,
    read_json_object,
    sha256_file,
    validate_durable_root,
)
from validation.dream_full_kinetic_reference import (
    DREAM_COMMIT,
    EXPECTED_FLAGS,
    ION_RATE_AUXILIARY_QUANTITIES,
    REQUIRED_AUXILIARY_QUANTITIES,
)
from validation.reference_data.dream.full_kinetic_radial_parity_deck import (
    REQUESTED_OTHER_QUANTITIES,
)

h5py = importlib.import_module("h5py")


def _text_dataset(handle: Any, name: str, value: str) -> None:
    handle.create_dataset(name, data=np.frombuffer(value.encode("utf-8"), dtype="S1"))


def _complete_output(path: Path, contract: DreamOutputContract) -> None:
    nt, nr, nxi, np_ = contract.nt, contract.nr, contract.nxi, contract.np
    with h5py.File(path, "w") as handle:
        _text_dataset(handle, "code/commit", DREAM_COMMIT)
        _text_dataset(handle, "settings/other/include", ";".join(REQUESTED_OTHER_QUANTITIES))
        for key, value in EXPECTED_FLAGS.items():
            handle.create_dataset(key, data=np.asarray([value], dtype=np.int64))
        handle.create_dataset("grid/t", data=np.linspace(0.0, contract.final_time_s, nt + 1))
        for key, shape in {
            "eqsys/f_re": (nt + 1, nr, nxi, np_),
            "eqsys/n_re": (nt + 1, nr),
            "eqsys/j_re": (nt + 1, nr),
            "eqsys/n_tot": (nt + 1, nr),
            "eqsys/E_field": (nt + 1, nr),
            "other/runaway/Ar": (nt, nr + 1, nxi, np_),
            "other/runaway/Drr": (nt, nr + 1, nxi, np_),
            "other/runaway/S_ava": (nt, nr, nxi, np_),
        }.items():
            fill = 1.0 if key.endswith(("Drr", "S_ava")) else 0.0
            handle.create_dataset(key, shape=shape, dtype=np.float64, fillvalue=fill, chunks=True)
        for name in (
            "Ap1",
            "Dpp",
            "Dpx",
            "lnLambda_ee_f1",
            "lnLambda_ei_f1",
            "nu_D_f1",
            "nu_s_f1",
            "nu_par_f1",
            "synchrotron_f1",
            "bremsstrahlung_f1",
        ):
            fill = 1.0 if name in {"Dpp", "synchrotron_f1", "bremsstrahlung_f1"} else 0.0
            handle.create_dataset(
                f"other/runaway/{name}",
                shape=(nt, nr, nxi, np_ + 1),
                dtype=np.float64,
                fillvalue=fill,
                chunks=True,
            )
        for name in (
            "Ap2",
            "Dxp",
            "Dxx",
            "lnLambda_ee_f2",
            "lnLambda_ei_f2",
            "nu_D_f2",
            "nu_s_f2",
            "nu_par_f2",
            "synchrotron_f2",
        ):
            fill = 1.0 if name in {"Dxx", "synchrotron_f2"} else 0.0
            handle.create_dataset(
                f"other/runaway/{name}",
                shape=(nt, nr, nxi + 1, np_),
                dtype=np.float64,
                fillvalue=fill,
                chunks=True,
            )
        active_auxiliary = {
            "fluid/GammaAva",
            "fluid/runawayRate",
            "fluid/W_re",
            "scalar/energyloss_f_re",
            "scalar/radialloss_f_re",
        }
        for name in REQUIRED_AUXILIARY_QUANTITIES:
            if name in ION_RATE_AUXILIARY_QUANTITIES:
                shape = (nt, 21, nr)
            elif name.startswith("fluid/"):
                shape = (nt, nr)
            else:
                shape = (nt, 1)
            handle.create_dataset(
                f"other/{name}",
                shape=shape,
                dtype=np.float64,
                fillvalue=1.0 if name in active_auxiliary else 0.0,
                chunks=True,
            )


def test_output_inspection_authenticates_complete_real_hdf5(tmp_path: Path) -> None:
    contract = DreamOutputContract("integration", nr=2, np=3, nxi=2, nt=2, final_time_s=0.2)
    output = tmp_path / "output.h5"
    _complete_output(output, contract)

    inspection = inspect_dream_output(output, contract)

    assert inspection["validated"] is True
    assert inspection["sha256"] == sha256_file(output)
    assert inspection["contract"]["resolution"] == "integration"
    assert all(inspection["active_physics"].values())


def test_output_inspection_rejects_missing_and_nonfinite_datasets(tmp_path: Path) -> None:
    contract = DreamOutputContract("integration", nr=2, np=3, nxi=2, nt=2, final_time_s=0.2)
    missing = tmp_path / "missing.h5"
    _complete_output(missing, contract)
    with h5py.File(missing, "a") as handle:
        del handle["other/runaway/Drr"]
    with pytest.raises(ValueError, match="missing DREAM dataset"):
        inspect_dream_output(missing, contract)

    nonfinite = tmp_path / "nonfinite.h5"
    _complete_output(nonfinite, contract)
    with h5py.File(nonfinite, "a") as handle:
        handle["eqsys/n_re"][0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        inspect_dream_output(nonfinite, contract)


def test_atomic_json_round_trip_and_volatile_root_rejection(tmp_path: Path) -> None:
    target = tmp_path / "state" / "campaign.json"
    atomic_write_json(target, {"schema": "test", "value": 1})
    first_digest = sha256_file(target)
    atomic_write_json(target, {"schema": "test", "value": 2})

    assert read_json_object(target) == {"schema": "test", "value": 2}
    assert sha256_file(target) != first_digest
    assert json.loads(target.read_text(encoding="utf-8"))["value"] == 2
    with pytest.raises(ValueError, match="volatile"):
        validate_durable_root(tmp_path)
    with pytest.raises(ValueError, match="regular file"):
        sha256_file(tmp_path)


def test_frozen_contract_and_durable_path_boundaries(tmp_path: Path) -> None:
    veryfine = frozen_output_contract("veryfine")
    superfine = frozen_output_contract("superfine")
    assert (veryfine.nr, veryfine.np, veryfine.nxi, veryfine.nt) == (12, 120, 48, 24)
    assert (superfine.nr, superfine.np, superfine.nxi, superfine.nt) == (14, 140, 56, 28)
    with pytest.raises(ValueError, match="veryfine then superfine"):
        frozen_output_contract("coarse")

    durable_relative = Path("data/external/not-created/deeper")
    assert validate_durable_root(durable_relative) == (Path.cwd() / durable_relative).resolve()
    link = tmp_path / "linked"
    link.symlink_to(tmp_path / "target")
    with pytest.raises(ValueError, match="symlink"):
        validate_durable_root(link / "campaign")


def test_json_boundaries_reject_non_objects_and_clean_failed_temporary_write(
    tmp_path: Path,
) -> None:
    non_object = tmp_path / "list.json"
    non_object.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be an object"):
        read_json_object(non_object)

    target = tmp_path / "state.json"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.partial")
    temporary.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError):
        atomic_write_json(target, {"value": 1})
    assert not temporary.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("commit", "unexpected DREAM commit"),
        ("flag", "expected"),
        ("requested", "diagnostic groups"),
        ("shape", "has shape"),
        ("inactive", "physics dataset is inactive"),
        ("nonnumeric", "non-numeric DREAM dataset"),
        ("time_start", "increasing trajectory"),
        ("time_order", "increasing trajectory"),
        ("final_time", "frozen final time"),
    ],
)
def test_output_inspection_rejects_every_custody_drift(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    contract = DreamOutputContract("integration", nr=2, np=3, nxi=2, nt=2, final_time_s=0.2)
    output = tmp_path / f"{mutation}.h5"
    _complete_output(output, contract)
    with h5py.File(output, "a") as handle:
        if mutation == "commit":
            del handle["code/commit"]
            _text_dataset(handle, "code/commit", "0" * 40)
        elif mutation == "flag":
            handle[next(iter(EXPECTED_FLAGS))][0] = -1
        elif mutation == "requested":
            del handle["settings/other/include"]
            _text_dataset(handle, "settings/other/include", "fluid")
        elif mutation == "shape":
            del handle["other/runaway/Drr"]
            handle.create_dataset("other/runaway/Drr", shape=(1,), dtype=np.float64)
        elif mutation == "inactive":
            del handle["other/runaway/Drr"]
            handle.create_dataset("other/runaway/Drr", shape=(2, 3, 2, 3), dtype=np.float64)
        elif mutation == "nonnumeric":
            del handle["eqsys/n_re"]
            handle.create_dataset("eqsys/n_re", shape=(3, 2), dtype="S1")
        elif mutation == "time_start":
            handle["grid/t"][:] = [0.1, 0.15, 0.2]
        elif mutation == "time_order":
            handle["grid/t"][:] = [0.0, 0.2, 0.1]
        else:
            handle["grid/t"][:] = [0.0, 0.1, 0.3]
    with pytest.raises(ValueError, match=message):
        inspect_dream_output(output, contract)


def test_output_inspection_rejects_non_file(tmp_path: Path) -> None:
    contract = DreamOutputContract("integration", nr=2, np=3, nxi=2, nt=2, final_time_s=0.2)
    with pytest.raises(ValueError, match="regular file"):
        inspect_dream_output(tmp_path / "absent.h5", contract)
