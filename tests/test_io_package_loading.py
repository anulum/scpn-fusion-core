# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Real IO Package Loading and Storage Regression
"""Exercise bounded files and deferred public adapters in a fresh interpreter."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from coverage import Coverage


def test_bounded_json_then_original_public_ids_storage(tmp_path: Path) -> None:
    """Keep optional bridges out of bounded reads, then use the real IDS exports.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Real isolated directory for both bounded JSON and IDS round trips.

    """
    active_coverage = Coverage.current()
    coverage_options = (
        None
        if active_coverage is None
        else {
            name: active_coverage.get_option(f"run:{name}")
            for name in ("data_file", "branch", "source", "include", "omit")
        }
    )
    program = r"""
import atexit
import json
import sys
from pathlib import Path

from coverage import Coverage

options = json.loads(sys.argv[2])
if options is not None and Coverage.current() is None:
    child_coverage = Coverage(config_file=False, data_suffix=True, **options)
    child_coverage.start()

    def finish_coverage():
        child_coverage.stop()
        child_coverage.save()

    atexit.register(finish_coverage)

from scpn_fusion.io.safe_loaders import checked_json_load
import scpn_fusion.io as io

root = Path(sys.argv[1])
source = root / "bounded.json"
source.write_text('{"plasma_current":1000000}', encoding="utf-8")
assert checked_json_load(source, max_bytes=128) == {"plasma_current": 1000000}
assert "omas" not in sys.modules
assert "scpn_fusion.io.imas_connector" not in sys.modules
assert "read_ids" in dir(io) and "HAS_OMAS" in dir(io)
assert "omas" not in sys.modules

storage = io.imas_connector_storage
assert "omas" not in sys.modules
payload = {"ids_properties": {"homogeneous_time": 1}, "shot": 170001}
target = root / "ids.json"
io.write_ids(payload, target)
assert io.read_ids(target) == payload
assert io.write_ids is storage.write_ids
assert io.read_ids is storage.read_ids
if io.HAS_OMAS:
    assert "omas" in sys.modules

exports = {}
exec("from scpn_fusion.io import *", exports)
assert exports["read_ids"](target) == payload
assert exports["write_ids"] is io.write_ids
assert exports["imas_connector_storage"] is storage
assert exports["CANONICAL_COCOS"] == io.imas_connector.CANONICAL_COCOS
try:
    getattr(io, "missing_public_adapter")
except AttributeError:
    pass
else:
    raise AssertionError("Unknown package attribute was accepted")
print(json.dumps({"bounded_read": True, "ids_roundtrip": True, "legacy_exports": True}))
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path), json.dumps(coverage_options)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "bounded_read": True,
        "ids_roundtrip": True,
        "legacy_exports": True,
    }
