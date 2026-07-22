#!/usr/bin/env bash
set -euo pipefail
cd /media/anulum/GOTM/aaa_God_of_the_Math_Collection/03_CODE/SCPN-FUSION-CORE
git add tests/test_eped_pb_kbm.py tests/test_eped_pb_kbm_miller_geometry.py
git commit -F .scratch_m3/commit_mypy_fix.txt
echo "committed: $(git rev-parse --short HEAD)"
echo "origin/main before: $(git rev-parse origin/main)"
git push origin HEAD:main
echo "origin/main after: $(git rev-parse origin/main)"
