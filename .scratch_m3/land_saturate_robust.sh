#!/usr/bin/env bash
set -euo pipefail
cd /media/anulum/GOTM/aaa_God_of_the_Math_Collection/03_CODE/SCPN-FUSION-CORE
git add tests/test_eped_pb_kbm_miller_geometry.py
git commit -F .scratch_m3/commit_saturate_robust.txt
echo "committed: $(git rev-parse --short HEAD)"
git push origin HEAD:main
echo "pushed origin/main: $(git rev-parse origin/main)"
