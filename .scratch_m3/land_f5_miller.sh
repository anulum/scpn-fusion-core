#!/usr/bin/env bash
set -euo pipefail
cd /media/anulum/GOTM/aaa_God_of_the_Math_Collection/03_CODE/SCPN-FUSION-CORE
git add \
  .gitignore \
  src/scpn_fusion/core/__init__.py \
  src/scpn_fusion/core/eped_pb_kbm.py \
  src/scpn_fusion/core/ballooning_second_stability.py \
  validation/ballooning_reference_runner.py \
  tests/test_ballooning_reference_runner.py \
  tests/test_ballooning_second_stability.py \
  tests/test_eped_pb_kbm_miller_geometry.py
git commit -F .scratch_m3/commit_f5_miller.txt
echo "committed: $(git rev-parse --short HEAD)"
echo "origin/main before: $(git rev-parse origin/main)"
git push origin HEAD:main
echo "origin/main after: $(git rev-parse origin/main)"
