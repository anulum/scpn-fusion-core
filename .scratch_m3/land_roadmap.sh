#!/usr/bin/env bash
set -euo pipefail
cd /media/anulum/GOTM/aaa_God_of_the_Math_Collection/03_CODE/SCPN-FUSION-CORE
git add docs/internal/roadmap_reconciliation_2026-07-15.md
git commit -F .scratch_m3/commit_roadmap.txt
echo "committed: $(git rev-parse --short HEAD)"
git push origin HEAD:main
echo "origin/main after: $(git rev-parse origin/main)"
