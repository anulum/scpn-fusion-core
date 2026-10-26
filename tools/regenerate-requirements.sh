#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
# Regenerate hash-pinned lock files from requirements/*.in
# Requires: uv 0.12.21 (pinned — see UV_PINNED below; https://docs.astral.sh/uv/)
#
# REGEN_UPGRADE=1 moves every lock to the newest versions allowed by its .in
# constraints (whole dependency chain); without it existing pins are kept.
set -euo pipefail

cd "$(dirname "$0")/.."

# ── Pinned uv version ─────────────────────────────────────────────────────────
# The committed locks are only bit-reproducible under a fixed uv: different uv
# releases can change universal-marker emission and dependency resolution, which
# silently churns the locks. Regenerate ONLY with this version so a fresh regen is
# a no-op against the committed files. Override deliberately (e.g. a vetted newer
# uv) with UV_VERSION_OVERRIDE=1.
readonly UV_PINNED="0.12.21"
uv_have="$(uv --version 2>/dev/null | awk '{print $2}')"
if [ "${uv_have}" != "${UV_PINNED}" ] && [ "${UV_VERSION_OVERRIDE:-0}" != "1" ]; then
    echo "ERROR: locks are pinned to uv ${UV_PINNED} but 'uv --version' is '${uv_have:-<none>}'." >&2
    echo "       Install it (e.g. 'uv self update ${UV_PINNED}' or 'pipx install uv==${UV_PINNED}')." >&2
    echo "       To override intentionally: UV_VERSION_OVERRIDE=1 $0" >&2
    exit 1
fi

# ci.in is special: it produces three per-Python-version Linux lock files
# (ci-py311/py312.txt), driven here rather than by a single header line.
upgrade_flags=()
if [ "${REGEN_UPGRADE:-0}" = "1" ]; then
    upgrade_flags=(--upgrade)
fi

# Closures that are not produced by uv from their .in file. Each is refreshed by
# hand from its upstream source and is audited by the dependency-policy workflow:
#   cfspopcon / process — observed runtime closures of pinned upstream sources;
#   gpu-bootstrap       — hash-locked installer bootstrap (pip, setuptools, wheel).
manual_closures=("cfspopcon" "process" "gpu-bootstrap")

compiled=0
ci_versions=("3.11" "3.12")
for pyver in "${ci_versions[@]}"; do
    pytag="${pyver//.}"
    outfile="requirements/ci-py${pytag}.txt"
    echo "Compiling requirements/ci.in -> ${outfile}  (Python ${pyver}, linux)"
    uv pip compile requirements/ci.in \
        --generate-hashes \
        --python-version "${pyver}" \
        --python-platform linux \
        "${upgrade_flags[@]}" \
        -o "${outfile}"
    compiled=$((compiled + 1))
done

# Every other *.in carries its EXACT canonical command in a "# Regenerate: uv pip
# compile ..." header line — the single source of truth for its per-file Python
# version / platform / universal flags (these genuinely differ: build/ci-interop/
# ci-stress are 3.11-linux, ci-benchmark/docs/studio are 3.12-linux, full/minimal
# are --universal with no pinned version). Execute that header command verbatim so
# the script can never drift from it — the drift that previously mis-locked
# ci-interop / ci-stress / build against 3.12 when their headers require 3.11.
for infile in requirements/*.in; do
    base="$(basename "${infile}" .in)"
    [ "${base}" = "ci" ] && continue  # handled above
    if [[ " ${manual_closures[*]} " == *" ${base} "* ]]; then
        echo "Skipping ${infile}: manually maintained closure"
        continue
    fi
    cmd="$(sed -n 's/^# Regenerate: \(uv pip compile .*\)$/\1/p' "${infile}" | head -1)"
    if [ -z "${cmd}" ]; then
        echo "ERROR: ${infile} lacks a '# Regenerate: uv pip compile ...' header line" >&2
        exit 1
    fi
    echo "Compiling ${infile}  (per its header: ${cmd})"
    eval "${cmd} ${upgrade_flags[*]:-}"
    compiled=$((compiled + 1))
done

echo "Done. ${compiled} lock files compiled; manual closures not touched: ${manual_closures[*]}."
