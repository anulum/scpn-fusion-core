#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
# Regenerate hash-pinned lock files from requirements/*.in
# Requires: uv 0.11.16 (pinned — see UV_PINNED below; https://docs.astral.sh/uv/)
set -euo pipefail

cd "$(dirname "$0")/.."

# ── Pinned uv version ─────────────────────────────────────────────────────────
# The committed locks are only bit-reproducible under a fixed uv: different uv
# releases can change universal-marker emission and dependency resolution, which
# silently churns the locks. Regenerate ONLY with this version so a fresh regen is
# a no-op against the committed files. Override deliberately (e.g. a vetted newer
# uv) with UV_VERSION_OVERRIDE=1.
readonly UV_PINNED="0.11.16"
uv_have="$(uv --version 2>/dev/null | awk '{print $2}')"
if [ "${uv_have}" != "${UV_PINNED}" ] && [ "${UV_VERSION_OVERRIDE:-0}" != "1" ]; then
    echo "ERROR: locks are pinned to uv ${UV_PINNED} but 'uv --version' is '${uv_have:-<none>}'." >&2
    echo "       Install it (e.g. 'uv self update ${UV_PINNED}' or 'pipx install uv==${UV_PINNED}')." >&2
    echo "       To override intentionally: UV_VERSION_OVERRIDE=1 $0" >&2
    exit 1
fi

# ci.in is special: it produces three per-Python-version Linux lock files
# (ci-py310/py311/py312.txt), driven here rather than by a single header line.
ci_versions=("3.10" "3.11" "3.12")
for pyver in "${ci_versions[@]}"; do
    pytag="${pyver//.}"
    outfile="requirements/ci-py${pytag}.txt"
    echo "Compiling requirements/ci.in -> ${outfile}  (Python ${pyver}, linux)"
    uv pip compile requirements/ci.in \
        --generate-hashes \
        --python-version "${pyver}" \
        --python-platform linux \
        -o "${outfile}"
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
    cmd="$(sed -n 's/^# Regenerate: \(uv pip compile .*\)$/\1/p' "${infile}" | head -1)"
    if [ -z "${cmd}" ]; then
        echo "ERROR: ${infile} lacks a '# Regenerate: uv pip compile ...' header line" >&2
        exit 1
    fi
    echo "Compiling ${infile}  (per its header: ${cmd})"
    eval "${cmd}"
done

locks=(requirements/*.txt)
echo "Done. ${#locks[@]} lock files regenerated."
