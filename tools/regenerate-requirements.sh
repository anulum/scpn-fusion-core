#!/usr/bin/env bash
# Regenerate hash-pinned lock files from requirements/*.in
# Requires: uv (https://docs.astral.sh/uv/)
set -euo pipefail

cd "$(dirname "$0")/.."

# ci.in gets per-Python-version lock files (Linux platform)
ci_versions=("3.9" "3.10" "3.11" "3.12")
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

# All other .in files: single lock file each (Python 3.12, linux)
for infile in requirements/*.in; do
    base="$(basename "${infile}" .in)"
    [ "${base}" = "ci" ] && continue  # handled above
    outfile="${infile%.in}.txt"
    echo "Compiling ${infile} -> ${outfile}  (Python 3.12, linux)"
    uv pip compile "${infile}" \
        --generate-hashes \
        --python-version "3.12" \
        --python-platform linux \
        -o "${outfile}"
done

echo "Done. $(ls requirements/*.txt | wc -l) lock files regenerated."
