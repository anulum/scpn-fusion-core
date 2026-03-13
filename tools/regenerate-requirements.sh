#!/usr/bin/env bash
# Regenerate hash-pinned lock files from requirements/*.in
# Requires: uv (https://docs.astral.sh/uv/)
set -euo pipefail

cd "$(dirname "$0")/.."

for infile in requirements/*.in; do
    outfile="${infile%.in}.txt"
    echo "Compiling $infile -> $outfile"
    uv pip compile "$infile" --generate-hashes --universal -o "$outfile"
done

echo "Done. $(ls requirements/*.txt | wc -l) lock files regenerated."
