#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance

set -euo pipefail

# Build every review manuscript from a disposable copy. The repository's final
# PDFs and ignored LaTeX auxiliaries are never used as build inputs or modified.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly REPO_ROOT
readonly SUBMISSIONS_DIR="${REPO_ROOT}/papers/submissions"
readonly PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
readonly PAPER_VERIFY_PREFIX="${TMPDIR:-/tmp}/scpn-paper-verify."
readonly PAPER_SOURCE_DATE_EPOCH="1787756047"

require_command() {
    local command_name="$1"
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Error: required command is unavailable: ${command_name}" >&2
        exit 1
    fi
}

cleanup() {
    if [[ -n "${PAPER_VERIFY_ROOT:-}" && -d "${PAPER_VERIFY_ROOT}" ]]; then
        case "${PAPER_VERIFY_ROOT}" in
            "${PAPER_VERIFY_PREFIX}"*) rm -rf -- "${PAPER_VERIFY_ROOT}" ;;
            *) echo "Error: refusing to remove unexpected path: ${PAPER_VERIFY_ROOT}" >&2 ;;
        esac
    fi
}

verify_pdf() {
    local pdf_path="$1"
    local text_path="$2"

    if [[ ! -s "${pdf_path}" ]]; then
        echo "Error: build did not produce a non-empty PDF: ${pdf_path}" >&2
        return 1
    fi

    if pdffonts "${pdf_path}" | awk 'NR > 2 && $1 != "" && $6 != "yes" { exit 1 }'; then
        :
    else
        echo "Error: PDF contains a font that is not embedded: ${pdf_path}" >&2
        return 1
    fi

    pdftotext "${pdf_path}" "${text_path}"
    if [[ ! -s "${text_path}" ]]; then
        echo "Error: PDF has no extractable text: ${pdf_path}" >&2
        return 1
    fi
}

verify_evidence_manifest() {
    local evidence_dir="$1"
    local repository_revision="$2"
    local manifest_path="${evidence_dir}/evidence_manifest.json"
    local evidence_name
    local generator_name
    local expected_sha256
    local actual_sha256

    [[ -f "${manifest_path}" ]] || return 0
    if ! jq -e --arg revision "${repository_revision}" \
        '.repository_revision == $revision' "${manifest_path}" >/dev/null; then
        echo "Error: evidence manifest revision differs from submission metadata." >&2
        return 1
    fi
    while IFS=$'\t' read -r evidence_name expected_sha256; do
        if [[ ! -f "${evidence_dir}/${evidence_name}" ]]; then
            echo "Error: evidence manifest references a missing file: ${evidence_name}" >&2
            return 1
        fi
        actual_sha256="$(sha256sum -- "${evidence_dir}/${evidence_name}" | awk '{print $1}')"
        if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
            echo "Error: evidence SHA-256 mismatch: ${evidence_name}" >&2
            return 1
        fi
    done < <(jq -r '.files | to_entries[] | [.key, .value.sha256] | @tsv' "${manifest_path}")
    while IFS=$'\t' read -r generator_name expected_sha256; do
        if [[ ! -f "${REPO_ROOT}/${generator_name}" ]]; then
            echo "Error: evidence manifest references a missing generator: ${generator_name}" >&2
            return 1
        fi
        actual_sha256="$(sha256sum -- "${REPO_ROOT}/${generator_name}" | awk '{print $1}')"
        if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
            echo "Error: generator SHA-256 mismatch: ${generator_name}" >&2
            return 1
        fi
    done < <(jq -r '(.generators // {}) | to_entries[] | [.key, .value.sha256] | @tsv' \
        "${manifest_path}")
}

verify_public_headers() {
    local source_dir="$1"
    local file_path

    while IFS= read -r -d '' file_path; do
        case "${file_path}" in
            *.json)
                if ! jq -e '
                    .["SPDX-License-Identifier"] == "AGPL-3.0-or-later" and
                    .commercialLicense == "available" and
                    (.conceptsCopyright | type == "string") and
                    (.codeCopyright | type == "string") and
                    .orcid == "0009-0009-3560-0851" and
                    .contact == "www.anulum.li | protoscience@anulum.li" and
                    (.projectDescription | type == "string" and length > 0)
                ' "${file_path}" >/dev/null; then
                    echo "Error: JSON lacks the Tier-0 structured header: ${file_path}" >&2
                    return 1
                fi
                ;;
            *)
                if ! head -n 9 -- "${file_path}" | grep -Fq \
                    'SPDX-License-Identifier: AGPL-3.0-or-later'; then
                    echo "Error: text file lacks the Tier-0 header: ${file_path}" >&2
                    return 1
                fi
                ;;
        esac
    done < <(
        find "${source_dir}" -type f \
            \( -name '*.md' -o -name '*.tex' -o -name '*.bib' \
            -o -name '*.cff' -o -name '*.py' -o -name '*.sh' -o -name '*.json' \) \
            -print0
    )
}

verify_submission_metadata() {
    local source_dir="$1"
    local package_name="$2"
    local metadata_path="${source_dir}/submission_metadata.json"
    local evidence_path
    local repository_revision

    if ! jq -e \
        --arg catalogue_id "${package_name:0:3}" '
            .catalogue_id == $catalogue_id and
            .status == "review_draft_not_submitted" and
            (.repository_revision | type == "string" and test("^[0-9a-f]{40}$")) and
            .author.orcid == "0009-0009-3560-0851" and
            .doi == null and
            .manuscript_content_license == null
        ' "${metadata_path}" >/dev/null; then
        echo "Error: submission metadata violates the review-package contract: ${package_name}" >&2
        return 1
    fi
    repository_revision="$(jq -r '.repository_revision' "${metadata_path}")"
    if ! git -C "${REPO_ROOT}" cat-file -e "${repository_revision}^{commit}" 2>/dev/null; then
        echo "Error: submission metadata references an unavailable commit: ${package_name}" >&2
        return 1
    fi
    if ! git -C "${REPO_ROOT}" merge-base --is-ancestor \
        "${repository_revision}" HEAD; then
        echo "Error: submission evidence revision is not an ancestor of HEAD: ${package_name}" >&2
        return 1
    fi
    if ! grep -Fq "${repository_revision:0:12}" "${source_dir}/manuscript.tex"; then
        echo "Error: manuscript text does not name its metadata revision: ${package_name}" >&2
        return 1
    fi

    while IFS= read -r evidence_path; do
        if [[ ! -f "${source_dir}/${evidence_path}" ]]; then
            echo "Error: submission metadata references a missing file: ${evidence_path}" >&2
            return 1
        fi
    done < <(jq -r '.evidence_files[]?' "${metadata_path}")
    printf '%s\n' "${repository_revision}"
}

compare_generated_artifacts() {
    local source_dir="$1"
    local scratch_dir="$2"
    local generated_path
    local relative_path

    cmp -- "${source_dir}/manuscript.pdf" "${scratch_dir}/manuscript.pdf"
    if [[ ! -d "${scratch_dir}/figures" ]]; then
        return 0
    fi
    while IFS= read -r -d '' generated_path; do
        relative_path="${generated_path#"${scratch_dir}"/}"
        if [[ ! -f "${source_dir}/${relative_path}" ]]; then
            echo "Error: generated figure is not committed: ${relative_path}" >&2
            return 1
        fi
        cmp -- "${source_dir}/${relative_path}" "${generated_path}"
    done < <(
        find "${scratch_dir}/figures" -maxdepth 1 -type f \
            \( -name '*.pdf' -o -name '*.png' \) -print0
    )

    if [[ ! -d "${scratch_dir}/evidence" ]]; then
        return 0
    fi
    while IFS= read -r -d '' generated_path; do
        relative_path="${generated_path#"${scratch_dir}"/}"
        if [[ ! -f "${source_dir}/${relative_path}" ]]; then
            echo "Error: generated evidence is not committed: ${relative_path}" >&2
            return 1
        fi
        cmp -- "${source_dir}/${relative_path}" "${generated_path}"
    done < <(
        find "${scratch_dir}/evidence" -maxdepth 1 -type f \
            \( -name '*.json' -o -name '*.tex' \) -print0
    )
}

build_submission() {
    local source_dir="$1"
    local package_name
    local repository_revision
    local scratch_dir

    package_name="$(basename -- "${source_dir}")"
    scratch_dir="${PAPER_VERIFY_ROOT}/papers/submissions/${package_name}"
    verify_public_headers "${source_dir}"
    repository_revision="$(verify_submission_metadata "${source_dir}" "${package_name}")"
    cp -a -- "${source_dir}" "${scratch_dir}"

    find "${scratch_dir}" -maxdepth 1 -type f \
        \( -name 'manuscript.pdf' -o -name 'manuscript.aux' \
        -o -name 'manuscript.bbl' -o -name 'manuscript.blg' \
        -o -name 'manuscript.log' -o -name 'manuscript.out' \) -delete
    if [[ -d "${scratch_dir}/figures" ]]; then
        find "${scratch_dir}/figures" -maxdepth 1 -type f \
            \( -name '*.pdf' -o -name '*.png' \) -delete
    fi

    (
        export SOURCE_DATE_EPOCH="${PAPER_SOURCE_DATE_EPOCH}"
        export FORCE_SOURCE_DATE=1
        export TZ=UTC
        cd -- "${scratch_dir}"
        if [[ -f figures/generate_figures.py ]]; then
            PYTHONPATH=../../../src ../../../.venv/bin/python figures/generate_figures.py
        fi
        if [[ -f generate_evidence_manifest.py ]]; then
            PYTHONPATH=../../../src ../../../.venv/bin/python generate_evidence_manifest.py
        fi
        pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex >/dev/null
        bibtex manuscript >/dev/null
        pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex >/dev/null
        pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex >/dev/null
        pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex >/dev/null

        if rg -n 'LaTeX Warning|Package .* Warning|Overfull|Underfull|undefined|multiply defined|Error' manuscript.log; then
            echo "Error: LaTeX log contains a warning or error: ${package_name}" >&2
            return 1
        fi

        cffconvert --validate -i CITATION.cff >/dev/null
        jq empty submission_metadata.json
        if [[ -d evidence ]]; then
            find evidence -maxdepth 1 -type f -name '*.json' -print0 \
                | xargs -0 --no-run-if-empty -n 1 jq empty
            verify_evidence_manifest evidence "${repository_revision}"
        fi
        verify_pdf manuscript.pdf manuscript.txt
        compare_generated_artifacts "${source_dir}" "${scratch_dir}"
    )

    echo "[OK] ${package_name}"
}

main() {
    local submission_dir
    local package_name
    local found=0

    for command_name in bibtex cffconvert git jq pdffonts pdflatex pdftotext rg sha256sum; do
        require_command "${command_name}"
    done
    if [[ ! -x "${PYTHON_BIN}" ]]; then
        echo "Error: project Python is unavailable: ${PYTHON_BIN}" >&2
        exit 1
    fi
    "${PYTHON_BIN}" "${SCRIPT_DIR}/generate_legacy_layout_manifest.py" --check

    PAPER_VERIFY_ROOT="$(mktemp -d "${PAPER_VERIFY_PREFIX}XXXXXX")"
    export PAPER_VERIFY_ROOT
    trap cleanup EXIT INT TERM

    mkdir -p -- "${PAPER_VERIFY_ROOT}/papers/submissions"
    ln -s -- "${REPO_ROOT}/src" "${PAPER_VERIFY_ROOT}/src"
    ln -s -- "${REPO_ROOT}/.venv" "${PAPER_VERIFY_ROOT}/.venv"
    ln -s -- "${REPO_ROOT}/validation" "${PAPER_VERIFY_ROOT}/validation"

    if [[ "$#" -gt 0 ]]; then
        for package_name in "$@"; do
            if [[ ! "${package_name}" =~ ^[0-9]{3}_[a-z0-9_]+$ ]]; then
                echo "Error: invalid package name: ${package_name}" >&2
                exit 1
            fi
            submission_dir="${SUBMISSIONS_DIR}/${package_name}"
            if [[ ! -d "${submission_dir}" ]]; then
                echo "Error: submission package does not exist: ${package_name}" >&2
                exit 1
            fi
            found=1
            build_submission "${submission_dir}"
        done
    else
        while IFS= read -r -d '' submission_dir; do
            found=1
            build_submission "${submission_dir}"
        done < <(find "${SUBMISSIONS_DIR}" -mindepth 1 -maxdepth 1 -type d \
            -name '[0-9][0-9][0-9]_*' -print0 | sort -z)
    fi

    if [[ "${found}" -eq 0 ]]; then
        echo "Error: no numbered submission packages found." >&2
        exit 1
    fi

    echo "All submission packages rebuilt successfully in disposable custody."
}

main "$@"
